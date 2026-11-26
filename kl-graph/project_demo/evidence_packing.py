"""Deterministic, source-addressed evidence packing for the B/F/G experiment."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

from .answer import _prompt, _terms, evidence_view
from .context import project_scope

WINDOW = 50
RANK_CONSTANT = 20


class Span(NamedTuple):
    source: str
    start: int
    end: int


class Hop(NamedTuple):
    origin: str
    relation: str
    target: str
    witness: tuple[str, ...]

    def as_reference(self):
        result = dict(origin=self.origin, relation=self.relation, target=self.target)
        if self.relation == 'shared_ABOUT':
            left, entity, name, right = self.witness
            result.update(shared_entity=name, entity_id=entity, origin_fact=left, target_fact=right)
        elif self.witness:
            result['matched_reference'] = list(self.witness)
        return result


@dataclass(frozen=True)
class PathEvidence:
    hops: tuple[Hop, ...]
    spans: tuple[Span, ...]
    # A fact at the destination can require a quoted reply parent of its own.
    target_support: tuple[Span, ...] = ()

    @property
    def target(self):
        return self.hops[-1].target

    def validates(self, source):
        endpoints = {s.source for s in self.spans}
        return bool(self.hops and self.target == source and all(
            h.origin in endpoints and h.target in endpoints
            and (i == 0 or self.hops[i-1].target == h.origin)
            for i, h in enumerate(self.hops)
        ))

    def target_excerpt(self):
        if self.target_support:
            return self.target_support
        if self.hops[-1].relation == "shared_ABOUT":
            # Older snapshots did not distinguish the destination fact's reply
            # dependencies from the other witnesses. Preserve them together.
            return self.spans
        return tuple(s for s in self.spans if s.source == self.target)


@dataclass
class Snapshot:
    originals: dict[str, tuple[Span, ...]]
    facts: dict[str, tuple[Span, ...]]
    paths: dict[str, PathEvidence]
    facets: tuple[str, ...] = ()
    policy: str = "rank"


def merge_spans(spans):
    merged = []
    for span in sorted(set(spans)):
        if merged and merged[-1].source == span.source and merged[-1].end >= span.start:
            prev = merged.pop()
            span = Span(span.source, prev.start, max(prev.end, span.end))
        merged.append(span)
    return tuple(merged)


@dataclass(frozen=True)
class Candidate:
    source: str
    ranks: tuple[int | None, ...]
    spans: tuple[Span, ...]
    path: PathEvidence | None = None
    kind: str = "source"

    @property
    def score(self):
        return sum(Fraction(1, RANK_CONSTANT + rank) for rank in self.ranks if rank)


def candidates(mode, snapshot, *, graph_delivery="bundled"):
    if graph_delivery not in ("bundled", "separate"):
        raise ValueError(f"未知图谱交付方式 {graph_delivery}")
    lanes = [snapshot.originals, snapshot.facts if mode in ('F', 'G') else {},
             {k: p for k, p in snapshot.paths.items() if p.validates(k)} if mode == 'G' else {}]
    ranks = [{key: i for i, key in enumerate(list(lane)[:WINDOW], 1)} for lane in lanes]
    entries = []
    for key in set().union(*ranks):
        o, t, h = (rank.get(key) for rank in ranks)
        fact_spans = merge_spans(snapshot.facts[key]) if t else ()
        direct = merge_spans([*(snapshot.originals[key] if o else ()), *fact_spans])
        path = snapshot.paths[key] if h else None
        if path and graph_delivery == "separate":
            # Each vote buys its own supporting text, even when two lanes
            # matched different paragraphs in the same destination record.
            target = merge_spans(path.target_excerpt())
            combined = merge_spans((*direct, *target))
            entries.append(Candidate(key, (o, t, h), combined))
            if direct and direct != combined:
                entries.append(Candidate(key, (o, t, None), direct))
            if target != combined:
                entries.append(Candidate(key, (None, None, h), target, kind="graph_target"))
            # Only H ranks the relationship. All witnesses travel together and
            # pack() charges only for spans not already in the final input.
            entries.append(Candidate(key, (None, None, h), path.spans, path, kind="path"))
        elif direct:
            entries.append(Candidate(key, (o, t, None), direct))
        if o and fact_spans and fact_spans != direct:
            # Schedule at T's own rank, including the complete reply support.
            entries.append(Candidate(key, (None, t, None), fact_spans, kind="fact_fallback"))
        if path and graph_delivery == "bundled":
            entries.append(Candidate(key, (o, t, h), merge_spans((*direct, *path.spans)), path, kind="path"))
    return sorted(entries, key=lambda c: (-c.score, min(r for r in c.ranks if r), c.source))


async def pack(model, mode, question, project, cutoff, records, snapshot, *, graph_delivery="bundled"):
    if mode not in ('B', 'F', 'G'):
        raise ValueError(f'未知实验方案 {mode}')
    limit = min(model.context_window - model.max_output, model.evidence_budget)
    rows = {key: dict(evidence_view(row), project_scope=project_scope(row, project))
            for key, row in records.items()}

    def render(intervals, paths):
        evidence = []
        for key, spans in intervals.items():
            row = rows[key].copy()
            body = row['content']
            row['content'] = '\n[…]\n'.join(body[s.start:s.end] for s in spans)
            row['source_ranges'] = [(s.start, s.end) for s in spans]
            if key in paths:
                row['paths'] = paths[key]
            evidence.append(row)
        return _prompt(question, project, cutoff, evidence, None, compact=True)

    intervals, paths = {}, {}
    queue = candidates(mode, snapshot, graph_delivery=graph_delivery)
    # Measure redundancy on the actual target excerpt. A shared path origin
    # or reply dependency is necessary context, not a second topical vote.
    def features(entry):
        body = ' '.join(rows[s.source]['content'][s.start:s.end] for s in entry.spans
                        if s.source == entry.source)
        terms = _terms(body)
        shingles = set(zip(terms, terms[1:], terms[2:]))
        title = rows[entry.source]['title']
        facets = {name for name in snapshot.facets if name in body or name in title}
        return shingles, facets

    diversify = snapshot.policy == "facet_novelty"
    features_by_entry = {entry: features(entry) for entry in queue} if diversify else {}
    initial_order = {entry: i for i, entry in enumerate(queue)}
    overlap = dict.fromkeys(queue, 0.0)
    represented = defaultdict(set)

    def utility(entry):
        _, facets = features_by_entry[entry]
        gain = max((1 / (1 + len(represented[name])) for name in facets), default=0)
        return float(entry.score) * (1 + gain) * (1 - overlap[entry] / 2)

    messages = render(intervals, paths)
    tokens = await model.count(messages)
    if tokens > limit:
        raise ValueError('固定提示与问句已超过输入预算')
    trace = []
    over_budget_unions = set()
    while queue:
        if diversify:
            queue.sort(key=lambda entry: (-utility(entry), initial_order[entry]))
        deferred = []
        waiting = []
        for index, entry in enumerate(queue):
            if entry.kind == "fact_fallback" and entry.source not in over_budget_unions:
                waiting.append(entry)
                continue
            additions = defaultdict(list)
            for span in entry.spans:
                if not 0 <= span.start < span.end <= len(rows[span.source]['content']):
                    raise ValueError(f'无效的原文区间 {span}')
                additions[span.source].append(span)
            trial = intervals.copy()
            for key, spans in additions.items():
                trial[key] = merge_spans((*trial.get(key, ()), *spans))
            trial_paths = paths.copy()
            if entry.path:
                witnesses = [hop.as_reference() for hop in entry.path.hops]
                trial_paths[entry.source] = witnesses
            audit = dict(source=entry.source, ranks=entry.ranks, score=str(entry.score),
                         kind=entry.kind)
            if diversify:
                audit.update(marginal_score=round(utility(entry), 6), facets=sorted(features_by_entry[entry][1]))
            if trial == intervals and trial_paths == paths:
                trace.append(dict(audit, result='covered'))
                continue
            proposed = render(trial, trial_paths)
            cost = await model.count(proposed)
            if cost > limit:
                if entry.kind == "source" and all(entry.ranks[:2]):
                    over_budget_unions.add(entry.source)
                deferred.append(entry)
                trace.append(dict(audit, result='over_budget', incremental_tokens=cost-tokens))
                continue
            if entry.kind == "source":
                over_budget_unions.discard(entry.source)
            trace.append(dict(audit, result='included', incremental_tokens=cost-tokens,
                              reused_sources=sorted(additions.keys() & intervals.keys())))
            intervals, paths, messages, tokens = trial, trial_paths, proposed, cost
            if not diversify:
                if deferred:
                    queue = deferred + queue[index + 1:]
                    break
                continue
            words, matched = features_by_entry[entry]
            for name in matched:
                represented[name].add(entry.source)
            queue = deferred + waiting + queue[index + 1:]
            for other in queue:
                other_words = features_by_entry[other][0]
                union_size = len(words) + len(other_words)
                common = len(words & other_words)
                if common:
                    overlap[other] = max(overlap[other], common / (union_size - common))
            break
        else:
            # A deferred union may unlock a shorter fact seen earlier in this pass.
            queue = [item for item in waiting if item.source in over_budget_unions]
    fingerprint = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return dict(messages=messages, source_ids=list(intervals), intervals=intervals, paths=paths,
                input_tokens=tokens, input_limit=limit, count_method=model.count_method,
                input_sha256=fingerprint, decisions=trace)
