"""Collect the three independent evidence rankings for a frozen corpus."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from contextlib import closing

import httpx
import numpy as np

from kl_graph.query.associated_sources import entity_name_pattern
from kl_graph.query.fact_evidence import fact_support, grounded_graph, qualified_spans, source_facts
from kl_graph.utils.helpers import litellm_base_url
from kl_graph.query.source_graph import expand_source_links, read_graph
from .answer import _terms, fuse_originals, rank_originals
from .context import ContextUnavailable, _fact_scopes, query_original_vectors, visible_records
from .evidence_packing import Hop, PathEvidence, Snapshot, Span, WINDOW, merge_spans

from .path_ranking import rank_light_paths
from .query_facets import combine_queries, plan_facets

FACT_DEPTH = 200
ENTITY_DEPTH = 10
SEED_LIMIT = 16
DEGREE_LIMIT = 64


def direct_excerpt(question_terms, key, records):
    """Use a complete matching paragraph; keep a single-paragraph record intact."""
    body = records[key]['content']
    paragraphs = [p for p in re.finditer(r'[^\n]+(?:\n|$)', body) if p[0].strip()]
    scores = [len(question_terms.intersection(_terms(p[0]))) for p in paragraphs]
    best = max(scores, default=0)
    if len(body) < 160 or not best:
        return (Span(key, 0, len(body)),)
    spans = qualified_spans(paragraphs, [i for i, score in enumerate(scores) if score == best])
    return tuple(Span(key, s['start'], s['end']) for s in spans)


async def search(client, collection, question, depth, id_field):
    response = await client.post('/search', json=dict(collection=collection, query=question, top_k=depth))
    response.raise_for_status()
    data = response.json()
    if data.get('collection') != collection or not isinstance(data.get('results'), list):
        raise ContextUnavailable(f'{collection} 向量查询响应格式无效')
    scores = {}
    for row in data['results']:
        key, score = row['id'], row['score']
        if not isinstance(key, str) or not key or key != row['payload'].get(id_field) or not math.isfinite(score):
            raise ContextUnavailable(f'{collection} 向量结果缺少有效分数或出处标识')
        scores[key] = max(score, scores.get(key, -math.inf))
    return sorted(scores, key=lambda key: (-scores[key], key))


def supported_spans(support, records):
    if support['status'] in ('unavailable', 'incomplete_context'):
        return ()
    spans = []
    for q in support['quotes']:
        key, start, end = q['unit_id'], q['start'], q['end']
        if key not in records or records[key]['content'][start:end] != q['quote'] or end <= start:
            return ()
        spans.append(Span(key, start, end))
    return tuple(spans)


def fact_lane(by_id, hit_ids, eligible):
    spans, witnesses = defaultdict(list), defaultdict(list)
    for fid in hit_ids:
        if fid not in by_id:
            continue
        fact = by_id[fid]
        key = fact['evidence_ids'][0]
        support = fact_support(fact['text'], key, eligible.get, fact['source_quote'])
        found = supported_spans(support, eligible)
        if found:
            spans[key].extend(found)
            witnesses[key].append(fid)
    return {key: merge_spans(items) for key, items in spans.items()}, dict(witnesses)


def graph_paths(conn, records, scopes, entity_hits, seeds, excerpts):
    """Walk native ABOUT edges and explicit source links; keep their actual witnesses."""
    local = {key: records[key] for key in sorted(scopes) if scopes[key] != 'comparison'}
    seed_rank = {key: i for i, key in enumerate(seeds) if key in local}
    entity_ids = set(entity_hits)
    anchor_facts = [f['id'] for f in source_facts(conn, local, seed_rank)]
    for start in range(0, len(anchor_facts), 400):
        batch = anchor_facts[start:start + 400]
        marks = ','.join('?' for _ in batch)
        entity_ids.update(row[0] for row in conn.execute(
            f"SELECT DISTINCT target_id FROM edges WHERE edge_type='ABOUT' AND source_type='fact' "
            f"AND target_type='entity' AND source_id IN ({marks})", batch))
    members = {}
    for eid in sorted(entity_ids):
        fids = [row[0] for row in conn.execute(
            "SELECT DISTINCT source_id FROM edges WHERE target_type='entity' AND target_id=? "
            "AND source_type='fact' AND edge_type='ABOUT' ORDER BY source_id LIMIT ?", (eid, DEGREE_LIMIT + 1))]
        if 0 < len(fids) <= DEGREE_LIMIT:
            members['native:' + eid] = fids
    selected = sorted({fid for fids in members.values() for fid in fids})
    entities, rows, _, _ = read_graph(conn, local, {k: scopes[k] for k in local}, selected, [], {})
    facts = {f["id"]: f for f in rows}
    grounded, relations = grounded_graph(rows, entities, local)
    supports = {fact['id']: fact['support'] for fact in grounded}
    entity_types = {e['id']: e['type'] for e in entities}
    origins = defaultdict(dict)
    for link in sorted(relations, key=lambda row: row['id']):
        entity = link['target_id']
        if entity in members:
            fid = link['source_id'].removeprefix('fact:')
            key = supports[fid]['primary_unit_id']
            spans = supported_spans(supports[fid], local)
            name = link['target_name']
            if entity_types[entity] != 'Person':
                match = entity_name_pattern(name).search
                if len(name.strip()) < 2 or not match(facts[fid]['text']) or not any(
                    match(local[s.source]['content'][s.start:s.end]) for s in spans if s.source == key
                ):
                    continue
            if spans:
                origins[entity].setdefault(key, (fid, spans, link['target_name']))

    paths, graph_seeds = [], []
    for entity, units in sorted(origins.items()):
        keys = sorted(units, key=lambda key: (seed_rank.get(key, len(seeds)), key))
        if entity.removeprefix('native:') in entity_hits:
            graph_seeds.extend(keys[:1])
        elif not any(key in seed_rank for key in keys):
            continue
        if len(keys) < 2:
            continue
        for target in keys:
            origin = keys[1] if target == keys[0] else keys[0]
            left, start_spans, name = units[origin]
            right, end_spans, _ = units[target]
            hop = Hop(origin, 'shared_ABOUT', target, (left, entity, name, right))
            paths.append(PathEvidence((hop,), merge_spans((*start_spans, *end_spans)), tuple(end_spans)))

    # Discovery prefixes belong to new sources. A link joining two reached
    # records is evidence in its own right and must not rewrite either prefix.
    seeds = list(dict.fromkeys([*seed_rank, *graph_seeds]))
    _, links = expand_source_links(local, seeds, include_known_links=True)
    chains = {key: () for key in seeds}
    for link in links:
        a, b, kind = link['from'], link['to'], link['type']
        if kind == 'same_thread':
            witness = (local[a]['conversation_id'], local[a]['thread_id'])
        elif kind == 'same_source':
            witness = (local[a]['source_id'],)
        elif kind in ('attachment', 'referenced_by_attachment'):
            owner, target = (a, b) if kind == 'attachment' else (b, a)
            witness = (target if target in local[owner]['attachment_refs'] else local[target]['source_id'],)
        else:
            owner = b if kind == 'has_reply' else a
            witness = (local[owner]['reply_to'],)
        hop = Hop(a, kind, b, witness)
        if b in chains:
            chain = (hop,)
        else:
            chain = (*chains[a], hop)
            chains[b] = chain
        units = [chain[0].origin, *(hop.target for hop in chain)]
        paths.append(PathEvidence(chain, merge_spans(s for key in units for s in excerpts[key])))
    return paths


def path_text(path, records):
    def quote(key):
        return '\n'.join(records[key]['content'][s.start:s.end] for s in path.spans if s.source == key)
    parts = [quote(path.hops[0].origin)]
    for hop in path.hops:
        parts.extend((hop.relation, hop.witness[2] if hop.relation == 'shared_ABOUT' else '', quote(hop.target)))
    return '\n'.join(filter(None, parts))


async def rank_paths(question, paths, records):
    """Use the same configured query instruction and index embedding model."""
    from kl_graph.config import cfg

    settings = cfg.services.embedding
    transport = cfg.pipelines.ingestion.embedding
    texts = {path: path_text(path, records) for path in paths}
    if not texts:
        return {}, []
    headers = {'Authorization': f'Bearer {settings.api_key}'} if settings.api_key else {}
    base = litellm_base_url('openai', settings.base_url)
    if not base:
        raise ContextUnavailable('未配置路径编码所用的向量服务')
    async with httpx.AsyncClient(base_url=base + '/', headers=headers,
                                 timeout=float(transport.timeout), trust_env=False) as client:
        async def embed(batch):
            payload = dict(model=settings.model, input=batch, encoding_format='float')
            if settings.send_dimensions:
                payload['dimensions'] = int(settings.dim)
            response = await client.post('embeddings', json=payload)
            if response.status_code == 413:
                if len(batch) == 1:
                    return [None]
                split = len(batch) // 2
                return await embed(batch[:split]) + await embed(batch[split:])
            response.raise_for_status()
            rows = sorted(response.json()['data'], key=lambda item: item['index'])
            if [r['index'] for r in rows] != list(range(len(batch))):
                raise ContextUnavailable('路径向量与输入条目未对齐')
            result = []
            for row in rows:
                vector = np.asarray(row['embedding'], dtype=np.float64)
                norm = np.linalg.norm(vector)
                if vector.shape != (settings.dim,) or not np.isfinite(norm) or norm == 0:
                    raise ContextUnavailable('路径查询返回无效向量')
                result.append(vector / norm)
            return result

        instruction = settings.query_instruction
        query = f'Instruct: {instruction}\nQuery:{question}' if instruction else question
        qvec, = await embed([query])
        if qvec is None:
            raise ContextUnavailable('问句超过向量模型输入上限')
        unique = list(dict.fromkeys(texts.values()))
        scores = {}
        size = min(128, max(1, int(transport.batch_size)))
        for start in range(0, len(unique), size):
            batch = unique[start:start + size]
            for passage, vector in zip(batch, await embed(batch), strict=True):
                if vector is not None:
                    scores[passage] = float(qvec @ vector)
    ranked = sorted((p for p in texts if texts[p] in scores),
                    key=lambda p: (-scores[texts[p]], sum(2 if h.relation == 'shared_ABOUT' else 1 for h in p.hops), p.hops))
    best = {}
    for path in ranked:
        best.setdefault(path.target, path)
    return best, [p.hops for p in texts if texts[p] not in scores]


async def retrieve(base, data_dir, dataset, question, project, modes=("B", "F", "G"), cutoff=None, *, policy="rank"):
    records = visible_records(dataset, project, cutoff or dataset.cutoff)
    if policy not in ("rank", "facet_novelty"):
        raise ValueError(f"未知装入规则 {policy}")
    facets = plan_facets(question, records) if policy == "facet_novelty" else {}
    queries = [question, *facets.values()]
    keywords, vectors, raw_lanes = [], [], []
    for query in queries:
        vector = await query_original_vectors(base, data_dir, query, records, allow_empty=True)
        keyword = rank_originals(query, records)
        keywords.append(keyword)
        vectors.append(vector)
        raw_lanes.append(fuse_originals(keyword, vector))
    original = combine_queries(raw_lanes[0], raw_lanes[1:], WINDOW)
    # The best ranking query also selects the excerpt, with whole-question ties first.
    best = {}
    for query, lane in zip(queries, raw_lanes):
        for rank, key in enumerate(lane[:WINDOW]):
            if key not in best or rank < best[key][0]:
                best[key] = (rank, query)
    terms = {q: set(_terms(q)) for q in queries}
    excerpts = {key: direct_excerpt(terms[best[key][1] if key in best else question], key, records)
                for key in records}
    snapshot = Snapshot({key: excerpts[key] for key in original}, {}, {}, tuple(facets), policy)
    audit = dict(keyword=keywords[0], vector=vectors[0], O=original, evidence_policy=policy, facet_queries=facets,
                 facet_originals={name: lane[:WINDOW] for name, lane in zip(facets, raw_lanes[1:])})
    if not set(modes).intersection(("F", "G")):
        return records, snapshot, audit
    use_graph = "G" in modes
    async with httpx.AsyncClient(base_url=base, timeout=120, trust_env=False) as client:
        async def lane(collection, depth, field):
            rankings = [await search(client, collection, q, depth, field) for q in queries]
            return combine_queries(rankings[0], rankings[1:], depth)

        fact_hits = await lane('facts', FACT_DEPTH, 'fact_id')
        entities = await lane('entities', ENTITY_DEPTH, 'entity_id') if use_graph else []
    unavailable = {r['conversation_id'] for k, r in dataset.records.items()
                   if k not in records and r.get('conversation_id')}
    scopes = _fact_scopes(records, project, unavailable)
    eligible = {key: records[key] for key in scopes}
    uri = (data_dir / 'knowledge.db').resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        by_id = {f['id']: f for f in source_facts(conn, eligible, fact_ids=fact_hits)}
        snapshot.facts, witness_ids = fact_lane(by_id, fact_hits, eligible)
        audit.update(fact_hits=fact_hits, fact_witnesses=witness_ids, T=list(snapshot.facts)[:WINDOW])
        if not use_graph:
            return records, snapshot, audit
        seeds = list(dict.fromkeys([*original[:SEED_LIMIT], *list(snapshot.facts)[:SEED_LIMIT]]))
        paths = graph_paths(conn, records, scopes, entities, seeds, excerpts)
    snapshot.paths = rank_light_paths(question, paths, records, snapshot.originals, snapshot.facts)
    audit.update(entity_hits=entities, H=list(snapshot.paths)[:WINDOW], path_ranking="light", path_candidates=len(paths), oversized_paths=[])
    return records, snapshot, audit
