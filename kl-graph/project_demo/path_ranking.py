"""Lightweight ranking for graph paths with verified source excerpts."""

import math
from collections import Counter
from fractions import Fraction

from .answer import _terms
from .evidence_packing import RANK_CONSTANT

_RELATION_STRENGTH = {
    'same_source': 1, 'same_thread': 1, 'shared_ABOUT': 2,
    'attachment': 3, 'referenced_by_attachment': 3, 'reply_to': 3, 'has_reply': 3,
}


def _competition_ranks(values):
    """Equal values receive the same rank, including when all candidates tie."""
    counts = Counter(values.values())
    next_rank = 1
    ranks = {}
    for value in sorted(counts, reverse=True):
        ranks[value] = next_rank
        next_rank += counts[value]
    return {path: ranks[value] for path, value in values.items()}


def _target_scores(question, paths, records):
    terms = sorted(set(_terms(question)))
    quotes = {
        path: '\n'.join(records[span.source]['content'][span.start:span.end]
                        for span in path.spans if span.source == path.target)
        for path in paths
    }
    frequencies = {quote: Counter(_terms(quote)) for quote in set(quotes.values())}
    size = len(frequencies)
    average = sum(freq.total() for freq in frequencies.values()) / max(1, size)
    document_frequency = Counter(term for freq in frequencies.values() for term in terms if term in freq)
    scores = {}
    for quote, freq in frequencies.items():
        length_factor = 1.2 * (0.25 + 0.75 * freq.total() / max(1, average))
        scores[quote] = sum(
            math.log1p((size - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            * freq[term] * 2.2 / (freq[term] + length_factor)
            for term in terms if term in freq
        )
    return {path: scores[quote] for path, quote in quotes.items()}


def rank_light_paths(question, paths, records, originals, facts):
    """Use equal RRF votes for seed relevance, relation specificity and target BM25."""
    origin_ranks = {}
    for lane in (originals, facts):
        for rank, source in enumerate(lane, 1):
            origin_ranks[source] = min(rank, origin_ranks.get(source, rank))
    signals = (
        {p: -origin_ranks.get(p.hops[0].origin, math.inf) for p in paths},
        {p: (min(_RELATION_STRENGTH[h.relation] for h in p.hops), -len(p.hops)) for p in paths},
        _target_scores(question, paths, records),
    )
    ranks = [_competition_ranks(signal) for signal in signals]

    def order(path):
        score = sum(Fraction(1, RANK_CONSTANT + lane[path]) for lane in ranks)
        return -score, len(path.hops), path.hops, path.spans

    best = {}
    for path in sorted(paths, key=order):
        best.setdefault(path.target, path)
    return best
