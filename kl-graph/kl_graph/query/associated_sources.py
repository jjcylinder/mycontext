"""Find complementary source records through two persisted ABOUT edges."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import NamedTuple

from kl_graph.utils.helpers import rrf

from .fact_evidence import grounded_graph
from .source_graph import read_graph

# Limit the query-time walk and avoid using high-degree entities as topic hubs.
_SEED_LIMIT = 16
_DEGREE_LIMIT = 64
_SOURCE_LIMIT = 8

# Whole-chunk extraction is attributable to a record only for a single-unit chunk.
_LINEAGE = """JOIN chunk_units cu ON cu.chunk_id = f.source_chunk_id
    AND (cu.unit_id = f.source_unit_id OR (NULLIF(f.source_unit_id, '') IS NULL AND
         (SELECT COUNT(*) FROM chunk_units u WHERE u.chunk_id = cu.chunk_id) = 1))"""
_ABOUT = """ed.source_type='fact' AND ed.edge_type='ABOUT' AND ed.target_type='entity'"""


class _Hop(NamedTuple):
    entity: str
    fact: str
    source: str
    source_fact: str
    text: str


def entity_name_pattern(name):
    """Match a stored entity name without treating a longer identifier as the same object."""
    pattern = re.escape(name)
    if name.isascii():
        pattern = r"(?<![a-z0-9_-])" + pattern + r"(?![a-z0-9_-])"
    return re.compile(pattern, re.I | re.ASCII)


def _associated_sources(conn, records, seeds, original_ids, question):
    """Return witnessed paths to additional sources, without inferring a business relation.

    records contains only eligible evidence as of the query. Both ends of a walk
    must name the same active entity in the stored fact and in its own source.
    """
    seed_set = set(seeds)
    first = list(dict.fromkeys(key for key in seeds if key in records))[:_SEED_LIMIT]
    if not first:
        return []
    seed_rank = {key: index for index, key in enumerate(first)}
    marks = ",".join("?" for _ in first)
    rows = conn.execute(f"""
        SELECT cu.unit_id, f.id, f.text, e.id, e.name FROM facts f
        {_LINEAGE} JOIN edges ed ON ed.source_id=f.id AND {_ABOUT}
        JOIN entities e ON e.id=ed.target_id
        WHERE cu.unit_id IN ({marks}) AND e.quality_status='active' AND e.entity_type!='Person'
        ORDER BY cu.unit_id, f.id, e.id
    """, first)
    anchors = defaultdict(dict)
    patterns = {}
    for unit, fid, claim, eid, name in rows:
        if not name or len(name.strip()) < 2:
            continue
        if eid not in patterns:
            patterns[eid] = entity_name_pattern(name)
        match = patterns[eid].search
        if match(claim) and match(records[unit]['content']):
            anchors[eid].setdefault(unit, fid)

    # One representative path per destination fact/entity pair.
    by_unit = defaultdict(list)
    for eid, origins in anchors.items():
        edges = conn.execute(f"""SELECT ed.source_id FROM edges ed WHERE {_ABOUT}
                              AND ed.target_id=? LIMIT ?""", (eid, _DEGREE_LIMIT + 1)).fetchall()
        if len(edges) > _DEGREE_LIMIT:
            continue
        fids = [edge[0] for edge in edges]
        marks = ','.join('?' for _ in fids)
        start = min(origins, key=seed_rank.__getitem__)
        for unit, fid, claim in conn.execute(f"""SELECT DISTINCT cu.unit_id, f.id, f.text
                FROM facts f {_LINEAGE} WHERE f.id IN ({marks})""", fids):
            if unit in seed_set or unit not in records:
                continue
            match = patterns[eid].search
            if match(claim) and match(records[unit]['content']):
                by_unit[unit].append(_Hop(eid, fid, start, origins[start], claim))

    # Combine independent ranks: corroboration by multiple entities, lexical
    # relevance of the linked observation, and the existing hybrid source rank.
    query_terms = {question[i:i + 2].casefold() for i in range(len(question) - 1)}

    def overlap(text):
        folded = text.casefold()
        return sum(term in folded for term in query_terms)

    leading_originals = set(original_ids[:_SEED_LIMIT])
    graph_scores = {}
    for unit, paths in by_unit.items():
        entities = {path.entity for path in paths}
        origins = {key for entity in entities for key in anchors[entity]}
        relevance = max(overlap(path.text) for path in paths)
        if len(entities) > 1 or len(origins) > 1 or relevance or unit in leading_originals:
            graph_scores[unit] = (len(entities), relevance, len(origins))
    # Stable source IDs break tied graph scores before reciprocal rank fusion.
    graph_rank = sorted(sorted(graph_scores), key=graph_scores.get, reverse=True)
    ranked = rrf([[(key, 0) for key in graph_rank], [(key, 0) for key in dict.fromkeys(original_ids)]])
    selected = [key for key, _ in ranked if key in graph_scores][:_SOURCE_LIMIT]
    paths = []
    for unit in selected:
        choices = sorted(by_unit[unit], key=lambda p: (-overlap(p.text), seed_rank[p.source], p.fact, p.entity))
        seen = set()
        for eid, fid, origin, anchor_fact, _ in choices:
            if fid not in seen:
                paths.append(dict(source=origin, source_fact=anchor_fact, entity_id=eid,
                                  target_fact=fid, target=unit))
                seen.add(fid)
            if len(seen) == 2:
                break
    return paths


def related_evidence(conn, records, scopes, seeds, original_ids, question):
    """Hydrate only witnessed facts. Never export an unverified participant link."""
    local_scopes = {key: scope for key, scope in scopes.items()
                    if key in records and scope not in {"comparison", "other"}}
    local = {key: records[key] for key in local_scopes}
    paths = _associated_sources(conn, local, seeds, original_ids, question)
    ids = list(dict.fromkeys(fid for path in paths for fid in (path["source_fact"], path["target_fact"])))
    if not ids:
        return [], []
    entities, rows, _, _ = read_graph(conn, local, local_scopes, ids, [], None)
    facts, relations = grounded_graph(rows, entities, local)
    links = {(relation["source_id"], relation["target_id"]) for relation in relations}
    names = {row["id"]: row["name"] for row in entities}
    verified = []
    for path in paths:
        entity = "native:" + path["entity_id"]
        if all(("fact:" + path[k], entity) in links for k in ("source_fact", "target_fact")):
            verified.append({**path, "entity_name": names[entity], "edge_type": "ABOUT"})
    targets = {p["target_fact"] for p in verified}
    return [fact for fact in facts if fact["id"] in targets], verified
