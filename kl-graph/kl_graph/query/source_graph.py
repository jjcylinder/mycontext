"""Read a source-scoped projection of native facts, participants and chunk lineage."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime

from .fact_evidence import SOURCE_QUOTE_SQL


def _select(conn: sqlite3.Connection, query: str, ids: list[str] | None):
    if ids is None:
        return conn.execute(query.replace("{filter}", "1")).fetchall()
    if not ids:
        return []
    rows = []
    for start in range(0, len(ids), 400):
        batch = ids[start:start + 400]
        marks = ",".join("?" for _ in batch)
        rows.extend(conn.execute(query.replace("{filter}", f"id IN ({marks})"), batch))
    return rows


def chunk_units(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, list[str]]:
    units: dict[str, list[str]] = defaultdict(list)
    for start in range(0, len(chunk_ids), 500):
        batch = chunk_ids[start:start + 500]
        marks = ",".join("?" for _ in batch)
        for chunk_id, unit_id in conn.execute(
            f"SELECT chunk_id, unit_id FROM chunk_units WHERE chunk_id IN ({marks}) "
            "ORDER BY chunk_id, unit_ordinal_in_chunk", batch,
        ):
            units[chunk_id].append(unit_id)
    return units


def read_graph(conn, records, scopes, fact_ids, chunk_ids, participants):
    """Read visible units. scopes contains only units eligible to support factual claims."""
    fact_rows = _select(conn, "SELECT f.*, " + SOURCE_QUOTE_SQL + " AS source_quote FROM facts f WHERE {filter}", fact_ids)
    # /ask returns a bounded graph. Recover the stored ABOUT links of those
    # exact fact hits even when their participant nodes fell outside that view.
    about = {fid: set(ids) for fid, ids in (participants or {}).items()}
    if fact_ids is not None or participants is None:
        hits = [row["id"] for row in fact_rows]
        for start in range(0, len(hits), 400):
            batch = hits[start:start + 400]
            marks = ",".join("?" for _ in batch)
            for fid, entity in conn.execute(
                "SELECT source_id, target_id FROM edges WHERE source_type='fact' AND target_type='entity' "
                f"AND edge_type='ABOUT' AND source_id IN ({marks})", batch
            ):
                about.setdefault(fid, set()).add(entity)
    chunks = list(dict.fromkeys(chunk_ids + [row["source_chunk_id"] for row in fact_rows]))
    units = chunk_units(conn, chunks)
    native_entities: dict[str, dict] = {}
    evidence_by_entity: dict[str, set[str]] = defaultdict(set)
    facts = []
    ranking = {fact_id: index for index, fact_id in enumerate(fact_ids or [])}
    fact_rows.sort(key=lambda row: (ranking.get(row["id"], 0), -(row["timestamp"] or 0), row["id"]))
    for row in fact_rows:
        unit_id = row["source_unit_id"]
        chunk_members = list(dict.fromkeys(units[row["source_chunk_id"]]))
        if not unit_id and len(chunk_members) == 1:
            unit_id = chunk_members[0]
        # A multi-message chunk is not precise evidence for a fact.
        if unit_id not in records or unit_id not in chunk_members:
            continue
        record = records[unit_id]
        if unit_id not in scopes:
            continue
        scope = scopes[unit_id]
        endpoint_ids = set(about.get(row["id"], ()))
        endpoint_ids.update(x for x in (row["subject_entity_id"], row["object_entity_id"]) if x)
        for entity_id in endpoint_ids:
            evidence_by_entity[entity_id].add(unit_id)
        facts.append({
            "id": row["id"], "text": row["text"], "type": row["fact_type"],
            "occurred_at": record.get("occurred_at"), "confidence": row["confidence"],
            "evidence_ids": [unit_id], "scope": scope, "origin": "mycontext",
            "source_quote": row["source_quote"],
            "entity_ids": sorted(f"native:{eid}" for eid in about.get(row["id"], ())),
        })
    speaker_ids = defaultdict(set)
    for key in scopes:
        row = records[key]
        identifier = row.get("speaker_id") or row.get("sender_id")
        if identifier and row.get("speaker"):
            speaker_ids[row["speaker"]].add(identifier)
    for row in _select(conn, "SELECT id, name, entity_type FROM entities WHERE quality_status = 'active' AND {filter}",
                       list(evidence_by_entity)):
        name = row["name"]
        if row["entity_type"] == "Person":
            # A name-only native id can represent homonyms. Bind it to a cited
            # speaker only where the source supplies a single stable identity.
            ids = speaker_ids.get(name, set())
            refs = sorted(ref for ref in evidence_by_entity[row["id"]] if len(ids) == 1
                          and records[ref].get("speaker") == name
                          and (records[ref].get("speaker_id") or records[ref].get("sender_id")) in ids)
        else:
            refs = sorted(ref for ref in evidence_by_entity[row["id"]]
                          if name.casefold() in records[ref]["content"].casefold())
        if refs:
            native_entities[row["id"]] = {
                "id": f"native:{row['id']}", "name": row["name"], "type": row["entity_type"],
                "evidence_ids": refs, "origin": "mycontext",
                "scopes": sorted({scopes[ref] for ref in refs}),
            }
    refs = list(dict.fromkeys(ref for chunk in chunk_ids for ref in units[chunk] if ref in records))
    return list(native_entities.values()), facts, refs, len(fact_rows) - len(facts)


def expand_source_links(records: dict[str, dict], seeds: list[str]) -> tuple[list[str], list[dict]]:
    """Follow exact source/thread, reply and attachment references, without inferring facts."""
    scopes = defaultdict(list)
    sources = defaultdict(list)

    def scope(row):
        if row.get("source_type") == "chat":
            if row.get("thread_id") and row.get("conversation_id"):
                return ("thread", row["conversation_id"], row["thread_id"])
        elif row.get("source_id") and row.get("source_type"):
            return ("source", row["source_type"], row["source_id"])
        return None

    for key, row in records.items():
        if value := scope(row):
            scopes[value].append(key)
        if row.get("source_id"):
            sources[row["source_id"]].append(key)
    selected = dict.fromkeys(key for key in seeds if key in records)
    queue = list(selected)
    paths = []
    expanded_scopes = set()
    for key in queue:
        row = records[key]
        neighbors = []
        value = scope(row)
        if value and value not in expanded_scopes:
            expanded_scopes.add(value)
            neighbors.extend((ref, "same_" + value[0]) for ref in scopes[value])
        if row.get("reply_to") in records:
            neighbors.append((row["reply_to"], "reply_to"))
        for ref in row.get("attachment_refs") or []:
            if isinstance(ref, str):
                neighbors.extend((target, "attachment") for target in ([ref] if ref in records else sources[ref]))
        for ref, kind in neighbors:
            if ref in selected:
                continue
            selected[ref] = None
            queue.append(ref)
            paths.append({"from": key, "to": ref, "type": kind})
    return list(selected), paths


def attachment_project_paths(records, scopes, candidates):
    """Find an auditable reply path to one unambiguous in-project attachment.

    records must include all visible comparison records. A missing scope also
    blocks the path (for example, a conversation excluded by the cutoff).
    """
    documents = defaultdict(list)
    for key, row in records.items():
        if row.get("source_type") == "document":
            documents[row.get("source_id")].append(key)

    paths = {}
    for key in dict.fromkeys(candidates):
        row = records.get(key, {})
        if row.get("source_type") != "chat" or scopes.get(key) != "unknown":
            continue
        thread = row.get("thread_id")
        conversation = row.get("conversation_id")
        if not thread or not conversation:
            continue
        path = [key]
        while len(path) <= 16:
            current = records[path[-1]]
            attachments = current.get("attachment_refs") or []
            if attachments:
                if len(attachments) != 1:
                    break
                sections = documents.get(attachments[0], [])
                if not sections or any(scopes.get(s) != "requested" or
                                       not records[s].get("project_hint") for s in sections):
                    break
                speaker = row.get("speaker")
                section = next((s for s in sections if speaker and speaker in records[s]["content"]), sections[0])
                paths[key] = {"reply_path": path, "attachment_record": section}
                break
            parent = current.get("reply_to")
            if parent in path or scopes.get(parent) not in {"requested", "unknown"}:
                break
            ancestor = records.get(parent, {})
            if ancestor.get("thread_id") != thread or ancestor.get("conversation_id") != conversation:
                break
            if not ancestor.get("occurred_at") or not current.get("occurred_at"):
                break
            if datetime.fromisoformat(ancestor["occurred_at"]) > datetime.fromisoformat(current["occurred_at"]):
                break
            path.append(parent)
    return paths
