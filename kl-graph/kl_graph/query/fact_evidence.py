"""Trace extracted claims to quotations in visible source units."""

from __future__ import annotations

import json
import re
from collections import defaultdict

_SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?\n]*")
_DATE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s*")
_CODE = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+")
_BUSINESS_TYPES = {"STATUS", "DECISION", "DELEGATE", "ACTION_ITEM", "CAUSAL", "REQUEST"}
_ANAPHORA = re.compile(r"该(?:包|项|版本|项目|方案|记录|样本|文件|对象|配置|环境|问题|任务|结果)|上述|前述|其中|这些|此项|其余")
_REPLY_REFERENCE = re.compile(r"这[几两三四一]?([项列条步点个]|部分)|那[个项包]|上述|前述|该包|these|those", re.I)


# SQLite is the authoritative store for source provenance on every graph backend.
SOURCE_QUOTE_SQL = """(SELECT json_extract(e.properties, '$.source_quote.quote') FROM edges e
    WHERE e.source_type='fact' AND e.source_id=f.id AND e.target_type='chunk'
      AND e.target_id=f.source_chunk_id AND e.edge_type='STATES')"""


def _anchor_span(content, quote):
    if not isinstance(quote, str) or not quote.strip():
        return None
    start = content.find(quote)
    if start < 0 or content.find(quote, start + 1) >= 0:
        return None
    return {"start": start, "end": start + len(quote)}


def _grams(text):
    text = re.sub(r"[^\w\u3400-\u9fff]", "", text.casefold())
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _matches(text, sentences):
    grams = _grams(text)
    codes = set(_CODE.findall(text))
    if not grams:
        return set()
    for width in (1, 2, 3):
        matches = []
        for start in range(len(sentences) - width + 1):
            window = sentences[start:start + width]
            if not all(_ANAPHORA.search(s[0]) for s in window[1:]):
                continue
            quote = "".join(s[0] for s in window)
            if codes.issubset(_CODE.findall(quote)):
                score = len(grams & _grams(quote)) / len(grams)
                if score >= 0.55:
                    matches.append((score, start))
        if matches:
            best = max(score for score, _ in matches)
            return {i for score, start in matches if score == best for i in range(start, start + width)}
    return set()


def _located(text, sentences, content, anchor):
    indices = _matches(text, sentences)
    span = _anchor_span(content, anchor)
    # A verbatim quote can still be unrelated to its claim. Only widen an
    # independently located excerpt; otherwise the record stays in full.
    if span and indices:
        indices.update(i for i, sentence in enumerate(sentences)
                       if sentence.start() < span["end"] and sentence.end() > span["start"])
    return indices


def _spans(sentences, selected):
    selected = set(selected)
    for i in list(selected):
        if i and (_ANAPHORA.search(sentences[i][0]) or re.match(
            r'\s*[”"]?(?:该|上述|前述|其中|这些|此|其|因此|只有|满足)', sentences[i][0]
        )):
            selected.add(i - 1)
        if i + 1 < len(sentences) and (_ANAPHORA.search(sentences[i + 1][0]) or re.match(
            r'\s*[”"]?(?:但|然而|仅|只限|除|否则|前提|尚未|不代表)', sentences[i + 1][0]
        )):
            selected.add(i + 1)
    spans = []
    for i in sorted(selected):
        start, end = sentences[i].span()
        if spans and start == spans[-1]["end"]:
            spans[-1]["end"] = end
        else:
            spans.append({"start": start, "end": end})
    return spans


def fact_excerpts(records: dict[str, dict], facts: list[dict], *, audit: dict | None = None) -> dict[str, dict]:
    """Compact a long source only when every eligible claim can be located."""
    by_source = defaultdict(list)
    for fact in facts:
        refs = fact.get("evidence_ids", [])
        if len(refs) == 1 and refs[0] in records and fact.get("id"):
            by_source[refs[0]].append(fact)
    projections = {}
    for key, source_facts in by_source.items():
        row = records[key]
        content = row["content"]
        if row.get("source_type") in {"chat", "message"} or len(content) < 160:
            continue
        sentences = list(_SENTENCE.finditer(content))
        selected, matched, unmatched = set(), [], []
        for fact in source_facts:
            text = _DATE.sub("", fact["text"])
            if fact.get("type") not in _BUSINESS_TYPES and not _CODE.search(text):
                continue
            indices = _located(text, sentences, content, fact.get("source_quote"))
            selected.update(indices)
            (matched if indices else unmatched).append(fact["id"])
        if audit is not None:
            audit[key] = {"matched_fact_ids": sorted(set(matched)), "unmatched_fact_ids": sorted(set(unmatched)),
                          "status": "full_text_unmatched_facts" if unmatched else "no_projection"}
        if unmatched or not selected:
            continue
        spans = _spans(sentences, selected)
        excerpt = "\n[…]\n".join(content[s["start"]:s["end"]] for s in spans)
        if len(excerpt) < len(content):
            projections[key] = {"content": excerpt, "spans": spans, "fact_ids": sorted(set(matched)),
                                "original_characters": len(content)}
            if audit is not None:
                audit[key]["status"] = "projected"
    return projections


def fact_support(text, primary, get_record, source_quote=None):
    """Return source quotations and explicit reply dependencies. Spans index each supplied body."""
    row = get_record(primary)
    if not row:
        return {"primary_unit_id": primary, "status": "unavailable", "quotes": []}
    body = row["content"]
    anchor = _anchor_span(body, source_quote)
    chat = row.get("source_type") in {"chat", "message"}
    indices = []
    if not chat:
        sentences = list(_SENTENCE.finditer(body))
        indices = _spans(sentences, _located(_DATE.sub("", text), sentences, body, source_quote))
    quotes = [{"unit_id": primary, "start": span["start"], "end": span["end"],
               "quote": body[span["start"]:span["end"]]} for span in indices]
    status = "located" if quotes else "whole_source"
    if not quotes:
        quotes.append({"unit_id": primary, "start": 0, "end": len(body), "quote": body})
    if chat:
        if speaker := row.get("speaker"):
            quotes[0]["speaker"] = speaker
        missing = set(_CODE.findall(text)) - set(_CODE.findall(body))
        needs_parent = bool(_REPLY_REFERENCE.search(body))
        seen = {primary}
        while missing or needs_parent:
            parent_id = row.get("reply_to")
            parent = get_record(parent_id) if parent_id and parent_id not in seen else None
            if not parent or not row.get("conversation_id") or parent.get("conversation_id") != row["conversation_id"]:
                status = "incomplete_context"
                break
            if isinstance(row.get("timestamp"), (int, float)) and isinstance(parent.get("timestamp"), (int, float)):
                if parent["timestamp"] > row["timestamp"]:
                    status = "incomplete_context"
                    break
            if row.get("thread_id") and parent.get("thread_id") and row["thread_id"] != parent["thread_id"]:
                status = "incomplete_context"
                break
            seen.add(parent_id)
            body = parent["content"]
            quotes.append({"unit_id": parent_id, "start": 0, "end": len(body), "quote": body})
            if speaker := parent.get("speaker"):
                quotes[-1]["speaker"] = speaker
            missing.difference_update(_CODE.findall(body))
            needs_parent = bool(_REPLY_REFERENCE.search(body))
            row = parent
            status = "reply_context"
    support = {"primary_unit_id": primary, "status": status, "quotes": quotes}
    if anchor:
        support["target_quote"] = {**anchor, "quote": source_quote}
    return support


def stored_fact_support(conn, text, chunk_id, primary, source_quote=None):
    """Read native lineage without guessing a target in a multi-unit chunk."""
    def chunk_support():
        row = conn.execute("SELECT content FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if not row:
            return {"primary_unit_id": None, "status": "unavailable", "quotes": []}
        support = fact_support(text, chunk_id, {chunk_id: {"content": row[0]}}.get, source_quote)
        for quote in support["quotes"]:
            quote.pop("unit_id")
            quote["chunk_id"] = chunk_id
        support.update(primary_unit_id=None, status="chunk_only", offset_basis="source_chunk")
        return support

    members = conn.execute(
        "SELECT source_id, source_type, unit_id FROM chunk_units WHERE chunk_id=?", (chunk_id,)
    ).fetchall()
    targets = [tuple(m) for m in members if m[2] == primary] if primary else list(map(tuple, members))
    if len(targets) != 1:
        return chunk_support()
    source, kind, primary = targets[0]
    if kind != "message" and len(members) != 1:
        return chunk_support()

    def read(key):
        if kind != "message":
            row = conn.execute("SELECT content FROM chunks WHERE id=?", (chunk_id,)).fetchone()
            return {"content": row[0], "source_type": kind} if row else None
        unit = conn.execute("SELECT metadata, timestamp FROM units WHERE source_id=? AND source_type=? AND unit_id=?",
                            (source, kind, key)).fetchone()
        meta = json.loads(unit[0] or "{}") if unit else {}
        content = meta.get("extraction_target_content")
        if not isinstance(content, str):
            return None
        return {"source_type": kind, "content": content, "conversation_id": meta.get("conversation_id"),
                "timestamp": unit[1], "speaker": meta.get("sender"), "thread_id": (meta.get("source_record") or {}).get("thread_id"),
                "reply_to": meta.get("reply_to")}

    support = fact_support(text, primary, read, source_quote)
    if support["status"] == "unavailable":
        return chunk_support()
    support["offset_basis"] = "extraction_target_content" if kind == "message" else "source_chunk"
    support["source_id"] = source
    support["source_type"] = kind
    if kind != "message":
        for q in support["quotes"]:
            q["chunk_id"] = chunk_id
    return support


def source_evidence(conn, records, candidates, *, question=""):
    """Read evidence from in-scope bodies. Archived callers may still pass question."""
    facts = []
    dependencies = defaultdict(dict)
    ids = list(dict.fromkeys(key for key in candidates if key in records))
    for start in range(0, len(ids), 400):
        batch = ids[start:start + 400]
        marks = ",".join("?" for _ in batch)
        rows = conn.execute(f"""
            SELECT cu.unit_id, f.id, f.fact_type, f.text, {SOURCE_QUOTE_SQL}
            FROM chunk_units cu JOIN facts f ON f.source_chunk_id = cu.chunk_id
            WHERE cu.unit_id IN ({marks}) AND (
                f.source_unit_id = cu.unit_id OR (NULLIF(f.source_unit_id, '') IS NULL AND
                (SELECT count(*) FROM chunk_units other WHERE other.chunk_id = cu.chunk_id) = 1))
            ORDER BY cu.unit_id, f.id
        """, batch)
        for key, fid, kind, text, anchor in rows:
            if records[key].get("source_type") in {"chat", "message"}:
                support = fact_support(text, key, records.get)
                if support["status"] == "reply_context":
                    dependencies[key].update(dict.fromkeys(q["unit_id"] for q in support["quotes"][1:]))
            else:
                facts.append({"id": fid, "text": text, "type": kind, "source_quote": anchor, "evidence_ids": [key]})
    audit = {}
    excerpts = fact_excerpts(records, facts, audit=audit)
    return {"projections": excerpts, "audit": audit,
            "dependencies": {key: list(parents) for key, parents in dependencies.items()}}


def grounded_graph(facts, entities, records):
    """Project native fact/participant links, keeping quotations distinct from extracted wording."""
    by_id = {item["id"]: item for item in entities}
    grounded, links = [], []
    for fact in facts:
        primary = fact["evidence_ids"][0]
        support = fact_support(fact["text"], primary, records.get, fact.get("source_quote"))
        quotes = support["quotes"]
        if not quotes:
            continue
        refs = list(dict.fromkeys(q["unit_id"] for q in quotes))
        quotation = "\n[…]\n".join(q["quote"] for q in quotes if q["unit_id"] == primary)
        item = {k: fact[k] for k in ("id", "type", "occurred_at", "scope", "origin") if k in fact}
        grounded.append({**item, "text": quotation, "evidence_ids": refs, "support": support})
        if support["status"] == "incomplete_context":
            continue
        for entity_id in fact.get("entity_ids", []):
            entity = by_id.get(entity_id)
            if entity and entity.get("type") == "Person" and primary not in entity["evidence_ids"]:
                continue
            if not entity or not any(entity["name"].casefold() in q["quote"].casefold()
                                     or (entity["name"] == q.get("speaker") and q["unit_id"] in entity["evidence_ids"])
                                     for q in quotes):
                continue
            links.append({"id": f"fact:{fact['id']}|ABOUT|{entity_id}", "source_id": f"fact:{fact['id']}",
                          "target_id": entity_id, "source_name": fact["type"], "target_name": entity["name"],
                          "type": "ABOUT", "certainty": "native_participant", "origin": "mycontext",
                          "evidence_ids": refs, "scopes": [fact["scope"]]})
    return grounded, links
