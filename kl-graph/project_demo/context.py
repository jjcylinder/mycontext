"""Project evidence retrieval over the pinned MyContext knowledge graph."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import httpx

from kl_graph.query.associated_sources import related_evidence
from kl_graph.query.fact_evidence import fact_support, grounded_graph, source_evidence
from kl_graph.query.source_graph import attachment_project_paths, chunk_units, expand_source_links, read_graph


class ContextUnavailable(RuntimeError):
    """Native retrieval could not be completed for this dataset."""


_CODE = re.compile(r"(?<![\w-])(?:[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)(?![A-Z0-9_-])", re.ASCII)
_CODE_TYPES = {
    "SK": "skill", "CASE": "task", "DEMO": "demo_task",
    "APP": "application_config", "ENV": "environment", "M": "model_version",
    "BUG": "issue", "ERR": "issue", "SEM": "issue", "QA": "issue",
    "EVAL": "case_set", "FR": "case_set", "LET": "case_set", "POL": "case_set",
    "STYLE": "case_set", "SU": "case_set", "TR": "case_set", "WR": "case_set",
    "DEDUP": "component", "LANG": "component", "ROUTE": "component", "TPL": "component",
    "DOC": "artifact", "IM": "source_record", "FWD": "source_record", "PO": "source_record",
    "JOB": "run", "REPORT": "artifact", "SN": "snapshot",
}
_DECLARE_ALIAS = re.compile(
    r"(?P<code>SK-[A-Z0-9]+)\s*(?:对应|为)\s*"
    r"(?P<slug>[a-z][a-z0-9]*(?:-[a-z0-9]+)+)(?:[，,]\s*别名(?P<aliases>[^；;。]+))?"
)
_ROLE_PERSON = re.compile(
    r"(?P<id>U\d+)\s*(?P<name>(?:应用|模型|测试|前方|数据|发布)[\u4e00-\u9fff]{1,3}工)"
)
_PROJECT = re.compile(r"项目\s*([A-Z0-9]+)", re.IGNORECASE | re.ASCII)
_BY_CLAUSE = re.compile(r"[。；;！？!?\n]+")
_COMPARISON_SCOPES = {"other", "comparison"}


def _time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)


def visible_records(dataset, project_id: str, cutoff: str | None = None) -> dict[str, dict]:
    """Return the same as-of corpus to A, B, and C, including distractors."""
    end = _time(cutoff or dataset.cutoff)
    if end is None:
        raise ValueError("核查截止时间须为 ISO 8601 格式")
    result = {}
    for record_id, record in dataset.records.items():
        observed = _time(record.get("available_at")) or _time(record.get("occurred_at"))
        if observed is None or observed <= end:
            result[record_id] = record
    return result


def project_scope(record: dict, project_id: str) -> str:
    """Describe project attribution for a source, without excluding comparisons."""
    target = _PROJECT.fullmatch(project_id.strip())
    hint = record.get("project_hint")
    if not target:
        return "requested" if hint == project_id else "other" if hint else "unknown"
    wanted = target[1].upper()
    mentioned = {match.upper() for match in _PROJECT.findall(record.get("content", ""))}
    if mentioned - {wanted}:
        return "comparison" if wanted in mentioned else "other"
    if hint and hint != project_id:
        return "comparison" if mentioned else "other"
    return "requested" if hint or mentioned else "unknown"


def _identifiers(text: str) -> list[str]:
    return list(dict.fromkeys(
        m[0] for m in _CODE.finditer(text) if m[0].split("-", 1)[0] in _CODE_TYPES
    ))


def _relation(first: dict, second: dict, text: str) -> tuple[dict, dict, str, str]:
    membership = {first["type"]: first, second["type"]: second}
    rules = (
        ("environment", "application_config", "deployment_binding", r"绑定|搭载|部署"),
        ("application_config", "model_version", "model_reference", r"调用|返回|服务"),
        ("skill", "task", "skill_task", r"对应|本轮|讨论|任务|用于"),
        ("case_set", "issue", "verification_reference", r"覆盖|验证|复测|通过|回归"),
        ("case_set", "application_config", "verification_reference", r"验证|复核|复测|通过|回归"),
        ("case_set", "environment", "verification_reference", r"验证|复核|复测|通过|回归"),
        ("person", "issue", "coordination_reference", r"负责|协助|跟进|分析|处理|复核"),
        ("project", "environment", "project_affiliation", r"属于|归属"),
        ("project", "application_config", "project_affiliation", r"属于|归属"),
    )
    for source, target, label, marker in rules:
        if source in membership and target in membership and re.search(marker, text):
            # Only label an explicit relation when the clause has just one
            # plausible endpoint of each kind. Relations with negation retain
            # their co-mention but cannot imply a positive binding.
            if not re.search(rf"(?:未|没|不|尚未|并非)[^，、()（）]{{0,6}}(?:{marker})", text):
                return membership[source], membership[target], label, "clause_reference"
    return first, second, "co_mentioned", "inferred"


def _work_graph(records: dict[str, dict], project_id: str) -> tuple[list[dict], list[dict]]:
    """Project business references extracted from the actual source excerpts."""
    entities: dict[str, dict] = {}
    relations: dict[tuple[str, str, str], dict] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    for record in records.values():
        for match in _DECLARE_ALIAS.finditer(record["content"]):
            aliases[match["code"]].add(match["slug"])
            aliases[match["code"]].update(
                alias.strip(" \t“”\"") for alias in re.split(r"[、，,]", match["aliases"] or "")
                if alias.strip()
            )

    people: dict[str, set[str]] = defaultdict(set)
    for record in records.values():
        for match in _ROLE_PERSON.finditer(record["content"]):
            people[match["name"]].add(match["id"])

    def entity(code, name, kind, record_id, scope):
        row = entities.setdefault(code, {
            "id": f"work:{code}", "name": name, "type": kind,
            "origin": "source_reference", "evidence_ids": [], "scopes": [],
        })
        if record_id not in row["evidence_ids"]:
            row["evidence_ids"].append(record_id)
        if scope not in row["scopes"]:
            row["scopes"].append(scope)
        return row

    for record_id, record in records.items():
        scope = project_scope(record, project_id)
        if hint := record.get("project_hint"):
            entity(f"project:{hint}", hint, "project", record_id, scope)
        for clause in _BY_CLAUSE.split(record["content"]):
            ids = _identifiers(clause)
            nodes = [entity(code, code, _CODE_TYPES[code.split("-", 1)[0]], record_id, scope) for code in ids]
            projects = sorted({f"项目{code.upper()}" for code in _PROJECT.findall(clause)})
            nodes.extend(entity(f"project:{name}", name, "project", record_id, scope) for name in projects)
            explicit_people = {(match["name"], match["id"]) for match in _ROLE_PERSON.finditer(clause)}
            for name, person_ids in people.items():
                if name not in clause:
                    continue
                for person_id in sorted(person_ids):
                    if len(person_ids) == 1 or (name, person_id) in explicit_people:
                        nodes.append(entity(person_id, name, "person", record_id, scope))
            # One clause containing parallel configurations can establish
            # co-occurrence but cannot identify which endpoint pair is bound.
            kinds = [node["type"] for node in nodes]
            for first, second in combinations(nodes, 2):
                if kinds.count(first["type"]) == kinds.count(second["type"]) == 1:
                    source, target, kind, certainty = _relation(first, second, clause)
                else:
                    source, target, kind, certainty = first, second, "co_mentioned", "inferred"
                if kind == "co_mentioned" and source["id"] > target["id"]:
                    source, target = target, source
                key = (source["id"], target["id"], kind)
                row = relations.setdefault(key, {
                    "id": "|".join(key), "source_id": source["id"], "target_id": target["id"],
                    "source_name": source["name"], "target_name": target["name"], "type": kind,
                    "origin": "source_reference", "certainty": certainty,
                    "evidence_ids": [], "scopes": [],
                })
                if record_id not in row["evidence_ids"]:
                    row["evidence_ids"].append(record_id)
                if scope not in row["scopes"]:
                    row["scopes"].append(scope)
    for code, names in aliases.items():
        if code in entities:
            entities[code]["aliases"] = sorted(names)
    return list(entities.values()), list(relations.values())


async def query_original_vectors(
    base_url: str, data_dir: Path, question: str, records: dict[str, dict], *, timeout: float = 30,
) -> list[str]:
    """Rank original records from native dense chunk hits and their lineage."""
    path = data_dir / "knowledge.db"
    if not path.is_file():
        raise ContextUnavailable("当前数据集还没有原文索引")
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(f"{base_url.rstrip('/')}/search", json={
                "query": question, "collection": "chunks", "top_k": 20,
            })
            response.raise_for_status()
            answer = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise ContextUnavailable("原文向量查询失败") from error

    if (not isinstance(answer, dict) or answer.get("collection") != "chunks"
            or not isinstance(answer.get("results"), list)):
        raise ContextUnavailable("原文向量查询返回了无法识别的数据")
    if not answer["results"]:
        raise ContextUnavailable("当前数据集的原文向量索引为空")
    chunk_ids = []
    for hit in answer["results"]:
        payload = hit.get("payload") if isinstance(hit, dict) else None
        chunk_id = payload.get("chunk_id") if isinstance(payload, dict) else None
        if not isinstance(chunk_id, str) or not chunk_id.strip() or hit.get("id") != chunk_id:
            raise ContextUnavailable("原文向量查询缺少可验证的片段标识")
        chunk_ids.append(chunk_id)
    chunk_ids = list(dict.fromkeys(chunk_ids))

    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)) as conn:
            units = chunk_units(conn, chunk_ids)
    except sqlite3.Error as error:
        raise ContextUnavailable("无法读取当前数据集的证据索引") from error

    ranked_records: dict[str, None] = {}
    for chunk_id in chunk_ids:
        members = units.get(chunk_id)
        if not members:
            raise ContextUnavailable("原文向量片段缺少来源记录")
        # Do not use the score of a chunk whose embedding saw later records.
        if all(unit_id in records for unit_id in members):
            ranked_records.update(dict.fromkeys(members))
    return list(ranked_records)


def _native_graph(
    data_dir: Path, records: dict[str, dict], project_id: str,
    fact_ids: list[str] | None, chunk_ids: list[str], participants: dict[str, set[str]] | None,
    adjacent_unavailable: set[str] | None = None,
) -> tuple[list[dict], list[dict], list[str], int]:
    path = data_dir / "knowledge.db"
    if not path.is_file():
        raise ContextUnavailable("当前数据集还没有工作图谱")
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            scopes = _fact_scopes(records, project_id, adjacent_unavailable or set())
            return read_graph(conn, records, scopes, fact_ids, chunk_ids, participants)
    except sqlite3.Error as error:
        raise ContextUnavailable("无法读取当前数据集的证据索引") from error


def _fact_scopes(records, project_id, unavailable_conversations):
    scopes = {key: project_scope(row, project_id) for key, row in records.items()
              if row.get("conversation_id") not in unavailable_conversations}
    return {key: scope for key, scope in scopes.items() if scope != "other"}


def _id(value: str) -> str:
    return value.split(":", 1)[1] if value.startswith(("ent:", "fact:", "cnk:")) else value


def _candidate_ids(answer: dict) -> tuple[list[str], list[str], dict[str, set[str]]]:
    fact_ids = [_id(row["id"]) for row in answer.get("items", [])
                if row.get("type") == "fact" and row.get("id")]
    chunk_ids = []
    about: dict[str, set[str]] = defaultdict(set)
    for component in answer.get("graph", {}).get("components", []):
        edges = component.get("edges", [])
        connected_ids = {endpoint for edge in edges for endpoint in (edge.get("from"), edge.get("to"))}
        for node in component.get("nodes", []):
            if node.get("type") == "fact":
                fact_ids.append(_id(node["id"]))
            elif node.get("type") == "chunk" and node["id"] in connected_ids:
                chunk_ids.append(_id(node["id"]))
        for edge in edges:
            if edge.get("type") != "ABOUT":
                continue
            left, right = edge.get("from", ""), edge.get("to", "")
            if left.startswith("ent:"):
                left, right = right, left
            if left.startswith("fact:") and right.startswith("ent:"):
                about[_id(left)].add(_id(right))
    return list(dict.fromkeys(fact_ids)), list(dict.fromkeys(chunk_ids)), about


async def query_context(
    base_url: str, data_dir: Path, dataset, question: str,
    project_id: str, cutoff: str | None = None, *, timeout: float = 90, original_ids=(),
) -> dict:
    records = visible_records(dataset, project_id, cutoff)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(f"{base_url.rstrip('/')}/ask", json={
                "query": question, "force_phase2": False, "top_k": 32,
                "radius": 1, "max_nodes": 64, "seed_k": 10,
            })
            response.raise_for_status()
            answer = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise ContextUnavailable("工作图谱查询失败") from error
    if not isinstance(answer, dict):
        raise ContextUnavailable("工作图谱查询返回了无法识别的数据")

    fact_ids, chunk_ids, about = _candidate_ids(answer)
    # The fixed native index was built against the entire profile, and its
    # chat extractor sees neighboring messages on both sides of each target.
    # Earlier as-of queries can show visible raw lines from such conversations
    # while withholding facts whose extraction may have seen later messages.
    adjacent_unavailable = {
        row["conversation_id"] for ref, row in dataset.records.items()
        if ref not in records and row.get("conversation_id")
    }
    scopes = _fact_scopes(records, project_id, adjacent_unavailable)
    eligible = {key: records[key] for key in scopes}
    try:
        path = (data_dir / "knowledge.db").resolve().as_uri()
        with closing(sqlite3.connect(f"{path}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            native_entities, facts, chunk_refs, omitted = read_graph(conn, records, scopes, fact_ids, chunk_ids, about)
            facts, relations = grounded_graph(facts, native_entities, eligible)
            fact_refs = [ref for fact in facts for ref in fact["evidence_ids"]]
            seeds = list(dict.fromkeys([*fact_refs, *chunk_refs]))
            references, source_paths = expand_source_links(records, seeds)
            related, association_paths = related_evidence(conn, eligible, scopes, seeds, original_ids, question)
            facts.extend(related)
            references = list(dict.fromkeys([*references, *(key for fact in related for key in fact["evidence_ids"])]))
            evidence = source_evidence(conn, eligible, [*original_ids, *references])
            evidence["project_paths"] = attachment_project_paths(records, scopes, [*original_ids, *references])
    except sqlite3.Error as error:
        raise ContextUnavailable("无法读取事实对应的原文依据") from error
    uncertainties = []
    if omitted:
        uncertainties.append("部分原生候选事实缺少本轮可用的精确来源，或仅指向其他项目。")
    if adjacent_unavailable:
        uncertainties.append("此截止时间早于图谱构建所用的材料范围，已跳过可能依赖后续消息的会话事实。")
    return {
        "entities": native_entities, "relations": relations, "facts": facts, "evidence_ids": references,
        "source_evidence": evidence,
        "source_expansion": {"seed_ids": seeds, "paths": source_paths, "relation_ids": [r["id"] for r in relations]},
        "uncertainties": uncertainties,
        "coverage": {
            "visible_records": len(records), "native_retrieval_mode": answer.get("mode"),
            "native_latency_ms": answer.get("latency_ms"), "native_query_calls": 1,
            "native_fact_candidates": len(fact_ids), "omitted_native_facts": omitted,
            "cutoff_affected_conversations": len(adjacent_unavailable),
            "expanded_source_records": len(source_paths),
            "associated_fact_paths": association_paths,
            "native_seed_ids": seeds, "native_participant_links": len(relations),
            "source_expansion_paths": source_paths,
            "comparison_evidence_ids": [
                ref for ref in references if project_scope(records[ref], project_id) in _COMPARISON_SCOPES
            ],
        },
    }


def _browse_relations(relations: list[dict], facts: list[dict]) -> list[dict]:
    """Group equivalent display rows while retaining all supporting sources."""
    claims = {"fact:" + fact["id"]: fact["text"] for fact in facts}
    groups = {}
    for relation in relations:
        claim = claims[relation["source_id"]]
        key = (claim, relation["source_name"], relation["type"], relation["target_id"],
               tuple(relation["scopes"]))
        if key not in groups:
            groups[key] = {**relation, "source_text": claim, "evidence_ids": []}
        row = groups[key]
        row["evidence_ids"] = list(dict.fromkeys([*row["evidence_ids"], *relation["evidence_ids"]]))
    return list(groups.values())


def list_context(
    data_dir: Path, dataset, view: str, query: str, offset: int, limit: int,
    project_id: str = "项目A", *, include_native: bool = True, group_relations: bool = False,
) -> tuple[list[dict], int]:
    records = visible_records(dataset, project_id)
    if view not in {"entities", "facts", "relations"}:
        raise ValueError("未知的上下文视图")
    unavailable = {row["conversation_id"] for key, row in dataset.records.items()
                   if key not in records and row.get("conversation_id")}
    scopes = _fact_scopes(records, project_id, unavailable)
    eligible = {key: records[key] for key in scopes}
    if include_native and (data_dir / "knowledge.db").is_file():
        entities, facts, _, _ = _native_graph(
            data_dir, records, project_id, None, [], None if view == "relations" else {}, unavailable)
        relations = grounded_graph(facts, entities, eligible)[1] if view == "relations" else []
        if view == "relations" and group_relations:
            relations = _browse_relations(relations, facts)
    else:
        entities, relations = _work_graph(records, project_id) if view != "facts" else ([], [])
        facts = []
    values = {"entities": entities, "facts": facts, "relations": relations}[view]
    query = query.strip().casefold()
    filtered = []
    for value in values:
        display = " ".join(str(value.get(key, "")) for key in (
            "id", "name", "type", "text", "source_name", "source_text", "target_name", "aliases",
        ))
        if group_relations and view == "relations":
            display += " " + " ".join(value["evidence_ids"])
        if not query or query in display.casefold():
            filtered.append(value)
    page = filtered[offset:offset + limit]
    if view == "facts":
        for fact in page:
            support = fact_support(fact["text"], fact["evidence_ids"][0], eligible.get, fact.get("source_quote"))
            fact["support"] = support
            fact["evidence_ids"] = list(dict.fromkeys(q["unit_id"] for q in support["quotes"]))
    return page, len(filtered)
