import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from project_demo import app as workbench
from project_demo import answer as answer_module
from project_demo.answer import AnswerModel, UnavailableError, parse_answer, prepare_answer, rank_originals
from project_demo.runtime import SOURCE_ID, audit_import


def record(key="E1", text="项目A 联调", **other):
    return {"record_id": key, "source_id": "G001", "source_type": "chat", "title": "联调讨论",
            "occurred_at": "2026-11-20T10:00:00+08:00", "is_simulated": True,
            "source_locator": f"corpus/chat/messages.jsonl#{key}", "content": text, **other}


def settings():
    return AnswerModel("", "test-model", "", 100_000, 3500, 1800, "", "")


async def fake_rank(model, question, project, cutoff, records, candidates, **kwargs):
    return {"groups": [{"ids": [key], "priority": 3, "topic": key,
                        "aspects": ["状态"], "reason": "test"} for key in candidates], "calls": []}


@pytest.mark.asyncio
async def test_answer_baselines_share_scope_and_retrieval_budget(monkeypatch):
    monkeypatch.setattr(answer_module, "rank_evidence", fake_rank)
    records = {"E1": record(), "E2": record("E2", "项目B 通过"),
               "E3": record("E3", "候选包", reply_to="E1")}
    graph = {"evidence_ids": ["E3"], "facts": [
        {"text": "联调候选包", "evidence_ids": ["E3"]},
        {"text": "没有原文的结论", "evidence_ids": ["not-in-application"]},
    ]}
    answer = settings()
    cutoff = "2026-11-20T23:59:00+08:00"
    messages, evidence, context, a = await prepare_answer(answer, "A", "联调", "项目A", cutoff, records)
    assert a["all_records_included"] and [row["id"] for row in evidence] == list(records)
    assert evidence[1]["project_scope"] == "other"
    assert "extracted_context" not in json.loads(messages[1]["content"])
    assert not context
    _, _, context, b = await prepare_answer(answer, "B", "联调", "项目A", cutoff, records, vector_refs=[])
    assert not context
    _, _, context, c = await prepare_answer(answer, "C", "联调", "项目A", cutoff, records, graph, vector_refs=[])
    assert b["input_budget"] == c["input_budget"] == answer.evidence_budget
    assert context["facts"] == graph["facts"][:1]
    assert c["used_graph_facts"] == 1


@pytest.mark.asyncio
async def test_legacy_count_facts_do_not_change_answer_input(monkeypatch):
    monkeypatch.setattr(answer_module, "rank_evidence", fake_rank)
    records = {"report": record("report", "甲组2个样例功能验收通过；乙组2个样例功能验收通过。",
                                source_type="document", project_hint="项目A")}
    legacy = {"source_evidence": {"dependencies": {}, "audit": {}, "projections": {},
                                  "count_facts": {"report": [["甲组", 2, "个样例", "功能验收"],
                                                             ["乙组", 2, "个样例", "功能验收"]]}}}
    args = ("甲组和乙组合计涉及多少个样例？", "项目A", "2026-11-20", records)
    b, *_ = await prepare_answer(settings(), "B", *args, vector_refs=["report"])
    c, evidence, _, _ = await prepare_answer(settings(), "C", *args, legacy, vector_refs=["report"])
    assert [row["id"] for row in evidence] == ["report"]
    assert c == b


@pytest.mark.asyncio
async def test_b_and_c_share_hybrid_originals_and_only_c_adds_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(answer_module, "rank_evidence", fake_rank)
    records = {
        "keyword": {**record("keyword", "待核查的事项"), "title": "关键词记录"},
        "semantic": {**record("semantic", "向量命中的原文"), "title": "向量记录"},
        "graph": {**record("graph", "关系发现的补充材料"), "title": "图谱记录"},
    }
    cutoff = "2026-11-20T23:59:00+08:00"
    dataset = SimpleNamespace(id="test", records=records, cutoff=cutoff)
    demo = workbench.Demo(tmp_path, tmp_path / "state")
    demo.datasets[dataset.id] = dataset
    demo.model = replace(settings(), base_url="http://localhost:9000/v1")
    calls = []
    prompts = []

    async def start():
        return "http://localhost:9001"

    async def query_original_vectors(base_url, data_dir, question, visible):
        calls.append((base_url, data_dir, question, list(visible)))
        return ["semantic", "semantic", "unavailable"]

    fact = {"text": "待核查的关联事实", "evidence_ids": ["graph"]}

    async def query_context(*args, **kwargs):
        calls.append("graph")
        return {"evidence_ids": ["graph"], "facts": [fact]}

    async def complete(self, messages):
        prompts.append(json.loads(messages[1]["content"]))
        return json.dumps({key: [] for key in ("progress", "changes", "open_issues",
                                               "coordination", "uncertainties")}), {}

    monkeypatch.setattr(demo, "graph", lambda _: SimpleNamespace(ready=True, data_dir=tmp_path, start=start))
    monkeypatch.setattr(workbench, "graph_is_local", lambda: True)
    monkeypatch.setattr(workbench, "query_original_vectors", query_original_vectors)
    monkeypatch.setattr(workbench, "query_context", query_context)
    monkeypatch.setattr(AnswerModel, "complete", complete)

    b, c = [await demo.check({"run_id": mode, "dataset_id": dataset.id, "mode": mode,
                              "project_id": "项目A", "question": "待核查"}) for mode in ("B", "C")]
    assert calls == [("http://localhost:9001", tmp_path, "待核查", list(records))] * 2 + ["graph"]
    assert b["coverage"]["original_retrieval"] == c["coverage"]["original_retrieval"]
    assert b["coverage"]["original_retrieval"]["candidate_ids"] == ["keyword", "semantic"]
    assert b["coverage"]["input_budget"] == c["coverage"]["input_budget"]
    assert [row["id"] for row in b["evidence"]] == ["keyword", "semantic"]
    assert {row["id"] for row in c["evidence"]} == set(records)
    assert "extracted_context" not in prompts[0]
    assert prompts[1]["extracted_context"]["facts"] == [fact]


@pytest.mark.asyncio
async def test_graph_supplements_cannot_displace_selected_originals(monkeypatch):
    monkeypatch.setattr(answer_module, "rank_evidence", fake_rank)
    records = {"E1": record(text="发布已完成"), "E2": record("E2", "发布仅限预演环境")}
    cutoff = "2026-11-20"
    model = replace(settings(), evidence_budget=10000)
    _, originals, _, baseline = await prepare_answer(
        model, "B", "发布范围", "项目A", cutoff, records, vector_refs=list(records),
    )
    model = replace(model, evidence_budget=baseline["preflight_tokens"])
    graph = {"evidence_ids": list(records), "facts": [
        {"text": "发布范围尚需核查" * 100, "evidence_ids": ["E1", "E2"]},
    ]}
    _, evidence, context, coverage = await prepare_answer(
        model, "C", "发布范围", "项目A", cutoff, records, graph, vector_refs=list(records),
    )
    assert evidence == originals
    assert not context.get("facts")
    assert coverage["used_graph_facts"] == 0
    assert coverage["preflight_tokens"] == model.evidence_budget


@pytest.mark.asyncio
async def test_compact_input_preserves_full_text_time_scope_and_source_lookup(monkeypatch):
    monkeypatch.setattr(answer_module, "rank_evidence", fake_rank)
    rows = {"E1": record(text="拟发布，尚未确认。", project_hint="项目A"),
            "E2": record("E2", "只有预演通过；正式环境未变。", project_hint="项目A", reply_to="E1")}
    messages, evidence, _, coverage = await prepare_answer(
        replace(settings(), evidence_budget=10000), "B", "发布范围", "项目A", "2026-11-20", rows,
        vector_refs=list(rows),
    )
    payload = json.loads(messages[1]["content"])
    assert len(payload["sources"]) == 1
    assert {i for group in payload["evidence_index"] for i in group["ids"]} == set(rows)
    assert all(set(group) == {"topic", "aspects", "ids"} for group in payload["evidence_index"])
    for excerpt in payload["source_excerpts"]:
        original = rows[excerpt["id"]]
        assert excerpt["content"] == original["content"]
        assert excerpt["occurred_at"] == original["occurred_at"]
        assert excerpt["project_scope"] == "requested"
        assert payload["sources"][excerpt["source"]]["source_id"] == original["source_id"]
    assert all(row["source_locator"] for row in evidence)
    assert coverage["preflight_tokens"] < await settings().count(
        answer_module._prompt("发布范围", "项目A", "2026-11-20", evidence, {},
                              evidence_index=payload["evidence_index"]))


@pytest.mark.asyncio
async def test_c_uses_fact_supported_source_before_budget_packing(monkeypatch):
    monkeypatch.setattr(answer_module, "rank_evidence", fake_rank)
    original = "背景介绍和操作指南。" * 300 + "APP-Z-7仅预演通过；正式未部署，仍待验证。"
    rows = {"doc": record("doc", original, source_type="document", project_hint="项目A")}
    graph = {"evidence_ids": ["doc"], "facts": [], "source_facts": [
        {"id": "native-fact", "type": "STATUS", "text": "APP-Z-7仅预演通过。", "evidence_ids": ["doc"]}
    ]}
    model = replace(settings(), evidence_budget=4000)
    _, b_evidence, _, b = await prepare_answer(model, "B", "发布范围", "项目A", "2026-11-20",
                                               rows, vector_refs=["doc"])
    messages, evidence, _, c = await prepare_answer(model, "C", "发布范围", "项目A", "2026-11-20",
                                                    rows, graph, vector_refs=["doc"])
    assert b["input_budget"] == c["input_budget"] == 4000
    assert not b_evidence and len(evidence) == 1
    assert evidence[0]["content"] == original  # The UI/audit retains the full source.
    payload = json.loads(messages[1]["content"])
    excerpt = payload["source_excerpts"][0]
    assert "正式未部署，仍待验证" in excerpt["content"]
    assert excerpt["project_scope"] == "requested"
    assert excerpt["representation"] == "fact_supported_verbatim_excerpt"
    assert c["fact_projection"]["fact_ids"] == ["native-fact"]
    assert c["preflight_tokens"] <= c["input_budget"]


@pytest.mark.asyncio
async def test_all_originals_refuse_instead_of_truncating_with_exact_window(monkeypatch):
    async def count(_self, messages):
        assert len(json.loads(messages[1]["content"])["source_excerpts"]) == 2
        return 5800

    monkeypatch.setattr(AnswerModel, "count", count)
    model = replace(settings(), context_window=6000, tokenize_url="http://model/tokenize")
    with pytest.raises(UnavailableError, match="5800 Token"):
        await prepare_answer(model, "A", "进度", "项目A", "2026-11-20", {"E1": record(), "E2": record("E2")})


def test_citations_and_reply_expansion():
    records = {"E1": record(text="旧消息"), "E2": record("E2", "公函验收回归", reply_to="E1"),
               "E3": record("E3", "候选包")}
    assert rank_originals("公函验收", records)[:2] == ["E2", "E1"]
    answer = {"progress": [{"text": "候选包已回归", "evidence_ids": ["E2"]}],
              "changes": [], "open_issues": [], "coordination": [], "uncertainties": []}
    assert parse_answer(json.dumps(answer), {"E2"})["progress"][0]["evidence_ids"] == ["E2"]
    with pytest.raises(ValueError, match="未提供给模型"):
        parse_answer(json.dumps(answer), {"E1"})


@pytest.mark.asyncio
async def test_raw_import_evidence_unavailable_state_and_restoration(tmp_path, monkeypatch):
    dataset = SimpleNamespace(root=tmp_path, profile="app_focus", id="app_focus-test", digest="digest",
                              cutoff="2026-11-20T23:59:00+08:00", records={"E1": record()})
    monkeypatch.setattr(workbench, "load_dataset", lambda root, profile: dataset)
    monkeypatch.setattr(workbench, "write_dws", lambda data, path: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(workbench, "graph_configured", lambda: False)
    monkeypatch.setattr(AnswerModel, "from_env", lambda: settings())
    app = workbench.create_app(tmp_path, tmp_path / "state")

    async def request_job(client, route, request):
        response = await client.post(route, json=request)
        assert response.status_code == 202
        run = response.json()
        for _ in range(20):
            if run["status"] not in {"running", "queued"}:
                break
            await asyncio.sleep(0)
            run = (await client.get(f"/demo/runs/{run['run_id']}")).json()
        return run

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://demo") as client:
            run = await request_job(client, "/demo/import", {"profile": "app_focus"})
            assert run["status"] == "completed" and run["result"]["graph_status"] == "unavailable"
            original = await client.get("/demo/evidence/E1", params={"dataset_id": dataset.id})
            assert original.json()["source_locator"].endswith("#E1")
            check = await request_job(client, "/demo/check", {"dataset_id": dataset.id,
                                                              "mode": "C", "question": "联调进展？"})
            assert check["status"] == "unavailable" and "未配置" in check["error"]

    restored = workbench.create_app(tmp_path, tmp_path / "state")
    async with restored.router.lifespan_context(restored):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored), base_url="http://demo") as client:
            profiles = (await client.get("/demo/profiles")).json()["profiles"]
            assert profiles[0]["loaded"] and profiles[0]["dataset_id"] == dataset.id


def test_native_completion_recovery_requires_exact_source_lineage_and_prior_success(tmp_path):
    from kl_graph.storage.sqlite_store import SQLiteStore

    database = tmp_path / "knowledge.db"
    store = SQLiteStore(database)
    store.close()
    with sqlite3.connect(database) as conn:
        conn.execute("INSERT INTO units (source_id, source_type, unit_id, content_hash, processed_at) "
                     "VALUES (?, 'message', 'E1', 'hash', 0)", (SOURCE_ID,))
        conn.execute("INSERT INTO chunk_units (chunk_id, source_id, source_type, unit_id, "
                     "unit_ordinal_in_chunk, chunk_ordinal_in_unit) VALUES ('chunk1', ?, 'message', 'E1', 0, 0)",
                     (SOURCE_ID,))
    with pytest.raises(RuntimeError, match="图谱导入对账"):
        audit_import(database, tmp_path, ["E1"], {"state": "done"})

    with sqlite3.connect(database) as conn:
        conn.execute("INSERT INTO ingest_runs (run_id, source_id, input_dir, state, phase, "
                     "started_at, updated_at, completed_at, units_discovered, units_processed, extraction_failed) "
                     "VALUES ('prior', ?, ?, 'done', '', 1, 2, 2, 1, 1, 0)", (SOURCE_ID, str(tmp_path)))
    counts, recovered = audit_import(database, tmp_path, ["E1"], {"state": "done"})
    assert counts["units_processed"] == 1 and recovered == "prior"
    with pytest.raises(RuntimeError, match="原文溯源对账"):
        audit_import(database, tmp_path, ["wrong-unit"], {"state": "done"})


def test_chatgpt_loopback_relay_applies_remote_report_guard(tmp_path):
    original_report = SimpleNamespace(records={"E1": {"is_simulated": False}})
    simulated = SimpleNamespace(records={"E1": {"is_simulated": True}})
    direct_local = workbench.Demo(tmp_path, tmp_path / "state")
    relay = workbench.Demo(tmp_path, tmp_path / "state", uses_chatgpt=True)

    assert direct_local.permit_model(original_report, local=True)
    assert not relay.permit_model(original_report, local=True)
    assert relay.permit_model(simulated, local=True)


@pytest.mark.asyncio
async def test_selected_reply_charged_with_its_grounding_sources(monkeypatch):
    async def rank(*args, **kwargs):
        return {"groups": [{"ids": ["reply"], "priority": 3, "topic": "交付",
                            "aspects": ["范围"], "reason": "承诺"}]}
    monkeypatch.setattr(answer_module, "rank_evidence", rank)
    rows = {k: record(k, text=t) for k, t in [("reply", "我补这两项。"), ("parent", "缺两项。")]}
    graph = {"evidence_ids": ["reply"], "source_evidence": {
        "projections": {}, "audit": {}, "dependencies": {"reply": ["parent"]}}}
    model = settings()
    _, evidence, _, coverage = await prepare_answer(model, "C", "什么范围", "项目A", "2026-11-20",
                                                    rows, graph, vector_refs=["reply"])
    assert [row["id"] for row in evidence] == ["reply", "parent"]
    assert coverage["evidence_selection"]["packed_dependencies"] == {"reply": ["parent"]}
    base = answer_module._prompt("什么范围", "项目A", "2026-11-20", evidence, {}, compact=True)
    base_tokens = await model.count(base)
    assert base_tokens < coverage["preflight_tokens"]
    exact = replace(model, evidence_budget=base_tokens)
    messages, same, _, _ = await prepare_answer(exact, "C", "什么范围", "项目A", "2026-11-20",
                                               rows, graph, vector_refs=["reply"])
    assert messages == base and same == evidence
    smaller = replace(model, evidence_budget=base_tokens - 1)
    _, evidence, _, _ = await prepare_answer(smaller, "C", "什么范围", "项目A", "2026-11-20",
                                              rows, graph, vector_refs=["reply"])
    assert not evidence
