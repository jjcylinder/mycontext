import json
from types import SimpleNamespace

import pytest

from project_demo.selection import _compact_sources, _messages, pack_evidence, rank_evidence


@pytest.mark.asyncio
async def test_packing_skips_oversize_group_and_preserves_supporting_pair():
    lengths = {"large": 100, "parent": 8, "reply": 8, "duplicate": 8, "new": 8}

    def build(ids):
        return [{"content": "x" * sum(lengths[key] for key in ids)}], [], {}

    async def count(messages):
        return len(messages[0]["content"])

    groups = [
        {"ids": ["large"], "priority": 3, "topic": "large", "aspects": ["状态"]},
        {"ids": ["parent", "reply"], "priority": 3, "topic": "发布", "aspects": ["项目", "版本"]},
        {"ids": ["duplicate"], "priority": 3, "topic": "发布", "aspects": ["项目"]},
        {"ids": ["new"], "priority": 3, "topic": "发布", "aspects": ["环境"]},
    ]
    selected, decisions = await pack_evidence(SimpleNamespace(count=count), groups, build, 24)
    assert set(selected) == {"parent", "reply", "new"}
    assert any(row.get("reason_excluded") == "budget" for row in decisions)


@pytest.mark.asyncio
async def test_rank_rejects_invented_source_and_sends_full_originals():
    async def count(messages):
        return len(json.dumps(messages))

    async def complete(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["records"][0]["content"] == "原文中的完整条件"
        return json.dumps({"groups": [{"ids": ["invented"], "priority": 3,
                         "topic": "发布", "aspects": ["状态"], "reason": "x"}]}), {}

    model = SimpleNamespace(context_window=10000, max_output=1000, count=count, complete=complete)
    records = {"source": {"title": "发布", "content": "原文中的完整条件", "source_id": "doc"}}
    with pytest.raises(ValueError, match="无效ID"):
        await rank_evidence(model, "何时发布", "项目A", "2026-11-20", records, ["source"])


@pytest.mark.asyncio
async def test_rank_batches_without_losing_originals_or_exceeding_capacity():
    seen = []
    stages = []

    async def count(messages):
        # Capacity here allows two whole records, including the output reserve.
        return 10 + 10 * len(json.loads(messages[1]["content"])["records"])

    async def complete(messages):
        rows = json.loads(messages[1]["content"])["records"]
        assert await count(messages) <= 30
        seen.extend(row["id"] for row in rows)
        return json.dumps({"groups": [
            {"ids": [row["id"]], "priority": 3, "topic": "发布",
             "aspects": ["环境"], "reason": "明确环境"} for row in rows
        ]}), {"total_tokens": 1}

    model = SimpleNamespace(context_window=40, max_output=10, count=count, complete=complete)
    records = {str(i): {"title": "发布", "content": f"完整原文{i}", "source_id": "doc"}
               for i in range(5)}
    result = await rank_evidence(model, "何时发布", "项目A", "2026-11-20", records, list(records),
                                 progress=lambda *stage: stages.append(stage))
    assert seen[:5] == list(records)
    assert [call["input_count"] for call in result["calls"][:3]] == [30, 30, 20]
    assert all(call["input_count"] <= 30 for call in result["calls"])
    assert result["global_ranking"] == "completed"
    assert result["global_review_method"] == "bounded_group_cards"
    assert result["global_review_batches"] == 3
    assert stages == [
        ("candidate_ranking", 1, 3), ("candidate_ranking", 2, 3), ("candidate_ranking", 3, 3),
        ("global_ranking", None, None),
        ("global_ranking", 1, 3), ("global_ranking", 2, 3), ("global_ranking", 3, 3),
    ]
    assert set(seen[5:]) == set(records)
    assert len(result["groups"]) == 5


@pytest.mark.asyncio
async def test_graph_sources_reconsidered_globally_and_identical_calls_cached():
    calls = []

    async def count(messages):
        return len(json.dumps(messages, ensure_ascii=False))

    async def complete(messages):
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        # A graph fact points to a necessary source omitted by initial ranking.
        ids = ["a", "b"] if payload.get("graph_hints") else ["a"]
        return json.dumps({"groups": [{"ids": ids, "priority": 3, "topic": "发布",
                                      "aspects": ["适用范围"], "reason": "条件需配套"}]}), {"total_tokens": 9}

    model = SimpleNamespace(context_window=10000, max_output=1000, count=count, complete=complete)
    rows = {key: {"title": "发布", "source_id": "doc", "content": key} for key in ("a", "b")}
    cache = {}
    b = await rank_evidence(model, "发布范围", "项目A", "2026-11-20", rows, list(rows), cache=cache)
    c = await rank_evidence(model, "发布范围", "项目A", "2026-11-20", rows, list(rows), cache=cache,
                            graph={"facts": [{"text": "仅预演", "evidence_ids": ["b"]}]})
    assert len(calls) == 3  # Initial ranking is reused; both modes perform global review.
    assert b["groups"][0]["ids"] == ["a"]
    assert c["groups"][0]["ids"] == ["a", "b"]
    assert c["calls"][0]["cache_hit"] and c["calls"][0]["usage"] == {}
    assert c["calls"][1]["graph_hints"] == 1


def test_global_compaction_preserves_sources_and_complete_records():
    rows = [{"id": str(i), "source_id": "doc", "title": "完整来源标题" * 10,
             "content": f"原文{i}：尚未确认，不能沿用。", "project_hint": "项目A",
             "occurred_at": "2026-11-20", "reply_to": "parent"} for i in range(8)]
    message = _messages("进展", "项目A", "2026-11-20", rows)[1]
    compact = _compact_sources(message)
    payload = json.loads(compact["content"])
    assert len(compact["content"].encode()) < len(message["content"].encode())
    compact_rows = [dict(zip(payload["record_columns"], values)) for values in payload["records"]]
    restored = [{**{k: v for k, v in row.items() if k != "source"},
                 **payload["sources"][row["source"]]} for row in compact_rows]
    assert restored == rows


@pytest.mark.asyncio
async def test_semantic_coverage_precedes_short_text_discount():
    lengths = {"unqualified_chat": 1, "scope": 7, "condition": 7}

    def build(ids):
        return [{"content": "x" * sum(lengths[k] for k in ids)}], [], {}

    async def count(messages):
        return len(messages[0]["content"])

    groups = [
        {"ids": ["scope", "condition"], "topic": "发布", "aspects": ["范围", "条件"], "priority": 3},
        {"ids": ["unqualified_chat"], "topic": "发布", "aspects": ["范围"], "priority": 3},
    ]
    selected, _ = await pack_evidence(SimpleNamespace(count=count), groups, build, 14)
    assert selected == ["scope", "condition"]


@pytest.mark.asyncio
async def test_coverage_audit_recovers_scoped_condition_without_auto_adding_background():
    seen = []
    records = {
        'main': {'title': '核查', 'source_id': 'doc', 'content': '预演通过。', 'project_hint': '项目A'},
        'scope': {'title': '核查', 'source_id': 'doc', 'content': '项目A正式仍未升级。', 'project_hint': '项目A'},
        'weak': {'title': '群聊', 'source_id': 'chat', 'content': '正式未升级。'},
        'noise': {'title': '指南', 'source_id': 'guide', 'content': '一般背景。'},
    }
    def group(ids, priority=3):
        return {'ids': ids, 'priority': priority, 'topic': '发布', 'aspects': ['适用范围'], 'reason': '版本限定'}

    async def count(messages):
        return len(json.dumps(messages, ensure_ascii=False))

    async def complete(messages):
        payload = json.loads(messages[1]['content']);seen.append(payload)
        if 'omitted_groups' in payload:
            assert set(payload['selected_ids']) == {'main', 'weak'}
            by_id = {row['id']: row for row in payload['records']}
            assert by_id['scope']['project_scope'] == 'requested'
            assert by_id['weak']['project_scope'] == 'unknown'
            assert 'noise' not in by_id
            result = [group(['main', 'scope'])]
        elif len(seen) == 1:
            result = [group(['main', 'scope']), group(['weak'], 2), group(['noise'], 1)]
        else:
            result = [group(['main', 'weak'])]
        return json.dumps({'groups': result}), {}

    model = SimpleNamespace(context_window=20000, max_output=1000, count=count, complete=complete)
    result = await rank_evidence(model, '是否正式发布', '项目A', '2026-11-20', records, list(records))
    assert result['calls'][-1]['stage'] == 'coverage_audit'
    assert result['coverage_audit']['review_batches'] == 1
    assert {i for g in result['groups'] for i in g['ids']} == {'main', 'weak', 'scope'}


@pytest.mark.asyncio
async def test_overflowing_audit_retains_necessary_bundle_for_budget_packing():
    group = {'ids': ['a', 'b'], 'priority': 3, 'topic': '部署', 'aspects': ['范围'], 'reason': '需配套'}
    calls = 0

    async def count(messages):
        return 1000 if 'omitted_groups' in json.loads(messages[1]['content']) else 20

    async def complete(messages):
        nonlocal calls
        calls += 1
        return json.dumps({'groups': [group if calls == 1 else {**group, 'ids': ['a']}]}), {}

    records = {key: {'title': '部署', 'source_id': 'doc', 'content': key} for key in ['a', 'b']}
    model = SimpleNamespace(context_window=100, max_output=10, count=count, complete=complete)
    result = await rank_evidence(model, '部署范围', '项目A', '2026-11-20', records, list(records))
    assert calls == 2
    assert result['coverage_audit']['unreviewed_retained_groups'] == [group]
    assert {i for g in result['groups'] for i in g['ids']} == {'a', 'b'}


@pytest.mark.asyncio
async def test_more_aspect_labels_do_not_override_semantic_order():
    sizes = {'scope': 5, 'required_next_step': 5, 'broad_addition': 7}
    def build(ids):
        return [{'content': 'x' * sum(sizes[i] for i in ids)}], [], {}
    async def count(messages):
        return len(messages[0]['content'])
    groups = [
        {'ids': ['scope'], 'topic': '发布', 'priority': 3, 'aspects': ['完成状态', '适用范围']},
        {'ids': ['required_next_step'], 'topic': '发布', 'priority': 3, 'aspects': ['未决问题']},
        {'ids': ['broad_addition'], 'topic': '发布', 'priority': 3,
         'aspects': ['未决问题', '责任分工', '归属', '变更']},
    ]
    selected, _ = await pack_evidence(SimpleNamespace(count=count), groups, build, 10)
    assert selected == ['scope', 'required_next_step']


@pytest.mark.asyncio
async def test_audit_compares_same_label_sources_before_treating_a_condition_as_covered():
    texts = {"primary": "夜间访客须登记。", "duplicate": "夜间来访者需要登记。",
             "extra": "夜间送货须预约。", "noise": "大楼去年落成。"}
    rows = {key: {"title": "通知", "source_id": "doc", "content": text} for key, text in texts.items()}
    stages = []
    def group(key, priority):
        return {"ids": [key], "priority": priority, "topic": "大楼", "aspects": ["夜间通行"], "reason": texts[key]}
    async def count(messages):
        return 10
    async def complete(messages):
        payload = json.loads(messages[1]["content"])
        if "omitted_groups" in payload:
            stages.append("audit")
            assert payload["selected_ids"] == ["primary"]
            assert [g["ids"] for g in payload["omitted_groups"]] == [["duplicate"], ["extra"]]
            assert {r["content"] for r in payload["records"]} == {texts[k] for k in ("primary", "duplicate", "extra")}
            groups = [group("extra", 2)]
        elif not stages:
            stages.append("initial")
            groups = [group("primary", 3), group("duplicate", 2), group("extra", 2), group("noise", 1)]
        else:
            stages.append("global")
            groups = [group("primary", 3)]
        return json.dumps({"groups": groups}), {}
    model = SimpleNamespace(context_window=1000, max_output=20, count=count, complete=complete)
    result = await rank_evidence(model, "夜间通行有哪些要求？", "项目A", "2026-11-20", rows, list(rows))
    assert stages == ["initial", "global", "audit"]
    assert result["coverage_audit"]["review_candidates"] == 2
    assert [g["ids"] for g in result["groups"]] == [["primary"], ["extra"]]
