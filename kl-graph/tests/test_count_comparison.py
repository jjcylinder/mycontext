from dataclasses import replace

import pytest

from kl_graph.query.count_comparison import asks_for_combined_count, compare_counts, observed_count_fact
from project_demo import answer

QUESTION = "甲组和乙组合计涉及多少个样例？"


def cohort_evidence(extra="", second_check="功能验收"):
    first = "甲组2个样例功能验收通过"
    second = f"乙组2个样例{second_check}通过"
    evidence = [{"id": "report", "content": f"{first}；{second}。{extra}", "project_scope": "requested"}]
    return evidence, {"report": [observed_count_fact(first), observed_count_fact(second)]}


@pytest.mark.parametrize(("details", "check", "total", "member_text", "criterion_text"), [
    ("", "数字检查", None, "是否重叠", "未建立统一"),
    ("甲组与乙组样例互斥。甲组和乙组验收标准相同。", "功能验收", 4, "互不重叠", "相同口径"),
    ("甲组通过样例完整清单：E1、E2。乙组通过样例完整清单：E2、E3。", "功能验收", 3, "有重叠", "未建立统一"),
    ("甲组通过样例完整清单：E1、E2。乙组通过样例完整清单：E3、E4。甲组和乙组验收标准不同。", "数字检查", 4, "互不重叠", "口径不同"),
])
def test_counts_keep_distinct_samples_and_check_criteria_separate(details, check, total, member_text, criterion_text):
    evidence, facts = cohort_evidence(details, check)
    result = compare_counts(QUESTION, evidence, facts)
    assert result["distinct_samples"] == total
    assert member_text in result["membership"] and criterion_text in result["criterion"]
    assert result["evidence_ids"] == ["report"]
    assert all(g["quote"] in evidence[0]["content"] for g in result["groups"])


def test_comparison_uses_selected_quotes_and_project_scope():
    evidence, facts = cohort_evidence("甲组与乙组样例互斥。")
    evidence[0]["model_excerpt"] = evidence[0]["content"].split("。")[0]
    assert compare_counts(QUESTION, evidence, facts)["distinct_samples"] is None
    assert compare_counts(QUESTION, evidence, {}) is None
    evidence[0]["project_scope"] = "comparison"
    assert compare_counts(QUESTION, evidence, facts) is None


def test_partial_lists_and_conditionals_do_not_establish_disjointness():
    evidence, facts = cohort_evidence("甲组样例举例：E1。若甲组与乙组样例互斥。")
    assert compare_counts(QUESTION, evidence, facts)["distinct_samples"] is None
    evidence[0]["content"] += "甲组通过样例完整清单：E1。"
    assert compare_counts(QUESTION, evidence, facts) is None


def test_disjoint_check_counts_do_not_give_distinct_sample_count():
    first, second = "甲组2条功能测试通过", "乙组2条安全检查通过"
    rows = [{"id": "s", "content": f"{first}。{second}。甲组与乙组样例互斥。", "project_scope": "requested"}]
    result = compare_counts(QUESTION, rows, {"s": list(map(observed_count_fact, (first, second)))})
    assert result["distinct_samples"] is None and "互不重叠" in result["membership"]


@pytest.mark.parametrize("question", [
    "请按六类应用汇总已完成、未决、演示范围及下一步责任。",
    "2个样例和3个样例的检查分别由谁负责？",
    "12包已经发布，是否可以沿用所有预演结果？",
    "两组各自的通过率如何？",
])
def test_ordinary_questions_bypass_comparison(question):
    assert not asks_for_combined_count(question)


@pytest.mark.asyncio
async def test_comparison_added_only_after_ordinary_budget_packing(monkeypatch):
    async def rank(*args, **kwargs):
        return {"groups": [{"ids": ["report"], "topic": "验收", "aspects": ["范围"], "priority": 3}]}

    monkeypatch.setattr(answer, "rank_evidence", rank)
    sources, facts = cohort_evidence()
    record = dict(record_id="report", title="项目A", project_hint="项目A", source_type="document",
                  source_id="doc", occurred_at="2026-01-01", source_locator="doc#1", is_simulated=True,
                  content=sources[0]["content"])
    model = answer.AnswerModel("", "test", "", 32768, 12000, 1800, "", "")
    async def prepare(model, mode, facts):
        ctx = {"source_evidence": {"dependencies": {}, "audit": {}, "projections": {}, "count_facts": facts}}
        return await answer.prepare_answer(model, mode, QUESTION, "项目A", "2026-01-02", {"report": record},
                                           ctx, vector_refs=["report"])
    base, evidence, _, cov = await prepare(model, "C", {})
    extended, same, graph, _ = await prepare(model, "C", facts)
    assert extended != base and same == evidence and graph["count_comparisons"]
    assert (await prepare(model, "B", facts))[0] == base
    tight = replace(model, evidence_budget=cov["preflight_tokens"])
    assert (await prepare(tight, "C", facts))[0] == base


def test_multi_group_total_requires_all_pairs_or_complete_rosters():
    evidence, facts = cohort_evidence("甲组与乙组样例互斥。")
    third = "丙组3个样例功能验收通过"
    evidence[0]["content"] += third + "。"
    facts["report"].append(observed_count_fact(third))
    question = "甲组、乙组和丙组合计多少个样例？"
    assert compare_counts(question, evidence, facts)["distinct_samples"] is None
    evidence[0]["content"] += "甲组与丙组样例互斥。乙组与丙组样例互斥。"
    assert compare_counts(question, evidence, facts)["distinct_samples"] == 7


def test_cannot_infer_operands_from_unrelated_counts_or_operation_names():
    evidence, facts = cohort_evidence("甲组和乙组检查互斥。")
    assert compare_counts(QUESTION, evidence, facts)["distinct_samples"] is None
    assert compare_counts("三条测试与四条评审通过，能否报七条完整通过？", evidence, facts) is None


def test_conflicting_claims_and_unquoted_native_counts_abstain():
    evidence, facts = cohort_evidence("甲组与乙组样例互斥。甲组与乙组样例有重叠。")
    assert compare_counts(QUESTION, evidence, facts) is None
    evidence[0]["model_excerpt"] = "甲组2个样例功能验收通过；乙组2个样例功能验收未通过。"
    assert compare_counts(QUESTION, evidence, facts) is None


def test_local_roster_ids_from_unrelated_documents_cannot_be_deduplicated():
    evidence, facts = cohort_evidence()
    for group in ("甲组", "乙组"):
        evidence.append({"id": group, "source_id": group, "project_scope": "requested",
                         "content": group + "通过样例完整清单：E1、E2。"})
    assert compare_counts(QUESTION, evidence, facts)["distinct_samples"] is None
    for row in evidence[1:]:
        row["source_id"] = "shared-register"
    assert compare_counts(QUESTION, evidence, facts)["distinct_samples"] == 2


def test_explicit_sample_count_and_nonexact_quantities():
    assert observed_count_fact("甲组有十二个不同样例通过。") == ("甲组", 12, "个不同样例", "")
    for value in ("甲组约12个样例通过。", "甲组至少3个样例通过。", "甲组2个样例未通过。",
                  "甲组三四个样例通过。", "甲组一百二个样例通过。"):
        assert observed_count_fact(value) is None
