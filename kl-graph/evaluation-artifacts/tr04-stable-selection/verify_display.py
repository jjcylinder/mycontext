"""Offline TR04 acceptance against a hand-authored rubric, independent of duty_plan."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import re

from project_demo.answer import parse_answer


OUT = Path(__file__).resolve().parent
BASE = OUT.parent
QUESTION = "若准备翻译Skill演示，应用、测试、模型、发布分别需要协调什么？"


def load(path):
    return json.loads(path.read_text())


def render(items, rows, expected=None):
    review = []
    answer = parse_answer(json.dumps({"items": items}), set(rows), require_items=True,
                          assignment_sources=rows, assignment_review=review, question=QUESTION,
                          answer_requirements=expected)
    return answer, review


def independent_check(answer, rows, *, require_commitment):
    duties = answer["coordination"]
    assert len(duties) == 4
    def duty(anchor):
        found, = [row for row in duties if row["text"].startswith(anchor)]
        return found["text"], set(found["evidence_ids"])
    app, refs = duty("格式及异常提示由应用")
    assert "由应用提供期望答案" in app and "ATR0-S04" in refs
    if require_commitment:
        assert all(s in app for s in ("QA-A27", "期望答案、判定说明还缺", "我补这两列"))
        assert {"T08-02", "T08-03", "T08-04"} <= refs
    test, refs = duty("整体回归由测试")
    assert "将译义和凭空补内容纳入整体回归范围" in test
    assert re.search(r"纳入回归范围[^。]*TR-X03[^。]*(?:尚未|还没有)回归", test)
    assert {"ATR0-S04", "T04-06", "T04-07"} <= refs and {"XTR2-04", "ATR1-S04"} & refs
    assert all(word not in test for word in ("等待应用侧发布申请", "解决且QA-A27补齐后", "测试负责提供期望答案"))
    model, = [i for i in duties if "模型分析还等QA-A27" in i["text"] and i["text"] != test]
    assert all(s in model["text"] for s in ("BUG-S11", "模型侧还要分析", "期望答案与判定说明"))
    assert {"T04-06", "T04-07", "XTR3-06"} <= set(model["evidence_ids"])
    release, refs = duty("发布孙工表示")
    assert all(s in release for s in ("等待应用侧发布申请", "不把预演复测记成正式上线",
                                      "正式环境的升级情况需单独核对", "正式环境未升级异常提示补丁"))
    assert {"XTR3-05", "XTR3-07"} <= refs and all("等待应用侧发布申请" not in i["text"] for i in answer["changes"])
    text = str(answer)
    assert all(s not in text for s in ("公函", "模板文号", "摘要Skill", "已批准演示", "已批准于"))
    for column in ("coordination", "progress", "changes", "open_issues"):
        assert all(set(item["evidence_ids"]) <= rows.keys() for item in answer[column])


def citation(ref, quote, **extra):
    return {"source_id": ref, "quote": quote, **extra}


def item(party, anchor, *details):
    refs = list(dict.fromkeys(c["source_id"] for c in (anchor, *details)))
    return {"object": party, "action": "协调", "scope": "翻译", "period": "current",
            "state": "known", "kind": "coordination", "text": "已批准于12月1日演示。",
            "assignment": anchor, "details": list(details), "evidence_ids": refs}


def verify_variants(control, rows, expected):
    counts = {"release": 0, "foreign_duty": 0, "unknown_party_spelling": 0}
    for kind, state, location, duplicate, obj in product(
        ("coordination", "changes", "open_issues", "progress"), ("known", "pending", "done"),
        ("assignment", "details", "next_step"), (False, True), ("发布方", "正式环境升级")
    ):
        items = deepcopy(control)
        release = items[-1]
        if duplicate:
            items.insert(0, deepcopy(release))
        release.update(kind=kind, state=state, object=obj)
        if location != "assignment":
            quote = release.pop("assignment")
            if location == "details":
                release["details"] = [quote]
            else:
                release.pop("details")
                release["next_step"] = {"basis": [quote]}
        answer, review = render(items, rows, expected)
        independent_check(answer, rows, require_commitment=True)
        assert not any(r["status"].startswith("missing") for r in review)
        assert not any("等待应用侧发布申请" in r["text"]
                       for field in ("changes", "progress", "open_issues") for r in answer[field])
        counts["release"] += 1

    for tag, position in product((None, "check", "related"), ("first", "last", "assignment")):
        items = deepcopy(control)
        test = items[1]
        foreign = [items[-1]["assignment"].copy(), items[2]["details"][-1].copy()]
        for quote in foreign:
            test["evidence_ids"].append(quote["source_id"])
            if tag:
                quote["use"] = tag
        if position == "assignment":
            test["details"] += [test["assignment"], foreign[1]]
            test["assignment"] = foreign[0]
        else:
            test["details"] = foreign + test["details"] if position == "first" else test["details"] + foreign
        answer, review = render(items, rows, expected)
        independent_check(answer, rows, require_commitment=True)
        test_display, = [r for r in answer["coordination"] if r["text"].startswith("整体回归由测试")]
        assert "模型分析还等" not in test_display["text"]
        assert not {"XTR3-06", "XTR3-07"}.intersection(test_display["evidence_ids"])
        assert not any(r["status"].startswith("missing") for r in review)
        counts["foreign_duty"] += 1

    for alias in ("发布", "发布方", "发布团队", " 发布侧 "):
        items = deepcopy(control)
        items[-1].update(object=alias, state="unknown", evidence_ids=[])
        answer, review = render(items, rows, expected)
        assert any(r["status"] == "missing_duty" and r["object"] == "发布" for r in review)
        assert len(answer["coordination"]) == 3
        counts["unknown_party_spelling"] += 1
    return counts


def main():
    rows = {r["id"]: r for r in load(OUT / "sources-C-1.json")}
    requirements = load(OUT / "cold-1.json")["C"]["requirements"]
    responsibilities = "ATR0-S04"
    handoff = citation(responsibilities, "译义问题由应用提供期望答案后交模型林工分析")
    waiting = citation("XTR3-06", "模型分析还等QA-A27的期望答案与判定说明")
    application = citation("XTR3-07", "我等待应用侧发布申请，不把预演复测记成正式上线")
    control = [
        item("应用", citation(responsibilities, "格式及异常提示由应用周工处理"), handoff),
        item("测试", citation(responsibilities, "整体回归由测试陈工组织"),
             citation("XTR2-04", "TR-X03的损坏附件分支还没有回归")),
        item("模型", handoff, citation("T04-07", "那张包含译义和凭空补内容的问题，模型侧还要分析"), waiting),
        item("发布", application),
    ]
    results = {}
    for label in ("minimal", "done", "changes", "both", "invalid_use", "combined"):
        items = deepcopy(control)
        if label in {"done", "both", "combined"}:
            items[3]["state"] = "done"
        if label in {"changes", "both", "combined"}:
            items[3]["kind"] = "changes"
        if label in {"invalid_use", "combined"}:
            for wrong in [application, waiting]:
                items[1]["details"].append({**wrong, "use": "check"})
                items[1]["evidence_ids"].append(wrong["source_id"])
        answer, review = render(items, rows, requirements)
        independent_check(answer, rows, require_commitment=True)
        assert not any(e["status"].startswith("missing") for e in review)
        assert len([e for e in review if e["status"] == "unsupported_check"]) == (
            2 if label in {"invalid_use", "combined"} else 0)
        assert any(e["status"] == "unsupported_change" for e in review) == (label in {"done", "changes", "both", "combined"})
        results[label] = {"answer": answer, "review": review}

    # Exercise both actual model outputs against their exact original inputs.
    run = load(BASE / "eval-tr04-v4-pro-grounded-c/run.json")
    raw = load(BASE / "v4-deepseek-runtime/runs" / (run["run_id"] + ".json"))["model_output"]
    fixed = BASE / "eval-tr04-v4-pro-grounded-fixed"
    output = load(next((fixed / "model_calls").glob("*.json")))["response"]["choices"][0]["message"]["content"]
    for label, raw, evidence in [("archive_run", raw, run["result"]["evidence"]),
                                 ("archive_fixed", output, load(fixed / "input.json")["evidence"])]:
        old_rows = {r["id"]: r for r in evidence}
        items = json.loads(raw)["items"]
        answer, review = render(items, old_rows)
        independent_check(answer, old_rows, require_commitment=False)
        assert "T08-04" not in old_rows and "我补这两列" not in str(answer)
        results[label] = {"answer": answer, "review": review}
        # Counterfactual replay: discard citations absent from the new input.
        # Never call this the model's answer to the new prompt.
        retained = []
        for entry in deepcopy(items):
            entry["evidence_ids"] = [key for key in entry["evidence_ids"] if key in rows]
            if entry["evidence_ids"]:
                retained.append(entry)
        answer, review = render(retained, rows, requirements)
        independent_check(answer, rows, require_commitment=True)
        assert not any(e["status"].startswith("missing") for e in review)
        results[label + "_current_sources"] = {"answer": answer, "review": review}
    variants = verify_variants(control, rows, requirements)
    (OUT / "display-audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (OUT / "display-variants.json").write_text(json.dumps(variants, ensure_ascii=False, indent=2))
    print("TR04 display checks passed:", ", ".join(results))
    print("TR04 representation variants passed:", variants)


if __name__ == "__main__":
    main()
