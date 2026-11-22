"""Offline regression on both real V4 Pro responses; labels never reach the demo."""
import json
from pathlib import Path
from project_demo.answer import parse_answer

out = Path(__file__).resolve().parent
base = out.parent
fixed = base / "eval-tr04-v4-pro-grounded-fixed"


def read(path):
    return json.loads(path.read_text())


def check(raw, evidence, question, target):
    sources = {r["id"]: r for r in evidence}
    review = []
    answer = parse_answer(raw, set(sources), require_items=True, assignment_sources=sources,
                          assignment_review=review, question=question)
    duties = [d["text"] for d in answer["coordination"]]
    assert len(duties) == 4
    application, = [t for t in duties if "异常提示由应用" in t]
    assert "由应用提供期望答案" in application and "处理BUG-S11" not in application
    assert "应用补齐QA-A27的期望答案与判定说明" not in application
    tests, = [t for t in duties if "纳入回归范围" in t]
    assert all(k in tests for k in ("测试", "TR-X03", "译义"))
    assert "解决且QA-A27补齐后" not in tests
    assert any(all(k in t for k in ("BUG-S11", "模型侧还要分析", "QA-A27", "期望答案", "判定说明")) for t in duties)
    assert any("发布申请" in t and "不把预演复测记成正式上线" in t and "尚未部署" in t for t in duties)
    assert not any(k in str(answer) for k in ("SU-EX06", "LET-S08", "公函", "模板文号", "摘要Skill", "EVAL-62"))
    for field in ("coordination", "open_issues", "progress", "changes"):
        assert all(set(item["evidence_ids"]) <= set(sources) for item in answer[field])
    target.write_text(json.dumps({"result": answer, "review": review, "additional_model_calls": 0},
                                 ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run = read(out / "run.json")
    archived = read(base / "v4-deepseek-runtime/runs" / (run["run_id"] + ".json"))
    check(archived["model_output"], run["result"]["evidence"], run["question"], out / "reviewed-answer.json")
    trace, = (fixed / "model_calls").glob("*.json")
    raw = read(trace)["response"]["choices"][0]["message"]["content"]
    check(raw, read(fixed / "input.json")["evidence"], run["question"], fixed / "final-display.json")
    print("Both archived V4 Pro answers passed the joint display checks.")
