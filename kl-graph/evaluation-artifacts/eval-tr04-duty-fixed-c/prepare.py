"""Rebuild the archived production prompt as a control before changing only the C evidence check."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from project_demo import answer
from project_demo.context import _native_graph
from project_demo.dataset import load_dataset

OUT = Path(__file__).resolve().parent
DATA = OUT.parent
RUNTIME = DATA / "v4-deepseek-runtime"


async def main():
    stats = json.loads((DATA/"eval-tr04-duty-final-4096/comparison.json").read_text())["off-C"]
    run = json.loads((RUNTIME/"runs"/(stats["run_id"]+".json")).read_text())
    initial = run["coverage"]["original_retrieval"]["candidate_ids"]
    selection = run["coverage"]["evidence_selection"]
    dataset = load_dataset(Path("/home/jcylinder/codex/datasets/项目A_Demo应用增强数据_v4_长文版"), "app_focus")
    assert dataset.id == run["dataset_id"] and all(dataset.records[k]["is_simulated"] for k in initial)
    records = {k: dataset.records[k] for k in initial}
    graph_dir = next((RUNTIME/"datasets"/dataset.id).glob("graph-*/ready.json")).parent
    _, facts, _, _ = _native_graph(graph_dir, records, run["project_id"], None, [], {})
    index = {(f["text"], tuple(f["evidence_ids"])): f for f in facts}
    for trace in stats["graph_and_answer"]:
        if trace["stage"] == "select":
            request = json.loads((RUNTIME/"model_calls"/trace["trace"]).read_text())["request"]
            data = json.loads(request["messages"][0]["content"])
            if "graph_hints" in data:
                hints = data["graph_hints"]
                break
    # This run returned only sourced fact hints, no relations.
    assert all("text" in h for h in hints)
    graph = {"facts": [index[h["text"], tuple(h["evidence_ids"])] for h in hints], "source_facts": facts}
    assert len(hints) == run["coverage"]["graph"]["native_fact_candidates"]

    async def frozen(*args, **kwargs):
        return {"groups": deepcopy(selection["groups"])}
    model = answer.AnswerModel("", "", "", 131072, 8192, 65536, "", "o200k_base")
    async def pack():
        return await answer.prepare_answer(model, "C", run["question"], run["project_id"], dataset.cutoff,
                                           records, graph, vector_refs=initial)
    real_finder = answer.delegation_evidence
    def before(retrieved, *args):
        return real_finder([f for f in retrieved if f["type"] == "DELEGATE"], *args)
    with patch.object(answer, "rank_evidence", frozen):
        with patch.object(answer, "delegation_evidence", before):
            control, _, _, _ = await pack()
        assert control == run["model_input"], "Frozen control differs from the archived bytes"
        messages, evidence, _, coverage = await pack()
    audit = coverage["evidence_selection"]["delegation_coverage"]
    assert all(row["covered_by"] for row in audit)
    assert any("QA-A27" in row["quote"] and row["missing_before_check"] for row in audit)
    (OUT/"input.json").write_text(json.dumps({"messages": messages, "evidence": evidence,
        "control_run": run["run_id"], "identical_control": True, "tokens": coverage["preflight_tokens"],
        "assignment_audit": audit}, ensure_ascii=False, indent=2))
    print("Identical frozen control; new input tokens", coverage["preflight_tokens"], "sources",len(evidence))


if __name__ == "__main__":
    asyncio.run(main())
