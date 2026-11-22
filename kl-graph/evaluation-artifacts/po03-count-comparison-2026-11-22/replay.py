"""Replay frozen rankings; --generate submits only the two saved PO03 final requests."""
import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

from project_demo import answer
from project_demo.codex_bridge import ChatRequest, CodexBridge
from project_demo.context import _fact_scopes, visible_records
from project_demo.dataset import load_dataset
from kl_graph.query.fact_evidence import source_evidence

out = Path(__file__).resolve().parent
root = out.parent
archive = root / "eval-native-evidence-first-bc-2026-11-22"
runtime = root / "v4-longform-runtime"

def dump(name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

async def replay():
    model = answer.AnswerModel("", "chatgpt-codex-default", "", 32768, 8192, 1800, "", "o200k_base")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(root / "tokenizers")
    ds = load_dataset(Path("/home/jcylinder/codex/datasets/项目A_Demo应用增强数据_v4_长文版"), "app_focus")
    visible = visible_records(ds, "项目A")
    eligible = {k:visible[k] for k in _fact_scopes(visible, "项目A", set())}
    db = next(runtime.glob("datasets/*/graph-abc47ce2060b88a7/knowledge.db"))
    conn = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    results = []
    for entry in json.loads((archive / "audit.json").read_text())["rows"]:
        q, mode = entry["question"], entry["mode"]
        run = json.loads((runtime / "runs" / (entry["run_id"] + ".json")).read_text())
        ranking = run["coverage"]["evidence_selection"]
        pool = [key for call in ranking["calls"] if call["stage"] == "candidate_ranking" for key in call["candidate_ids"]]
        ctx = None
        if mode == "C":
            ctx = {**run["result"]["context"], "evidence_ids": pool,
                   "source_evidence": source_evidence(conn, eligible, pool, question=run["question"])}

        async def fixed(*args, **kwargs):
            return {"groups": ranking["groups"]}

        with patch.object(answer, "rank_evidence", fixed):
            msgs, ev, graph, cov = await answer.prepare_answer(
                model, mode, run["question"], run["project_id"], run["cutoff"], visible, ctx,
                vector_refs=[], keyword_refs=run["coverage"]["original_retrieval"]["candidate_ids"])
        ref = run["model_input"]
        assert digest(ref) == entry["prompt_sha256"]
        assert ev == run["result"]["evidence"], (q, mode, "evidence changed")
        target = q == "PO03" and mode == "C"
        assert (msgs != ref) == target, (q, mode)
        assert cov["preflight_tokens"] <= model.evidence_budget
        if target:
            old = json.loads(ref[1]["content"])
            new = json.loads(msgs[1]["content"])
            assert msgs[0] == ref[0] and old == {k:v for k,v in new.items() if k != "extracted_context"}
            operation = new["extracted_context"].pop("count_comparisons")
            assert operation == graph["count_comparisons"] and not any(new["extracted_context"].values())
            assert operation[0]["distinct_samples"] is None
            dump("comparison.json", operation)
            dump("requests.json", {"control": ref, "enhanced": msgs})
        results.append(dict(question=q, mode=mode, run_id=entry["run_id"], before=digest(ref),
                            after=digest(msgs), same_sources=True, sources=len(ev), before_tokens=entry["final_tokens_estimate"],
                            after_tokens=cov["preflight_tokens"]))
    conn.close()
    dump("replay.json", results)
    print([(r["question"]+r["mode"], r["before_tokens"], r["after_tokens"]) for r in results], flush=True)

async def generate():
    assert len(json.loads((out / "replay.json").read_text())) == 10
    bridge = CodexBridge()
    bridge.verify_login()
    requests = json.loads((out / "requests.json").read_text())
    for arm, msgs in requests.items():
        if (out / (arm + ".json")).exists():
            raise RuntimeError("Answer already recorded for " + arm)
        start = time.monotonic()
        req = ChatRequest(model=bridge.model_name, messages=msgs, temperature=0,
                          max_tokens=1800)
        raw, usage = await bridge._run(req)
        refs = {x["id"] for x in json.loads(msgs[1]["content"])["source_excerpts"]}
        dump(arm + ".json", dict(seconds=round(time.monotonic()-start,3), prompt_sha256=digest(msgs),
                                 answer=answer.parse_answer(raw, refs), raw=raw, usage=usage))
        print(arm, "completed", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    asyncio.run(generate() if parser.parse_args().generate else replay())
