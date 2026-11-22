"""Run the V3 regression questions and prepare a separate human review sheet."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .runtime import write_json


REGRESSION_QUESTION_IDS = ("TR04", "AM03", "PO03", "X01", "X03")
SECOND_TIER_QUESTION_IDS = ("SU04", "LE03", "WR04", "X05", "X07")


def read_jsonl(path: Path, snapshot: bytes | None = None) -> list[dict]:
    content = path.read_bytes() if snapshot is None else snapshot
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def load_rubric_snapshot(dataset: Path) -> tuple[dict, dict[str, bytes]]:
    """Verify the published grading files and keep the exact inputs for this run."""
    fingerprint = json.loads((dataset / "dataset.json").read_text(encoding="utf-8")).get("grading_rubric_sha256")
    if not fingerprint:
        return {}, {}
    directory = dataset / "evaluation"
    raw = (directory / "rubric_lock.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != fingerprint:
        raise ValueError("评分冻结清单与数据集登记的 SHA-256 不一致")
    lock = json.loads(raw)
    files = {}
    for name, digest in lock["sha256"].items():
        content = (directory / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"评分文件与冻结清单不一致：{name}")
        files[name] = content
    return lock, files


async def run_job(client: httpx.AsyncClient, route: str, request: dict) -> dict:
    response = await client.post(route, json=request)
    response.raise_for_status()
    run = response.json()
    while run["status"] in {"queued", "running"}:
        await asyncio.sleep(0.5)
        response = await client.get(f"/demo/runs/{run['run_id']}")
        response.raise_for_status()
        run = response.json()
    return run


def write_review(path: Path, references: dict[str, dict], runs: list[dict]) -> None:
    fields = ["question_id", "mode", "run_id", "run_status", "fact_no", "reference_fact",
              "sufficient_evidence_sets", "forbidden_errors", "fact_correct", "evidence_supports",
              "attribution_error", "review_seconds", "review_note"]
    frozen = any(r.get("rubric_version") for r in references.values())
    if frozen:
        fields.extend(["rubric_version", "grading_policy", "acceptance"])
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in runs:
            reference = references[row["question_id"]]
            for index, fact in enumerate(reference["facts"], start=1):
                writer.writerow({
                    "question_id": row["question_id"], "mode": row["mode"],
                    "run_id": row["run"]["run_id"], "run_status": row["run"]["status"], "fact_no": index,
                    "reference_fact": fact["text"],
                    "sufficient_evidence_sets": json.dumps(fact["evidence_sets"], ensure_ascii=False),
                    "forbidden_errors": json.dumps(reference["forbidden_errors"], ensure_ascii=False),
                    **({"rubric_version": reference.get("rubric_version", ""),
                        "grading_policy": reference.get("evidence_policy", ""),
                        "acceptance": fact.get("acceptance", "")} if frozen else {}),
                })


async def evaluate(args: argparse.Namespace, questions: list[dict], rubric: dict, files: dict[str, bytes]) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"profile": args.profile, "split": args.split, "modes": list(args.modes),
                "second_tier": getattr(args, "second_tier", False),
                "question_ids": [row["question_id"] for row in questions]}
    if rubric:
        manifest["grading_rubric"] = rubric
    write_json(args.output / "manifest.json", manifest)
    runs = []
    async with httpx.AsyncClient(base_url=args.url, timeout=30, trust_env=False) as client:
        imported = await run_job(client, "/demo/import", {"profile": args.profile})
        write_json(args.output / "import.json", imported)
        if not imported.get("dataset_id"):
            raise RuntimeError(f"导入没有生成可查询数据集：{imported.get('error')}")
        write_json(args.output / "manifest.json", {**manifest, "dataset_id": imported["dataset_id"]})
        with (args.output / "runs.jsonl").open("w", encoding="utf-8") as output:
            for row in questions:
                for mode in args.modes:
                    # Evaluation labels, evidence keys and expected answers never go to this route.
                    run = await run_job(client, "/demo/check", {
                        "dataset_id": imported["dataset_id"], "project_id": "项目A",
                        "mode": mode, "question": row["question"],
                    })
                    result = {"question_id": row["question_id"], "mode": mode, "run": run}
                    runs.append(result)
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"{row['question_id']} {mode} {run['status']}", flush=True)

    # Interpret the grading references only after all model calls have finished.
    reference_rows = read_jsonl(args.dataset / "evaluation" / "reference_answers.jsonl", files.get("reference_answers.jsonl"))
    references = {row["question_id"]: row for row in reference_rows}
    write_review(args.output / "review.csv", references, runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--profile", choices=("app_focus", "full_v3", "full_with_report"), default="app_focus")
    parser.add_argument("--split", choices=("dev", "eval"), default="eval")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all-questions", action="store_true", help="运行所选分组的全部题目")
    scope.add_argument("--second-tier", action="store_true", help="运行固定第二梯队五题，保留原默认五题")
    parser.add_argument("--modes", default="BC", help="运行的方案，默认 BC，也可指定 B 或 C")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="只打印题目 ID，不调用服务")
    args = parser.parse_args()
    if args.second_tier and args.split != "eval":
        parser.error("--second-tier 只适用于 eval 分组")
    url = urlsplit(args.url)
    if url.scheme not in {"http", "https"} or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("评测 URL 必须是本机 HTTP 地址")
    if not args.modes or set(args.modes) - set("ABC") or len(set(args.modes)) != len(args.modes):
        parser.error("--modes 必须由不重复的 A、B、C 字母组成")
    if not args.dry_run and (args.output is None or args.output.exists()):
        parser.error("请指定尚未存在的 --output 路径")
    try:
        rubric, files = load_rubric_snapshot(args.dataset)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    question_rows = read_jsonl(args.dataset / "evaluation" / "questions.jsonl", files.get("questions.jsonl"))
    questions = [row for row in question_rows if row["split"] == args.split]
    if args.split == "eval" and not args.all_questions:
        selected_ids = SECOND_TIER_QUESTION_IDS if args.second_tier else REGRESSION_QUESTION_IDS
        by_id = {row["question_id"]: row for row in questions}
        missing = [question_id for question_id in selected_ids if question_id not in by_id]
        if missing:
            parser.error(f"评测题库缺少所选回归题：{', '.join(missing)}")
        questions = [by_id[question_id] for question_id in selected_ids]
    if args.dry_run:
        print(json.dumps({"profile": args.profile, "split": args.split, "question_count": len(questions),
                          "question_ids": [row["question_id"] for row in questions],
                          "modes": list(args.modes)}, ensure_ascii=False))
        return
    asyncio.run(evaluate(args, questions, rubric, files))


if __name__ == "__main__":
    main()
