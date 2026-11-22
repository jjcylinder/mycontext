"""Build an independent, source-preserving long-form synthetic corpus.

The author receives one business source at a time, never evaluation questions.
Original assertions are inserted verbatim between newly authored prose. Model
responses and review decisions live outside the retrievable business corpus.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from .codex_bridge import ChatRequest, CodexBridge

AUTHOR = """你是模拟工作材料的编辑。把给定的业务记录扩写成自然的工作文档段落或会议讨论记录。
只根据所给来源；不知道任何评测题、参考答案或检索方案，也不要猜测它们。
每条返回 before 和 after，程序会在两者中间原样放入原句：before + 原句 + after。
要求：
1. 原句中的事实、数字、日期、编号、责任、状态、否定、范围、待确认条件不变。
2. 新文字限于相关背景说明、操作考虑、核查方法、可能的疑问、方案权衡；没有依据的过程必须用“可、若、建议、需”等明确表达设想。禁止编造已发生的操作、会谈、批准、检测结果、解决进展、人员分工或期限。
3. 不把别的段落的结论搬来，不重复原句，不添加新的业务编号、人物、数量或具体日期。不能把工作建议写成已确定要求。不能根据文档标题推导额外事实。
4. 文档应按材料类型写出有内容的使用背景、操作说明、定位步骤或限制，避免每段同一套话；登记表可较短。会议用议题展开、待核实的问题、不同处理思路的讨论体；不捏造具名发言人及达成一致，不能冒充真实逐字稿。
5. 不出现“评测、标准答案、B方案、C方案、图谱、事实提炼、扩写、原句、锚点、为了证明”等实验用语（原有模型评测业务术语可以保留）。不人为强调某个原句、不汇总成答案卡。
6. 每条含原句总计：文档260—380汉字左右，会议340—430汉字左右；不要靠复述凑字。before 与 after 均需非空，原句位置各段有变化。
7. 返回严格JSON：{"records":[{"record_id":"原编号","before":"文字","after":"文字"}]}。每条文字不含换行。记录顺序与输入一致。
"""

REVIEW = """审核模拟工作材料的扩写是否忠实。输入包含原文和扩写全文，不含评测题。
逐段检查新增文字，特别查找捏造已经发生的动作/检查/批准、追加责任/期限/计数、把未决说成已完成、否定变肯定、把其他记录的结论移到本段、缩窄或放宽范围。
允许常识性背景说明、明确的假设/建议/核查方法；不要求这些建议逐字出现在原文。禁止因为原句保留就忽略周边新增的矛盾或过度承诺。
有问题时给出修正后的 before 和 after（程序会把原文插在中间），保留自然且具体的背景说明，删改没有依据的内容即可，不用重写整篇。
返回JSON：{"approved":true或false,"issues":[{"record_id":"编号","reason":"问题","before":"修改后前文","after":"修改后后文"}]}。
approved为true时issues为空；false时只列出需要修订的段落。不要输出其它字段。
"""


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(rows: list[dict], generated: dict) -> list[dict]:
    parts = generated["records"]
    if [p["record_id"] for p in parts] != [r["record_id"] for r in rows]:
        raise ValueError("Record identity/order changed")
    output = []
    for row, part in zip(rows, parts):
        before, after = part["before"].strip(), part["after"].strip()
        if not before or not after or "\n" in before + after:
            raise ValueError("Expected nonempty single-line prose")
        content = before + row["content"] + after
        if content.count(row["content"]) != 1:
            raise ValueError("Original assertion must occur exactly once")
        if not 170 <= len(content) <= 650:
            raise ValueError(f"Unexpected length {row['record_id']}: {len(content)}")
        added = before + after
        # New concrete numeric claims must not slip into narrative framing.
        old_numbers = set(re.findall(r"[0-9]+", row["content"]))
        if set(re.findall(r"[0-9]+", added)) - old_numbers:
            raise ValueError(f"New numbers in {row['record_id']}")
        if any(word in added for word in ("标准答案", "B方案", "C方案", "事实提炼", "为了证明", "扩写")):
            raise ValueError("Experimental instructions in business text")
        output.append({**row, "content": content})
    return output


async def generate(source: Path, staging: Path, only: set[str]) -> None:
    groups = defaultdict(list)
    for row in read_rows(source / "normalized/records.jsonl"):
        if row["source_type"] != "chat" and row["is_simulated"]:
            groups[row["source_id"]].append(row)
    bridge = CodexBridge()
    bridge.verify_login()
    gate = asyncio.Semaphore(2)

    async def call(system: str, payload: dict, log: Path) -> dict:
        request = ChatRequest(model=bridge.model_name, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], response_format={"type": "json_object"}, max_tokens=6500)
        response, usage = await bridge._run(request)
        save(log, {"request": request.model_dump(), "response": response, "usage": usage})
        return json.loads(response)

    async def one(sid: str, rows: list[dict]) -> None:
        async with gate:
            output = staging / "approved" / f"{sid}.json"
            if output.exists():
                approval = json.loads(output.read_text())
                if assemble(rows, approval["prose"]) != approval["records"]:
                    raise ValueError(f"Source changed after review: {sid}")
                print(f"cached {sid}", flush=True)
                return
            payload = {"title": rows[0]["title"], "type": rows[0]["source_type"],
                       "records": [{"record_id": r["record_id"], "content": r["content"]} for r in rows]}
            error = None
            for attempt in range(3):
                try:
                    authored = await call(AUTHOR, {**payload, "correction": str(error) if error else None},
                                          staging / "generation" / f"{sid}-{attempt}.json")
                    expanded = assemble(rows, authored)
                    for review_round in range(3):
                        review = await call(REVIEW, {"source": payload, "expanded": [
                            {"record_id": r["record_id"], "content": r["content"]} for r in expanded]},
                            staging / "review" / f"{sid}-{attempt}-{review_round}.json")
                        if review.get("approved") is True and review.get("issues") == []:
                            save(output, {"source_id": sid, "prose": authored, "records": expanded,
                                          "review_approved": True})
                            print(f"approved {sid}: {len(rows)} records, {sum(len(r['content']) for r in expanded)} chars", flush=True)
                            return
                        fixes = {r["record_id"]: r for r in review["issues"]}
                        if not fixes or set(fixes) - {r["record_id"] for r in rows}:
                            raise ValueError("Invalid review response")
                        for part in authored["records"]:
                            if part["record_id"] in fixes:
                                fix = fixes[part["record_id"]]
                                part.update(before=fix["before"], after=fix["after"])
                        expanded = assemble(rows, authored)
                    raise ValueError("Semantic review did not approve after revisions")
                except Exception as exc:
                    error = exc
                    print(f"retry {sid} {attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
            raise RuntimeError(f"Could not approve {sid}: {error}")

    results = await asyncio.gather(*(one(sid, rows) for sid, rows in groups.items()
                                     if not only or sid in only), return_exceptions=True)
    failures = [str(r) for r in results if isinstance(r, Exception)]
    if failures:
        raise RuntimeError("; ".join(failures))


def publish(source: Path, staging: Path, target: Path) -> None:
    if target.exists():
        raise ValueError("Target already exists; do not overwrite a frozen corpus")
    original = read_rows(source / "normalized/records.jsonl")
    by_source = defaultdict(list)
    for row in original:
        if row["source_type"] != "chat":
            by_source[row["source_id"]].append(row)
    changed = {}
    for sid, rows in by_source.items():
        approval = json.loads((staging / "approved" / f"{sid}.json").read_text())
        assert approval["review_approved"] is True
        expanded = assemble(rows, approval["prose"])
        if expanded != approval["records"]:
            raise ValueError(f"Source changed after review: {sid}")
        changed.update((r["record_id"], r) for r in expanded)
    target.mkdir(parents=True)
    for name in ("corpus", "normalized", "evaluation"):
        shutil.copytree(source / name, target / name)
    (target / "audit").mkdir()
    shutil.copy2(source / "audit/周报原文_段落定位.md", target / "audit/周报原文_段落定位.md")
    for name in ("records.jsonl", "application_focus.jsonl", "application_added.jsonl"):
        path = target / "normalized" / name
        lines = []
        for line in path.read_text(encoding="utf-8-sig").splitlines(keepends=True):
            row = json.loads(line)
            if row["record_id"] in changed:
                line = json.dumps(changed[row["record_id"]], ensure_ascii=False) + "\n"
            lines.append(line)
        path.write_text("".join(lines), encoding="utf-8")
    for rows in by_source.values():
        path = target / rows[0]["source_locator"].split("#")[0]
        content = path.read_text(encoding="utf-8-sig")
        for row in rows:
            old = f"[{row['record_id']}] {row['content']}"
            assert content.count(old) == 1
            content = content.replace(old, f"[{row['record_id']}] {changed[row['record_id']]['content']}")
        path.write_text(content, encoding="utf-8")
    manifest = json.loads((source / "dataset.json").read_text())
    manifest.update(dataset_id="project_a_application_v4_expanded", baseline_dataset_id="project_a_application_v3",
                    status="独立扩写语料；模型逐来源生成并复核；尚未进行长文版 B/C 评测",
                    reuse="保留原记录身份、元数据和原始断言；仅扩写模拟文档及会议正文；聊天和评测材料不变",
                    expansion={"generator": "project_demo.expand_dataset", "source": str(source),
                               "source_sha256": sha(source / "normalized/records.jsonl"),
                               "generation_audit": str(staging.resolve()), "approved_sources": len(by_source),
                               "changed_records": len(changed), "evaluation_seen_by_author": False})
    manifest["profiles"]["app_focus"]["purpose"] = "六类应用及必要模型背景；扩写正文；B/C 使用相同输入"
    manifest["profiles"]["full_v3"]["purpose"] = "兼容原 profile 名称；选取当前版本的全部模拟业务材料"
    manifest["profiles"]["full_with_report"]["purpose"] = "扩写模拟材料加原样保留的非模拟周报；外发限制不变"
    save(target / "dataset.json", manifest)
    save(target / "audit/expansion_map.json", [{
        "record_id": row["record_id"], "source_id": row["source_id"],
        "original_content": row["content"], "original_chars": len(row["content"]),
        "expanded_chars": len(changed[row["record_id"]]["content"]),
        "original_offset": changed[row["record_id"]]["content"].index(row["content"]),
    } for row in original if row["record_id"] in changed])
    save(target / "audit/baseline_sha256.json", {
        str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()})
    print(f"Published {len(changed)} expanded records to {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "publish"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    if args.action == "generate":
        asyncio.run(generate(args.source, args.staging, set(args.only)))
    elif args.target:
        publish(args.source, args.staging, args.target)
    else:
        parser.error("publish requires --target")


if __name__ == "__main__":
    main()
