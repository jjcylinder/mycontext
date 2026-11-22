"""Audit expansion invariants, actual native input, and embedding truncation."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import PROFILES, load_dataset, write_dws
from .expand_dataset import read_rows, save, sha


def length_stats(rows: list[dict]) -> dict:
    result = {}
    for kind in ("document", "meeting", "chat"):
        subset = [r for r in rows if r["source_type"] == kind]
        totals = defaultdict(int)
        lengths = [len(r["content"]) for r in subset]
        for row in subset:
            totals[row["source_id"]] += len(row["content"])
        result[kind] = {
            "records": len(subset), "sources": len(totals), "total_chars": sum(lengths),
            "record_median_chars": statistics.median(lengths), "record_max_chars": max(lengths),
            "source_median_chars": statistics.median(totals.values()),
            "source_min_chars": min(totals.values()), "source_max_chars": max(totals.values()),
        }
    return result


def audit(baseline: Path, target: Path, export: Path, tokenizer_path: Path) -> dict:
    from tokenizers import Tokenizer
    from kl_graph.ingest.loaders.wiki_loader import load_wiki
    from kl_graph.ingest.loaders.minutes_loader import load_minutes
    from kl_graph.ingest.loaders.message_loader import load_all_messages
    from kl_graph.ingest.session_chunker import slice_chat_sessions
    from kl_graph.ingest import chunker

    checks = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise ValueError(label)
        checks.append(label)

    profiles = {}
    for profile in PROFILES:
        before, after = load_dataset(baseline, profile), load_dataset(target, profile)
        check(list(before.records) == list(after.records), f"{profile}: same record identities/order")
        check(before.cutoff == after.cutoff, f"{profile}: same cutoff")
        check(before.id != after.id and before.digest != after.digest, f"{profile}: isolated changed corpus")
        changed = 0
        for rid, old in before.records.items():
            new = after.records[rid]
            check({k: v for k, v in old.items() if k != "content"} ==
                  {k: v for k, v in new.items() if k != "content"}, f"{profile}/{rid}: metadata unchanged")
            if old["source_type"] == "chat" or not old["is_simulated"]:
                check(old == new, f"{profile}/{rid}: protected content unchanged")
            else:
                check(new["content"].count(old["content"]) == 1, f"{profile}/{rid}: original assertion preserved once")
                check(len(new["content"]) >= len(old["content"]) + 120, f"{profile}/{rid}: actual expansion")
                changed += 1
        profiles[profile] = {
            "baseline_dataset_id": before.id, "expanded_dataset_id": after.id,
            "changed_records": changed, "before": length_stats(list(before.records.values())),
            "after": length_stats(list(after.records.values())),
        }
    unchanged = {}
    protected = [*list((baseline / "corpus/chat").rglob("*")),
                 *list((baseline / "evaluation").rglob("*")),
                 baseline / "normalized/seed_report_records.jsonl", baseline / "audit/周报原文_段落定位.md"]
    for path in protected:
        if path.is_file():
            relative = path.relative_to(baseline)
            check(path.read_bytes() == (target / relative).read_bytes(), f"{relative}: byte-identical")
            unchanged[str(relative)] = sha(path)
    # Redundant normalized views must carry the same expanded text.
    canonical = {r["record_id"]: r for r in read_rows(target / "normalized/records.jsonl")}
    for name in ("application_focus.jsonl", "application_added.jsonl"):
        check(all(r == canonical[r["record_id"]] for r in read_rows(target / "normalized" / name)),
              f"{name}: agrees with canonical records")
    frozen_baseline = json.loads((target / "audit/baseline_sha256.json").read_text())
    check(all(sha(baseline / name) == checksum for name, checksum in frozen_baseline.items()),
          "V3 baseline files unchanged")

    dataset = load_dataset(target, "app_focus")
    write_dws(dataset, export)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    chunker.num_tokens_from_string("检查聊天分段的实际分词器")
    check(chunker._encoder is not False, "Native chat tokenizer available without char/4 fallback")
    chat_chunks = slice_chat_sessions(load_all_messages(export / "chat"))
    baseline_export = export.with_name(export.name + "-baseline")
    write_dws(load_dataset(baseline, "app_focus"), baseline_export)
    baseline_chat_chunks = slice_chat_sessions(load_all_messages(baseline_export / "chat"))
    check([(c.id, c.content) for c in chat_chunks] == [(c.id, c.content) for c in baseline_chat_chunks],
          "Stored chat session chunks identical to baseline")
    chunks = [*load_wiki(export / "wiki"), *load_minutes(export / "minutes"), *chat_chunks]
    token_counts = [{"id": chunk.id, "source_type": chunk.source_type,
                     "tokens": len(tokenizer.encode(chunk.content).ids),
                     "unit_id": chunk.metadata.get("unit_id")} for chunk in chunks]
    prose_overflow = [r for r in token_counts if r["source_type"] in ("wiki", "minutes") and r["tokens"] > 512]
    check(not prose_overflow, "Expanded document/meeting chunks fit actual 512-token embedding input")
    full_export = export.with_name(export.name + "-full")
    write_dws(load_dataset(target, "full_v3"), full_export)
    full_prose = [*load_wiki(full_export / "wiki"), *load_minutes(full_export / "minutes")]
    full_prose_tokens = [len(tokenizer.encode(chunk.content).ids) for chunk in full_prose]
    check(max(full_prose_tokens) <= 512, "All expanded sources fit embedding input, including non-focus sources")
    # Compare the literal exported body, not merely counts or source IDs.
    exported = [*read_rows(export / "wiki/records.jsonl"), *read_rows(export / "minutes/records.jsonl")]
    check(all(r["data"]["text"] == dataset.records[r["id"]]["content"] for r in exported),
          "Actual DWS input equals expanded prose, without evaluation/audit inputs")
    return {
        "status": "passed", "scope": "dataset integrity and local import; not answer accuracy",
        "checks_passed": len(checks), "profiles": profiles, "unchanged_file_sha256": unchanged,
        "native_dws_path": str(export.resolve()), "native_chunks": dict(Counter(r["source_type"] for r in token_counts)),
        "embedding_tokenizer_sha256": sha(tokenizer_path), "embedding_max_input": 512,
        "chat_chunk_tokenizer": chunker._encoder.name,
        "document_meeting_max_tokens": max(r["tokens"] for r in token_counts if r["source_type"] in ("wiki", "minutes")),
        "full_document_meeting_max_tokens": max(full_prose_tokens),
        "existing_chat_chunks_over_limit": [r for r in token_counts if r["source_type"] not in ("wiki", "minutes") and r["tokens"] > 512],
        "chunk_token_counts": token_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "target", "export", "tokenizer"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.baseline, args.target, args.export, args.tokenizer)
    save(args.target / "audit/validation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("chunk_token_counts", "unchanged_file_sha256", "profiles")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
