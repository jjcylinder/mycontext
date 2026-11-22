from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from project_demo.dataset import load_dataset, write_dws


OCCURRED = "2026-11-20T09:00:00+08:00"
CUTOFF = "2026-11-20T23:59:00+08:00"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _record(record_id: str, source_id: str, source_type: str, locator: str, text: str, **fields: object) -> dict:
    return {
        "record_id": record_id, "source_id": source_id, "source_type": source_type,
        "title": f"来源 {source_id}", "project_hint": None, "occurred_at": OCCURRED,
        "source_locator": locator, "content": text, "is_simulated": True, "time_basis": "synthetic",
        **fields,
    }


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / "项目资料"
    for name in ("normalized", "evaluation", "audit", "corpus/chat", "corpus/documents", "corpus/meetings"):
        (root / name).mkdir(parents=True)
    (root / "dataset.json").write_text(json.dumps({
        "dataset_id": "project_a_application_v3", "cutoff": CUTOFF,
        "profiles": {"app_focus": {"records": ["evaluation/answers.jsonl"]}},
    }), encoding="utf-8")
    with (root / "corpus/chat/msgs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["message_id", "content"])
        writer.writerows([("MSG-1", "建议调整时间。"), ("MSG-2", "下周确认，\n先查看附件。")])
    (root / "corpus/documents/记录.md").write_text("[DOC-1] 切换待确认。\n", encoding="utf-8")
    (root / "corpus/meetings/纪要.md").write_text("[MIN-1] 讨论事项。\n\n[MIN-2] 不批准切换。\n", encoding="utf-8")
    (root / "audit/周报原文_段落定位.md").write_text("[P0001] 原周报正文。\n", encoding="utf-8")
    (root / "evaluation/answers.jsonl").write_text('{"expected_facts":["参考答案"]}\n', encoding="utf-8")
    chat = {"conversation_id": "G1", "thread_id": "T1", "sender_id": "U1", "speaker": "用户A"}
    rows = [
        _record("MSG-1", "G1", "chat", "corpus/chat/msgs.csv#MSG-1", "建议调整时间。", **chat),
        _record(
            "MSG-2", "G1", "chat", "corpus/chat/msgs.csv#MSG-2", "下周确认，\n先查看附件。",
            **{**chat, "sender_id": "U2", "speaker": "用户B"},
            occurred_at="2026-11-20T09:01:00+08:00", reply_to="MSG-1", attachment_refs=["D1"],
        ),
        _record("DOC-1", "D1", "document", "corpus/documents/记录.md#DOC-1", "切换待确认。", project_hint="项目A"),
        _record("MIN-1", "M1", "meeting", "corpus/meetings/纪要.md#MIN-1", "讨论事项。"),
        _record("MIN-2", "M1", "meeting", "corpus/meetings/纪要.md#MIN-2", "不批准切换。"),
    ]
    _jsonl(root / "normalized/application_focus.jsonl", rows)
    _jsonl(root / "normalized/records.jsonl", rows + [rows[0]])
    _jsonl(root / "normalized/seed_report_records.jsonl", [
        _record("W001-P0001", "W001", "document", "audit/周报原文_段落定位.md#P0001", "原周报正文。", is_simulated=False),
    ])
    return root, rows


def test_allowlisted_profile_preserves_record_identity(corpus: tuple[Path, list[dict]]) -> None:
    root, rows = corpus
    focus = load_dataset(root, "app_focus")
    full = load_dataset(root, "full_v3")
    report = load_dataset(root, "full_with_report")
    assert focus.root == root.resolve()
    assert focus.cutoff == CUTOFF
    assert list(focus.records.values()) == rows
    assert full.records == focus.records
    assert full.digest == focus.digest
    assert full.id != focus.id
    assert focus.id == load_dataset(root, "app_focus").id
    assert list(report.records) == [*full.records, "W001-P0001"]
    assert not report.records["W001-P0001"]["is_simulated"]
    with pytest.raises(ValueError, match="Unknown V3 profile"):
        load_dataset(root, "evaluation/answers.jsonl")


def test_expanded_version_isolates_index_and_exports_actual_prose(corpus, tmp_path):
    root, rows = corpus
    baseline = load_dataset(root, "app_focus")
    manifest = json.loads((root / "dataset.json").read_text())
    manifest["dataset_id"] = "project_a_application_v4_expanded"
    (root / "dataset.json").write_text(json.dumps(manifest))
    expanded = load_dataset(root, "app_focus")
    assert expanded.id != baseline.id
    assert expanded.id.startswith("project-a-v4-expanded-")
    # Identity separation also applies if someone copies a V3 payload unchanged.
    assert expanded.digest == baseline.digest
    text = "讨论涉及切换前的核对工作。切换待确认。可继续核对配置与使用范围。"
    rows[2]["content"] = text
    _jsonl(root / "normalized/application_focus.jsonl", rows)
    (root / "corpus/documents/记录.md").write_text(f"[DOC-1] {text}\n")
    expanded = load_dataset(root, "app_focus")
    assert expanded.digest != baseline.digest
    output = tmp_path / "expanded-dws"
    write_dws(expanded, output)
    exported = [json.loads(line) for line in (output / "wiki/records.jsonl").read_text().splitlines()]
    assert exported[0]["data"]["text"] == text
    assert exported[0]["id"] == "DOC-1"
    assert not (output / "evaluation").exists()


@pytest.mark.parametrize(("change", "message"), [
    ({"content": "篡改证据"}, "differs"),
    ({"source_locator": "evaluation/answers.jsonl#MSG-1"}, "unsupported evidence locator"),
    ({"occurred_at": "2026-11-21T09:00:00+08:00"}, "after the dataset cutoff"),
    ({"expected_facts": ["不应进入语料"]}, "unexpected source type or record fields"),
])
def test_rejects_invalid_business_evidence(corpus: tuple[Path, list[dict]], change: dict, message: str) -> None:
    root, rows = corpus
    _jsonl(root / "normalized/application_focus.jsonl", [{**rows[0], **change}, *rows[1:]])
    with pytest.raises(ValueError, match=message):
        load_dataset(root, "app_focus")


def test_rejects_conflicting_duplicate_id(corpus: tuple[Path, list[dict]]) -> None:
    root, rows = corpus
    _jsonl(root / "normalized/application_focus.jsonl", [*rows, {**rows[0], "project_hint": "项目B"}])
    with pytest.raises(ValueError, match="conflicting duplicate record"):
        load_dataset(root, "app_focus")


def test_dws_export_retries_without_overwriting(corpus: tuple[Path, list[dict]], tmp_path: Path) -> None:
    root, _ = corpus
    dataset = load_dataset(root, "app_focus")
    output = tmp_path / "dws_export"
    write_dws(dataset, output)
    assert {path.name for path in output.iterdir()} == {"chat", "wiki", "minutes"}
    for source in output.iterdir():
        assert {path.name for path in source.iterdir()} == {
            "manifest.json", "scopes.jsonl", "records.jsonl", "resources.jsonl",
        }
    write_dws(dataset, output)
    resources = (output / "chat/resources.jsonl").read_text(encoding="utf-8").splitlines()
    attachment = json.loads(resources[0])
    assert attachment["id"] == "D1"
    assert attachment["refs"] == ["MSG-2"]
    assert attachment["uri"] == "corpus/documents/记录.md"

    manifest = output / "wiki/manifest.json"
    manifest.write_text("do not replace", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists with different contents"):
        write_dws(dataset, output)
    assert manifest.read_text(encoding="utf-8") == "do not replace"
    with pytest.raises(ValueError, match="outside the source dataset"):
        write_dws(dataset, root / "dws_export")


def test_native_chunks_preserve_record_time_and_lineage(corpus: tuple[Path, list[dict]], tmp_path: Path) -> None:
    from kl_graph.ingest.loaders import load_all_messages, load_minutes, load_wiki
    from kl_graph.ingest.loaders.base import to_unix_ms
    from kl_graph.ingest.session_chunker import slice_chat_sessions

    root, _ = corpus
    dataset = load_dataset(root, "app_focus")
    output = tmp_path / "dws_export"
    write_dws(dataset, output)

    messages = load_all_messages(output / "chat")
    assert [message.id for message in messages] == ["MSG-1", "MSG-2"]
    reply = messages[1]
    assert reply.metadata["sender_id"] == "U2"
    assert reply.metadata["reply_to"] == "MSG-1"
    assert reply.metadata["source_record"]["attachment_refs"] == ["D1"]
    assert reply.source_ref == "corpus/chat/msgs.csv#MSG-2"
    assert "建议调整时间。" in reply.content
    assert "建议调整时间。" not in reply.metadata["extraction_target_content"]
    assert slice_chat_sessions(messages)[0].metadata["member_message_ids"] == ["MSG-1", "MSG-2"]

    [wiki] = load_wiki(output / "wiki")
    assert wiki.metadata["unit_id"] == "DOC-1"
    assert wiki.timestamp == to_unix_ms(OCCURRED)
    assert wiki.source_ref == "corpus/documents/记录.md#DOC-1"
    assert wiki.content == "切换待确认。"

    meeting = load_minutes(output / "minutes")
    assert [chunk.metadata["unit_id"] for chunk in meeting] == ["MIN-1", "MIN-2"]
    assert {chunk.metadata["meeting_id"] for chunk in meeting} == {"M1"}
    assert all(chunk.timestamp == to_unix_ms(OCCURRED) for chunk in meeting)
    assert len({chunk.id for chunk in meeting}) == 2
    assert meeting[1].content.endswith("不批准切换。")
    assert meeting[1].source_ref == "corpus/meetings/纪要.md#MIN-2"
