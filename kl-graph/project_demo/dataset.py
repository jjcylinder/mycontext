"""Load a frozen Project A corpus and adapt it to the native DWS layout."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

PROFILES = {
    "app_focus": ("normalized/application_focus.jsonl",),
    "full_v3": ("normalized/records.jsonl",),
    "full_with_report": ("normalized/records.jsonl", "normalized/seed_report_records.jsonl"),
}
DATASET_VERSIONS = {
    "project_a_application_v3": "v3",
    "project_a_application_v4_expanded": "v4-expanded",
    "project_a_application_v5_natural": "v5-natural",
}
_REPORT = "audit/周报原文_段落定位.md"
_SOURCE_FORMATS = {
    "chat": ("chat", "corpus/chat", ".csv"),
    "document": ("wiki", "corpus/documents", ".md"),
    "meeting": ("minutes", "corpus/meetings", ".md"),
}
_RECORD_FIELDS = {
    "record_id", "source_id", "source_type", "title", "project_hint", "occurred_at",
    "is_simulated", "time_basis", "content", "source_locator", "conversation_id",
    "thread_id", "sender_id", "speaker", "reply_to", "attachment_refs",
    "available_at", "source_updated_at", "record_version",
}
_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")


@dataclass(frozen=True)
class Dataset:
    root: Path
    profile: str
    digest: str
    cutoff: str
    records: dict[str, dict]
    version: str = "v3"

    @property
    def id(self) -> str:
        return f"project-a-{self.version}-{self.profile}-{self.digest[:16]}"


def _source_file(root: Path, relative: str) -> Path:
    path = root / relative
    if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid or missing dataset file: {relative}")
    return path


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: expected a JSON object")
            yield row


def _timestamp(value: object, label: str) -> datetime:
    if isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label}: invalid ISO timestamp") from exc
        if result.utcoffset() is not None:
            return result
    raise ValueError(f"{label}: expected a timestamp with a time zone")


def _source_excerpts(path: Path) -> dict[str, str]:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            if not {"message_id", "content"}.issubset(rows.fieldnames or ()):
                raise ValueError(f"{path.name}: missing message_id/content headers")
            pairs = [(row["message_id"], row["content"]) for row in rows]
    else:
        pairs = re.findall(r"^\[([A-Za-z0-9_.-]+)\] ([^\n]*)$", path.read_text(encoding="utf-8-sig"), re.M)
    excerpts = dict(pairs)
    if len(excerpts) != len(pairs):
        raise ValueError(f"{path.name}: duplicate evidence anchors")
    return excerpts


def _check_record(
    row: dict, root: Path, profile: str, excerpts: dict[str, dict[str, str]], cutoff: datetime,
) -> datetime | None:
    record_id = row.get("record_id")
    for name in ("record_id", "source_id", "source_type", "title", "time_basis", "content", "source_locator"):
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{record_id}: missing or invalid {name}")
    source_type = row["source_type"]
    if source_type not in _SOURCE_FORMATS or row.keys() - _RECORD_FIELDS:
        raise ValueError(f"{record_id}: unexpected source type or record fields")
    if not isinstance(row.get("is_simulated"), bool) or not _ID.fullmatch(record_id) or not _ID.fullmatch(row["source_id"]):
        raise ValueError(f"{record_id}: invalid source identity or simulation marker")
    if row.get("project_hint") is not None and not isinstance(row["project_hint"], str):
        raise ValueError(f"{record_id}: invalid project hint")

    occurred = row.get("occurred_at")
    when = _timestamp(occurred, record_id) if occurred is not None else None
    if when is not None and when > cutoff:
        raise ValueError(f"{record_id}: occurrence is after the dataset cutoff")
    for name in ("available_at", "source_updated_at"):
        if row.get(name) is not None:
            _timestamp(row[name], f"{record_id}.{name}")

    relative, separator, anchor = row["source_locator"].rpartition("#")
    path = PurePosixPath(relative)
    _, directory, suffix = _SOURCE_FORMATS[source_type]
    if relative == _REPORT:
        allowed = profile == "full_with_report" and source_type == "document" and record_id == f"W001-{anchor}"
    else:
        allowed = str(path.parent) == directory and path.suffix == suffix and anchor == record_id
    if not separator or not allowed:
        raise ValueError(f"{record_id}: unsupported evidence locator {row['source_locator']!r}")
    if relative not in excerpts:
        excerpts[relative] = _source_excerpts(_source_file(root, relative))
    if excerpts[relative].get(anchor) != row["content"]:
        raise ValueError(f"{record_id}: normalized text differs from its source excerpt")
    return when


def _check_chat_links(records: dict[str, dict], times: dict[str, datetime | None]) -> None:
    sources: dict[str, list[datetime | None]] = defaultdict(list)
    for record_id, row in records.items():
        sources[row["source_id"]].append(times[record_id])
    for record_id, row in records.items():
        if row["source_type"] != "chat":
            continue
        for name in ("conversation_id", "thread_id", "sender_id", "speaker"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"{record_id}: invalid {name}")
        when = times[record_id]
        reply = row.get("reply_to")
        if reply is not None:
            parent = records.get(reply) if isinstance(reply, str) else None
            if not parent or reply == record_id or any(
                parent.get(key) != row[key] for key in ("conversation_id", "thread_id")
            ) or (when is not None and times[reply] is not None and times[reply] >= when):
                raise ValueError(f"{record_id}: invalid reply target {reply!r}")
        attachments = row.get("attachment_refs", [])
        if not isinstance(attachments, list) or any(
            not isinstance(ref, str) or ref not in sources
            or (when is not None and all(ts is not None and ts > when for ts in sources[ref]))
            for ref in attachments
        ):
            raise ValueError(f"{record_id}: invalid attachment references")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def load_dataset(root: Path, profile: str) -> Dataset:
    """Load one allowlisted V3 profile and verify each evidence excerpt against its source."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown V3 profile: {profile!r}")
    root = root.resolve()
    manifest = json.loads(_source_file(root, "dataset.json").read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("dataset_id") not in DATASET_VERSIONS:
        raise ValueError("Expected a supported Project A dataset directory")
    version = DATASET_VERSIONS[manifest["dataset_id"]]
    cutoff = manifest.get("cutoff")
    deadline = _timestamp(cutoff, "dataset cutoff")
    records: dict[str, dict] = {}
    times: dict[str, datetime | None] = {}
    excerpts: dict[str, dict[str, str]] = {}
    for filename in PROFILES[profile]:
        for row in _jsonl(_source_file(root, filename)):
            when = _check_record(row, root, profile, excerpts, deadline)
            record_id = row["record_id"]
            if record_id in records and records[record_id] != row:
                raise ValueError(f"{record_id}: conflicting duplicate record")
            records[record_id] = row
            times[record_id] = when
    if not records:
        raise ValueError(f"V3 profile {profile!r} is empty")
    _check_chat_links(records, times)
    digest = hashlib.sha256(_json_bytes({"cutoff": cutoff}))
    for row in records.values():
        digest.update(_json_bytes(row))
    return Dataset(root=root, profile=profile, digest=digest.hexdigest(), cutoff=cutoff, records=records,
                   version=version)


def _record_data(row: dict) -> dict:
    return {
        key: row[key] for key in (
            "record_id", "source_id", "source_locator", "project_hint", "occurred_at",
            "is_simulated", "time_basis", "thread_id", "attachment_refs", "record_version",
            "available_at", "source_updated_at",
        ) if key in row
    }


def _write_lines(path: Path, rows: Iterable[dict]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(_json_bytes(row))


def _write_bundles(dataset: Dataset, root: Path) -> None:
    scopes: dict[str, dict[str, dict]] = {name: {} for name, _, _ in _SOURCE_FORMATS.values()}
    records: dict[str, list[dict]] = {name: [] for name in scopes}
    resources: dict[str, dict] = {}
    sources = {row["source_id"]: row for row in dataset.records.values()}
    for record_id, row in dataset.records.items():
        source_id, source_type, title = row["source_id"], row["source_type"], row["title"]
        name, _, _ = _SOURCE_FORMATS[source_type]
        scope_data = {"title": title}
        data: dict = {"source_record": _record_data(row)}
        record_type = "document_unit"
        if source_type == "chat":
            record_type = "message"
            scope_type = "chat"
            scope_id = f"chat:{row['conversation_id']}"
            scope_data["chat_kind"] = "group"
            data.update(
                openMessageId=record_id, openConversationId=row["conversation_id"], sender=row["speaker"],
                senderOpenDingTalkId=row["sender_id"], content=row["content"], createTime=row.get("occurred_at"),
            )
            if row.get("reply_to"):
                parent = dataset.records[row["reply_to"]]
                data["quotedMessage"] = {
                    "openMessageId": parent["record_id"], "sender": parent["speaker"], "content": parent["content"],
                }
            for ref in row.get("attachment_refs", []):
                if ref not in resources:
                    resources[ref] = {
                        "id": ref, "kind": "source_reference", "uri": sources[ref]["source_locator"].rpartition("#")[0],
                        "local_path": None, "refs": [], "data": {"source_id": ref},
                    }
                resources[ref]["refs"].append(record_id)
        elif source_type == "document":
            scope_type = "document"
            scope_id = f"document:wiki/{source_id}"
            scope_data["node"] = {"nodeId": source_id, "name": title}
            data.update(text=row["content"], docUrl=row["source_locator"])
        else:
            scope_type = "meeting"
            scope_id = f"meeting:{source_id}"
            data.update(unit_kind="minutes_summary_section", text=row["content"])

        scopes[name].setdefault(scope_id, {"id": scope_id, "type": scope_type, "parent_id": None, "data": scope_data})
        records[name].append({"id": record_id, "scope_id": scope_id, "type": record_type, "data": data})

    for name in records:
        directory = root / name
        directory.mkdir()
        attachments = list(resources.values()) if name == "chat" else []
        files = {"scopes": list(scopes[name].values()), "records": records[name], "resources": attachments}
        for key, rows in files.items():
            _write_lines(directory / f"{key}.jsonl", rows)
        (directory / "manifest.json").write_bytes(_json_bytes({
            "source": f"project-a-{dataset.version}", "dataset": name, "profile": dataset.profile,
            "corpus_sha256": dataset.digest, "counts": {key: len(rows) for key, rows in files.items()},
        }))


def _same_export(left: Path, right: Path) -> bool:
    if not right.is_dir():
        return False
    files = {path.relative_to(left) for path in left.rglob("*")}
    if files != {path.relative_to(right) for path in right.rglob("*")}:
        return False
    return all(
        not (right / relative).is_symlink()
        and ((right / relative).is_dir() if (left / relative).is_dir()
             else (right / relative).is_file() and (left / relative).read_bytes() == (right / relative).read_bytes())
        for relative in files
    )


def write_dws(dataset: Dataset, output: Path) -> None:
    """Publish deterministic DWS quartets under output, without replacing unrelated data."""
    if output.is_symlink():
        raise ValueError(f"DWS output cannot be a symlink: {output}")
    output = output.resolve()
    if output.is_relative_to(dataset.root):
        raise ValueError("DWS output must be outside the source dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".project-demo-dws-", dir=output.parent) as temporary:
        staged = Path(temporary) / "export"
        staged.mkdir()
        _write_bundles(dataset, staged)
        try:
            staged.rename(output)
        except OSError:
            if not _same_export(staged, output):
                if output.exists():
                    raise ValueError(f"DWS output already exists with different contents: {output}") from None
                raise
