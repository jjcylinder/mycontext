"""Run native KL-Graph in an isolated, loopback-only subprocess."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections.abc import Callable, Collection
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from omegaconf import OmegaConf

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_ID = "project-a-demo-v3"
_PORT = re.compile(rb"Uvicorn running on http://127\.0\.0\.1:(\d+)")
LOGGER = logging.getLogger(__name__)
_COUNTS = ("units_discovered", "units_skipped", "units_processed", "chunks_created",
           "extraction_total", "extraction_succeeded", "extraction_failed")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def graph_configured() -> bool:
    from kl_graph.config import cfg

    return bool(cfg.services.llm_flash.base_url and cfg.services.embedding.base_url)


def graph_is_local() -> bool:
    from kl_graph.config import cfg

    urls = [cfg.services.llm_flash.base_url, cfg.services.embedding.base_url,
            cfg.services.reranker.base_url]
    return all(urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"} for url in urls if url)


def graph_path(dataset_path: Path) -> Path:
    """Partition incompatible extraction/embedding runs without exposing credentials."""
    from kl_graph.config import cfg

    settings = OmegaConf.to_container(cfg, resolve=True)
    for service in settings["services"].values():
        service.pop("api_key", None)
    settings["storage"]["vector"]["qdrant"].pop("api_key", None)
    identity = {key: settings[key] for key in ("services", "storage", "pipelines")}
    identity["project_demo_version"] = 1
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return dataset_path / f"graph-{fingerprint}"


def audit_import(
    database: Path, source_directory: Path, record_ids: Collection[str], run: dict,
) -> tuple[dict, str | None]:
    expected = set(record_ids)
    counts = {key: run.get(key, 0) for key in _COUNTS}
    recovered_from = None
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        units = {row[0] for row in connection.execute(
            "SELECT unit_id FROM units WHERE source_id = ?", (SOURCE_ID,),
        )}
        lineage = {row[0] for row in connection.execute(
            "SELECT DISTINCT unit_id FROM chunk_units WHERE source_id = ?", (SOURCE_ID,),
        )}
        if units != expected or lineage != expected:
            raise RuntimeError(f"原文溯源对账未通过，期望 {len(expected)} 条，实际索引 {len(units)} 条、可追溯 {len(lineage)} 条")

        if not any(counts.values()) and run.get("outcome") != "skipped":
            prior = connection.execute(
                f"SELECT run_id, {', '.join(_COUNTS)} FROM ingest_runs WHERE source_id = ? AND input_dir = ? "
                "AND state = 'done' AND coalesce(outcome, '') != 'skipped' AND extraction_failed = 0 "
                "AND units_processed + units_skipped = ? ORDER BY completed_at DESC LIMIT 1",
                (SOURCE_ID, str(source_directory), len(expected)),
            ).fetchone()
            if prior is not None:
                counts = {key: prior[key] for key in _COUNTS}
                recovered_from = prior["run_id"]

    accounted = counts["units_skipped"] + counts["units_processed"]
    if accounted != len(expected) or counts["extraction_failed"] or run.get("outcome") == "skipped":
        raise RuntimeError(f"图谱导入对账未通过，期望 {len(expected)} 条原文，实际 {counts}")
    return counts, recovered_from


class GraphRuntime:
    def __init__(self, data_dir: Path, input_dir: Path):
        self.data_dir = data_dir
        self.input_dir = input_dir
        self.process: asyncio.subprocess.Process | None = None
        self.base_url = ""
        self.lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return (self.data_dir / "ready.json").is_file() and (self.data_dir / "knowledge.db").is_file()

    async def start(self) -> str:
        async with self.lock:
            if self.process is not None and self.process.returncode is None and self.base_url:
                return self.base_url

            self.data_dir.mkdir(parents=True, exist_ok=True)
            override = self.data_dir / "demo-native-config.json"
            write_json(override, {"application": {"data_dir": str(self.data_dir),
                                                   "dws_export_dir": str(self.input_dir),
                                                   "current_user": "", "current_user_aliases": []}})
            log_path = self.data_dir / "server.log"
            with log_path.open("wb") as log:
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable, str(PROJECT_DIR / "kl_server.py"),
                    "--config", str(override), "--port", "0", cwd=PROJECT_DIR,
                    stdout=log, stderr=asyncio.subprocess.STDOUT,
                    env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
                )
            deadline = asyncio.get_running_loop().time() + 90
            while asyncio.get_running_loop().time() < deadline:
                if self.process.returncode is not None:
                    break
                if match := _PORT.search(log_path.read_bytes()):
                    self.base_url = f"http://127.0.0.1:{int(match[1])}"
                    return self.base_url
                await asyncio.sleep(0.25)
            await self.stop()
            LOGGER.error("工作图谱服务启动失败，详情见 %s", log_path)
            raise RuntimeError("工作图谱服务启动失败")

    async def ingest(self, progress: Callable[[str, dict], None], record_ids: Collection[str]) -> dict:
        timeout_seconds = int(os.getenv("DEMO_INGEST_TIMEOUT_SECONDS", "1800"))
        if timeout_seconds <= 0:
            raise ValueError("DEMO_INGEST_TIMEOUT_SECONDS 必须为正整数")
        start = time.monotonic()
        base_url = await self.start()
        async with httpx.AsyncClient(base_url=base_url, timeout=30, trust_env=False) as client:
            response = await client.post(
                "/ingest", json={"input_dir": str(self.input_dir), "source_id": SOURCE_ID,
                                 "concurrency": 2, "improve_mode": "off"},
            )
            response.raise_for_status()
            run_id = response.json()["run_id"]
            deadline = asyncio.get_running_loop().time() + timeout_seconds
            while asyncio.get_running_loop().time() < deadline:
                response = await client.get("/status")
                response.raise_for_status()
                status = response.json()
                run = status["ingest"]
                if run.get("run_id") != run_id:
                    raise RuntimeError("工作图谱返回了其他导入批次")
                if run["state"] == "done":
                    counts, recovered_from = audit_import(
                        self.data_dir / "knowledge.db", self.input_dir, record_ids, run,
                    )
                    result = {"native_run_id": run_id, "counts": counts, "knowledge": status["knowledge"],
                              "elapsed_ms": round((time.monotonic() - start) * 1000),
                              "graph_backend": status["graph_backend"], "vector_backend": status["vector_backend"]}
                    if recovered_from:
                        result["recovered_from_native_run_id"] = recovered_from
                    write_json(self.data_dir / "ready.json", result)
                    return result
                if run["state"] == "error":
                    LOGGER.error("工作图谱导入失败，详情见 %s", self.data_dir / "server.log")
                    raise RuntimeError("工作图谱导入失败")
                progress(run.get("phase") or "indexing", {key: run.get(key, 0) for key in _COUNTS})
                await asyncio.sleep(1)
        await self.stop()
        raise RuntimeError("工作图谱导入超时，可重新导入")

    async def stop(self) -> None:
        self.base_url = ""
        if self.process is None or self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()
