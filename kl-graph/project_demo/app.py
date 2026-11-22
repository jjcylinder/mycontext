"""Local API and workbench for inspecting a frozen project corpus."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .answer import AnswerModel, UnavailableError, evidence_view, fuse_originals, parse_answer, prepare_answer, rank_originals
from .context import ContextUnavailable, list_context, query_context, query_original_vectors, visible_records
from .dataset import Dataset, load_dataset, write_dws
from .runtime import GraphRuntime, graph_configured, graph_is_local, graph_path, write_json


PROFILES = {
    "app_focus": ("应用核查", "六类应用 Skill 及所需模型背景", 349),
    "full_v3": ("全量材料", "含其他项目与非应用材料，用于检查范围判断", 497),
    "full_with_report": ("全量与原周报", "含 57 条标记为非模拟的原周报摘录", 554),
}
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOGGER = logging.getLogger(__name__)


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Literal["app_focus", "full_v3", "full_with_report"]


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    project_id: Literal["项目A"] = "项目A"
    mode: Literal["A", "B", "C"]
    question: str = Field(min_length=1, max_length=3000)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Demo:
    def __init__(
        self, dataset_root: Path, data_dir: Path, allow_original_report: bool = False,
        *, uses_chatgpt: bool = False,
    ):
        self.dataset_root = dataset_root.resolve()
        self.data_dir = data_dir.resolve()
        self.allow_original_report = allow_original_report
        self.uses_chatgpt = uses_chatgpt
        self.model = AnswerModel.from_env()
        self.datasets: dict[str, Dataset] = {}
        self.profile_ids: dict[str, str] = {}
        self.graphs: dict[str, GraphRuntime] = {}
        self.graph_states: dict[str, str] = {}
        self.import_locks = {profile: asyncio.Lock() for profile in PROFILES}
        self.jobs: dict[str, dict] = {}
        self.tasks: set[asyncio.Task] = set()
        self.rank_cache: dict = {}

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def dataset_dir(self, dataset: Dataset) -> Path:
        return self.data_dir / "datasets" / dataset.id

    def graph(self, dataset: Dataset) -> GraphRuntime:
        if dataset.id not in self.graphs:
            path = self.dataset_dir(dataset)
            self.graphs[dataset.id] = GraphRuntime(graph_path(path), path / "dws")
        return self.graphs[dataset.id]

    def graph_state(self, dataset: Dataset) -> str:
        if self.graph(dataset).ready:
            return "ready"
        return self.graph_states.get(dataset.id, "unavailable" if not graph_configured() else "not_indexed")

    def restore_imports(self) -> None:
        for profile in PROFILES:
            marker = self.data_dir / "profiles" / f"{profile}.json"
            if not marker.is_file():
                continue
            try:
                saved = json.loads(marker.read_text(encoding="utf-8"))
                dataset = load_dataset(self.dataset_root, profile)
                if saved["dataset_id"] == dataset.id and (self.dataset_dir(dataset) / "dws").is_dir():
                    self.datasets[dataset.id] = dataset
                    self.profile_ids[profile] = dataset.id
            except (KeyError, OSError, ValueError):
                LOGGER.warning("Could not restore the previous import for %s", profile)

    def update(self, run: dict, **fields) -> None:
        run.update(updated_at=_timestamp(), **fields)
        write_json(self.runs_dir / f"{run['run_id']}.json", run)

    def run(self, run_id: str) -> dict:
        if run_id in self.jobs:
            return self.jobs[run_id]
        path = self.runs_dir / f"{run_id}.json"
        if not path.is_file():
            raise HTTPException(404, "找不到该运行记录")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] in {"running", "queued"}:
            self.update(result, status="failed", stage="interrupted", error="服务已重启，请重新运行")
        return result

    def launch(self, worker: Callable[[dict], Awaitable[dict]], **fields) -> dict:
        run = {"run_id": str(uuid4()), "status": "queued", "stage": "queued",
               "error": None, "created_at": _timestamp(), **fields}
        self.update(run)
        self.jobs[run["run_id"]] = run

        async def execute() -> None:
            self.update(run, status="running")
            try:
                result = await worker(run)
            except (UnavailableError, ContextUnavailable) as exc:
                self.update(run, status="unavailable", stage="unavailable", error=str(exc))
            except asyncio.CancelledError:
                self.update(run, status="failed", stage="interrupted", error="运行被服务停止中断")
                raise
            except Exception as exc:
                LOGGER.exception("Demo run %s failed", run["run_id"])
                self.update(run, status="failed", stage="failed", failed_stage=run["stage"], error=str(exc))
            else:
                self.update(run, status="completed", stage="completed", result=result)
            finally:
                self.jobs.pop(run["run_id"], None)

        task = asyncio.create_task(execute())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return run

    async def close(self) -> None:
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(graph.stop() for graph in self.graphs.values()))

    async def import_dataset(self, run: dict) -> dict:
        async with self.import_locks[run["profile"]]:
            return await self._import_dataset(run)

    async def _import_dataset(self, run: dict) -> dict:
        self.update(run, stage="parsing")
        dataset = load_dataset(self.dataset_root, run["profile"])
        path = self.dataset_dir(dataset)
        write_dws(dataset, path / "dws")
        self.datasets[dataset.id] = dataset
        self.profile_ids[dataset.profile] = dataset.id
        counts = {source: sum(row["source_type"] == source for row in dataset.records.values())
                  for source in ("chat", "meeting", "document")}
        stats = {"dataset_id": dataset.id, "digest": dataset.digest, "profile": dataset.profile,
                 "record_count": len(dataset.records), "source_counts": counts, "cutoff": dataset.cutoff}
        write_json(self.data_dir / "profiles" / f"{dataset.profile}.json", stats)
        self.update(run, stage="raw_ready", dataset_id=dataset.id, import_stats=stats)

        graph = self.graph(dataset)
        if graph.ready:
            stats["native"] = json.loads((graph.data_dir / "ready.json").read_text(encoding="utf-8"))
        elif graph_configured() and self.permit_model(dataset, graph_is_local()):
            self.graph_states[dataset.id] = "indexing"
            self.update(run, stage="indexing")
            try:
                stats["native"] = await graph.ingest(
                    lambda phase, counts: self.update(run, stage=phase, native_counts=counts),
                    dataset.records.keys(),
                )
            except Exception:
                self.graph_states[dataset.id] = "failed"
                raise
        else:
            self.graph_states[dataset.id] = "unavailable"
            stats["graph_reason"] = (
                "未配置图谱抽取模型和向量接口" if not graph_configured()
                else "当前服务仅允许向本机模型发送原周报"
            )
        return {**stats, "graph_status": self.graph_state(dataset)}

    def permit_model(self, dataset: Dataset, local: bool) -> bool:
        return (local and not self.uses_chatgpt) or self.allow_original_report or all(
            row.get("is_simulated") is True for row in dataset.records.values()
        )

    async def check(self, run: dict) -> dict:
        dataset = self.get_dataset(run["dataset_id"])
        if not self.model.configured:
            raise UnavailableError("未配置最终回答模型。原文仍可在资料侧栏回查。")
        if not self.permit_model(dataset, self.model.local):
            raise UnavailableError("当前服务仅允许向本机模型发送原周报")
        start = time.monotonic()
        mode, question, project = run["mode"], run["question"], run["project_id"]
        self.update(run, stage="retrieving", model=self.model.model)

        def report(stage, batch=None, batch_total=None):
            progress = {"batch": batch, "total": batch_total} if batch is not None else None
            self.update(run, stage=stage, stage_progress=progress)

        originals = visible_records(dataset, project, dataset.cutoff)
        context = None
        vector_refs = None
        keyword_refs = None
        if mode in {"B", "C"}:
            graph = self.graph(dataset)
            if not graph.ready:
                if mode == "C":
                    raise UnavailableError("该数据集的图谱尚未就绪，配置模型和向量服务后重新导入")
                raise UnavailableError("该数据集的原文向量索引尚未就绪，配置模型和向量服务后重新导入")
            if not self.permit_model(dataset, graph_is_local()):
                raise UnavailableError("当前服务仅允许对原周报使用本机检索和图谱模型")
            base = await graph.start()
            vector_refs = await query_original_vectors(base, graph.data_dir, question, originals)
            keyword_refs = rank_originals(question, originals)
            if mode == "C":
                report("retrieving_graph")
                context = await query_context(
                    base, graph.data_dir, dataset, question, project, dataset.cutoff,
                    timeout=240 if self.uses_chatgpt else 90, original_ids=fuse_originals(keyword_refs, vector_refs),
                )

        if mode == "A":
            report("preparing")
        messages, evidence, used_graph, coverage = await prepare_answer(
            self.model, mode, question, project, dataset.cutoff, originals, context, vector_refs=vector_refs,
            rank_cache=self.rank_cache, keyword_refs=keyword_refs, progress=report,
        )
        if context:
            coverage["graph"] = context.get("coverage", {})
        self.update(run, stage="answering", stage_progress=None, model_input=messages, coverage=coverage)
        raw, usage = await self.model.complete(messages)
        elapsed_ms = round((time.monotonic() - start) * 1000)
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        metrics = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                   "elapsed_ms": elapsed_ms, "cost": _cost(input_tokens, output_tokens)}
        self.update(run, stage="checking_citations", model_output=raw, usage=metrics)
        answer = parse_answer(raw, {row["id"] for row in evidence})
        return {**answer, "model": self.model.model, "usage": metrics, "coverage": coverage,
                "evidence": evidence, "context": used_graph, "cutoff": dataset.cutoff}

    def get_dataset(self, dataset_id: str) -> Dataset:
        if dataset_id not in self.datasets:
            raise HTTPException(404, "数据集未导入，或资料内容已变化，请重新导入")
        return self.datasets[dataset_id]


def _cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
    input_rate = os.getenv("DEMO_USD_INPUT_PER_MILLION")
    output_rate = os.getenv("DEMO_USD_OUTPUT_PER_MILLION")
    if None in (input_tokens, output_tokens, input_rate, output_rate):
        return None
    return (input_tokens * float(input_rate) + output_tokens * float(output_rate)) / 1_000_000


def _public(run: dict) -> dict:
    return {key: value for key, value in run.items() if key not in {"model_input", "model_output"}}


def create_app(
    dataset_root: Path, data_dir: Path, *, allow_original_report: bool = False,
    model_routers: tuple[APIRouter, ...] = (),
) -> FastAPI:
    demo = Demo(dataset_root, data_dir, allow_original_report, uses_chatgpt=bool(model_routers))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        demo.data_dir.mkdir(parents=True, exist_ok=True)
        with (demo.data_dir / ".server.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("该 Demo 数据目录已被另一个服务使用，请更换 --data-dir") from exc
            demo.restore_imports()
            try:
                yield
            finally:
                await demo.close()

    app = FastAPI(title="重点项目进展核查", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/demo/profiles")
    async def profiles():
        items = []
        for profile, (label, description, count) in PROFILES.items():
            dataset_id = demo.profile_ids.get(profile)
            dataset = demo.datasets.get(dataset_id)
            items.append({"id": profile, "label": label, "description": description,
                          "dataset_id": dataset_id, "record_count": len(dataset.records) if dataset else count,
                          "loaded": dataset is not None,
                          "graph_status": demo.graph_state(dataset) if dataset else "not_indexed"})
        return {"profiles": items, "answer_model": demo.model.model if demo.model.configured else None,
                "graph_configured": graph_configured()}

    @app.post("/demo/import", status_code=202)
    async def import_dataset(request: ImportRequest):
        return _public(demo.launch(demo.import_dataset, kind="import", profile=request.profile, dataset_id=None))

    @app.get("/demo/runs/{run_id}")
    async def run(run_id: UUID):
        return _public(demo.run(str(run_id)))

    @app.post("/demo/check", status_code=202)
    async def check(request: CheckRequest):
        dataset = demo.get_dataset(request.dataset_id)
        question = request.question.strip()
        if not question:
            raise HTTPException(422, "请输入需要核查的问题")
        return _public(demo.launch(
            demo.check, kind="check", profile=dataset.profile, dataset_id=dataset.id,
            mode=request.mode, project_id=request.project_id, question=question, cutoff=dataset.cutoff,
        ))

    @app.get("/demo/evidence/{record_id}")
    async def evidence(record_id: str, dataset_id: str):
        dataset = demo.get_dataset(dataset_id)
        if record_id not in dataset.records:
            raise HTTPException(404, "该数据集未收录这条原文")
        return evidence_view(dataset.records[record_id])

    @app.get("/demo/context")
    async def context(
        dataset_id: str,
        view: Literal["sources", "entities", "facts", "relations"] = "sources",
        q: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
        group_relations: bool = False,
    ):
        dataset = demo.get_dataset(dataset_id)
        state = demo.graph_state(dataset)
        if view == "sources":
            needle = q.casefold()
            items = [evidence_view(row) for row in dataset.records.values() if needle in
                     f"{row['title']} {row['record_id']} {row['content']}".casefold()]
            items, total = items[offset:offset + limit], len(items)
        elif state == "ready" or view in {"entities", "relations"}:
            items, total = list_context(
                demo.graph(dataset).data_dir, dataset, view, q, offset, limit,
                include_native=state == "ready", group_relations=group_relations,
            )
        else:
            items, total = [], 0
        return {"items": items, "total": total, "graph_status": state, "cutoff": dataset.cutoff}

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    for router in model_routers:
        app.include_router(router)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 MyContext 项目进展核查工作台")
    parser.add_argument("--dataset", type=Path, required=True, help="项目 A 数据集目录（V3、V4 或 V5）")
    parser.add_argument("--data-dir", type=Path, default=Path("data/project_demo"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--chatgpt", action="store_true",
                        help="使用已登录的 Codex 和本机中文向量模型")
    parser.add_argument("--codex-model", help="指定 Codex 模型，省略时使用官方默认模型")
    parser.add_argument("--allow-original-report", action="store_true",
                        help="允许将非模拟的原周报提供给外部模型")
    args = parser.parse_args()
    if not (args.dataset / "normalized").is_dir():
        parser.error("资料目录缺少 normalized 子目录")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1–65535 之间")
    if args.codex_model and not args.chatgpt:
        parser.error("--codex-model 需要配合 --chatgpt")

    model_routers = ()
    if args.chatgpt:
        from .model_setup import setup_chatgpt

        try:
            model_routers = setup_chatgpt(args.port, args.data_dir.resolve(), args.codex_model)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"模型启动失败：{exc}\n")

    uvicorn.run(
        create_app(args.dataset, args.data_dir, allow_original_report=args.allow_original_report,
                   model_routers=model_routers), host="127.0.0.1", port=args.port,
    )


if __name__ == "__main__":
    main()
