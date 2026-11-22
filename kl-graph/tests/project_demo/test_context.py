"""Verify graph retrieval against the native JSON and SQLite lineage contracts."""

import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from project_demo.context import (
    ContextUnavailable,
    list_context,
    query_context,
    query_original_vectors,
    visible_records,
    expand_source_links,
)


def record(content, project="项目A", **fields):
    return {
        "content": content, "project_hint": project,
        "occurred_at": "2026-11-20T08:00:00+08:00", **fields,
    }


class ContextTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.dataset = SimpleNamespace(
            cutoff="2026-11-20T10:00:00+08:00",
            records={
                "skill": record("SK-SU对应doc-summarizer，别名文档摘要、摘要助手。讨论SK-SU/CASE-S24任务。"),
                "run": record("APP-A-12绑定至ENV-A-PREVIEW，SU-EX06在该配置复测；模板通过。"),
                "other": record("项目B的APP-B-12已发布。", "项目B"),
                "late": record("ENV-A-DEMO绑定APP-A-12。", available_at="2026-11-20T10:10:00+08:00"),
                "people": record("U21应用林工跟进SK-SU；U26模型林工协助。"),
                "conflicting_binding": record("ENV-A-DEMO尚未绑定APP-A-12。"),
                "second_application_identity": record("U99应用林工；不是U21。"),
                "ambiguous_person": record("应用林工与CASE-S24待协调。"),
                "flat": record("仅出现在平铺结果的原文。"),
                "recalled": record("仅出现在近邻召回的原文。"),
                "isolate": record("仅出现在孤立片段节点的原文。"),
            },
        )

    def make_native_store(self):
        with closing(sqlite3.connect(self.path / "knowledge.db")) as db:
            db.executescript("""
                CREATE TABLE facts (id TEXT, text TEXT, fact_type TEXT,
                    timestamp INTEGER, confidence REAL, subject_entity_id TEXT,
                    object_entity_id TEXT, source_chunk_id TEXT, source_unit_id TEXT);
                CREATE TABLE chunk_units (chunk_id TEXT, unit_id TEXT, unit_ordinal_in_chunk INTEGER);
                CREATE TABLE edges (source_type TEXT, source_id TEXT, target_type TEXT, target_id TEXT,
                    edge_type TEXT, properties TEXT);
                CREATE TABLE entities (id TEXT, name TEXT, entity_type TEXT, quality_status TEXT);
                INSERT INTO chunk_units VALUES ('slice', 'run', 0), ('slice', 'late', 1),
                    ('other_chunk', 'other', 0), ('flat_chunk', 'flat', 0),
                    ('recalled_chunk', 'recalled', 0), ('isolated_chunk', 'isolate', 0),
                    ('first_hit', 'other', 1), ('first_hit', 'run', 0),
                    ('second_hit', 'other', 0), ('second_hit', 'skill', 1);
                INSERT INTO facts VALUES
                    ('run-fact','复测','STATUS',1,0.9,NULL,NULL,'slice','run'),
                    ('late-fact','正式环境更新','STATUS',1,0.9,NULL,NULL,'slice','late'),
                    ('ambiguous-fact','没有精确单元','STATUS',1,0.9,NULL,NULL,'slice',NULL),
                    ('other-fact','其他项目已发布','STATUS',1,0.9,NULL,NULL,'other_chunk','other');
                INSERT INTO entities VALUES ('version','APP-A-12','System','active'),
                    ('person','林工','Person','active'), ('env','ENV-A-PREVIEW','System','active');
                INSERT INTO edges VALUES ('fact','run-fact','entity','env','ABOUT',NULL);
            """)
            db.commit()

    def test_review_recovers_grounded_sibling_fact_through_a_native_entity(self):
        self.make_native_store()
        self.dataset.records.update({
            "related": record("ENV-A-PREVIEW 在次日恢复，附有回归记录。"),
            "prefix": record("ENV-A-PREVIEW-X 在次日恢复。"),
            "unlinked": record("ENV-A-PREVIEW 在次日恢复。"),
            "mixed": record("项目A 和项目B 都讨论 ENV-A-PREVIEW。"),
        })
        with closing(sqlite3.connect(self.path / "knowledge.db")) as db:
            db.execute("UPDATE facts SET text='ENV-A-PREVIEW 曾暂停。' WHERE id='run-fact'")
            for key in ("related", "prefix", "unlinked", "mixed", "late", "other"):
                chunk = "chunk-" + key
                db.execute("INSERT INTO chunk_units VALUES (?,?,0)", (chunk, key))
                db.execute("INSERT INTO facts VALUES (?,?,'STATUS',1,1,NULL,NULL,?,NULL)",
                           ("f-" + key, "ENV-A-PREVIEW 在次日恢复。", chunk))
                if key != "unlinked":
                    db.execute("INSERT INTO edges VALUES ('fact',?,'entity','env','ABOUT',NULL)", ("f-" + key,))
            db.commit()
        def ask(request):
            return httpx.Response(200, json={"items": [{"type": "fact", "id": "run-fact"}]})

        async def retrieve():
            with patch("project_demo.context.httpx.AsyncClient",
                       return_value=httpx.AsyncClient(transport=httpx.MockTransport(ask))):
                return await query_context("http://local", self.path, self.dataset, "何时恢复？", "项目A")

        result = asyncio.run(retrieve())
        paths = result["coverage"]["associated_fact_paths"]
        self.assertEqual([(p["source"], p["source_fact"], p["entity_name"], p["target_fact"], p["target"])
                          for p in paths], [("run", "run-fact", "ENV-A-PREVIEW", "f-related", "related")])
        observation = next(f for f in result["facts"] if f["id"] == "f-related")
        self.assertEqual(observation["text"], self.dataset.records["related"]["content"])
        self.assertEqual(observation["evidence_ids"], ["related"])
        self.assertIn("related", result["evidence_ids"])
        self.assertEqual(asyncio.run(retrieve())["coverage"]["associated_fact_paths"], paths)

    def query_vectors(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("project_demo.context.httpx.AsyncClient", return_value=client):
            return asyncio.run(query_original_vectors(
                "http://localhost", self.path, "发布版本", visible_records(self.dataset, "项目A"),
            ))

    def test_relation_list_groups_duplicate_claims_and_keeps_all_citations(self):
        self.make_native_store()
        self.dataset.records.update({
            "copy": record("ENV-A-PREVIEW 已恢复。"),
            "original": record("ENV-A-PREVIEW 已恢复。"),
            "pending": record("ENV-A-PREVIEW 尚未恢复。"),
        })
        with closing(sqlite3.connect(self.path / "knowledge.db")) as db:
            for key in ("copy", "original", "pending"):
                db.execute("INSERT INTO chunk_units VALUES (?, ?, 0)", (key, key))
                db.execute("INSERT INTO facts VALUES (?, ?, 'STATUS', 1, 1, NULL, NULL, ?, NULL)",
                           ("fact-" + key, self.dataset.records[key]["content"], key))
                db.execute("INSERT INTO edges VALUES ('fact', ?, 'entity', 'env', 'ABOUT', NULL)",
                           ("fact-" + key,))
            db.commit()

        edges, total = list_context(self.path, self.dataset, "relations", "ENV-A-PREVIEW", 0, 25)
        self.assertEqual(total, 4)
        self.assertEqual(len({edge["source_id"] for edge in edges}), 4)
        rows, total = list_context(self.path, self.dataset, "relations", "STATUS", 0, 25, group_relations=True)
        self.assertEqual(total, 3)
        self.assertIn("ENV-A-PREVIEW 尚未恢复。", {row["source_text"] for row in rows})
        same_claim, total = list_context(self.path, self.dataset, "relations", "ENV-A-PREVIEW 已恢复", 0, 1,
                                         group_relations=True)
        self.assertEqual(total, 1)
        self.assertEqual(set(same_claim[0]["evidence_ids"]), {"copy", "original"})

    @staticmethod
    def dense_results(*chunk_ids):
        return httpx.Response(200, json={"collection": "chunks", "results": [
            {"id": chunk_id, "point_id": f"backend-{index}", "payload": {"chunk_id": chunk_id}}
            for index, chunk_id in enumerate(chunk_ids)
        ]})

    def test_query_original_vectors_preserves_rank_and_rejects_mixed_cutoff_chunks(self):
        self.make_native_store()

        def search(request):
            self.assertEqual((request.method, request.url.path), ("POST", "/search"))
            self.assertEqual(json.loads(request.content), {
                "query": "发布版本", "collection": "chunks", "top_k": 20,
            })
            return self.dense_results("slice", "first_hit", "second_hit", "first_hit")

        self.assertEqual(self.query_vectors(search), ["run", "other", "skill"])
        self.assertEqual(self.query_vectors(lambda _: self.dense_results("slice")), [])

    def test_query_original_vectors_fails_closed_on_service_or_lineage_errors(self):
        self.make_native_store()
        cases = [
            (httpx.Response(502), "查询失败"),
            (httpx.Response(200, content="non-json"), "查询失败"),
            (httpx.Response(200, json={"collection": "facts", "results": []}), "无法识别"),
            (httpx.Response(200, json={"collection": "chunks", "results": ["first_hit"]}), "片段标识"),
            (httpx.Response(200, json={"collection": "chunks", "results": [
                {"id": "first_hit", "payload": {"chunk_id": "second_hit"}}
            ]}), "片段标识"),
            (self.dense_results(), "向量索引为空"),
            (self.dense_results("first_hit", "missing"), "缺少来源记录"),
        ]
        for response, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ContextUnavailable, error):
                self.query_vectors(lambda _: response)

    def test_cutoff_and_scope_preserve_comparison_sources(self):
        result = visible_records(self.dataset, "项目A")
        self.assertIn("other", result)
        self.assertNotIn("late", result)

    def test_source_expansion_follows_exact_threads_attachments_and_cutoff(self):
        rows = {
            "child": record("见回复", source_type="chat", source_id="g", conversation_id="g",
                            thread_id="t1", reply_to="parent", attachment_refs=["doc"]),
            "parent": record("另一条回复", source_type="chat", source_id="g", conversation_id="g",
                             thread_id="t1", reply_to="child"),
            "unrelated": record("同群其他主题", source_type="chat", source_id="g", conversation_id="g",
                                thread_id="t2"),
            "doc1": record("前提", source_type="document", source_id="doc"),
            "doc2": record("条件", source_type="document", source_id="doc"),
            "future": record("未来状态", source_type="document", source_id="doc",
                             available_at="2026-11-21T10:00:00+08:00"),
        }
        dataset = SimpleNamespace(records=rows, cutoff=self.dataset.cutoff)
        selected, paths = expand_source_links(visible_records(dataset, "项目A"), ["child"])
        self.assertEqual(set(selected), {"child", "parent", "doc1", "doc2"})
        self.assertEqual({path["to"] for path in paths}, set(selected) - {"child"})

    def test_domain_graph_aliases_and_ambiguous_person(self):
        entities, _ = list_context(self.path, self.dataset, "entities", "文档摘要", 0, 25)
        self.assertEqual([row["name"] for row in entities], ["SK-SU"])
        people, _ = list_context(self.path, self.dataset, "entities", "林工", 0, 25)
        self.assertEqual({row["id"] for row in people}, {"work:U21", "work:U26", "work:U99"})
        refs = {row["id"]: row["evidence_ids"] for row in people}
        self.assertNotIn("ambiguous_person", refs["work:U21"])
        self.assertNotIn("ambiguous_person", refs["work:U99"])
        bindings, _ = list_context(self.path, self.dataset, "relations", "deployment_binding", 0, 25)
        self.assertEqual([row["evidence_ids"] for row in bindings], [["run"]])

    def test_query_projects_candidates_on_precise_visible_lineage(self):
        self.make_native_store()

        def ask(request):
            self.assertEqual(request.url.path, "/ask")
            self.assertFalse(json.loads(request.content)["force_phase2"])
            return httpx.Response(200, json={
                "items": [
                    {"type": "fact", "id": value}
                    for value in ("late-fact", "ambiguous-fact", "other-fact")
                ] + [{"type": "message", "id": "flat_chunk"}],
                "recalled_chunks": [{"id": "cnk:recalled_chunk"}],
                "graph": {"components": [{
                    "nodes": [
                        {"type": "fact", "id": "fact:run-fact"},
                        {"type": "chunk", "id": "cnk:slice"},
                        {"type": "chunk", "id": "cnk:other_chunk"},
                        {"type": "chunk", "id": "cnk:isolated_chunk"},
                    ],
                    "edges": [
                        {"type": "ABOUT", "from": "ent:version", "to": "fact:run-fact"},
                        {"type": "STATES", "from": "fact:run-fact", "to": "cnk:slice"},
                        {"type": "REPLY_TO", "from": "cnk:slice", "to": "cnk:other_chunk"},
                    ],
                }]},
                "mode": "graph",
            })

        simulated_client = httpx.AsyncClient(transport=httpx.MockTransport(ask))
        with patch("project_demo.context.httpx.AsyncClient", return_value=simulated_client):
            result = asyncio.run(query_context(
                "http://localhost", self.path, self.dataset, "APP-A-12 文档摘要", "项目A"
            ))
        self.assertEqual([row["id"] for row in result["facts"]], ["run-fact"])
        self.assertEqual(result["facts"][0]["evidence_ids"], ["run"])
        self.assertIn("native:version", {row["id"] for row in result["entities"]})
        self.assertEqual({link["target_id"] for link in result["relations"]}, {"native:version", "native:env"})
        self.assertTrue(all(link["type"] == "ABOUT" for link in result["relations"]))
        self.assertTrue({"flat", "recalled", "isolate", "late"}.isdisjoint(result["evidence_ids"]))
        self.assertEqual(result["coverage"]["comparison_evidence_ids"], ["other"])
        self.assertEqual(result["coverage"]["omitted_native_facts"], 3)
        all_refs = set(result["evidence_ids"])
        for field in ("entities", "relations", "facts"):
            self.assertTrue(all(set(row["evidence_ids"]) <= all_refs for row in result[field]))

        self.dataset.records["run"]["conversation_id"] = "conversation"
        self.dataset.records["late"]["conversation_id"] = "conversation"
        simulated_client = httpx.AsyncClient(transport=httpx.MockTransport(ask))
        with patch("project_demo.context.httpx.AsyncClient", return_value=simulated_client):
            historical = asyncio.run(query_context(
                "http://localhost", self.path, self.dataset, "APP-A-12 文档摘要", "项目A"
            ))
        self.assertEqual(historical["facts"], [])
        self.assertEqual(historical["coverage"]["cutoff_affected_conversations"], 1)
