from kl_graph.query.source_graph import attachment_project_paths


def test_attribution_requires_visible_unambiguous_reply_and_attachment():
    chat = dict(source_type="chat", thread_id="t", conversation_id="c",
                occurred_at="2026-11-01T08:00:00+08:00")
    doc = dict(source_type="document", source_id="d", project_hint="项目A", content="陈工负责核对。")
    rows = {"a": dict(chat, attachment_refs=["d"]), "b": dict(chat, reply_to="a"),
            "c": dict(chat, reply_to="b", speaker="陈工"), "d1": doc}
    scopes = {"a": "unknown", "b": "unknown", "c": "unknown", "d1": "requested"}
    def paths(records, scope=scopes):
        return attachment_project_paths(records, scope, ["c"])
    assert paths(rows)["c"] == {"reply_path": ["c", "b", "a"], "attachment_record": "d1"}
    assert not paths({k: v for k, v in rows.items() if k != "d1"})
    assert not paths(rows, {**scopes, "b": "comparison"})
    assert not paths({**rows, "b": dict(chat, reply_to="c")})
    assert not paths({**rows, "b": dict(chat, thread_id="other", reply_to="a")})
    assert not paths({**rows, "d2": dict(doc, project_hint="项目B")})
    assert not paths({**rows, "a": dict(rows["a"], occurred_at="2026-11-02T08:00:00+08:00")})
