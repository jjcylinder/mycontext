import sqlite3

from kl_graph.ingest.pipeline import IngestionPipeline
from kl_graph.query.fact_evidence import fact_excerpts, fact_support, grounded_graph, stored_fact_support


def test_only_unique_target_quote_is_persisted_on_primary_projection():
    body = "登记备注。该补丁已交给预演包。"
    quote = "该补丁已交给预演包。"
    for target, expected in [(body, True), (body + quote, False), ("只读引用中出现了这句话。", False)]:
        fact = {"fact_text": "记录了补丁的交付。", "source_quote": quote}
        edges = IngestionPipeline._fact_edges("primary", fact, {}, state_chunk_ids=["primary", "secondary"],
                                              target_content=target)
        assert edges[1].properties is None
        if expected:
            anchor = edges[0].properties["source_quote"]
            assert target[anchor["start"]:anchor["end"]] == anchor["quote"] == quote
        else:
            assert edges[0].properties is None


def test_document_quote_is_relocated_in_source_and_retains_negative_condition():
    text = "阅读须知。" * 40 + "这次没有发布。仅限预演。"
    fact = {"id": "f", "type": "STATUS", "text": "正式升级尚待批准",
            "source_quote": "这次没有发布。", "evidence_ids": ["doc"]}
    # A literal quotation alone cannot establish a paraphrased claim.
    assert fact_excerpts({"doc": {"content": text}}, [fact]) == {}
    fact["text"] = "这次没有发布。"
    projection = fact_excerpts({"doc": {"content": text}}, [fact])["doc"]
    assert projection["content"] == "这次没有发布。仅限预演。"


def test_reply_support_keeps_source_authorship_and_stops_at_unavailable_parent():
    def chat(body, speaker, parent=None, conversation="g"):
        return {"content": body, "speaker": speaker, "source_type": "chat",
                "reply_to": parent, "conversation_id": conversation}
    records = {"commit": chat("我补这两列。", "应用", "columns"),
               "columns": chat("缺预期值和判定条件。", "测试", "package"),
               "package": chat("已收到PKG-R8。", "模型")}
    claim = "应用会补PKG-R8的预期值。"
    support = fact_support(claim, "commit", records.get)
    assert support["status"] == "reply_context"
    assert [(q["unit_id"], q["speaker"]) for q in support["quotes"]] == [
        ("commit", "应用"), ("columns", "测试"), ("package", "模型")]
    records["package"]["conversation_id"] = "other"
    support = fact_support(claim, "commit", records.get)
    assert support["status"] == "incomplete_context"
    assert [q["unit_id"] for q in support["quotes"]] == ["commit", "columns"]


def test_unverified_extractor_direction_never_becomes_a_query_claim():
    records = {"s": {"content": "PATCH-R2已经给预演包。", "source_type": "chat", "speaker": "发布"}}
    rows = [{"id": "f", "type": "STATUS", "scope": "requested", "entity_ids": ["native:p"],
             "text": "PATCH-R2提供了预演包。", "evidence_ids": ["s"]}]
    entities = [{"id": "native:p", "name": "PATCH-R2", "evidence_ids": ["s"]}]
    facts, relations = grounded_graph(rows, entities, records)
    assert facts[0]["text"] == records["s"]["content"]
    assert relations[0]["source_id"] == "fact:f" and relations[0]["type"] == "ABOUT"


def test_ambiguous_unit_in_a_legacy_chunk_keeps_chunk_level_provenance():
    with sqlite3.connect(":memory:") as conn:
        conn.executescript("CREATE TABLE chunks(id,content); INSERT INTO chunks VALUES('c','原始片段');"
                           "CREATE TABLE chunk_units(chunk_id,source_id,source_type,unit_id);"
                           "INSERT INTO chunk_units VALUES('c','one','message','x'),('c','two','message','x');")
        result = stored_fact_support(conn, "内容", "c", "x")
        assert result["status"] == "chunk_only" and result["primary_unit_id"] is None
        assert result["quotes"][0] == {"chunk_id": "c", "start": 0, "end": 4, "quote": "原始片段"}


def test_new_extraction_quote_roundtrips_from_cache_shape_to_native_read_and_packing():
    import json

    from kl_graph.ingest.llm_extractor import _expand_compact_result
    from kl_graph.query.fact_evidence import SOURCE_QUOTE_SQL, source_evidence

    quote = "本次延期。审批仍在处理中。"
    content = "背景说明。" * 40 + quote + "仅限预演。"
    fact = _expand_compact_result({"事实": [{"内容": "本次延期。", "事类": "STATUS", "原句": quote}]})["facts"][0]
    edge = IngestionPipeline._fact_edges("c", fact, {}, target_content=content)[0]
    with sqlite3.connect(":memory:") as conn:
        conn.executescript("""
            CREATE TABLE edges(source_type,source_id,target_type,target_id,edge_type,properties);
            CREATE TABLE facts(id, text, fact_type, source_unit_id, source_chunk_id);
            CREATE TABLE chunks(id,content);
            CREATE TABLE chunk_units(chunk_id,source_id,source_type,unit_id);
            INSERT INTO chunk_units VALUES('c','import','document','doc');
        """)
        conn.execute("INSERT INTO chunks VALUES('c',?)", (content,))
        conn.execute("INSERT INTO facts VALUES(?,?,?,NULL,'c')", (edge.source_id, fact["fact_text"], "STATUS"))
        conn.execute("INSERT INTO edges VALUES('fact',?,'chunk','c','STATES',?)",
                     (edge.source_id, json.dumps(edge.properties)))
        anchor = conn.execute("SELECT " + SOURCE_QUOTE_SQL + " FROM facts f").fetchone()[0]
        assert anchor == quote
        support = stored_fact_support(conn, fact["fact_text"], "c", None, anchor)
        assert support["quotes"][0]["quote"] == "本次延期。审批仍在处理中。仅限预演。"
        assert support["target_quote"]["quote"] == quote
        evidence = source_evidence(conn, {"doc": {"content": content}}, ["doc"])
        assert evidence["projections"]["doc"]["content"] == support["quotes"][0]["quote"]
