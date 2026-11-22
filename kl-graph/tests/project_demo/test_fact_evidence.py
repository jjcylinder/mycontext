from project_demo.fact_evidence import fact_excerpts


def test_fact_projection_keeps_verbatim_qualifications_and_provenance():
    text = "背景阅读说明。" * 30 + "APP-Z-7仅在预演验证通过；正式未升级，仍需回归。" + "资料检索说明。" * 30
    records = {"doc": {"content": text, "source_type": "document"}}
    facts = [{"id": "fact", "type": "STATUS", "text": "APP-Z-7在预演验证通过。", "evidence_ids": ["doc"]}]
    projection = fact_excerpts(records, facts)["doc"]
    assert "正式未升级，仍需回归" in projection["content"]
    assert projection["fact_ids"] == ["fact"]
    assert all(text[s['start']:s['end']] in projection['content'] for s in projection['spans'])
    assert len(projection['content']) < len(text) / 2


def test_uncertain_or_wrong_lineage_falls_back_and_chat_is_not_shortened():
    records = {"doc": {"content": "项目测试，状态待核查。" * 50, "source_type": "document"},
               "chat": {"content": "APP-Z-7未发布。" * 50, "source_type": "chat"}}
    facts = [
        {"id": "wrong_code", "type": "STATUS", "text": "APP-Z-7已发布。", "evidence_ids": ["doc"]},
        {"id": "ambiguous", "type": "STATUS", "text": "项目测试。", "evidence_ids": ["doc", "chat"]},
        {"id": "chat_fact", "type": "STATUS", "text": "APP-Z-7未发布。", "evidence_ids": ["chat"]},
    ]
    assert fact_excerpts(records, facts) == {}


def test_fact_wording_cannot_replace_source_negation_or_drop_next_qualification():
    text = "背景说明。" * 40 + "APP-Z-7验证通过。仅适用于预演，正式尚未升级。" + "阅读说明。" * 40
    records = {"doc": {"content": text, "source_type": "document"}}
    facts = [{"id": "f", "type": "STATUS", "text": "APP-Z-7验证通过。", "evidence_ids": ["doc"]}]
    assert "正式尚未升级" in fact_excerpts(records, facts)["doc"]["content"]


def test_general_procedural_facts_do_not_force_whole_document_back_in():
    content = "建议复核时查看完整的背景材料。" * 30 + "任务仍未关闭；发布须等待验证。"
    records = {"doc": {"content": content, "source_type": "document"}}
    facts = [{"id": "guide", "type": "GENERAL", "text": "建议复核时查看完整的背景材料。", "evidence_ids": ["doc"]},
             {"id": "state", "type": "STATUS", "text": "任务仍未关闭。", "evidence_ids": ["doc"]}]
    result = fact_excerpts(records, facts)["doc"]
    assert result["fact_ids"] == ["state"]
    assert "发布须等待验证" in result["content"]


def test_anaphoric_sentence_group_preserves_excluded_membership():
    content = ('阅读背景。' * 40 + '样本包PKG-R8缺少判定说明。'
               '新样本EVT-G4尚未补入该包。' + '一般使用指导。' * 40)
    records = {'doc': {'content': content, 'source_type': 'document'}}
    facts = [
        {'id': 'missing-review', 'type': 'STATUS', 'text': '样本包PKG-R8缺少判定说明。', 'evidence_ids': ['doc']},
        {'id': 'excluded', 'type': 'STATUS', 'text': '新样本EVT-G4尚未补入PKG-R8。', 'evidence_ids': ['doc']},
    ]
    audit = {}
    projected = fact_excerpts(records, facts, audit=audit)['doc']
    assert '样本包PKG-R8缺少判定说明。' in projected['content']
    assert '新样本EVT-G4尚未补入该包。' in projected['content']
    assert set(projected['fact_ids']) == {'missing-review', 'excluded'}
    assert len(projected['spans']) == 1  # No artificial omission between antecedent and anaphor.
    assert audit['doc']['unmatched_fact_ids'] == []
    assert '\n[…]\n'.join(content[s['start']:s['end']] for s in projected['spans']) == projected['content']
    assert len(projected['content']) < len(content) / 2


def test_one_unmatched_business_fact_prevents_partial_projection():
    content = '查阅说明。' * 40 + 'PKG-R8仍待审核。' + '收尾说明。' * 40
    facts = [
        {'id': 'known', 'type': 'STATUS', 'text': 'PKG-R8仍待审核。', 'evidence_ids': ['doc']},
        {'id': 'unresolved', 'type': 'STATUS', 'text': 'PKG-R8尚未覆盖EVT-G4。', 'evidence_ids': ['doc']},
    ]
    audit = {}
    assert fact_excerpts({'doc': {'content': content, 'source_type': 'document'}}, facts, audit=audit) == {}
    assert audit['doc']['status'] == 'full_text_unmatched_facts'
    assert audit['doc']['unmatched_fact_ids'] == ['unresolved']


def test_unrelated_adjacent_code_mentions_are_not_resolved_as_coreference():
    content = ('背景说明。' * 40 + 'PKG-R8已经发布。EVT-G4仍待审核。' + '使用说明。' * 40)
    facts = [{'id': 'unresolved', 'type': 'STATUS', 'text': 'PKG-R8的EVT-G4仍待审核。', 'evidence_ids': ['doc']}]
    assert fact_excerpts({'doc': {'content': content, 'source_type': 'document'}}, facts) == {}
