import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from project_demo import app, evidence_retrieval as retrieval
from project_demo.answer import AnswerModel
from project_demo.evidence_packing import Hop, PathEvidence, Snapshot, Span, candidates, pack
from project_demo.evidence_retrieval import fact_lane, rank_paths


def record(key, body, **fields):
    return dict(record_id=key, content=body, source_type='chat', source_id='team',
                title='会议', source_locator=key, occurred_at='2026-01-01',
                is_simulated=True, conversation_id='team', **fields)


MODEL = AnswerModel('', '', '', 12000, 8192, 1000, '', '')
ROWS = {k: record(k, body) for k, body in {'a': '型号甲未通过。', 'b': '另一条证据。',
                                          'p': '前文。', 'x': '另一对象。'}.items()}


def full(k):
    return Span(k, 0, len(ROWS[k]['content']))


def snapshot():
    path = PathEvidence((Hop('a', 'reply_to', 'x', ()),), (full('a'), full('x')))
    return Snapshot({'a': (full('a'),), 'b': (full('b'),)},
                    {'b': (full('b'), full('p'))}, {'x': path})


def test_light_path_ranking_keeps_graph_only_endpoint_and_resolves_ties():
    paths = [
        PathEvidence((Hop('a', relation, target, ()),), (full('a'), full(target)))
        for relation, target in [('reply_to', 'x'), ('same_thread', 'x'), ('reply_to', 'b')]
    ]
    for order in (paths, paths[::-1]):
        ranked = retrieval.rank_light_paths('另一对象', order, ROWS, ['a'], [])
        assert list(ranked) == ['x', 'b']
        assert ranked['x'] == paths[0]


async def packed(mode, snap, model=MODEL):
    return await pack(model, mode, '能否通过？', '项目A', '2026-01-01', ROWS, snap)


@pytest.mark.asyncio
async def test_lanes_and_disable_graph():
    snap = snapshot()
    b, f, g = [await packed(mode, snap) for mode in 'BFG']
    assert b['source_ids'] == ['a', 'b']
    assert f['source_ids'][:2] == ['b', 'p']
    assert set(g['source_ids']) == set(ROWS)
    assert g['paths']['x'] == [snap.paths['x'].hops[0].as_reference()]
    assert 'paths' not in f['messages'][1]['content']
    assert g['messages'][1]['content'].count(ROWS['a']['content']) == 1
    assert (await packed('G', replace(snap, paths={}))) == f
    assert (await packed('F', replace(snap, facts={}))) == b
    assert await packed('G', snap) == g
    broken = replace(snap.paths['x'], spans=(full('x'),))
    assert (await packed('G', replace(snap, paths={'x': broken}))) == f


@pytest.mark.asyncio
async def test_over_budget_path_drops_graph_vote_and_keeps_independent_hit():
    snap = snapshot()
    snap.originals = {'x': (full('x'),)}
    snap.facts = {}
    direct = await packed('B', snap)
    cap = replace(MODEL, evidence_budget=direct['input_tokens'])
    checked = await packed('G', snap, cap)
    assert checked['messages'] == direct['messages']
    assert [(d['kind'], d['result']) for d in checked['decisions']][:2] == [
        ('path', 'over_budget'), ('source', 'included')]
    assert not (await packed('G', replace(snap, originals={}), cap))['source_ids']
    assert not (await packed('B', snap, replace(cap, evidence_budget=cap.evidence_budget-1)))['source_ids']


def test_outer_rrf_window_and_order():
    keys = [f'd{i:02}' for i in range(50)]
    ranks = Snapshot({k: (full('a'),) for k in keys}, {'only_fact': (full('b'),),
                     **{f'f{i}': (full('b'),) for i in range(48)}, keys[-1]: (full('a'),),
                     'outside': (full('p'),)}, {})
    entries = [e.source for e in candidates('F', ranks)]
    assert entries.index('only_fact') < entries.index(keys[-1])
    assert 'outside' not in entries


@pytest.mark.asyncio
async def test_separate_delivery_can_pack_graph_only_destination_without_claiming_a_link():
    path = snapshot().paths['x']
    snap = Snapshot({}, {}, {'x': path})
    direct = await packed('B', Snapshot({'x': (full('x'),)}, {}, {}))
    cap = replace(MODEL, evidence_budget=direct['input_tokens'])
    result = await pack(cap, 'G', '能否通过？', '项目A', '2026-01-01', ROWS, snap,
                        graph_delivery='separate')
    assert result['messages'] == direct['messages'] and result['paths'] == {}
    assert result['decisions'][0]['ranks'] == (None, None, 1)
    assert any(d['kind'] == 'path' and d['result'] == 'over_budget' for d in result['decisions'])


def test_separate_votes_require_the_matching_excerpt_and_relation_has_only_graph_rank():
    path = PathEvidence((Hop('a', 'shared_ABOUT', 'x', ('f1', 'e', '模型', 'f2')),),
                        (full('a'), full('p'), full('x')), (full('p'), full('x')))
    first = Span('x', 0, 1)
    snap = Snapshot({'x': (first,)}, {'x': (first,)}, {'x': path})
    entries = candidates('G', snap, graph_delivery='separate')
    assert entries[0].ranks == (1, 1, 1)
    assert set(entries[0].spans) == {full('p'), full('x')}
    assert [(e.ranks, e.spans) for e in entries if e.path] == [((None, None, 1), path.spans)]
    assert any(e.ranks == (1, 1, None) and e.spans == (first,) for e in entries)
    for mode in 'BF':
        assert candidates(mode, snap, graph_delivery='separate') == candidates(mode, snap)


def test_fact_lineage_and_incomplete_replies():
    rows = dict(ROWS, reply=record('reply', '上述两项需要重测。', reply_to='a'))
    facts = {key: dict(id=key, evidence_ids=[source], text=text, source_quote=None)
             for key, source, text in [('f0', 'a', '型号甲未通过'), ('f1', 'reply', '需要重测')]}
    spans, witness = fact_lane(facts, ['missing', 'f1', 'f0'], rows)
    assert list(spans) == ['reply', 'a'] and witness['reply'] == ['f1']
    assert {s.source for s in spans['reply']} == {'a', 'reply'}
    assert not fact_lane(facts, ['f1'], {'reply': rows['reply']})[0]


@pytest.mark.asyncio
async def test_path_encoder_rejects_overflow_and_propagates_failure(monkeypatch):
    from kl_graph import config
    monkeypatch.setattr(retrieval, 'httpx', SimpleNamespace(AsyncClient=lambda **kw: httpx.AsyncClient(
        **kw, transport=httpx.MockTransport(handle))))
    from omegaconf import OmegaConf
    settings = OmegaConf.create({'base_url':'http://local.test/v1','model':'test','dim':2,'api_key':'',
                                 'query_instruction':'rank','send_dimensions':False})
    monkeypatch.setattr(config, 'cfg', OmegaConf.merge(config.cfg, {'services': {'embedding': settings}}))
    sent = []
    failure = False

    def handle(request):
        texts = json.loads(request.content)['input']
        sent.extend(texts)
        if failure:
            return httpx.Response(503)
        if any(ROWS['b']['content'] in text for text in texts):
            return httpx.Response(413)
        return httpx.Response(200, json={'data':[{'index':i, 'embedding':[1,0]} for i in range(len(texts))]})

    short = snapshot().paths['x']
    large = PathEvidence((Hop('b', 'reply_to', 'x', ()),), (full('b'), full('x')))
    lane, skipped = await rank_paths('问题', [short, large, short], ROWS)
    assert lane == {'x': short} and skipped == [large.hops]
    assert sent[0] == 'Instruct: rank\nQuery:问题'
    failure = True
    with pytest.raises(httpx.HTTPStatusError):
        await rank_paths('问题', [short], ROWS)


@pytest.mark.asyncio
@pytest.mark.parametrize('generate, expected', [(False, 0), (True, 3)])
async def test_comparison_worker_shares_snapshot_and_one_answer_per_group(tmp_path, monkeypatch, generate, expected):
    from unittest.mock import AsyncMock
    demo = app.Demo(tmp_path, tmp_path)
    ds = SimpleNamespace(id='ds', records=ROWS, digest='revision', cutoff='2026-01-01')
    demo.datasets['ds'] = ds
    demo.model = replace(MODEL, base_url='http://localhost', model='test')
    demo.graphs['ds'] = SimpleNamespace(ready=True, start=AsyncMock(return_value='http://local'), data_dir=tmp_path)
    retrieve = AsyncMock(return_value=(ROWS, snapshot(), {}))
    monkeypatch.setattr(retrieval, 'retrieve', retrieve)
    answer = dict(progress=[], changes=[], open_issues=[], coordination=[], uncertainties=[])
    complete = AsyncMock(return_value=(json.dumps(answer), {}))
    monkeypatch.setattr(AnswerModel, 'complete', complete)
    run = dict(run_id='trial', cutoff=ds.cutoff, modes=list('BFG'), dataset_id='ds', generate_answers=generate, question='核查', project_id='项目A')
    result = await demo.experiment(run)
    assert retrieve.await_count == 1 and complete.await_count == expected
    assert len(result['anonymous_answers']) == expected
    assert set(result['groups']) == set('BFG')
    assert all('mode' not in a for a in result['anonymous_answers'])


@pytest.mark.asyncio
async def test_retrieval_uses_only_enabled_channels_and_verifies_graph_witnesses(tmp_path, monkeypatch):
    import sqlite3
    from unittest.mock import AsyncMock

    records = {
        'seed': record('seed', '组件 ALPHA-01 已提交。'),
        'linked': record('linked', '组件 ALPHA-01 尚未批准。'),
        'similar': record('similar', '组件 ALPHA-01-X 已批准。'),
        'independent': record('independent', '服务需补签。'),
    }
    dataset = SimpleNamespace(records=records, cutoff='2026-02-01')
    with sqlite3.connect(tmp_path / 'knowledge.db') as db:
        db.executescript("""
            CREATE TABLE facts (id TEXT, text TEXT, fact_type TEXT, source_chunk_id TEXT,
                source_unit_id TEXT, timestamp INTEGER, confidence REAL, subject_entity_id TEXT, object_entity_id TEXT);
            CREATE TABLE chunk_units (chunk_id TEXT, unit_id TEXT, unit_ordinal_in_chunk INTEGER);
            CREATE TABLE entities (id TEXT, name TEXT, entity_type TEXT, quality_status TEXT);
            CREATE TABLE edges (source_type TEXT, source_id TEXT, target_type TEXT, target_id TEXT,
                                edge_type TEXT, properties TEXT);
            INSERT INTO entities VALUES ('e','ALPHA-01','System','active');
        """)
        for key, row in records.items():
            db.execute("INSERT INTO chunk_units VALUES (?,?,0)", (key, key))
            db.execute("INSERT INTO facts VALUES (?,?, 'STATUS',?,?,0,1,NULL,NULL)",
                       ('f'+key, row['content'], key, key))
            if key != 'independent':
                db.execute("INSERT INTO edges VALUES ('fact',?,'entity','e','ABOUT',NULL)", ('f'+key,))
    monkeypatch.setattr(retrieval, 'query_original_vectors', AsyncMock(return_value=['seed']))
    searched = []

    def handler(request):
        collection = json.loads(request.content)['collection']
        searched.append(collection)
        field, key = ('fact_id', 'findependent') if collection == 'facts' else ('entity_id', 'e')
        return httpx.Response(200, json={'collection':collection, 'results':[
            dict(id=key, score=0.9, payload={field:key})]})

    monkeypatch.setattr(retrieval, 'httpx', SimpleNamespace(AsyncClient=lambda **kw: httpx.AsyncClient(
        **kw, transport=httpx.MockTransport(handler))))
    for modes, expected in [('B', []), ('F', ['facts']), ('G', ['facts', 'entities'])]:
        searched.clear()
        _, snap, _ = await retrieval.retrieve('http://local', tmp_path, dataset, 'ZZZ', '项目A', modes)
        assert searched == expected and list(snap.originals) == ['seed']
        if modes in ('F', 'G'):
            assert list(snap.facts) == ['independent']
        if modes == 'G':
            assert set(snap.paths) == {'seed', 'linked'}
            assert snap.paths['linked'].hops[0].witness == ('fseed', 'native:e', 'ALPHA-01', 'flinked')
        else:
            assert not snap.paths


@pytest.mark.asyncio
async def test_experiment_api_defaults_to_evidence_only_and_freezes_cutoff(tmp_path, monkeypatch):
    ds = SimpleNamespace(id='d', profile='app_focus', cutoff='2026-11-20T23:59:00+08:00')
    monkeypatch.setattr(app.Demo, 'get_dataset', lambda self, key: ds)
    monkeypatch.setattr(app.Demo, 'launch', lambda self, worker, **fields: fields)
    web = app.create_app(tmp_path, tmp_path)
    body = {'dataset_id': 'd', 'question': ' 核查进展 '}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web), base_url='http://local') as client:
        async def submit(**fields):
            return await client.post('/demo/experiments/bfg', json={**body, **fields})
        response = await submit()
        assert response.status_code == 202
        run = response.json()
        assert run['question'] == '核查进展' and run['generate_answers'] is False
        assert run['modes'] == list('BFG') and run['cutoff'] == ds.cutoff
        assert (await submit(modes=['F', 'F'])).status_code == 422
        assert (await submit(cutoff='2026-11-21T00:00:00+08:00')).status_code == 422


@pytest.mark.parametrize('body', ['按上面那个版本重跑。', '按上面说的版本重跑。',
                                  '照刚才提到的配置执行。', '沿用上一条的设置。', '按上一个版本重跑。', '按上文所述版本重跑。'])
@pytest.mark.asyncio
async def test_dependent_reply_reaches_answer_input(body):
    records = {'base': record('base', '指定版本为第三版。'),
               'reply': record('reply', body, reply_to='base')}
    fact = {'f': dict(text=body, source_quote=None, evidence_ids=['reply'])}
    lane, _ = fact_lane(fact, ['f'], records)
    for mode in ('F', 'G'):
        result = await pack(MODEL, mode, '如何执行？', '项目A', '2026-01-01', records, Snapshot({}, lane, {}))
        assert set(result['source_ids']) == {'base', 'reply'}
    assert not fact_lane(fact, ['f'], {'reply': records['reply']})[0]


@pytest.mark.asyncio
async def test_graph_keeps_explicit_links_between_seeds_and_through_new_sources():
    import sqlite3
    rows = {k: record(k, k, thread_id='thread') for k in ('child', 'parent')}
    rows['child'].update(reply_to='parent', attachment_refs=['doc', 'doc'])
    rows.update({k: dict(record(k, k), source_type='document', source_id='doc') for k in ('doc1', 'doc2')})
    rows['doc2']['attachment_refs'] = ['leaf']
    rows['leaf'] = record('leaf', 'leaf')
    scope = {k: 'requested' for k in rows}
    spans = {k: (Span(k, 0, len(k)),) for k in rows}
    # Empty fact index isolates native reply, thread and attachment evidence.
    with sqlite3.connect(':memory:') as db:
        db.executescript("""
            CREATE TABLE facts (id, text, fact_type, source_chunk_id, source_unit_id);
            CREATE TABLE chunk_units (chunk_id, unit_id);
            CREATE TABLE edges (source_type, source_id, target_type, target_id, edge_type, properties);
        """)
        for seeds in (['child'], ['parent'], ['child', 'parent', 'doc1'], ['parent', 'child', 'doc1']):
            paths = retrieval.graph_paths(db, rows, scope, [], seeds, spans)
            chains = {tuple((h.origin, h.relation, h.target) for h in p.hops) for p in paths}
            reply = (('child', 'reply_to', 'parent'),)
            assert reply in chains
            assert any(c[-1] == ('child', 'attachment', 'doc1') for c in chains)
            if seeds == ['parent']:
                assert (('parent', 'has_reply', 'child'),) in chains
            assert not any(h.relation == 'same_thread' for p in paths for h in p.hops)
            assert any(len(c) > 1 and c[-1] == ('doc2', 'attachment', 'leaf') for c in chains)
            assert len(chains) == len(paths)
            path = next(p for p in paths if p.hops[-1].relation == 'reply_to')
            result = await pack(MODEL, 'G', '核查', '项目A', '2026-01-01', rows,
                                Snapshot(spans, {}, {'parent': path}))
            assert result['paths']['parent'][0]['relation'] == 'reply_to'


def test_reply_supplies_named_context_without_a_reference_phrase():
    rows = {'p': record('p', '请核对设备移交清单。'),
            'r': record('r', '明天完成核对。', reply_to='p')}
    fact = dict(text='明天完成设备移交清单核对。', evidence_ids=['r'], source_quote=None)
    spans, _ = fact_lane({'f': fact}, ['f'], rows)
    assert {s.source for s in spans['r']} == {'p', 'r'}
    rows['r']['content'] = '设备移交清单明天完成核对。'
    assert {s.source for s in fact_lane({'f': fact}, ['f'], rows)[0]['r']} == {'r'}
    rows['r']['content'] = '刚才完成了检查，上面有水印。'
    assert {s.source for s in fact_lane({'f': dict(fact, text=rows['r']['content'])}, ['f'], rows)[0]['r']} == {'r'}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["F", "G"])
async def test_fact_fallback_competes_at_fact_rank_and_keeps_required_context(mode):
    text = '独立背景段。' * 150 + '仅限预演。'
    rows = dict(ROWS, s=record('s', text), q=record('q', '优先证据。' * 40))
    fact = (Span('s', len(text) - 5, len(text)), full('p'))
    originals = {'s': (Span('s', 0, len(text)),)}
    facts = {'q': (Span('q', 0, len(rows['q']['content'])),), 's': fact}
    path = PathEvidence((Hop('x', 'reply_to', 's', ()),), (full('x'), *fact))
    snap = Snapshot(originals, facts, {'s': path})

    async def run(snapshot, budget=MODEL.evidence_budget):
        return await pack(replace(MODEL, evidence_budget=budget), mode, '核查', '项目A', '2026-01-01', rows, snapshot)

    # If both single-channel candidates just fit, T#1 must enter before the
    # fallback at T#2 even though s leads the original and graph rankings.
    fact_only = await run(Snapshot({}, facts, {}))
    result = await run(snap, fact_only['input_tokens'])
    assert result['messages'] == fact_only['messages']
    assert result['source_ids'][0] == 'q'
    taken = [d for d in result['decisions'] if d['result'] == 'included']
    assert [(d['source'], d['kind']) for d in taken] == [('q', 'source'), ('s', 'fact_fallback')]
    assert taken[1]['ranks'] == (None, 2, None) and taken[1]['score'] == '1/22'
    assert any(d['source'] == 's' and d['kind'] == 'source' and d['result'] == 'over_budget'
               for d in result['decisions'])

    # Enough for the fact group alone, but not after its higher-ranked competitor.
    q_only = await run(Snapshot({}, {'q': facts['q']}, {}))
    group_only = await run(Snapshot({}, {'s': fact}, {}))
    assert group_only['input_tokens'] <= q_only['input_tokens']
    assert (await run(snap, q_only['input_tokens']))['source_ids'] == ['q']

    # A missing token must not split off the parent. A roomy budget keeps the
    # full original, including qualifications outside the fact excerpt.
    dual = Snapshot(originals, {'s': fact}, {})
    assert not (await run(dual, group_only['input_tokens'] - 1))['source_ids']
    roomy = await run(dual)
    assert roomy['intervals']['s'] == originals['s']
    assert roomy['intervals']['p'] == (full('p'),)
    assert [d['kind'] for d in roomy['decisions'] if d['result'] == 'included'] == ['source']


def test_facet_queries_preserve_scope_and_require_corroborated_catalogue():
    from project_demo.query_facets import plan_facets

    assert plan_facets('升级前，财务、采购分别核对什么？', {}) == {
        '财务': '升级前，财务核对什么', '采购': '升级前，采购核对什么'}
    question = '请按两类合同汇总生效时间、例外和责任。'
    rows = {str(i): {'title': name + ending, 'content': name + '合同'}
            for i, (name, ending) in enumerate((n, e) for n in ('租赁', '采购')
                                                for e in ('申请记录', '复查纪要'))}
    assert set(plan_facets(question, rows)) == {'租赁', '采购'}
    assert not plan_facets(question, {'0': rows['0'], '2': rows['2']})
    assert not plan_facets('这两笔金额能否相加？', rows)


@pytest.mark.asyncio
async def test_marginal_packing_prefers_a_new_object_over_duplicate_excerpt():
    rows = {key: record(key, body) for key, body in [
        ('a', '甲方采购计划的签发时间已经确定。'), ('b', '甲方采购计划的签发时间已经确定。'),
        ('c', '乙方设备审批仍在等待合同签署。')]}
    snap = Snapshot({k: (Span(k, 0, len(v['content'])),) for k, v in rows.items()},
                    {}, {}, ('甲方', '乙方'), 'facet_novelty')
    result = await pack(MODEL, 'B', '甲方、乙方分别进展如何？', '项目A', '2026-01-01', rows, snap)
    assert result['source_ids'] == ['a', 'c', 'b']
    normal = await pack(MODEL, 'B', '甲方、乙方分别进展如何？', '项目A', '2026-01-01', rows,
                        replace(snap, policy='rank'))
    assert normal['source_ids'] == ['a', 'b', 'c']
