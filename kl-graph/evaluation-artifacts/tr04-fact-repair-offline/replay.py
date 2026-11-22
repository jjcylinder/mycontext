"""Exercise current evidence packing with archived ranking, without model calls."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
from project_demo import answer
from project_demo.context import _native_graph, visible_records
from project_demo.dataset import load_dataset
from project_demo.fact_evidence import commitment_evidence, duty_scope_sources

ROOT = Path('data/project_demo')
OUT = ROOT / 'tr04-fact-repair-offline'
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT/'tokenizers'))

async def main():
    dataset = load_dataset(Path('/home/jcylinder/codex/datasets/项目A_Demo应用增强数据_v4_长文版'), 'app_focus')
    graph_dir = ROOT / 'v4-deepseek-runtime/datasets' / dataset.id / 'graph-ac48d6bc8cf79275'
    rows = visible_records(dataset, '项目A')
    _, facts, _, _ = _native_graph(graph_dir, rows, '项目A', None, [], {})
    summaries = []
    for name in ('eval-tr04-v4-pro-c','eval-tr04-v4-pro-grounded-c'):
        archive = ROOT / name
        run = json.loads((archive/'run.json').read_text())
        records = visible_records(dataset, run['project_id'], run['cutoff'])
        traces = json.loads((archive/'timing.json').read_text())['traces']
        hints = []
        for trace in traces:
            req = json.loads((ROOT/'v4-deepseek-runtime/model_calls'/trace).read_text())['request']
            try:
                payload = json.loads(req['messages'][0]['content'])
            except ValueError:
                continue
            if 'graph_hints' in payload:
                hints = payload['graph_hints']
                break
        index = {(f['text'],f['source_unit_id']):f for f in facts}
        graph = {'facts':[index[h['text'],h['evidence_ids'][-1]] for h in hints], 'source_facts':facts}
        async def frozen(*args, **kwargs):
            return {'groups':deepcopy(run['coverage']['evidence_selection']['groups'])}
        model = answer.AnswerModel('', '', '', 131072, 8192, 65536, '', 'o200k_base')
        with patch.object(answer,'rank_evidence',frozen):
            messages,evidence,used,coverage = await answer.prepare_answer(
                model,'C',run['question'],run['project_id'],run['cutoff'],records,graph,
                vector_refs=run['coverage']['original_retrieval']['candidate_ids'])
        with patch.object(answer,'rank_evidence',frozen):
            repeated = await answer.prepare_answer(
                model,'C',run['question'],run['project_id'],run['cutoff'],records,graph,
                vector_refs=run['coverage']['original_retrieval']['candidate_ids'])
        assert repeated[0] == messages, 'Fixed inputs produced a different prompt'
        ids = [r['id'] for r in evidence]
        required = {'T08-02','T08-03','T08-04','T04-06','T04-07',
                    'ATR0-S04','XTR3-06','XTR3-07'}
        assert required <= set(ids)
        assert all(d['covered_by'] for d in coverage['evidence_selection']['delegation_coverage'])
        assert coverage['preflight_tokens'] <= 8192
        handoffs = [f['handoff'] for f in used.get('facts',[]) if 'handoff' in f]
        assert any(h['provider'] == '应用侧' and h['deliverable'] == '发布申请' for h in handoffs)
        assert any(h['provider'] is None and 'QA-A27' in h['deliverable'] for h in handoffs)
        synthetic = {'items':[{'object':'应用','action':'assignment','scope':'翻译Skill演示准备',
            'period':'current','state':'known','kind':'coordination','text':'应用补齐两列。',
            'evidence_ids':['T08-02','T08-03','T08-04'],
            'details':[{'source_id':'T08-04','quote':'我补这两列，当前还不是可训练性检查通过。'}]}]}
        displayed = answer.parse_answer(json.dumps(synthetic,ensure_ascii=False), set(ids),
            require_items=True, assignment_sources={r['id']:r for r in evidence}, question=run['question'])
        rendered = displayed['coordination'][0]['text']
        assert all(word in rendered for word in ('应用协调','QA-A27','期望答案','判定说明','我补这两列'))

        print(name,coverage['preflight_tokens'],len(ids),'chain',set(['T08-02','T08-03','T08-04']) <= set(ids))
        print('missing',[(d['source'],d.get('evidence_ids')) for d in coverage['evidence_selection']['delegation_coverage'] if not d['covered_by']])
        summary = {'archive':name,'tokens':coverage['preflight_tokens'],'ids':ids,
                   'duties':coverage['evidence_selection']['delegation_coverage'],
                   'handoffs':handoffs,'synthetic_display_check':rendered,'repeat_identical':True,
                   'model_calls':0}
        summaries.append(summary)
        (OUT/(name+'.json')).write_text(json.dumps({'messages':messages,'coverage':coverage},ensure_ascii=False,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    asyncio.run(main())
