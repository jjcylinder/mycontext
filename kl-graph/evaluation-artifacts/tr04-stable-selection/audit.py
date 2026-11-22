"""Offline checks on cold inputs and on omissions injected into a grounded control."""
import json
from copy import deepcopy
from pathlib import Path

from project_demo.answer import parse_answer

OUT = Path(__file__).resolve().parent
load = lambda file: json.loads(file.read_text())
a, b = (load(OUT / f'cold-{s}.json') for s in (1, 97))
for m in ('B', 'C'):
    assert a[m] == b[m] and a[m]['tokens'] <= 8192 and not a[m]['calls']
    assert set(a['requests']) == set(b['requests']) == {'/v1/embeddings'}
r = a['C']
assert {'T08-02','T08-03','T08-04','T04-06','T04-07','XTR3-06','XTR3-07','ATR0-S04',
        'XTR2-04','XTR3-05'} <= set(r['ids'])
sources = {s['id']:s for s in load(OUT/'sources-C-1.json')}
run = load(OUT.parent/'eval-tr04-v4-pro-grounded-c/run.json')

def check(raw, rows):
    review = []
    result = parse_answer(json.dumps(raw, ensure_ascii=False), set(rows), require_items=True,
                          assignment_sources=rows, assignment_review=review, question=run['question'])
    return [e for e in review if e['status'] in {'missing_party','missing_duty'}]

items = []
for party, required in r['requirements'].items():
    citations = [{'source_id': refs[0], 'quote': quote} for quote, refs in required.items()
                 if refs != ['XTR2-04']]  # The other scoped source independently covers the same check.
    # Choose an actual responsibility as the assignment, retaining all supporting clauses.
    anchor = next(c for c in citations if ('由' in c['quote'] and party in c['quote'])
                  or (party == '发布' and '申请' in c['quote']))
    items.append({'object':party, 'action':'分工', 'scope':'翻译演示', 'period':'current',
                  'state':'known','kind':'coordination','text':anchor['quote'],
                  'evidence_ids':list(dict.fromkeys(c['source_id'] for c in citations)),
                  'assignment':anchor, 'details':citations})
assert check({'items':items},sources) == []
omissions={}
for name in ['remove_party','remove_wait','unknown','remove_check']:
    changed = deepcopy(items)
    party = '测试' if name == 'remove_check' else '模型'
    item = next(i for i in changed if i['object']==party)
    if name == 'remove_party':
        changed.remove(item)
    elif name == 'unknown':
        item.update(state='unknown',text='待确认',evidence_ids=[])
    else:
        word = 'QA-A27' if name=='remove_wait' else 'TR-X03'
        item['details'] = [c for c in item['details'] if word not in c['quote']]
    misses = check({'items':changed},sources)
    assert len(misses)==1 and misses[0]['object']==party, (name,misses)
    assert misses[0]['status']==('missing_party' if name=='remove_party' else 'missing_duty')
    omissions[name] = misses
archived = load(OUT.parent/'v4-deepseek-runtime/runs'/(run['run_id']+'.json'))
old = json.loads(archived['model_output'])
old['items'] = [i for i in old['items'] if i['object']!='模型']
old_result = check(old,{s['id']:s for s in run['result']['evidence']})
assert any(e['status']=='missing_party' and e['object']=='模型' for e in old_result)
def other_docs(rows):
    return sorted(s['id'] for s in rows if s['source_type'] in {'meeting','document'} and
                  any(word in s['title'] for word in ['公函','摘要','润色','早报','周报生成']))
stats={'current_other_titles':other_docs(sources.values()),'historical_other_titles':other_docs(run['result']['evidence'])}
assert len(stats['current_other_titles']) <= len(stats['historical_other_titles'])
(OUT/'audit.json').write_text(json.dumps({'control':'grounded synthetic control passed','omissions':omissions,
    'mutated_archive':old_result,'scope_sample':stats},ensure_ascii=False,indent=2))
print('Cold prompts identical, all 10 key C sources present, 4 omissions detected, title sample',stats)
