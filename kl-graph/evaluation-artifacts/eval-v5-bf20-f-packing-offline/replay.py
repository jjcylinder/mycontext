"""Offline F packing experiment. The selection phase does not read grading keys."""
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from project_demo import evidence_retrieval as retrieval
from project_demo.answer import AnswerModel, _prompt, evidence_view
from project_demo.context import project_scope
from project_demo.dataset import load_dataset
from project_demo.evidence_packing import Span, candidates, merge_spans, pack

BASE = OUT.parent / 'eval-v5-bf-all20'
DENSE = OUT.parent / 'eval-v5-bf20-f-hybrid'
DATA = ROOT / 'data/project_demo/v5-natural-runtime'
os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'data/project_demo/tokenizers')

def read(path):
    return json.loads(path.read_text())

def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n')

def extend(intervals, spans):
    trial = intervals.copy()
    for span in spans:
        trial[span.source] = merge_spans((*trial.get(span.source, ()), span))
    return trial

def render(question, cutoff, rows, intervals, *, columns):
    evidence=[]
    for key,spans in intervals.items():
        row=rows[key].copy()
        row['content']='\n[…]\n'.join(row['content'][s.start:s.end] for s in spans)
        row['source_ranges']=[(s.start,s.end) for s in spans]
        evidence.append(row)
    messages=_prompt(question,'项目A',cutoff,evidence,None,compact=True)
    if columns:
        payload=json.loads(messages[1]['content'])
        items=payload['source_excerpts']
        fields=list(dict.fromkeys(f for row in items for f in row))
        # Place frequently present fields first to reduce trailing empty slots.
        fields.sort(key=lambda f:-sum(f in row for row in items))
        values=[]
        for row in items:
            cells=[row.get(f) for f in fields]
            while cells and cells[-1] is None:
                cells.pop()
            assert {f:v for f,v in zip(fields,cells) if v is not None}==row
            values.append(cells)
        payload.pop('source_excerpts')
        payload['source_excerpt_columns']=fields
        payload['source_excerpts']=values
        messages[1]['content']=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    return messages

async def main():
    folder=next((ROOT.parent/'datasets').glob('*v5*'))
    ds=load_dataset(folder,'app_focus')
    meta=read(BASE/'manifest.json')
    assert ds.digest==meta['digest']
    model=AnswerModel('',meta['model'],'',32768,8192,1800,'','o200k_base')
    index=DATA/'datasets'/ds.id/meta['index']
    old={r['bf_id']:r['input_sha256'] for r in csv.DictReader((BASE/'summary.csv').open(encoding='utf-8-sig')) if r['mode']=='F'}
    question_rows=(folder/'evaluation/bf_all20/questions.v1.jsonl').read_text().splitlines()
    for line in question_rows:
        row=json.loads(line)
        k,q=row['bf_id'],row['question']
        cache=read(DENSE/f'{k}-retrieval.json')
        with patch.object(retrieval,'query_original_vectors',AsyncMock(return_value=cache['vector'])),patch.object(retrieval,'search',AsyncMock(return_value=cache['fact_vector_hits'])):
            records,snapshot,_=await retrieval.retrieve('http://unused',index,ds,q,'项目A','F',ds.cutoff)
        group=await pack(model,'F',q,'项目A',ds.cutoff,records,snapshot)
        assert group['input_sha256']==old[k]
        rows={key:dict(evidence_view(r),project_scope=project_scope(r,'项目A')) for key,r in records.items()}
        intervals=group['intervals']
        def messages_for(spans,columns=True):
            return render(q,ds.cutoff,rows,spans,columns=columns)
        assert group['messages']==messages_for(intervals,False)
        msgs=messages_for(intervals)
        compact=await model.count(msgs)
        if compact>group['input_tokens']:
            raise ValueError('Compression did not save space')
        pool=candidates('F',snapshot)
        trace=[]
        for entry in pool:
            trial=extend(intervals,entry.spans)
            if trial==intervals:
                continue
            proposed=messages_for(trial)
            cost=await model.count(proposed)
            if cost<=model.evidence_budget:
                intervals,msgs=trial,proposed
                trace.append([entry.source,entry.kind,cost])
        assert extend(intervals,[s for spans in group['intervals'].values() for s in spans])==intervals
        save(OUT/f'{k}.json',dict(old_sha256=old[k],old_tokens=group['input_tokens'],compact_tokens=compact,
             final_tokens=await model.count(msgs),old_intervals=group['intervals'],intervals=intervals,
             extra_candidates=trace,messages=msgs,pool=[dict(target=c.source,kind=c.kind,ranks=c.ranks,spans=c.spans) for c in pool]))
if __name__=='__main__': asyncio.run(main())
