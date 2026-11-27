"""Paired F evaluation: use archived dense rankings, recompute hybrid facts and packing."""
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import AsyncMock, patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from project_demo import evidence_retrieval as retrieval
from project_demo.answer import AnswerModel, parse_answer
from project_demo.codex_bridge import ChatRequest, CodexBridge
from project_demo.dataset import load_dataset
from project_demo.evidence_packing import pack
from project_demo.runtime import write_json

OLD = OUT.parent / 'eval-v5-bf-all20'
DATA = ROOT / 'data/project_demo/v5-natural-runtime'
DS = next((ROOT.parent / 'datasets').glob('*v5*'))
os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'data/project_demo/tokenizers')

def read(path):
    return json.loads(path.read_text())

async def main():
    meta = read(OLD / 'manifest.json')
    ds = load_dataset(DS, 'app_focus')
    assert ds.digest == meta['digest'] and all(r['is_simulated'] for r in ds.records.values())
    index = DATA / 'datasets' / ds.id / meta['index']
    rows = [json.loads(s) for s in (DS / 'evaluation/bf_all20/questions.v1.jsonl').read_text().splitlines()]
    old = {r['bf_id']:r for r in csv.DictReader((OLD/'summary.csv').open(encoding='utf-8-sig')) if r['mode']=='F'}
    archives = {}
    for p in (DATA/'runs').glob('*.json'):
        a = read(p)
        g = a.get('groups',{}).get('F',{})
        if g.get('input_sha256') and a.get('retrieval'):
            archives[g['input_sha256']] = a['retrieval']
    model = AnswerModel('',meta['model'],'',32768,8192,1800,'','o200k_base')
    bridge = CodexBridge()
    assert model.model == bridge.model_name and len(rows)==len(old)==20
    if not hasattr(retrieval, "rank_texts"):
        raise RuntimeError("此脚本用于归档的事实混合检索版本。当前实现不支持该实验。")
    write_json(OUT/'manifest.json', dict(dataset_digest=ds.digest, index=index.name, cutoff=ds.cutoff,
               model=model.model, budget=8192, mode='F', policy='rank', fact_depth=200,
               dense_rankings='from original scored runs', max_delivery_attempts=2,
               frozen_atoms_sha256=hashlib.sha256((OLD/'atom-review.csv').read_bytes()).hexdigest()))
    async def answer(key, group):
        path=OUT/f'{key}-answer.json'
        if path.exists() and read(path).get('answer') and read(path)['input_sha256']==group['input_sha256']:return
        result=dict(input_sha256=group['input_sha256'],attempts=[])
        for _ in range(2):
            start=perf_counter()
            attempt={}
            try:
                raw,usage=await bridge._run(ChatRequest(model=model.model,messages=group['messages'],temperature=0,max_tokens=1800))
                attempt.update(raw=raw,usage=usage)
                result['answer']=parse_answer(raw,set(group['source_ids']))
            except Exception as exc:
                attempt['error']=str(exc)
            attempt['seconds']=round(perf_counter()-start,2)
            result['attempts'].append(attempt)
            if 'answer' in result:break
        write_json(path,result)
        print(key,'answer' if result.get('answer') else 'failed',len(result['attempts']),flush=True)
    jobs=[]
    for row in rows:
        key,q,question=row['bf_id'],row['question_id'],row['question']
        stored=OLD/f'{q}-retrieval.json'
        audit=read(stored) if stored.exists() else archives[old[key]['input_sha256']]
        assert not audit.get('facet_queries')
        with patch.object(retrieval,'query_original_vectors',AsyncMock(return_value=audit['vector'])), patch.object(retrieval,'search',AsyncMock(return_value=audit['fact_hits'])):
            rec,snap,new=await retrieval.retrieve('http://unused',index,ds,question,'项目A',('F',),ds.cutoff)
        assert new['O']==audit['O'] and not snap.paths
        group=await pack(model,'F',question,'项目A',ds.cutoff,rec,snap)
        write_json(OUT/f'{key}-input.json',group)
        write_json(OUT/f'{key}-retrieval.json',new)
        assert group['input_sha256']!=old[key]['input_sha256']
        jobs.append(asyncio.create_task(answer(key,group)))
        await asyncio.sleep(0)
    await asyncio.gather(*jobs)
if __name__=='__main__':asyncio.run(main())
