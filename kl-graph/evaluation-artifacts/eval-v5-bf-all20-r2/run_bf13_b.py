"""Rebuild the B baseline for FG02 with the frozen original-source rankings."""
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic
from unittest.mock import AsyncMock, patch

OUT = Path(__file__).resolve().parent
FG = OUT.parent / 'eval-v5-fg-main10'
sys.path.insert(0,str(OUT.parent/'eval-v5-bf20-f-packing-offline'))
import replay as r
from project_demo.answer import parse_answer
from project_demo.codex_bridge import ChatRequest, CodexBridge

def digest(messages):
    return hashlib.sha256(json.dumps(messages,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

async def main():
    ds=r.load_dataset(next((r.ROOT.parent/'datasets').glob('*v5*')),'app_focus')
    meta=r.read(FG/'manifest.json')
    assert ds.digest==meta['digest'] and all(x['is_simulated'] for x in ds.records.values())
    raw=r.read(FG/'FG02-retrieval.json')
    hist=r.read(FG/'FG02-F-input.json')
    q=json.loads(hist['messages'][1]['content'])['question']
    question=next(x for x in map(json.loads,(ds.root/'evaluation/bf_all20/questions.jsonl').read_text().splitlines()) if x['bf_id']=='BF13')
    assert question['question']==q and question['question_id']=='FG02'
    for source,ans in [(hist,r.read(FG/'FG02-F-answer.json')),(r.read(OUT.parent/'eval-v5-fg20-f-packing-offline/FG02.json'),r.read(OUT.parent/'eval-v5-fg20-f-packing-online/FG02-answer.json'))]:
        assert digest(source['messages'])==ans['input_sha256'] and json.loads(source['messages'][1]['content'])['question']==q
    model=r.AnswerModel('',meta['model'],'',32768,8192,1800,'','o200k_base')
    with patch.object(r.retrieval,'query_original_vectors', AsyncMock(return_value=raw['vector'])):
        records,snap,audit=await r.retrieval.retrieve('http://unused',r.DATA/'datasets'/ds.id/meta['index'],ds,q,'项目A',('B',),ds.cutoff)
    assert audit['O']==raw['O'] and not snap.facts and not snap.paths
    group=await r.pack(model,'B',q,'项目A',ds.cutoff,records,snap)
    r.save(OUT/'BF13-B-input.json',group)
    if '--offline' in sys.argv:return
    bridge=CodexBridge()
    assert bridge.model_name==meta['model']
    bridge.verify_login()
    path=OUT/'BF13-B-answer.json'
    result=r.read(path) if path.exists() else dict(input_sha256=group['input_sha256'],attempts=[])
    assert result['input_sha256']==group['input_sha256']
    for _ in range(len(result['attempts']),2):
        if result.get('answer'):break
        start=monotonic()
        raw,usage=await bridge._run(ChatRequest(model=model.model,messages=group['messages'],temperature=0,max_tokens=1800))
        item=dict(raw=raw,usage=usage,seconds=round(monotonic()-start,2))
        result['attempts'].append(item)
        try:result['answer']=parse_answer(raw,set(group['source_ids']))
        except ValueError as exc:item['error']=str(exc)
        r.save(path,result)
    print('valid',bool(result.get('answer')),'attempts',len(result['attempts']))
if __name__=='__main__':asyncio.run(main())
