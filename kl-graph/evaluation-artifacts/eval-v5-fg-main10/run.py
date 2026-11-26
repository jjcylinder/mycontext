"""Run the V5 F/G exam with shared retrieval, frozen packing and light G."""
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import uvicorn
from fastapi import FastAPI

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from project_demo.model_setup import setup_chatgpt
from project_demo.runtime import GraphRuntime, write_json
from project_demo.dataset import load_dataset
from project_demo.answer import AnswerModel, parse_answer
from project_demo.codex_bridge import ChatRequest, CodexBridge

DATA = ROOT/'data/project_demo/v5-natural-runtime'
DS = next((ROOT.parent/'datasets').glob('*v5*'))
EXAM = DS/'evaluation/fg_main10'
TWO = OUT.parent/'eval-v5-fg01-fg06-path-ranking'
OLD = OUT.parent/'eval-v5-bfg-light-default5-2026-11-25'
os.environ.update(KL_GRAPH_BACKEND='sqlite', KL_VECTOR_BACKEND='zvec', LITELLM_LOCAL_MODEL_COST_MAP='True')

def read(path):
    return json.loads(path.read_text())

def prior(q, mode, group, ds, model, index):
    if q in ('FG01','FG06') and mode=='G':
        path=TWO/(q+'-light-answer.json')
        old_input=read(TWO/(q+'-light-input.json'))
        old=read(path)
        pool=read(TWO/(q+'-pool.json'))
        same=pool['dataset_digest']==ds.digest and pool['index']==index.name and old['model']==model.model
    elif q=='FG07':
        path=OLD/f'PO03-{mode}.json'
        old_input=old=read(path)
        meta=read(OLD/'manifest.json')
        same=meta['model']==model.model and meta['dataset_id']==ds.id and meta['path_ranking']=='light'
    else:
        return
    if same and old_input['messages']==group['messages'] and old['input_sha256']==group['input_sha256']:
        return dict(answer=old['answer'],usage=old.get('usage'),historical_seconds=old.get('seconds'),
                    provenance=str(path.relative_to(OUT.parent)),attempts=[])

async def main():
    app=FastAPI()
    for router in setup_chatgpt(8768,DATA,None):
        app.include_router(router)
    from project_demo.evidence_retrieval import retrieve
    from project_demo.evidence_packing import pack
    ds=load_dataset(DS,'app_focus')
    assert len(ds.records)==349 and all(v['is_simulated'] is True for v in ds.records.values())
    data_dir=DATA/'datasets'/ds.id
    index=next(data_dir.glob('*/ready.json')).parent
    graph=GraphRuntime(index,data_dir/'dws')
    model=AnswerModel.from_env()
    bridge=CodexBridge()
    assert (model.model,model.evidence_budget,model.max_output)==(bridge.model_name,8192,1800)
    rows=list(map(json.loads,(EXAM/'questions.jsonl').read_text().splitlines()))
    write_json(OUT/'manifest.json',dict(dataset=ds.id,digest=ds.digest,index=index.name,cutoff=ds.cutoff,
               model=model.model,modes=['F','G'],policy='rank',g_ranking='light',budget=8192,
               rubric_sha256=sha256((EXAM/'reference_answers.jsonl').read_bytes()).hexdigest(),
               questions_sha256=sha256((EXAM/'questions.jsonl').read_bytes()).hexdigest()))
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=8768,log_level='warning',access_log=False))
    serving=asyncio.create_task(server.serve())
    slots=asyncio.Semaphore(2)
    async def generate(q,mode,group):
        name=f'{q}-{mode}';path=OUT/(name+'-answer.json')
        if path.exists() and read(path).get('answer') and read(path)['input_sha256']==group['input_sha256']:
            return
        result=prior(q,mode,group,ds,model,index)
        if result is None:
            result=dict(provenance='new',attempts=[])
            async with slots:
                for _ in range(2):
                    t=perf_counter()
                    attempt={}
                    try:
                        raw,usage=await bridge._run(ChatRequest(model=model.model,messages=group['messages'],temperature=0,max_tokens=model.max_output))
                        attempt.update(raw=raw,usage=usage)
                        result['answer']=parse_answer(raw,set(group['source_ids']))
                    except Exception as exc:
                        attempt['error']=str(exc)
                    attempt['seconds']=round(perf_counter()-t,3)
                    result['attempts'].append(attempt)
                    if 'answer' in result:break
        write_json(path,dict(input_sha256=group['input_sha256'],**result))
        print(name,result['provenance'],'ok' if 'answer' in result else 'failed',len(result['attempts']),flush=True)
    jobs=[]
    try:
        while not server.started:
            if serving.done(): await serving
            await asyncio.sleep(.1)
        base=await graph.start()
        for row in rows:
            q=row['question_id'];question=row['question']
            rec,snap,audit=await retrieve(base,index,ds,question,'项目A',('F','G'),ds.cutoff,policy='rank')
            assert audit['path_ranking']=='light'
            write_json(OUT/(q+'-retrieval.json'),audit)
            for mode in ('F','G'):
                group=await pack(model,mode,question,'项目A',ds.cutoff,rec,snap)
                assert mode=='G' or not group['paths']
                write_json(OUT/(f'{q}-{mode}-input.json'),group)
                jobs.append(asyncio.create_task(generate(q,mode,group)))
        await asyncio.gather(*jobs)
    finally:
        await graph.stop()
        server.should_exit=True
        await serving

if __name__=='__main__':asyncio.run(main())
