"""Compare both G rankers on one captured candidate pool per question."""
import asyncio
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI
import uvicorn
from project_demo.model_setup import setup_chatgpt
from project_demo.runtime import GraphRuntime, write_json
from project_demo.dataset import load_dataset
from project_demo.answer import AnswerModel, parse_answer
from project_demo.codex_bridge import ChatRequest, CodexBridge

DATA = ROOT / 'data/project_demo/v5-natural-runtime'
DS = next((ROOT.parent/'datasets').glob('*v5*'))
EXAM = DS / 'evaluation/fg_main10'
os.environ.update(KL_GRAPH_BACKEND='sqlite', KL_VECTOR_BACKEND='zvec', LITELLM_LOCAL_MODEL_COST_MAP='True')
def save(name, value):
    write_json(OUT / (name+'.json'),value)
def log(*msg):
    print(*msg,flush=True)

async def answer(q, variant, group, model, bridge):
    name=f'{q}-{variant}'
    file=OUT/(name+'-answer.json')
    if file.exists():
        assert json.loads(file.read_text())['input_sha256']==group['input_sha256']
        return
    t=perf_counter()
    try:
        raw,usage=await bridge._run(ChatRequest(model=model.model,messages=group['messages'],temperature=0,max_tokens=model.max_output))
        result=dict(raw_answer=raw,input_sha256=group['input_sha256'],usage=usage,seconds=perf_counter()-t,model=model.model)
        try:
            result['answer']=parse_answer(raw,set(group['source_ids']))
        except Exception as exc:
            result['validation_error']=str(exc)
        save(name+'-answer',result)
        log(name,'answer',round(result['seconds'],1),result.get('validation_error','ok'))
    except Exception as exc:
        save(name+'-error',dict(error=str(exc),seconds=perf_counter()-t))
        log(name,'error',str(exc))

async def main():
    port=8768
    app=FastAPI()
    for router in setup_chatgpt(port, DATA, None):
        app.include_router(router)
    from project_demo import evidence_retrieval as er
    from project_demo.evidence_packing import pack
    ds=load_dataset(DS,'app_focus')
    assert all(v['is_simulated'] is True for v in ds.records.values())
    data_dir=DATA/'datasets'/ds.id
    index=next(data_dir.glob('*/ready.json')).parent
    graph=GraphRuntime(index,data_dir/'dws')
    assert graph.ready
    model=AnswerModel.from_env()
    bridge=CodexBridge()
    assert model.model==bridge.model_name and model.evidence_budget==8192
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning',access_log=False))
    serving=asyncio.create_task(server.serve())
    jobs=[]
    questions={x['question_id']:x['question'] for x in map(json.loads,(EXAM/'questions.jsonl').read_text().splitlines())}
    try:
        while not server.started:
            if serving.done(): await serving
            await asyncio.sleep(.1)
        base=await graph.start()
        log('ready',ds.id,ds.cutoff)
        cases=[]
        for q in ('FG01','FG06'):
            captured={}
            default=er.rank_light_paths
            def capture(question,paths,records,originals,facts):
                captured['paths']=tuple(paths)
                start=perf_counter()
                value=default(question,paths,records,originals,facts)
                captured['rank_seconds']=perf_counter()-start
                return value
            question=questions[q]
            start=perf_counter()
            with patch.object(er,'rank_light_paths',capture):
                records,snap,audit=await er.retrieve(base,index,ds,question,'项目A',('G',),ds.cutoff,policy='rank')
            shared=perf_counter()-start-captured['rank_seconds']
            paths=captured['paths']
            save(q+'-pool',dict(dataset_digest=ds.digest,index=index.name,question=question,cutoff=ds.cutoff,
                 retrieval=audit,originals=snap.originals,facts=snap.facts,paths=[asdict(p) for p in paths]))
            t=perf_counter()
            group=await pack(model,'G',question,'项目A',ds.cutoff,records,snap)
            group.update(rank_seconds=captured['rank_seconds'],packing_seconds=perf_counter()-t,shared_seconds=shared,
                         ranked_targets=list(snap.paths),candidate_paths=len(paths))
            save(q+'-light-input',group)
            jobs.append(asyncio.create_task(answer(q,'light',group,model,bridge)))
            cases.append((q,question,records,snap,paths,group,shared))
            log(q,'pool',len(paths),'unique texts',len({er.path_text(p,records) for p in paths}),'light packed')
        for q,question,records,snap,paths,light,shared in cases:
            log(q,'vector starting')
            t=perf_counter()
            ranked,oversize=await er.rank_paths(question,paths,records)
            rank_seconds=perf_counter()-t
            t=perf_counter()
            group=await pack(model,'G',question,'项目A',ds.cutoff,records,replace(snap,paths=ranked))
            group.update(rank_seconds=rank_seconds,packing_seconds=perf_counter()-t,shared_seconds=shared,
                         ranked_targets=list(ranked),candidate_paths=len(paths),oversize=oversize)
            save(q+'-vector-input',group)
            if light['messages']==group['messages']:
                await asyncio.gather(*jobs)
                result=json.loads((OUT/(q+'-light-answer.json')).read_text())
                save(q+'-vector-answer',dict(result,reused_from=q+'-light-answer.json'))
            else:
                jobs.append(asyncio.create_task(answer(q,'vector',group,model,bridge)))
            log(q,'vector packed',round(rank_seconds,2))
        await asyncio.gather(*jobs)
    finally:
        await graph.stop()
        server.should_exit=True
        await serving

if __name__=='__main__':
    asyncio.run(main())
