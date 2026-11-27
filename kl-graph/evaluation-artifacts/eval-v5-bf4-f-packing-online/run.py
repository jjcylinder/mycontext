"""Four frozen paired prompts, randomized labels, scoring blind to packing mode."""
import asyncio
import csv
import hashlib
import json
import secrets
import sys
from pathlib import Path
from time import monotonic

OUT = Path(__file__).resolve().parent
PACK = OUT.parent / 'eval-v5-bf20-f-packing-offline'
sys.path.insert(0, str(PACK))
import replay
from project_demo.codex_bridge import ChatRequest, CodexBridge
from project_demo.answer import parse_answer

CASES = ('BF18', 'BF13', 'BF07', 'BF20')

def fingerprint(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def prepare():
    folder = next((replay.ROOT.parent / 'datasets').glob('*v5*'))
    ds = replay.load_dataset(folder, 'app_focus')
    meta = replay.read(replay.BASE / 'manifest.json')
    assert ds.digest == meta['digest'] and all(r['is_simulated'] for r in ds.records.values())
    rows = {k:dict(replay.evidence_view(r), project_scope=replay.project_scope(r, '项目A')) for k,r in ds.records.items()}
    prompts = {}
    baseline = {r['bf_id']: r for r in csv.DictReader((replay.BASE/'summary.csv').open(encoding='utf-8-sig')) if r['mode']=='F'}
    for case in CASES:
        archive = replay.read(PACK / f'{case}.json')
        new = archive['messages']
        question = json.loads(new[1]['content'])['question']
        spans = {k:[replay.Span(*s) for s in v] for k,v in archive['old_intervals'].items()}
        old = replay.render(question, ds.cutoff, rows, spans, columns=False)
        assert fingerprint(old) == archive['old_sha256'] == baseline[case]['input_sha256']
        prompts[case] = {'old': (old, set(spans)), 'new': (new, set(archive['intervals']))}
    keyfile = OUT / 'key.json'
    if not keyfile.exists():
        key = {}
        order = ['old','new','new','old']
        secrets.SystemRandom().shuffle(order)
        for case, first in zip(CASES, order):
            key[case]={'A':first,'B':'new' if first=='old' else 'old'}
        replay.save(keyfile,key)
    key = replay.read(keyfile)
    replay.save(OUT/'manifest.json',dict(cases=CASES, model=meta['model'],reasoning='low', requested_max_tokens=1800,
                evidence_budget=8192,max_invalid_output_retries=1, frozen_rubric_sha256=hashlib.sha256((replay.BASE/'atom-review.csv').read_bytes()).hexdigest(),
                hashes={k:{v:fingerprint(p[0]) for v,p in variants.items()} for k,variants in prompts.items()}))
    return prompts,key,meta

async def main():
    prompts,key,meta=prepare()
    if '--prepare' in sys.argv:
        return
    bridge=CodexBridge()
    bridge.verify_login()
    assert bridge.model_name==meta['model']
    async def generate(case,label):
        path=OUT/f'{case}-{label}.json'
        if path.exists():
            return
        messages,ids=prompts[case][key[case][label]]
        attempts=[]
        answer=None
        for _ in range(2):
            start=monotonic()
            raw,usage=await bridge._run(ChatRequest(model=meta['model'], messages=messages,temperature=0,max_tokens=1800))
            item=dict(raw=raw,usage=usage,seconds=round(monotonic()-start,2))
            attempts.append(item)
            try:
                answer=parse_answer(raw,ids)
                break
            except ValueError as err:
                item['error']=str(err)
        replay.save(path,dict(answer=answer,attempts=attempts))
        print(case,label,'valid' if answer else 'invalid',len(attempts),flush=True)
    results=await asyncio.gather(*(generate(c,l) for c in CASES for l in ('A','B')),return_exceptions=True)
    for x in results:
        if isinstance(x,BaseException):
            raise x
if __name__=='__main__':asyncio.run(main())
