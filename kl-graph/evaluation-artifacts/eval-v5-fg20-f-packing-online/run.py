"""Submit the twenty frozen optimized F prompts to the same ChatGPT bridge."""
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OFF = OUT.parent / 'eval-v5-fg20-f-packing-offline'
sys.path.insert(0, str(ROOT))
from project_demo.answer import parse_answer
from project_demo.codex_bridge import ChatRequest, CodexBridge
from project_demo.dataset import load_dataset
from project_demo.runtime import write_json

def read(path):
    return json.loads(path.read_text())

async def main():
    dataset = load_dataset(next((ROOT.parent/'datasets').glob('*v5*')), 'app_focus')
    assert all(r['is_simulated'] for r in dataset.records.values())
    bridge = CodexBridge()
    bridge.verify_login()
    inputs = {}
    hashes = {}
    for i in range(1, 21):
        case = f'FG{i:02}'
        old = OUT.parent / ('eval-v5-fg-main10' if i<=10 else 'eval-v5-fg-next10')
        meta = read(old/'manifest.json')
        assert meta['digest']==dataset.digest and meta['model']==bridge.model_name
        data = read(OFF/f'{case}.json')
        base = read(old/f'{case}-F-input.json')
        assert base['input_sha256']==data['original_sha256'] and data['final_tokens']<=8192
        messages = data['messages']
        payload = json.loads(messages[1]['content'])
        assert payload['question']==json.loads(base['messages'][1]['content'])['question']
        refs={dict(zip(payload['source_excerpt_columns'],row))['id'] for row in payload['source_excerpts']}
        assert refs==set(data['intervals']) and refs<=dataset.records.keys()
        inputs[case] = messages,refs
        hashes[case] = hashlib.sha256(json.dumps(messages,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    write_json(OUT/'manifest.json',dict(model=bridge.model_name,reasoning='low',requested_max_tokens=1800,budget=8192,input_hashes=hashes,digest=dataset.digest,mode='F'))
    workers=asyncio.Semaphore(2)
    async def run(case):
        async with workers:
            path = OUT/f'{case}-answer.json'
            if path.exists():
                prior=read(path)
                assert prior['input_sha256']==hashes[case]
                if prior.get('answer') or len(prior['attempts'])>=2:
                    return
            messages,refs=inputs[case]
            result=dict(input_sha256=hashes[case],attempts=[])
            for _ in range(2):
                start=monotonic()
                raw,usage=await bridge._run(ChatRequest(model=bridge.model_name,messages=messages,temperature=0,max_tokens=1800))
                item=dict(raw=raw,usage=usage,seconds=round(monotonic()-start,2))
                result['attempts'].append(item)
                try:
                    result['answer']=parse_answer(raw,refs)
                except ValueError as exc:
                    item['error']=str(exc)
                write_json(path,result)
                if result.get('answer'):
                    break
            print(case,'valid' if result.get('answer') else 'invalid',len(result['attempts']),flush=True)
    outcomes=await asyncio.gather(*(run(case) for case in inputs),return_exceptions=True)
    for result in outcomes:
        if isinstance(result,BaseException):
            raise result
if __name__=='__main__':asyncio.run(main())
