"""Replay optimized F on frozen FG01–FG20 inputs with no remote calls."""
import asyncio
import csv
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

OUT = Path(__file__).resolve().parent
BF = OUT.parent / 'eval-v5-bf20-f-packing-offline'
sys.path.insert(0, str(BF))
import replay as r


async def main():
    folder = next((r.ROOT.parent / 'datasets').glob('*v5*'))
    ds = r.load_dataset(folder, 'app_focus')
    for batch in ('main10', 'next10'):
        base = OUT.parent / ('eval-v5-fg-' + batch)
        meta = r.read(base/'manifest.json')
        assert meta['digest']==ds.digest and meta['cutoff']==ds.cutoff
        model = r.AnswerModel('', meta['model'], '', 32768, 8192, 1800, '', 'o200k_base')
        index = r.DATA/'datasets'/ds.id/meta['index']
        for path in sorted(base.glob('FG??-F-input.json')):
            case = path.name[:4]
            run = r.read(path)
            archive = r.read(base/f'{case}-retrieval.json')
            assert not archive['facet_queries']
            question = json.loads(run['messages'][1]['content'])['question']
            with patch.object(r.retrieval, 'query_original_vectors', AsyncMock(return_value=archive['vector'])), patch.object(r.retrieval, 'search', AsyncMock(return_value=archive['fact_hits'])):
                rec, snap, audit = await r.retrieval.retrieve('http://unused', index, ds, question, '项目A', 'F', ds.cutoff)
            assert all(audit[k]==archive[k] for k in ('O','T'))
            rows = {key: dict(r.evidence_view(record), project_scope=r.project_scope(record,'项目A')) for key, record in rec.items()}
            old = {key:[r.Span(*s) for s in spans] for key,spans in run['intervals'].items()}
            old = {k:r.merge_spans(v) for k,v in old.items()}
            def render(intervals, columns=True):
                return r.render(question,ds.cutoff,rows,intervals,columns=columns)
            assert render(old,False)==run['messages']
            assert hashlib.sha256(json.dumps(run['messages'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==run['input_sha256']
            intervals=old
            msgs=render(old)
            tokens=await model.count(msgs)
            compact_tokens=tokens
            assert tokens<=run['input_tokens']<=8192
            for c in r.candidates('F',snap):
                trial=r.extend(intervals,c.spans)
                if trial==intervals:
                    continue
                proposed=render(trial)
                cost=await model.count(proposed)
                if cost<=8192:
                    tokens,intervals,msgs=cost,trial,proposed
            assert r.extend(intervals,[s for v in old.values() for s in v])==intervals
            if case=='FG01':
                assert msgs==r.read(BF/'BF18.json')['messages']
            r.save(OUT/f'{case}.json',dict(original_sha256=run['input_sha256'],old_intervals=old,intervals=intervals,
                   old_tokens=run['input_tokens'], compact_tokens=compact_tokens,final_tokens=tokens,messages=msgs))
            print(case,run['input_tokens'],tokens,flush=True)

if __name__=='__main__': asyncio.run(main())
