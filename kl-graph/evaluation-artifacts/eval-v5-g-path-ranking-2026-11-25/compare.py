"""Replay the two frozen G snapshots. Run prepare before answer."""
import argparse
import asyncio
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT/'data/project_demo/tokenizers'))
from project_demo.answer import AnswerModel, _terms, parse_answer
from project_demo.codex_bridge import ChatRequest, CodexBridge
from project_demo.dataset import load_dataset
from project_demo.evidence_packing import Hop, PathEvidence, Snapshot, Span, pack
from project_demo.evidence_retrieval import (
    SEED_LIMIT, _fact_scopes, direct_excerpt, graph_paths, path_text, visible_records,
)
from project_demo.evaluate import load_rubric_snapshot

OLD = OUT.parent/'eval-v5-bfg-default5-2026-11-25'
DATA = ROOT/'data/project_demo/v5-natural-runtime'
MODEL = AnswerModel('', 'chatgpt-codex-default', '', 32768, 8192, 1800, '', 'o200k_base')


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    path = OUT/name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def as_path(row):
    return PathEvidence(tuple(Hop(a,b,c,tuple(w)) for a,b,c,w in row['hops']),
                        tuple(Span(*s) for s in row['spans']))


def tied_ranks(values):
    """Competition ranks. Equal signal values receive the same vote."""
    counts = Counter(values.values())
    rank, table = 1, {}
    for value in sorted(counts, reverse=True):
        table[value] = rank
        rank += counts[value]
    return {p:table[v] for p,v in values.items()}


def light_ranking(question, paths, records, originals, facts):
    query = sorted(set(_terms(question)))
    # Use each distinct rendered target once for the lexical corpus.
    quotes = {p:'\n'.join(records[s.source]['content'][s.start:s.end]
                          for s in p.spans if s.source == p.target) for p in paths}
    freqs = {quote:Counter(_terms(quote)) for quote in sorted(set(quotes.values()))}
    df = Counter(word for freq in freqs.values() for word in query if word in freq)
    size = len(freqs)
    avg = sum(f.total() for f in freqs.values())/max(1,size)
    bm25 = {}
    for quote,f in freqs.items():
        length = 1.2*(0.25+0.75*f.total()/max(1,avg))
        bm25[quote] = sum(math.log1p((size-df[t]+0.5)/(df[t]+0.5)) * f[t]*2.2/(f[t]+length)
                          for t in query if t in f)
    origin_rank = {}
    for lane in (originals, facts):
        for rank,key in enumerate(lane,1):
            origin_rank[key] = min(rank,origin_rank.get(key,rank))
    def strength(h):
        if h.relation in ('same_source','same_thread'):
            return 1
        if h.relation == 'shared_ABOUT':
            return 2
        if h.relation in ('attachment','referenced_by_attachment','reply_to','has_reply'):
            return 3
        raise ValueError(h.relation)
    signals = [
        {p:-origin_rank.get(p.hops[0].origin,10**8) for p in paths},
        {p:(min(map(strength,p.hops)), -len(p.hops)) for p in paths},
        {p:bm25[quotes[p]] for p in paths},
    ]
    ranks = [tied_ranks(s) for s in signals]
    def order(p):
        score = sum(Fraction(1,20+r[p]) for r in ranks)
        return -score,len(p.hops),p.hops,p.spans
    best = {}
    for p in sorted(paths,key=order):
        best.setdefault(p.target,p)
    return best


async def prepare():
    manifest = read(OLD/'manifest.json')
    for f,h in manifest['code_sha256'].items():
        assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h, f
    dataset_path = next((ROOT.parent/'datasets').glob('*v5*'))
    assert load_rubric_snapshot(dataset_path)[0] == manifest['grading_rubric']
    ds = load_dataset(dataset_path,'app_focus')
    assert all(v['is_simulated'] is True for v in ds.records.values())
    for q,rid in manifest['runs'].items():
        if q not in ('TR04','AM03'):continue
        run = read(DATA/'runs'/(rid+'.json'))
        assert ds.digest == run['dataset_digest'] and run['answer_model'] == MODEL.model
        project,cutoff,question = run['project_id'],run['cutoff'],run['question']
        rec = visible_records(ds,project,cutoff)
        orig,facts,pvec = (run['snapshot'][k] for k in ('originals','facts','paths'))
        spans = lambda lane:{key:tuple(Span(*s) for s in v) for key,v in lane.items()}
        vec = Snapshot(spans(orig),spans(facts),{k:as_path(v) for k,v in pvec.items()})
        old = run['groups']['G']
        rebuilt = await pack(MODEL,'G',question,project,cutoff,rec,vec)
        assert rebuilt['input_sha256'] == old['input_sha256'] and rebuilt['messages']==old['messages']
        meta = run['retrieval']
        start = perf_counter()
        terms = set(_terms(question))
        excerpts = {k:direct_excerpt(terms,k,rec) for k in rec}
        unavail = {v['conversation_id'] for k,v in ds.records.items() if k not in rec and v.get('conversation_id')}
        scopes = _fact_scopes(rec,project,unavail)
        seeds = list(dict.fromkeys([*meta['O'][:SEED_LIMIT],*meta['T'][:SEED_LIMIT]]))
        db = DATA/'datasets'/ds.id/run['index']/'knowledge.db'
        with closing(sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            paths = graph_paths(conn,rec,scopes,meta['entity_hits'],seeds,excerpts)
        found = perf_counter()
        assert not meta['oversized_paths'] and set(vec.paths.values()) <= set(paths)
        light = light_ranking(question,paths,rec,orig,facts)
        ranked = perf_counter()
        assert light.keys()==vec.paths.keys()
        new = await pack(MODEL,'G',question,project,cutoff,rec,Snapshot(vec.originals,vec.facts,light))
        done = perf_counter()
        audit = dict(question=question,run_id=rid,dataset=ds.id,index=run['index'],
                     rule_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     rubric=manifest['grading_rubric'],candidates=len(paths),unique_targets=len(light),
                     path_texts=len({path_text(p,rec) for p in paths}),candidate_paths=[asdict(p) for p in paths],
                     old_ranking=list(vec.paths),new_ranking=list(light),
                     seconds=dict(reconstruct=found-start,light_rank=ranked-found,light_pack=done-ranked))
        save(q+'-audit.json',audit)
        save(q+'-vector-input.json',rebuilt)
        save(q+'-light-input.json',new)
        if 'answer' in old:
            save(q+'-vector-answer.json',dict(answer=old['answer'],usage=old['usage'],source_run=rid,
                                            input_sha256=old['input_sha256']))
        print(q,len(paths),audit['seconds'],new['input_tokens'],flush=True)


async def answer(q,variant):
    name = q+'-'+variant
    if (OUT/(name+'-answer.json')).exists():
        raise ValueError('Existing answer, skipping second generation')
    group = read(OUT/(name+'-input.json'))
    bridge = CodexBridge()
    bridge.verify_login()
    assert bridge.model_name == MODEL.model and await MODEL.count(group['messages'])==group['input_tokens']
    payload = ChatRequest(model=MODEL.model,messages=group['messages'],temperature=0,max_tokens=MODEL.max_output)
    start = perf_counter()
    raw,usage = await bridge._run(payload)
    result = dict(answer=parse_answer(raw,set(group['source_ids'])),raw_answer=raw,usage=usage,
                  seconds=perf_counter()-start,input_sha256=group['input_sha256'],model=MODEL.model)
    save(name+'-answer.json',result)
    print(name,'completed',round(result['seconds'],2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','answer'])
    parser.add_argument('case',nargs='?',choices=['TR04','AM03'])
    parser.add_argument('variant',nargs='?',choices=['light','vector'])
    args=parser.parse_args()
    asyncio.run(prepare() if args.action=='prepare' else answer(args.case,args.variant))
