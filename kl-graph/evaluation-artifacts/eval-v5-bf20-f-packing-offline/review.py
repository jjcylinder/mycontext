"""Grade fixed inputs after replay. Witnesses are used only for offline review."""
from collections import defaultdict
from replay import (OUT, ROOT, BASE, AnswerModel, Span, asyncio, csv, json, extend,
                    read, save, render, project_scope, evidence_view, load_dataset)

# New evidence actually packed by the experimental rule. Existing positives retain their cited text.
GAINS = {
    ('BF13', '1.1'): 'NTR-S02',
    ('BF13', '7.3'): 'A01-S02 XSH2-02',
    ('BF18', '1.2'): 'XSH2-07 AAM2-S02 XAM2-08',
}
# Additional support found by inspecting excerpts in the frozen candidate pool.
POOL_GAINS = {
    ('BF13', '3.1'): 'NAM-S03', ('BF13', '3.2'): 'NAM-S03', ('BF13', '3.4'): 'NAM-S03',
    ('BF13', '4.1'): 'XLE2-02', ('BF13', '4.4'): 'XLE2-07 XLE2-08',
    ('BF13', '6.3'): 'XPO1-02', ('BF13', '6.4'): 'NPO-S03', ('BF13', '7.4'): 'XTR3-07',
    ('BF14', '2.2'): 'ALE0-S04',
}

def read_csv(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def write_csv(name, rows):
    with (OUT / name).open('w',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def positions(spans):
    return {(key,i) for key,start,end in spans for i in range(start,end)}

async def diagnostic(k, q, ds, rows, model, a, demand):
    pool=a['pool']
    sets=[positions(c['spans']) for c in pool]
    missing=set(demand)
    chosen=[]
    while missing:
        gains=[len(s & missing)/(90+len(s)) for s in sets]
        best=max(range(len(pool)),key=lambda i:gains[i])
        assert gains[best]>0
        chosen.append(best)
        missing-=sets[best]
    for i in list(reversed(chosen)):
        others=[sets[j] for j in chosen if j!=i]
        if demand.issubset(set().union(*others)):
            chosen.remove(i)
    pack={}
    for i in sorted(chosen):
        pack=extend(pack,[Span(*s) for s in pool[i]['spans']])
    msgs=render(q,ds.cutoff,rows,pack,columns=False)
    size=await model.count(msgs)
    assert size<=8192 and demand.issubset(positions(s for v in pack.values() for s in v))
    save(OUT/f'{k}-diagnostic.json',dict(tokens=size,indices_zero_based=sorted(chosen),intervals=pack,messages=msgs))
    return size

async def main():
    folder=next((ROOT.parent/'datasets').glob('*v5*'))
    ds=load_dataset(folder,'app_focus')
    rows={k:dict(evidence_view(r),project_scope=project_scope(r,'项目A')) for k,r in ds.records.items()}
    meta=read(BASE/'manifest.json')
    model=AnswerModel('',meta['model'],'',32768,8192,1800,'','o200k_base')
    baseline=[r for r in read_csv(BASE/'summary.csv') if r['mode']=='F']
    atoms=[r for r in read_csv(BASE/'atom-review.csv') if r['mode']=='F']
    summary=[]; ledger=[]
    for r in baseline:
        k=r['bf_id']; a=read(OUT/f'{k}.json')
        q=json.loads(a['messages'][1]['content'])['question']
        union={}
        for c in a['pool']:
            union=extend(union,[Span(*s) for s in c['spans']])
        points=defaultdict(list); demand=set()
        for b in (b for b in atoms if b['bf_id']==k):
            key=(k,b['atom']); old=int(b['input_support'])
            assert not (old and (key in GAINS or key in POOL_GAINS))
            actual=int(bool(old or key in GAINS)); pool=int(bool(actual or key in POOL_GAINS))
            refs=GAINS.get(key,POOL_GAINS.get(key,b['source_ids'] if old else ''))
            lookup=a['old_intervals'] if old else a['intervals'] if actual else union
            for ref in refs.split():
                demand.update(positions(lookup[ref]))
            points[b['point']].append((old,actual,pool))
            ledger.append(dict(bf_id=k,atom=b['atom'],criterion=b['criterion'],baseline=old,packed=actual,pool_supported=pool,source_ids=refs))
        atom_counts=[sum(v[j] for pt in points.values() for v in pt) for j in range(3)]
        full_counts=[sum(all(v[j] for v in pt) for pt in points.values()) for j in range(3)]
        assert atom_counts[0]==int(r['input_atoms']) and full_counts[0]==int(r['input_complete'])
        cost=await diagnostic(k,q,ds,rows,model,a,demand) if r['answer_pass']=='0' else ''
        summary.append(dict(bf_id=k,question=r['question'],atoms=r['atoms'],old_atoms=atom_counts[0],new_atoms=atom_counts[1],points=r['points'],old_points=full_counts[0],new_points=full_counts[1],old_sources=len(a['old_intervals']),new_sources=len(a['intervals']),old_tokens=a['old_tokens'],compact_tokens=a['compact_tokens'],new_tokens=a['final_tokens'],diagnostic_atoms=atom_counts[2] if cost else '',diagnostic_points=full_counts[2] if cost else '',diagnostic_tokens=cost))
    write_csv('summary.csv',summary)
    write_csv('atom-review.csv',ledger)
if __name__=='__main__': asyncio.run(main())
