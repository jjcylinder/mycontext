"""Join anonymous answer reviews to the packing labels only after scoring."""
import csv
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
def read_csv(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))
def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
def load(path):
    return json.loads(path.read_text())
reviews={}
for line in (OUT/'blind-scores.txt').read_text().splitlines():
    if line and not line.startswith('#'):
        tag,*grades=line.split()
        assert tag not in reviews
        reviews[tag]=grades
key=load(OUT/'key.json')
assert set(reviews)=={f'{case}-{label}' for case in key for label in key[case]}
atoms=read_csv(OUT.parent/'eval-v5-bf-all20/atom-review.csv')
inputs={r['bf_id']:r for r in read_csv(OUT.parent/'eval-v5-bf20-f-packing-offline/summary.csv')}
fields=dict(p='progress',c='changes',o='open_issues',t='coordination',u='uncertainties')
items=[]; sums=[]
for case in key:
    criteria=[r for r in atoms if r['mode']=='F' and r['bf_id']==case]
    for label,version in sorted(key[case].items(),key=lambda pair:pair[1]=='new'):
        tag=f'{case}-{label}'
        delivery=load(OUT/(tag+'.json'))
        points={}; total=0
        for r,g in zip(criteria,reviews[tag],strict=True):
            ptr=[]
            if g!='-':
                for part in g.split('+'):
                    field=fields[part[0]]; index=int(part[1:])
                    assert delivery['answer'][field][index]
                    ptr.append(f'/answer/{field}/{index}')
            hit=bool(ptr)
            points.setdefault(r['point'],[]).append(hit)
            total+=hit
            items.append(dict(case=case,version=version,label=label,point=r['point'],atom=r['atom'],criterion=r['criterion'],covered=int(hit),pointers=' '.join(ptr)))
        complete=sum(map(all,points.values()))
        s=inputs[case]
        sums.append(dict(case=case,version=version,label=label,input_atoms=s[version+'_atoms'],answer_atoms=total,atoms=len(criteria),input_points=s[version+'_points'],answer_points=complete,points=len(points),full=int(complete==len(points)),attempts=len(delivery['attempts'])))
write_csv('atom-review.csv',items)
write_csv('summary.csv',sums)
