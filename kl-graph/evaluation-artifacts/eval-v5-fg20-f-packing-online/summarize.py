"""Assemble frozen atom-level answer reviews; keep coverage and delivery separate."""
import csv
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
PREP=OUT.parent/'eval-v5-fg20-f-packing-offline'

def csv_rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def save_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

old={}
for batch in ('main10','next10'):
    for row in csv_rows(OUT.parent/('eval-v5-fg-'+batch)/'summary.csv'):
        if row['方案']=='F':
            old[row['题号']]=row
inputs={r['case']:r for r in csv_rows(PREP/'summary.csv')}
rubric=csv_rows(PREP/'atomic-review.csv')
reviews={}
for line in (OUT/'review.txt').read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        case,*refs=line.split()
        assert case not in reviews
        reviews[case]=refs
assert reviews.keys()==old.keys()==inputs.keys()
fields=dict(p='progress',c='changes',o='open_issues',t='coordination',u='uncertainties')
details=[];summary=[]
for case in sorted(old):
    data=json.loads((OUT/(case+'-answer.json')).read_text())
    answers=data.get('answer') or {}
    atom_scores=[]
    points={}
    for r,ref in zip([r for r in rubric if r['case']==case],reviews[case],strict=True):
        pointers=[]
        if ref!='-':
            for loc in ref.split('+'):
                name=fields[loc[0]]
                index=int(loc[1:])
                assert answers[name][index]
                pointers.append(f'/answer/{name}/{index}')
        hit=int(bool(pointers)); atom_scores.append(hit)
        points.setdefault(r['atom'].split('.')[0],[]).append(hit)
        details.append(dict(case=case,atom=r['atom'],criterion=r['criterion'],input_support=r['new'],answer_coverage=hit,pointers=' '.join(pointers)))
    s=inputs[case]
    complete=sum(all(x) for x in points.values())
    summary.append(dict(case=case,atoms=int(s['atoms']),points=len(points),input_atoms=int(s['new_atoms']),answer_atoms=sum(atom_scores),input_points=int(s['new_points']),answer_points=complete,input_full=int(s['new_all']),answer_full=int(complete==len(points)),old_answer_atoms=int(old[case]['答卷原子项']),old_answer_points=int(old[case]['答卷完整要点']),old_answer_full=int(old[case]['整题答全']),attempts=len(data['attempts'])))
save_csv('atomic-review.csv',details)
save_csv('summary.csv',summary)
total={k:sum(r[k] for r in summary) for k in summary[0] if k!='case'}
print(total)
table=['| 题号 | 优化版输入原子项 | 优化版答卷原子项 | 优化版输入完整要点 | 优化版答卷完整要点 | 原版答卷原子项 | 原版答卷完整要点 |','|---|---:|---:|---:|---:|---:|---:|']
for r in summary+[dict(case='合计',**total)]:
    row=dict(r)
    if r['case']!='合计':
        row['case']=f"[{r['case']}]({r['case']}-answer.json)"
    table.append('| {case} | {input_atoms}/{atoms} | {answer_atoms}/{atoms} | {input_points}/{points} | {answer_points}/{points} | {old_answer_atoms}/{atoms} | {old_answer_points}/{points} |'.format(**row))
(OUT/'report.md').write_text('# 优化版 F 在 FG01–FG20 上的线上验证\n\n'+(OUT/'findings.txt').read_text()+'\n## 逐题结果\n\n'+'\n'.join(table)+'\n\n整题答全要求该题所有完整业务要点均得分。点击题号可查看答卷。\n\n[逐原子项评分和答卷位置](atomic-review.csv) · [逐题计数](summary.csv) · [配置和输入指纹](manifest.json) · [输入离线核查](../eval-v5-fg20-f-packing-offline/report.md)\n')
