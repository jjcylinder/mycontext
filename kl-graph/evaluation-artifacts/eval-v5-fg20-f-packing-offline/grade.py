"""Input review under the original FG atomic rubrics."""
import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
def read_csv(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))
def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

# Inspection of every original F omission, after packing was complete.
review = {
    ('FG01','1.2'): ('XSH2-07', '消息已可检索，但旧快照未重跑', '晨报任务与快照另由 AAM2-S02、XAM2-08 对应。'),
    ('FG06','2.1'): ('', '', 'XPO1-06 仅说 Skill 已进入，未指明实际入口；APO1-S03 未入选。'),
    ('FG18','1.1'): ('', '', 'XTR3-01 提及翻译线，输入没有交代法语文档翻阿语。'),
    ('FG18','1.2'): ('D07-S02','邮电部阿语公文写作','登记明写第二类场景。'),
}
details=[]; summary=[]; seen=set()
for batch in ('main10','next10'):
    base=OUT.parent/('eval-v5-fg-'+batch)
    rubrics=read_csv(base/'atomic-review.csv')
    for row in read_csv(base/'summary.csv'):
        if row['方案']!='F':
            continue
        case=row['题号']
        data=json.loads((OUT/f'{case}.json').read_text())
        payload=json.loads(data['messages'][1]['content'])
        recs=[dict(zip(payload['source_excerpt_columns'],r)) for r in payload['source_excerpts']]
        text={r['id']:r['content'] for r in recs}
        points=defaultdict(list)
        for a in rubrics:
            if a['题号']!=case or a['方案']!='F':
                continue
            atom=a['原子项']
            old=int(a['输入覆盖']); new=old; src=a['输入依据']; note=''
            if not old:
                seen.add((case,atom))
                src,quote,note=review[case,atom]
                new=int(bool(src))
                if src:
                    assert quote in text[src]
                if case=='FG06':
                    assert 'APO1-S03' not in text and 'Skill也确实进入' in text['XPO1-06']
            points[atom.split('.')[0]].append((old,new))
            details.append(dict(case=case,atom=atom,criterion=a['评分口径'],old=old,new=new,sources=src,note=note))
        counts=[sum(v[j] for pt in points.values() for v in pt) for j in range(2)]
        full=[sum(all(v[j] for v in pt) for pt in points.values()) for j in range(2)]
        assert counts[0]==int(row['输入原子项']) and full[0]==int(row['输入完整要点'])
        summary.append(dict(case=case,atoms=int(row['原子项满分']),old_atoms=counts[0],new_atoms=counts[1],points=len(points),old_points=full[0],new_points=full[1],old_all=int(full[0]==len(points)),new_all=int(full[1]==len(points))))
assert seen==set(review) and len(summary)==20
write_csv(OUT/'atomic-review.csv',details)
write_csv(OUT/'summary.csv',summary)
total={k:sum(r[k] for r in summary) for k in summary[0] if k!='case'}
print(total)
report=['# FG01–FG20 优化版 F 的最终输入覆盖','',
'沿用 V5 两批 FG 题库与原子评分标准，逐题核对原版 F 的输入指纹及原文榜、事实榜。在相同的 8,192 token 上限内保留全部原版原文区间，合并重复字段名，按原候选顺序补装。以下覆盖指模型作答前收到的材料。','',
'| 题号 | 原版原子项 | 优化版原子项 | 原版完整要点 | 优化版完整要点 |','|---|---:|---:|---:|---:|']
for r in [*summary,dict(case='合计',**total)]:
    report.append('| {case} | {old_atoms}/{atoms} | {new_atoms}/{atoms} | {old_points}/{points} | {new_points}/{points} |'.format(**r))
report+=['',f"输入足以支撑整题答全，由 {total['old_all']}/20 提升至 {total['new_all']}/20。一个要点的所有原子项均有证据才计为完整，一题的所有要点完整才计为整题有充分依据。",'',
'FG01 补入 XSH2-07，明确晨报来源后来可检索，该题补齐最后一项。它复用了 BF18 的问句，本次已比对两份优化版输入完全一致。FG18 补入 D07-S02，支持“邮电部阿语公文写作”，同一要点仍缺“法语文档翻阿语”。FG06 新装入 XPO1-06，群回复仅称进入了某个 Skill；能够确认实际润色入口的调用记录 APO1-S03 仍未装入。其余 17 题原输入已足以支撑全题。','',
'本次只做输入核查，无线上作答或 G 重跑。[逐题统计](summary.csv) · [逐原子项审核](atomic-review.csv) · [离线回放](verify.py)']
(OUT/'report.md').write_text('\n'.join(report)+'\n')
