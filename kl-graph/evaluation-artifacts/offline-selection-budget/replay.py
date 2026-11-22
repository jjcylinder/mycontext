"""Replay archived successful selections without network access or provider calls."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
DATA = OUT.parent
TRACES = DATA / 'v4-deepseek-runtime' / 'model_calls'
CAPS = [4096, 8192, 16384, 24576, 32768, 65536]
WATCH = {'TR04': ['D08-S02', 'ATR0-S04', 'ATR2-S03'], 'AM03': ['AAM1-S04'],
         'PO03': ['APO0-S02', 'APO1-S04', 'APO2-S02'], 'X01': ['AAM1-S04'], 'X03': ['AWR1-S04']}


def digest(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def main():
    traces = {}
    profiles = {}
    for path in sorted(TRACES.glob('*.json')):
        entry = json.loads(path.read_text())
        request = entry['request']
        if not request['system'].startswith('你负责为问题挑选'):
            continue
        messages = [{'role': 'system', 'content': request['system']}, *request['messages']]
        key = digest(messages)
        profile = (request['thinking']['type'], request.get('output_config', {}).get('effort'))
        profiles[profile] = profiles.get(profile, 0) + 1
        traces.setdefault(key, []).append((path.name, request, entry['response']))

    results = []
    for cohort, dirname in [('历史五题', 'eval-v4-deepseek-high-first-bc-2026-11-21'),
                            ('当前首题', 'eval-v4-action-scope-bc')]:
        for line in (DATA / dirname / 'runs.jsonl').read_text().splitlines():
            row = json.loads(line)
            run, qid = row['run'], row['question_id']
            if run['status'] != 'completed':
                continue
            sel = run['coverage']['evidence_selection']
            calls, candidates = [], []
            for call in sel['calls']:
                usage = call['usage'] or call['cached_usage']
                matches = [(name, req, resp) for name, req, resp in traces[call['request_hash']]
                           if resp['usage'] == usage and req['output_config']['effort'] == 'high']
                assert matches, (qid, call['request_hash'])
                name, req, resp = matches[0]
                assert resp['choices'][0]['finish_reason'] == 'stop'
                raw = resp['choices'][0]['message']['content'].strip()
                if raw.startswith('```'):
                    raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
                groups = json.loads(raw)['groups']
                allowed = set(call['candidate_ids'])
                assert all(set(g['ids']) <= allowed for g in groups)
                if call['stage'] == 'candidate_ranking':
                    candidates.extend(call['candidate_ids'])
                out = usage['completion_tokens']
                thinking = usage['provider_usage']['output_tokens_details']['thinking_tokens']
                calls.append({'stage': call['stage'], 'request_hash': call['request_hash'],
                              'trace': name, 'cache_hit': call['cache_hit'], 'total': out,
                              'thinking': thinking, 'nonthinking': out-thinking,
                              'returned_ids': sorted({k for g in groups for k in g['ids']})})
            snapshot = json.loads((DATA/'v4-deepseek-runtime'/'runs'/(run['run_id']+'.json')).read_text())
            excerpts = json.loads(snapshot['model_input'][1]['content'])['source_excerpts']
            ids = {r['id'] for r in excerpts}
            assert ids == {r['id'] for r in run['result']['evidence']}
            assert ids <= set(candidates)
            watch = {k: {'candidate': k in candidates,
                         'stages_returning_id': [c['stage'] for c in calls if k in c['returned_ids']],
                         'final': k in ids,
                         'content': next((r['content'] for r in excerpts if r['id']==k), None)}
                     for k in WATCH[qid]}
            results.append({'cohort': cohort, 'qid': qid, 'mode': row['mode'], 'run': run['run_id'],
                            'candidate_digest': digest(candidates), 'candidates': len(set(candidates)),
                            'peak_total': max(c['total'] for c in calls),
                            'peak_thinking': max(c['thinking'] for c in calls),
                            'peak_nonthinking': max(c['nonthinking'] for c in calls),
                            'fits': {cap: all(c['total'] <= cap for c in calls) for cap in CAPS},
                            'calls': calls, 'watch': watch})
    (OUT/'verified.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    print('profiles', profiles)
    for x in results:
        print(x['cohort'], x['qid'], x['mode'], x['peak_total'], x['peak_thinking'], x['peak_nonthinking'],
              {k:v['final'] for k,v in x['watch'].items()})
    for cap in CAPS:
        print(cap, [(cohort, mode, sum(x['fits'][cap] for x in results if (x['cohort'],x['mode']) == (cohort,mode)))
                    for cohort in ('历史五题','当前首题') for mode in ('B','C')])


if __name__ == '__main__':
    main()
