#!/usr/bin/env python3
"""1단계 자동 규칙 채점 — 의미 판정 없이 계산 가능한 지표만."""
import json, re, glob, sys, collections, statistics, pathlib

Q_PATH, FROZEN, RESULTS, OUT = sys.argv[1:5]
qs = {json.loads(l)['question_id']: json.loads(l) for l in open(Q_PATH, encoding='utf-8') if l.strip()}
fr = {json.loads(l)['question_id']: json.loads(l) for l in open(FROZEN, encoding='utf-8') if l.strip()}

REFUSE = re.compile(r'권한이 없|조회할 수 없|제공할 수 없|답할 수 없|답변할 수 없|확인할 수 없|근거가 부족|자료가 부족|명시되어 있지 않|근거에 없')
NUM = re.compile(r'\d{4}-\d{2}-\d{2}|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?')  # 날짜는 한 덩어리
PURE_CITE = re.compile(r'^[\s\[\]\d,、및·/]+$')

def clean(a):
    a = re.sub(r'<think>.*?</think>', '', a or '', flags=re.S)
    a = a.replace('<|eot_id|>', '').replace('<end_of_turn>', '')
    return a.strip()

def evidence_blocks(user):
    body = user.split('[질문]')[0]
    parts = re.split(r'\n\[(\d+)\] 출처: [^\n]*\n', '\n' + body)
    return {int(parts[i]): parts[i+1] for i in range(1, len(parts), 2)}

def score_row(r):
    q = qs[r['question_id']]; f = fr[r['question_id']]
    a = clean(r.get('answer'))
    ev = evidence_blocks(f['user']); n_ev = len(ev)
    out = {'question_id': r['question_id'], 'status': r.get('status'),
           'finish': r.get('finish_reason'), 'chars': len(a),
           'sentences': len([s for s in re.split(r'(?<=[.!?다])\s+|\n', a) if len(s.strip()) > 5])}
    if r.get('status') != 'ok':
        return out
    out['truncated'] = r.get('finish_reason') == 'length'
    # 형식
    hits = re.findall(r'근거:\s*([^\n]*)', a)
    out['has_answer_tag'] = '답변:' in a
    out['has_ground_line'] = bool(hits)
    out['format_ok'] = out['has_answer_tag'] and out['has_ground_line']
    # 인용 — 대괄호 번호 + 숫자만 나열된 마지막 '근거:' 줄만 인용으로 인정. 번호가 없으면 0
    cited = {int(x) for x in re.findall(r'\[(\d+)\]', a)}
    if hits and PURE_CITE.match(hits[-1].strip() or 'x'):
        cited |= {int(x) for x in re.findall(r'\d+', hits[-1])}
    valid = {c for c in cited if 1 <= c <= n_ev}
    out['n_cited'] = len(cited); out['n_invalid'] = len(cited - valid)
    out['cite_valid_rate'] = (len(valid) / len(cited)) if cited else 0.0
    out['has_valid_cite'] = bool(valid)
    out['says_none'] = bool(hits) and '없음' in hits[-1]
    # 원문 복사: 답변의 60자 창이 근거에 그대로 있는지
    joined = ''.join(ev.values()).replace(' ', '')
    aa = a.replace(' ', '')
    out['verbatim_copy'] = any(aa[i:i+60] in joined for i in range(0, max(1, len(aa)-60), 20)) if len(aa) >= 60 else False
    # 표면 기대동작
    act = q['expected_action']
    refusing = bool(REFUSE.search(a)) or out['says_none']
    out['expected_action'] = act
    out['surface_refuse'] = refusing
    if act == '답변': out['surface_match'] = not refusing
    elif act == '자료부족 안내': out['surface_match'] = refusing
    else: out['surface_match'] = None   # 조건부는 의미 판정 필요
    # 기대 수치 재현
    exp_nums = {x.replace(',', '') for x in NUM.findall(q['expected_answer']) if len(x.replace(',', '')) >= 2}
    if exp_nums:
        got = {x.replace(',', '') for x in NUM.findall(a)}
        out['num_recall'] = len(exp_nums & got) / len(exp_nums)
    else:
        out['num_recall'] = None
    # 속도 (GPU 병렬, 참고)
    out['sec'] = r.get('total_seconds'); out['ctok'] = r.get('completion_tokens')
    out['chars_per_sec'] = (len(a) / r['total_seconds']) if r.get('total_seconds') else None
    return out

def pct(xs): 
    xs = [x for x in xs if x is not None]
    return (100 * sum(xs) / len(xs)) if xs else None
def med(xs):
    xs = [x for x in xs if x is not None]; return statistics.median(xs) if xs else None
def p95(xs):
    xs = sorted(x for x in xs if x is not None); return xs[int(.95*len(xs))] if xs else None

summary, detail = {}, {}
for f in sorted(glob.glob(f'{RESULTS}/*/answers.jsonl')):
    m = pathlib.Path(f).parent.name
    rows = [score_row(json.loads(l)) for l in open(f, encoding='utf-8') if l.strip()]
    ok = [r for r in rows if r['status'] == 'ok']
    detail[m] = rows
    summary[m] = {
        'n': len(rows), 'n_ok': len(ok),
        'truncated_pct': pct([r['truncated'] for r in ok]),
        'format_ok_pct': pct([r['format_ok'] for r in ok]),
        'cite_valid_pct': pct([r['cite_valid_rate'] for r in ok]),
        'has_valid_cite_pct': pct([r['has_valid_cite'] for r in ok]),
        'invalid_cites_total': sum(r['n_invalid'] for r in ok),
        'verbatim_copy_pct': pct([r['verbatim_copy'] for r in ok]),
        'says_none_pct': pct([r['says_none'] for r in ok]),
        'surface_match_answer_pct': pct([r['surface_match'] for r in ok if r['expected_action']=='답변']),
        'surface_match_insufficient_pct': pct([r['surface_match'] for r in ok if r['expected_action']=='자료부족 안내']),
        'num_recall_pct': pct([r['num_recall'] for r in ok]),
        'n_num_questions': sum(1 for r in ok if r['num_recall'] is not None),
        'chars_median': med([r['chars'] for r in ok]),
        'sentences_median': med([r['sentences'] for r in ok]),
        'sec_p50': med([r['sec'] for r in ok]), 'sec_p95': p95([r['sec'] for r in ok]),
        'chars_per_sec_median': med([r['chars_per_sec'] for r in ok]),
    }
json.dump(summary, open(f'{OUT}/auto_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(f'{OUT}/auto_detail.jsonl', 'w', encoding='utf-8') as fh:
    for m, rows in detail.items():
        for r in rows: fh.write(json.dumps({'model': m, **r}, ensure_ascii=False) + '\n')

cols = [('format_ok_pct','형식준수%'),('cite_valid_pct','인용유효%'),('has_valid_cite_pct','유효인용有%'),
        ('invalid_cites_total','무효인용수'),('verbatim_copy_pct','원문복사%'),('says_none_pct','근거없음%'),
        ('surface_match_answer_pct','답변문항비거절%'),('surface_match_insufficient_pct','부족문항안내%'),
        ('num_recall_pct','기대수치재현%'),('truncated_pct','잘림%'),('chars_median','글자p50')]
print(f"{'모델':16}" + ''.join(f"{c[1]:>12}" for c in cols))
for m in sorted(summary, key=lambda k: -(summary[k]['format_ok_pct'] or 0)):
    s = summary[m]
    print(f"{m:16}" + ''.join(f"{(('%.0f' % s[k]) if isinstance(s[k], float) else str(s[k])) if s[k] is not None else 'N/A':>12}" for k,_ in cols))
