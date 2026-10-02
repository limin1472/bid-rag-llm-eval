#!/usr/bin/env python3
"""
판정 기록 + 규칙 기반 채점 + 런타임 측정 → 모델별 최종 지표

  python scripts/compute_metrics.py \
      --judgments data/judgments/judgments_100.jsonl \
      --auto-dir  data/auto_scoring \
      --runtime   data/runtime \
      --out       results/final_metrics.json

정의
  정답 / 환각 / 무응답 : 판정 j 가 'c' / {EC,SC,EB,SB} / 'F:*'. 세 비율의 합 = 100%.
  환각 4유형            : EC(Evident Conflict) SC(Subtle Conflict) EB(Evident Baseless) SB(Subtle Baseless)
  무응답 4유형          : 근거반복 · 절차서술 · 조각(파편) · 질문반복
  정답 안정도           : 지시문 5종 정답률의 평균 − 표준편차
  모든 비율은 지시문 5종(P1~P5) 각각 계산 후 평균 ± 표본표준편차.
"""
import argparse, collections, json, pathlib, statistics

PROMPTS = ["P1", "P2", "P3", "P4", "P5"]
HALLU = ["EC", "SC", "EB", "SB"]
NOANS = ["근거반복", "절차서술", "조각", "질문반복"]


def cat(j):
    if j == "c":
        return "정답"
    return "무응답" if j.startswith("F") else "환각"


def mean_sd(vals):
    return [round(statistics.mean(vals), 1), round(statistics.stdev(vals), 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--auto-dir", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.judgments, encoding="utf-8") if l.strip()]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["model"]][r["prompt"]].append(r["j"])
    nq = len({r["question_id"] for r in rows})

    auto = {p: json.load(open(pathlib.Path(a.auto_dir) / p / "auto_summary.json", encoding="utf-8")) for p in PROMPTS}
    speed = json.load(open(pathlib.Path(a.runtime) / "speed_workstation.json", encoding="utf-8"))["models"]
    mem = json.load(open(pathlib.Path(a.runtime) / "memory_ops_profile.json", encoding="utf-8"))["peak_rss_gb"]

    out = {}
    for m, per in by.items():
        pct = lambda pred: [100 * sum(pred(j) for j in per[p]) / nq for p in PROMPTS]
        r = {c: mean_sd(pct(lambda j, c=c: cat(j) == c)) for c in ["정답", "환각", "무응답"]}
        r["환각유형"] = {t: round(statistics.mean(pct(lambda j, t=t: j == t)), 1) for t in HALLU}
        r["무응답유형"] = {t: round(statistics.mean(pct(lambda j, t=t: j == f"F:{t}")), 1) for t in NOANS}
        r["정답안정도"] = round(r["정답"][0] - r["정답"][1], 1)
        am = auto[PROMPTS[0]].get(m) and m or m.replace("Mi:dm-2.0-Mini", "Midm-2.0-Mini")
        for key, src in [("형식준수", "format_ok_pct"), ("원문복사", "verbatim_copy_pct")]:
            vals = [auto[p][am][src] for p in PROMPTS if am in auto[p] and src in auto[p][am]]
            if vals:
                r[key] = mean_sd(vals)
        sp = speed.get(m, {})
        r["응답p50"] = round(sp["total_p50"], 1) if "total_p50" in sp else None
        r["응답p95"] = round(sp["total_p95"], 1) if "total_p95" in sp else None
        r["RAM"] = mem.get(m)
        r["n_questions"] = nq
        out[m] = r

    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"저장: {a.out}  ({len(out)}개 모델, {nq}문항)")
    print("※ 인용 정확도·수치 재현은 scripts/score_auto.py 로 data/responses/ 에서 계산한다.")


if __name__ == "__main__":
    main()
