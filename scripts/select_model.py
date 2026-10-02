#!/usr/bin/env python3
"""
모델 선정: 관문 → 지표별 순위(결론 근거) → 가중 점수(참고) → 민감도

  python scripts/select_model.py --metrics results/final_metrics.json --out results/selection.json

1) 관문     : 운영 설정 메모리 ≤ --ram-gate (기본 10GB)
2) 지표별 순위: 정답 안정도↑ · 환각률↓ · 인용 정확도↑ · 응답 시간↓ 각각 줄 세움
             평가 지표 관리 규정("임의 가중 합산 점수로 순위를 만들지 않는다")에 따라 결론은 여기서 내린다.
3) 가중 점수 : 참고용. 통과 모델 중 최고 1·최저 0으로 환산 후 3·3·1·1 가중, ÷8×100
4) 민감도    : 네 가중치를 0.5~3.0 무작위로 3,000회 재추출해 각 모델이 1위인 비율
"""
import argparse, json, random

METRICS = {  # 이름: (값 추출, 방향 +1 높을수록 좋음 / -1 낮을수록 좋음)
    "정답 안정도": (lambda r: r["정답안정도"], +1),
    "환각률": (lambda r: r["환각"][0], -1),
    "인용 정확도": (lambda r: r["인용정확도"][0], +1),
    "응답 시간": (lambda r: r["응답p50"], -1),
}
WEIGHTS = {"정답 안정도": 3, "환각률": 3, "인용 정확도": 1, "응답 시간": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ram-gate", type=float, default=10.0)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    F = json.load(open(a.metrics, encoding="utf-8"))
    passed = [m for m, r in F.items() if r.get("RAM") is not None and r["RAM"] <= a.ram_gate and r.get("응답p50")]
    failed = {m: r.get("RAM") for m, r in F.items() if m not in passed}

    val = {k: {m: f(F[m]) for m in passed} for k, (f, _) in METRICS.items()}
    rank = {k: {m: i + 1 for i, m in enumerate(sorted(passed, key=lambda x: v[x] * (-1 if METRICS[k][1] > 0 else 1)))}
            for k, v in val.items()}

    def norm(k):
        v = val[k]; lo, hi = min(v.values()), max(v.values())
        return {m: ((v[m] - lo) / (hi - lo) if METRICS[k][1] > 0 else (hi - v[m]) / (hi - lo)) for m in passed}
    N = {k: norm(k) for k in METRICS}
    score = {m: round(100 * sum(WEIGHTS[k] * N[k][m] for k in METRICS) / sum(WEIGHTS.values()), 1) for m in passed}

    random.seed(a.seed); win = {m: 0 for m in passed}
    for _ in range(a.trials):
        w = {k: random.uniform(0.5, 3.0) for k in METRICS}
        s = {m: sum(w[k] * N[k][m] for k in METRICS) for m in passed}
        win[max(s, key=s.get)] += 1

    order = sorted(passed, key=lambda m: (-sum(rank[k][m] == 1 for k in METRICS if k != "응답 시간"), -score[m]))
    res = {"gate": {"ram_gb_max": a.ram_gate, "passed": passed, "failed": failed},
           "metric_rank": rank,
           "quality_firsts": {m: sum(rank[k][m] == 1 for k in ["정답 안정도", "환각률", "인용 정확도"]) for m in passed},
           "weighted_score_reference": score,
           "sensitivity_top1_pct": {m: round(100 * win[m] / a.trials, 1) for m in passed},
           "order": order}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"관문 통과 {len(passed)}종 / 탈락 {list(failed)}")
    print(f"{'모델':18}" + "".join(f"{k:>12}" for k in METRICS) + f"{'품질1위':>8}{'가중(참고)':>10}{'민감도':>8}")
    for m in order:
        print(f"{m:18}" + "".join(f"{val[k][m]:7.1f}({rank[k][m]})" for k in METRICS)
              + f"{res['quality_firsts'][m]:8}{score[m]:10.1f}{res['sensitivity_top1_pct'][m]:7.1f}%")


if __name__ == "__main__":
    main()
