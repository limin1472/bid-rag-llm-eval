# 계산식

[metrics.md](metrics.md)의 지표 정의를 식으로 옮긴 것이다. 같은 내용을 논문 형식으로 정리한 PDF: [report/metrics-equations.pdf](../report/metrics-equations.pdf)

## 가중 점수 (참고용)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/score-formula-dark.png">
  <img alt="S = 100·[w1·M_stab + w2·(1 − M_hallu) + w3·M_cite + w4·(1 − M_latency)], Σw = 1" src="assets/score-formula-light.png" width="850">
</picture>

$`M`$은 관문을 통과한 9종 안에서 최소–최대 정규화한 값(0~1)이다. 가중치는 $`(w_1,w_2,w_3,w_4)=\tfrac{1}{8}(3,3,1,1)`$이고, 모델별로 계산한 $`S`$가 아래 5절의 $`S_m`$이다(Qwen3-8B 89.0). 이 점수는 참고용이며, **결론은 지표별 순위로 낸다**([method.md](method.md#선정)).

## 기호

| 기호 | 의미 |
|---|---|
| $`\mathcal{Q}`$ | 평가 문항 집합, $`\lvert\mathcal{Q}\rvert=295`$ (접근통제 S04 5문항 제외) |
| $`\mathcal{Q}^{\mathrm{ok}}_{m,p}`$ | 모델 $`m`$·지시문 $`p`$에서 정상 생성된 답변의 문항 집합 |
| $`\mathcal{Q}^{\mathrm{num}}`$ | 기대 답에 수치가 있는 문항 집합, $`\lvert\mathcal{Q}^{\mathrm{num}}\rvert=58`$ |
| $`\mathcal{M},\ \mathcal{M}_{\mathrm{pass}}`$ | 후보 모델 10종, 메모리 관문 통과 9종 |
| $`\mathcal{P}`$ | 지시문 5종 $`\{P_1,\dots,P_5\}`$ |
| $`j_{m,p,q}`$ | 답변 $`a_{m,p,q}`$의 판정 라벨 |
| $`\mathcal{H}`$ | 환각 유형 $`\{\mathrm{EC},\mathrm{SC},\mathrm{EB},\mathrm{SB}\}`$ |
| $`\mathcal{F}`$ | 무응답 유형 $`\{\mathrm{RE},\mathrm{PR},\mathrm{FR},\mathrm{QR}\}`$ (근거 반복·절차 서술·조각·질문 반복) |
| $`\mathbf{1}[\cdot]`$ | 조건이 참이면 1, 거짓이면 0 |

## 1. 판정 기반 지표

답변이 질문에 대한 판단을 담았는지를 $`\delta(a)`$, 근거를 벗어난 내용의 유형 집합을 $`T(a)\subseteq\mathcal{H}`$라 하면

```math
j(a)=
\begin{cases}
f(a)\in\mathcal{F}, & \delta(a)=0\\
\max_{\succ} T(a), & \delta(a)=1,\ T(a)\neq\varnothing\\
\mathrm{c}, & \delta(a)=1,\ T(a)=\varnothing
\end{cases}
\qquad \mathrm{EC}\succ\mathrm{EB}\succ\mathrm{SC}\succ\mathrm{SB}
```

```math
\mathrm{Acc}_{m,p}=\frac{100}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\mathbf{1}[j_{m,p,q}=\mathrm{c}],\quad
\mathrm{Hal}_{m,p}=\frac{100}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\mathbf{1}[j_{m,p,q}\in\mathcal{H}],\quad
\mathrm{Abs}_{m,p}=\frac{100}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\mathbf{1}[j_{m,p,q}\in\mathcal{F}]
```

세 비율의 합은 100이다. 유형별 비율 $`R^{(t)}_{m,p}`$는 $`\mathbf{1}[j_{m,p,q}=t]`$로 같은 방식으로 구하며, $`\sum_{t\in\mathcal{H}}R^{(t)}_{m,p}=\mathrm{Hal}_{m,p}`$이다.

## 2. 지시문 집계와 정답 안정도

```math
\bar{X}_m=\frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}}X_{m,p},\qquad
s_X(m)=\sqrt{\frac{1}{|\mathcal{P}|-1}\sum_{p\in\mathcal{P}}\left(X_{m,p}-\bar{X}_m\right)^2}
```

```math
\mathrm{Stab}_m=\overline{\mathrm{Acc}}_m-s_{\mathrm{Acc}}(m)
```

## 3. 규칙 기반 자동 지표

정상 생성된 답변만 대상으로 하고, 추론 블록과 특수 토큰을 지운 뒤 계산한다. $`x\sqsubseteq y`$는 문자열 포함이다.

**형식 준수**

```math
\mathrm{Fmt}_{m,p}=\frac{100}{|\mathcal{Q}^{\mathrm{ok}}_{m,p}|}\sum_{q\in\mathcal{Q}^{\mathrm{ok}}_{m,p}}\mathbf{1}\left[\text{“답변:”}\sqsubseteq a\ \wedge\ \text{“근거:”}\sqsubseteq a\right]
```

**인용 정확도** — $`C(a)=B(a)\cup G(a)`$. $`B(a)`$는 대괄호 번호 $`[k]`$ 전체, $`G(a)`$는 마지막 `근거:` 줄이 숫자·괄호·구분 기호로만 되어 있을 때 그 숫자들. 제공 범위 $`\mathcal{E}_q=\{1,\dots,n_q\}`$.

```math
\pi(a)=
\begin{cases}
\dfrac{|C(a)\cap\mathcal{E}_q|}{|C(a)|}, & C(a)\neq\varnothing\\[6pt]
0, & C(a)=\varnothing
\end{cases}
\qquad
\mathrm{Cite}_{m,p}=\frac{100}{|\mathcal{Q}^{\mathrm{ok}}_{m,p}|}\sum_{q\in\mathcal{Q}^{\mathrm{ok}}_{m,p}}\pi(a_{m,p,q})
```

**수치 재현** — $`\nu(x)`$는 수치 토큰 추출(날짜는 한 덩어리, 천 단위 쉼표 제거). $`N_q=\{u\in\nu(e^{\ast}_q):|u|\ge2\}`$.

```math
\mathrm{Num}_{m,p}=\frac{100}{|\mathcal{Q}^{\mathrm{num}}\cap\mathcal{Q}^{\mathrm{ok}}_{m,p}|}\sum_{q\in\mathcal{Q}^{\mathrm{num}}\cap\mathcal{Q}^{\mathrm{ok}}_{m,p}}\frac{|N_q\cap\nu(a_{m,p,q})|}{|N_q|}
```

**원문 복사** — 공백을 지운 답변 $`\hat{a}`$와 근거 전체 $`\hat{e}_q`$에 대해, 60자 창을 20자 간격으로 옮긴다.

```math
\mathrm{Copy}(a)=\mathbf{1}\left[|\hat{a}|\ge60\ \wedge\ \exists\,i\in\{0,20,40,\dots\},\ i<\max(1,|\hat{a}|-60):\ \hat{a}[i,i+60)\sqsubseteq\hat{e}_q\right]
```

## 4. 운영 지표

```math
T^{50}_m=\operatorname{median}_{q\in\mathcal{Q}^{\mathrm{spd}}}\tau_{m,q},\qquad
\mathrm{Mem}_m=\max_t\ \mathrm{RSS}_m(t)
```

$`\tau_{m,q}`$는 요청 시작부터 생성 종료까지의 시간(워크스테이션 CPU, 30문항), $`\mathrm{Mem}_m`$은 운영 설정의 최대 상주 메모리(VmHWM)다.

## 5. 선정

| 지표 $`k`$ | 값 $`v_{k,m}`$ | 방향 $`d_k`$ | 가중치 |
|---|---|---|---|
| 정답 안정도 | $`\mathrm{Stab}_m`$ | +1 | 3 |
| 환각률 | $`\overline{\mathrm{Hal}}_m`$ | −1 | 3 |
| 인용 정확도 | $`\overline{\mathrm{Cite}}_m`$ | +1 | 1 |
| 응답 시간 | $`T^{50}_m`$ | −1 | 1 |

```math
\mathcal{M}_{\mathrm{pass}}=\{m\in\mathcal{M}:\mathrm{Mem}_m\le10\,\mathrm{GB}\}
```

```math
\mathrm{rank}_k(m)=1+\left|\{m'\in\mathcal{M}_{\mathrm{pass}}:d_kv_{k,m'}>d_kv_{k,m}\}\right|,\qquad
W_m=\sum_{k\in\{\text{안정도},\text{환각},\text{인용}\}}\mathbf{1}[\mathrm{rank}_k(m)=1]
```

최종 순서는 $`(W_m,S_m)`$ 내림차순이다. 가중 점수 $`S_m`$은 다음과 같다.

```math
\tilde{v}_{k,m}=\frac{d_kv_{k,m}-\min_{m'}d_kv_{k,m'}}{\max_{m'}d_kv_{k,m'}-\min_{m'}d_kv_{k,m'}},\qquad
S_m=100\cdot\frac{\sum_k w_k\tilde{v}_{k,m}}{\sum_k w_k}
```

**민감도** — $`w^{(t)}_k\sim\mathcal{U}(0.5,3.0)`$, $`T=3000`$회(시드 3). 격자 검증은 $`\Omega=\{1,\dots,5\}^4`$의 625조합.

```math
\mathrm{Win}_m=\frac{100}{T}\sum_{t=1}^{T}\mathbf{1}\left[m=\operatorname*{arg\,max}_{m'\in\mathcal{M}_{\mathrm{pass}}}\sum_k w^{(t)}_k\tilde{v}_{k,m'}\right]
```

## 계산 예시: Qwen3-8B

$`\mathrm{Stab}=84.0-3.1=80.9`$. 품질 세 지표 모두 최선값이라 $`\tilde{v}=1`$이고, 응답 시간은 $`(29.2-26.4)/(29.2-6.1)=0.121`$이다.

```math
S=100\cdot\frac{3(1)+3(1)+1(1)+1(0.121)}{8}=89.0
```

## 구현과의 대응

- 판정 지표의 분모는 생성 실패를 포함한 295로 고정한다. 자동 지표의 분모는 정상 생성 답변 수(290~295)다. 수치 재현은 57 또는 58이다.
- 정답 안정도와 선정 계산은 소수 첫째 자리로 반올림한 값을 입력으로 쓴다. 응답 시간이 12.3초로 같은 Gemma-3-4B·Kanana-1.5-2.1B는 입력 순서로 4·5위가 갈리며, 반올림 전 값(12.26·12.31초)의 순서와 같다.
- 민감도 구현은 $`\sum_k w_k`$로 나누지 않는다. 양의 상수배는 1위를 바꾸지 않는다.
- 코드: 판정·집계 `scripts/compute_metrics.py`, 자동 지표 `scripts/score_auto.py`, 선정 `scripts/select_model.py`.
