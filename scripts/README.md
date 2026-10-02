# scripts

| 파일 | 용도 |
|---|---|
| `build_frozen_inputs.py` | 문항 + 근거 인용문 + 지시문 → 고정 입력(jsonl)과 해시 manifest 생성 |
| `run_eval.py` | llama-server로 10개 모델 × 고정 입력 실행, 답변 저장 |
| `score_auto.py` | 규칙 기반 채점 (형식 준수 · 인용 · 수치 재현 · 원문 복사). 인자: 문항 파일, 고정 입력, 답변 폴더, 출력 폴더 |
| `speed_bench.py` | 응답 시간 측정 (TTFT · 전체 p50/p95 · tok/s) |
| `mem_ops.py` | 운영 설정(문맥 4,096 · KV q8_0 · FA)으로 최대 메모리 측정 |
| `judge/blind_judge.py` | 블라인드 전문 판정 (라벨 섞기 · 표시 · 기록) |
| `compute_metrics.py` | 판정 + 자동 채점 + 런타임 → 모델별 지표 |
| `select_model.py` | 관문 · 지표별 순위 · 가중 점수(참고) · 민감도 |

모든 스크립트는 이 저장소의 `data/`만으로 실행된다. 모델 실행(`run_eval.py` · `speed_bench.py` · `mem_ops.py`)에는 llama.cpp와 모델 가중치가 추가로 필요하다.
