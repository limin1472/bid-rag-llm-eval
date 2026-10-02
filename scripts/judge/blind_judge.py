#!/usr/bin/env python3
"""
블라인드 판정 도구 — 문항 하나의 답변 50개(10모델 × 5지시문)를 섞어 전문으로 보여주고, 판정을 기록한다.

  # 1) 라벨 키 생성 (문항별로 모델·지시문 순서를 해시 시드로 섞음)
  python scripts/judge/blind_judge.py prep  --sample data/judgments/scoring_sample_100.json --responses data/responses --out work/key.json

  # 2) 한 문항 표시 (모델·지시문은 가려지고 [01]~[50] 라벨만 보임)
  python scripts/judge/blind_judge.py show  --key work/key.json --questions data/questions/questions_300.jsonl \
         --frozen data/frozen_inputs --responses data/responses --qid Q300-C01-N06-E1

  # 3) 판정 기록 (라벨 → 판정 코드). 지정하지 않은 라벨은 'c'(정답)
  echo '{"Q300-C01-N06-E1": {"06":"SC","11":"F:근거반복"}}' | \
    python scripts/judge/blind_judge.py record --key work/key.json --out data/judgments/new.jsonl --judge "Claude Fable 5.1"

판정 코드: c | EC SC EB SB | F:근거반복 F:절차서술 F:조각 F:질문반복   (기준서: docs/judgment_rubric.md)

"""
import argparse, glob, hashlib, json, pathlib, random, re, sys

MODELS = ["Gemma-3-1B", "Gemma-3-4B", "Kanana-1.5-2.1B", "Kanana-1.5-8B", "Llama-3.1-8B",
          "Midm-2.0-Mini", "Qwen3-1.7B", "Qwen3-14B", "Qwen3-4B", "Qwen3-8B"]
PROMPTS = ["P1", "P2", "P3", "P4", "P5"]


def clean(a):
    a = re.sub(r"<think>.*?</think>", "", a or "", flags=re.S)
    for t in ("<|eot_id|>", "<end_of_turn>", "<|im_end|>"):
        a = a.replace(t, "")
    return a.strip()


def load_responses(root):
    ans = {}
    for f in glob.glob(f"{root}/results_*/*/answers.jsonl"):
        p = pathlib.Path(f); prompt = p.parent.parent.name.replace("results_", ""); model = p.parent.name
        for l in open(f, encoding="utf-8"):
            d = json.loads(l); ans[(prompt, model, d["question_id"])] = d
    return ans


def cmd_prep(a):
    qids = json.load(open(a.sample, encoding="utf-8"))["full_100"]
    key = {}
    for q in qids:
        items = [(m, p) for m in MODELS for p in PROMPTS]
        random.Random(int(hashlib.sha256((q + a.salt).encode()).hexdigest()[:8], 16)).shuffle(items)
        key[q] = {f"{i+1:02d}": mp for i, mp in enumerate(items)}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(key, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"키 저장: {a.out} ({len(key)}문항 × 50)")


def cmd_show(a):
    key = json.load(open(a.key, encoding="utf-8"))
    qs = {json.loads(l)["question_id"]: json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()}
    fr = json.loads(next(l for l in open(f"{a.frozen}/frozen_P1.jsonl", encoding="utf-8") if a.qid in l))
    ans = load_responses(a.responses)
    q = qs[a.qid]
    print(f"{a.qid} | {q['expected_action']} | Q: {q['question']}")
    print("기대:", q["expected_answer"])
    print("근거:", fr["user"].split("[질문]")[0].strip())
    for lab, (m, p) in sorted(key[a.qid].items()):
        d = ans.get((p, m, a.qid))
        t = clean(d.get("answer")) if d and d.get("status") == "ok" else "(생성 실패)"
        print(f"[{lab}] {t}")


def cmd_record(a):
    key = json.load(open(a.key, encoding="utf-8")); d = json.load(sys.stdin); n = 0
    with open(a.out, "a", encoding="utf-8") as fh:
        for qid, labs in d.items():
            for lab, (m, p) in key[qid].items():
                fh.write(json.dumps({"question_id": qid, "model": m, "prompt": p, "j": labs.get(lab, "c"), "judge": a.judge},
                                    ensure_ascii=False) + "\n"); n += 1
    print(f"기록 {n}건 → {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("prep"); x.add_argument("--sample", required=True); x.add_argument("--out", required=True); x.add_argument("--salt", default="J100")
    x = sp.add_parser("show"); [x.add_argument(f"--{k}", required=True) for k in ["key", "questions", "frozen", "responses", "qid"]]
    x = sp.add_parser("record"); x.add_argument("--key", required=True); x.add_argument("--out", required=True); x.add_argument("--judge", default="Claude Fable 5.1")
    a = ap.parse_args(); {"prep": cmd_prep, "show": cmd_show, "record": cmd_record}[a.cmd](a)
