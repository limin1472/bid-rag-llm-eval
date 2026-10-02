#!/usr/bin/env python3
"""
공공입찰 RAG LLM 비교 평가 — 9종 일괄 실행기

frozen_inputs.jsonl을 읽어 모델 하나씩 llama-server를 띄우고 295문항을 돌린다.
표준 라이브러리만 사용한다.

  python run_eval.py --models-dir models --frozen frozen/frozen_inputs.jsonl --out results

주요 동작
  - 문항별 즉시 저장. 중단 후 같은 명령을 다시 실행하면 남은 문항만 이어서 진행
  - 모델 파일 SHA-256, 실행 설정, llama.cpp 커밋을 결과에 함께 기록
  - 실패·타임아웃·잘림을 구분해 기록. 0점으로 치환하지 않는다
  - 속도 수치는 GPU 병렬 실행값이므로 참고용. 도입 속도는 CPU에서 따로 측정한다
"""
import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, time, urllib.request, urllib.error

CTX_PER_SLOT = 24576
MAX_TOKENS = 512
TEMPERATURE = 0.0
SEED = 1234
REQUEST_TIMEOUT = 600

# 모델 등록부: 파일명 → (표시명, 계열, 크기대)
REGISTRY = [
    ("Qwen3-1.7B-Q4_K_M.gguf",                  "Qwen3-1.7B",    "qwen3",  "2B"),
    ("kanana-1.5-2.1b-Q4_K_M.gguf",             "Kanana-1.5-2.1B", "kanana", "2B"),
    ("Midm-2.0-Mini-Q4_K_M.gguf",               "Midm-2.0-Mini", "midm",   "2B"),
    ("gemma-4-E2B-Q4_K_M.gguf",                 "Gemma-4-E2B",   "gemma4", "2B"),
    ("Qwen3-4B-Q4_K_M.gguf",                    "Qwen3-4B",      "qwen3",  "4B"),
    ("gemma-4-E4B-Q4_K_M.gguf",                 "Gemma-4-E4B",   "gemma4", "4B"),
    ("Qwen3-8B-Q4_K_M.gguf",                    "Qwen3-8B",      "qwen3",  "8B"),
    ("kanana-1.5-8b-Q4_K_M.gguf",               "Kanana-1.5-8B", "kanana", "8B"),
    ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",  "Llama-3.1-8B",  "llama",  "8B"),
]


def slots_for(size_bytes):
    """가중치 크기에 따라 병렬 슬롯 수를 정한다.
    llama.cpp의 -c는 전체 컨텍스트이고 슬롯이 나눠 쓰므로,
    슬롯당 24,576을 보장하려면 -c = 24576 * slots 로 준다.
    A5000 24GB 안에 들어오도록 크기별로 낮춘다."""
    gb = size_bytes / 2**30
    if gb <= 2.0:   return 4
    if gb <= 3.5:   return 3
    return 2


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def wait_health(port, proc, timeout=300):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server 비정상 종료 (exit {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return time.time() - start
        except Exception:
            time.sleep(2)
    raise TimeoutError("llama-server 기동 시간 초과")


def ask(port, system, user):
    body = json.dumps({
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        data = json.loads(r.read().decode("utf-8"))
    elapsed = time.time() - t0
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    return {
        "answer": (choice.get("message") or {}).get("content"),
        "finish_reason": choice.get("finish_reason"),
        "total_seconds": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
    }


def run_model(entry, args, rows, build_commit):
    fname, display, family, band = entry
    model_path = pathlib.Path(args.models_dir) / fname
    if not model_path.exists():
        print(f"  [건너뜀] 파일 없음: {model_path}")
        return

    out_dir = pathlib.Path(args.out) / display
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"

    done = set()
    if answers_path.exists():
        with answers_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["question_id"])
                except Exception:
                    pass

    targets = [r for r in rows if not r["blocked_by_acl"] and r["question_id"] not in done]
    print(f"\n{'='*64}\n{display}  ({band}급, {family})")
    print(f"  파일 {fname}  ({model_path.stat().st_size/2**30:.2f} GB)")
    print(f"  남은 문항 {len(targets)} / 295  (완료 {len(done)})")
    if not targets:
        print("  이미 완료")
        return

    size = model_path.stat().st_size
    slots = slots_for(size)
    cmd = [args.server, "-m", str(model_path),
           "-c", str(CTX_PER_SLOT * slots), "-np", str(slots),
           "-ngl", "99", "--host", "127.0.0.1", "--port", str(args.port),
           "--log-disable"]

    print(f"  슬롯 {slots}개, 전체 컨텍스트 {CTX_PER_SLOT*slots:,} (슬롯당 {CTX_PER_SLOT:,})")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)
    try:
        load_s = wait_health(args.port, proc)
        print(f"  기동 완료 {load_s:.1f}초. 실행 시작")

        meta = {
            "model_display": display, "model_file": fname, "model_family": family,
            "size_band": band, "model_sha256": sha256_file(model_path),
            "size_bytes": size, "quantization": "Q4_K_M",
            "context_per_slot": CTX_PER_SLOT, "slots": slots,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "seed": SEED,
            "runtime": "llama.cpp", "build_commit": build_commit,
            "load_seconds": round(load_s, 2),
        }
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

        t_start, ok, fail = time.time(), 0, 0
        with answers_path.open("a", encoding="utf-8") as fh:
            for i, row in enumerate(targets, 1):
                rec = {"question_id": row["question_id"], "model_display": display,
                       "context_sha256": row["context_sha256"]}
                try:
                    rec.update(ask(args.port, row["system"], row["user"]))
                    rec["status"] = "ok"
                    ok += 1
                except urllib.error.URLError as e:
                    rec.update({"status": "error", "error": f"URLError: {e}"}); fail += 1
                except TimeoutError:
                    rec.update({"status": "timeout", "error": "request timeout"}); fail += 1
                except Exception as e:
                    rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"}); fail += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 25 == 0 or i == len(targets):
                    el = time.time() - t_start
                    print(f"    {i}/{len(targets)}  경과 {el/60:.1f}분  "
                          f"성공 {ok} 실패 {fail}  예상잔여 {(el/i)*(len(targets)-i)/60:.1f}분",
                          flush=True)
        print(f"  완료: 성공 {ok}, 실패 {fail}")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
        time.sleep(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="models")
    p.add_argument("--frozen", default="data/frozen_inputs/frozen_P1.jsonl")
    p.add_argument("--out", default="data/responses/results_P1")
    p.add_argument("--server", default="llama.cpp/build/bin/llama-server")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--limit", type=int, default=0, help="사전검증용: 앞 N문항만")
    p.add_argument("--only", default="", help="쉼표로 구분한 표시명. 일부만 실행")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.frozen, encoding="utf-8") if l.strip()]
    if args.limit:
        keep = [r for r in rows if not r["blocked_by_acl"]][:args.limit]
        rows = keep
    try:
        build_commit = subprocess.run(["git", "-C", "llama.cpp", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
    except Exception:
        build_commit = "unknown"

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    entries = [e for e in REGISTRY if not only or e[1] in only]

    print(f"동결 입력 {len(rows)}행 | 대상 모델 {len(entries)}종 | 빌드 {build_commit[:12]}")
    t0 = time.time()
    for e in entries:
        run_model(e, args, rows, build_commit)
    print(f"\n전체 완료. 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
