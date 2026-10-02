#!/usr/bin/env python3
"""
공공입찰 RAG LLM 비교 — CPU 속도·메모리 측정 (노트북, -ngl 0)

각 모델을 llama-server로 띄우고(GPU 오프로드 0, 단일 슬롯) 길이별로 고른 30문항을
스트리밍으로 요청해 TTFT·생성 속도·글자/초·peak RSS를 기록한다.

  python speed_bench.py --server llama-server.exe --models-dir models --frozen frozen_inputs.jsonl --out speed_results

품질 실행과 같은 동결 입력·같은 설정(컨텍스트 24,576, temp 0, seed 1234, 출력 512)을 쓴다.
차이는 단 두 가지: -ngl 0 (CPU 전용), -np 1 (병렬 없음). 속도를 재는 것이므로 답변 내용은 저장만 한다.
"""
import argparse, json, os, pathlib, subprocess, sys, time, urllib.request, statistics, platform

CTX = 24576; MAX_TOKENS = 512; TEMPERATURE = 0.0; SEED = 1234
SAMPLE_N = 30; WARMUP = 1

REGISTRY = [
    ("Qwen3-1.7B-Q4_K_M.gguf",                 "Qwen3-1.7B",      "2B"),
    ("kanana-1.5-2.1b-Q4_K_M.gguf",            "Kanana-1.5-2.1B", "2B"),
    ("Midm-2.0-Mini-Q4_K_M.gguf",              "Midm-2.0-Mini",   "2B"),
    ("gemma-3-1b-it-Q4_K_M.gguf",              "Gemma-3-1B",      "2B"),
    ("Qwen3-4B-Q4_K_M.gguf",                   "Qwen3-4B",        "4B"),
    ("gemma-3-4b-it-Q4_K_M.gguf",              "Gemma-3-4B",      "4B"),
    ("Qwen3-8B-Q4_K_M.gguf",                   "Qwen3-8B",        "8B"),
    ("kanana-1.5-8b-Q4_K_M.gguf",              "Kanana-1.5-8B",   "8B"),
    ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Llama-3.1-8B",    "8B"),
    ("Qwen3-14B-Q4_K_M.gguf",                  "Qwen3-14B",       "14B"),
]

try:
    import psutil
except ImportError:
    psutil = None


def pick_sample(rows):
    """생성 대상 문항을 프롬프트 길이순으로 정렬해 30개를 균등 간격으로 뽑는다(짧은~긴 문항 고루)."""
    gen = sorted([r for r in rows if not r["blocked_by_acl"]], key=lambda r: r["prompt_chars"])
    step = len(gen) / SAMPLE_N
    return [gen[int(i * step)] for i in range(SAMPLE_N)]


def wait_health(port, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server 종료 (exit {proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200: return time.time() - t0
        except Exception:
            time.sleep(1)
    raise TimeoutError("서버 기동 시간 초과")


def peak_rss(proc):
    if not psutil: return None
    try:
        p = psutil.Process(proc.pid)
        return max([p.memory_info().rss] + [c.memory_info().rss for c in p.children(recursive=True)])
    except Exception:
        return None


def ask_stream(port, system, user):
    body = json.dumps({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       "temperature": TEMPERATURE, "seed": SEED, "max_tokens": MAX_TOKENS, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; text = []; finish = None; usage = {}; timings = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            d = json.loads(payload)
            ch = (d.get("choices") or [{}])[0]
            delta = (ch.get("delta") or {}).get("content")
            if delta:
                if t_first is None: t_first = time.time()
                text.append(delta)
            if ch.get("finish_reason"): finish = ch["finish_reason"]
            if d.get("usage"): usage = d["usage"]
            if d.get("timings"): timings = d["timings"]
    t_end = time.time()
    answer = "".join(text)
    ttft = (t_first - t0) if t_first else None
    gen_s = (t_end - t_first) if t_first else None
    return {
        "answer": answer, "finish_reason": finish,
        "ttft_s": round(ttft, 3) if ttft else None,
        "generation_s": round(gen_s, 3) if gen_s else None,
        "total_s": round(t_end - t0, 3),
        "prompt_tokens": usage.get("prompt_tokens") or timings.get("prompt_n"),
        "completion_tokens": usage.get("completion_tokens") or timings.get("predicted_n"),
        "chars": len(answer),
        "tok_per_s": (round(timings["predicted_n"] / (timings["predicted_ms"] / 1000), 2)
                      if timings.get("predicted_ms") else None),
        "prompt_tok_per_s": (round(timings["prompt_n"] / (timings["prompt_ms"] / 1000), 2)
                             if timings.get("prompt_ms") else None),
        "chars_per_s": round(len(answer) / gen_s, 2) if gen_s and gen_s > 0 else None,
    }


def p(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def run_model(entry, args, sample, sysinfo):
    fname, display, band = entry
    path = pathlib.Path(args.models_dir) / fname
    if not path.exists():
        print(f"[건너뜀] {display}: 파일 없음"); return
    out_dir = pathlib.Path(args.out) / display; out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.server, "-m", str(path), "-c", str(CTX), "-np", "1", "-ngl", "0",
           "-t", str(args.threads), "--host", "127.0.0.1", "--port", str(args.port),
           "--reasoning", "off", "--reasoning-budget", "0", "--reasoning-format", "none", "--jinja",
           "--log-disable"]
    print(f"\n=== {display} ({band}급) {path.stat().st_size/2**30:.2f}GB")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    recs, peak = [], 0
    try:
        load_s = wait_health(args.port, proc)
        print(f"  기동 {load_s:.1f}초")
        for i, row in enumerate(sample):
            r = {"question_id": row["question_id"], "prompt_chars": row["prompt_chars"], "warmup": i < WARMUP}
            try:
                r.update(ask_stream(args.port, row["system"], row["user"])); r["status"] = "ok"
            except Exception as e:
                r.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            m = peak_rss(proc)
            if m: peak = max(peak, m)
            recs.append(r)
            if r["status"] == "ok":
                print(f"  {i+1:2}/{len(sample)} 입력 {r['prompt_tokens']}tok  TTFT {r['ttft_s']}s  "
                      f"생성 {r['generation_s']}s  {r['tok_per_s']} tok/s  {'(warmup)' if r['warmup'] else ''}", flush=True)
            else:
                print(f"  {i+1:2}/{len(sample)} 실패: {r.get('error','')[:80]}", flush=True)
    finally:
        try: proc.terminate(); proc.wait(timeout=30)
        except Exception: proc.kill()
        time.sleep(2)

    ok = [r for r in recs if r["status"] == "ok" and not r["warmup"]]
    summary = {
        "model": display, "band": band, "file": fname, "size_gb": round(path.stat().st_size/2**30, 2),
        "load_seconds": round(load_s, 1), "n_ok": len(ok), "n_fail": sum(r["status"] != "ok" for r in recs),
        "truncated": sum(r.get("finish_reason") == "length" for r in ok),
        "ttft_p50": p([r["ttft_s"] for r in ok], .5), "ttft_p95": p([r["ttft_s"] for r in ok], .95),
        "generation_p50": p([r["generation_s"] for r in ok], .5), "generation_p95": p([r["generation_s"] for r in ok], .95),
        "total_p50": p([r["total_s"] for r in ok], .5), "total_p95": p([r["total_s"] for r in ok], .95),
        "tok_per_s_median": p([r["tok_per_s"] for r in ok], .5),
        "prompt_tok_per_s_median": p([r["prompt_tok_per_s"] for r in ok], .5),
        "chars_per_s_median": p([r["chars_per_s"] for r in ok], .5),
        "peak_rss_gb": round(peak/2**30, 2) if peak else None,
        "settings": {"ngl": 0, "threads": args.threads, "ctx": CTX, "np": 1, "max_tokens": MAX_TOKENS},
        "system": sysinfo,
    }
    (out_dir / "requests.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → TTFT p50 {summary['ttft_p50']}s / p95 {summary['ttft_p95']}s | 전체 p50 {summary['total_p50']}s / p95 {summary['total_p95']}s "
          f"| {summary['tok_per_s_median']} tok/s | peak RSS {summary['peak_rss_gb']}GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--frozen", default="data/frozen_inputs/frozen_P1.jsonl")
    ap.add_argument("--out", default="data/runtime/speed_results")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.frozen, encoding="utf-8") if l.strip()]
    sample = pick_sample(rows)
    sysinfo = {"platform": platform.platform(), "cpu": platform.processor(), "cpu_count": os.cpu_count(),
               "ram_gb": round(psutil.virtual_memory().total/2**30, 1) if psutil else None,
               "psutil": bool(psutil), "sampled_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("시스템:", sysinfo)
    print(f"표본 {len(sample)}문항, 프롬프트 길이 {sample[0]['prompt_chars']}~{sample[-1]['prompt_chars']}자, 스레드 {args.threads}")
    if not psutil: print("※ psutil 없음 — peak RSS는 기록되지 않습니다. `pip install psutil` 권장")
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    for e in REGISTRY:
        if only and e[1] not in only: continue
        run_model(e, args, sample, sysinfo)
    print("\n완료.")


if __name__ == "__main__":
    main()
