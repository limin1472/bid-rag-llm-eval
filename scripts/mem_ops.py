#!/usr/bin/env python3
"""
운영 프로필 메모리 측정
  - 설정: ctx 4096 · KV 캐시 q8_0 · CPU 전용 · Flash Attention · batch 128 / ubatch 32
  - 모델을 띄우고 짧은 요청 하나만 보낸 뒤 최대 점유량(peak RSS)을 읽는다.
  - 메모리는 문맥 길이로 미리 확보되므로 긴 근거를 넣지 않아도 값이 같다.

  python mem_ops.py --server <llama-server 경로> --models-dir <모델 폴더> [--only 모델명]
"""
import argparse, json, os, pathlib, subprocess, sys, time, urllib.request

CTX = 4096
PROMPT = "다음 문장에 한 줄로 답하라. 오늘 날씨가 좋은가?"

def wait_health(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server 종료 (exit {proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if json.load(r).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            time.sleep(1)
    raise TimeoutError("서버 기동 시간 초과")

def ask(port):
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 64, "temperature": 0, "seed": 1234}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

def peak_rss_gb(pid):
    """서버 프로세스가 지금까지 점유한 최대 메모리(GB)"""
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024 / 1024, 2)
    except FileNotFoundError:
        pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="mem_ops.json")
    a = ap.parse_args()

    models = sorted(pathlib.Path(a.models_dir).glob("*.gguf"))
    if a.only:
        models = [m for m in models if a.only.lower() in m.name.lower()]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = {}

    for path in models:
        name = path.stem.replace("-Q4_K_M", "")
        print(f"=== {name} ({path.stat().st_size/1024**3:.2f}GB)", flush=True)
        cmd = [a.server, "-m", str(path), "-c", str(CTX), "-np", "1", "-ngl", "0",
               "-t", str(a.threads), "--host", "127.0.0.1", "--port", str(a.port),
               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
               "-fa", "on", "-b", "128", "-ub", "32"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        try:
            load_s = wait_health(a.port, proc)
            after_load = peak_rss_gb(proc.pid)
            ask(a.port)
            time.sleep(1)
            peak = peak_rss_gb(proc.pid)
            out[name] = {"peak_rss_gb": peak, "after_load_gb": after_load,
                         "load_s": round(load_s, 1), "file_gb": round(path.stat().st_size/1024**3, 2)}
            print(f"    로드 후 {after_load}GB / 최대 {peak}GB (기동 {load_s:.0f}초)", flush=True)
        except Exception as e:
            out[name] = {"error": str(e)}
            print(f"    실패: {e}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(2)

    json.dump({"settings": {"ctx": CTX, "kv_cache": "q8_0", "ngl": 0,
                            "threads": a.threads, "flash_attn": True, "batch": 128, "ubatch": 32},
               "models": out}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {a.out}")
    print(f"{'모델':22}{'최대 RSS':>10}")
    for k, v in out.items():
        print(f"{k:22}{(str(v.get('peak_rss_gb'))+'GB') if v.get('peak_rss_gb') else v.get('error','—'):>10}")

if __name__ == "__main__":
    main()
