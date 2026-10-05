"""
Load test: N concurrent streaming clients sharing a long system prompt.
Reports throughput and time-to-first-token percentiles.
    python reference/load_test.py [clients] [max_tokens] [base_url]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 128
BASE = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080"
SYSTEM = ("You are a helpful assistant for a bookshop. " * 40).strip()
QUESTIONS = ["Recommend a novel about the sea.", "What is a good book for learning Python?",
             "Suggest a Romanian classic.", "Which poetry book should I read first?"]

results = []
lock = threading.Lock()


def client(i):
    body = json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": QUESTIONS[i % len(QUESTIONS)] + f" (#{i})"}],
                       "max_tokens": MAX_TOKENS, "temperature": 0.7, "stream": True, "ignore_eos": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"]
            if first is None and j["choices"] and j["choices"][0]["delta"].get("content"):
                first = time.perf_counter() - t0
    with lock:
        results.append((first or 0, time.perf_counter() - t0, usage))


t0 = time.perf_counter()
threads = [threading.Thread(target=client, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - t0
gen = sum(u["completion_tokens"] for _, _, u in results if u)
cached = sum(u["prompt_tokens_details"]["cached_tokens"] for _, _, u in results if u)
prompt = sum(u["prompt_tokens"] for _, _, u in results if u)
ttft = sorted(r[0] * 1000 for r in results)
print(f"{N} clients, {gen} tokens generated in {wall:.2f} s = {gen / wall:.0f} tokens/s")
print(f"time to first token: median {statistics.median(ttft):.0f} ms, p90 {ttft[int(0.9 * (len(ttft) - 1))]:.0f} ms")
print(f"prompt tokens {prompt}, of which {cached} came from the prefix cache ({100 * cached / max(prompt, 1):.0f}%)")
