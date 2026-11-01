"""Fire N concurrent requests at the server to watch continuous batching live.

    python -m minivllm.server                  # terminal 1
    python examples/client_demo.py -n 32       # terminal 2
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import requests

TOPICS = ["the KV cache", "PagedAttention", "continuous batching", "memory fragmentation",
          "prefix caching", "GPU memory bandwidth", "speculative decoding", "quantization"]

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=32)
ap.add_argument("--url", default="http://127.0.0.1:8000")
a = ap.parse_args()

SYSTEM = "You are a concise systems-engineering tutor. Answer in at most three sentences. "


def one(i):
    t0 = time.perf_counter()
    r = requests.post(f"{a.url}/v1/completions", json={
        "prompt": SYSTEM + f"Explain {TOPICS[i % len(TOPICS)]}.", "chat": True,
        "max_tokens": 64 + 16 * (i % 8)}).json()
    return i, time.perf_counter() - t0, r


t0 = time.perf_counter()
with ThreadPoolExecutor(a.n) as ex:
    results = list(ex.map(one, range(a.n)))
wall = time.perf_counter() - t0
toks = sum(r["usage"]["completion_tokens"] for _, _, r in results)
for i, dt, r in results[:4]:
    print(f"[{i}] {dt:5.2f}s ttft={r['timing']['ttft_s']:.2f}s  {r['choices'][0]['text'][:90]!r}")
print(f"\n{a.n} concurrent requests, {toks} tokens in {wall:.2f}s -> {toks / wall:.0f} tok/s")
print("server stats:", requests.get(f"{a.url}/stats").json())
