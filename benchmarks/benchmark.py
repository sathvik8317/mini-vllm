"""Benchmarks mini-vllm against plain HuggingFace generate().

Run from the repo root:  python benchmarks/benchmark.py   (writes results/benchmark.md)

Experiment 1 - throughput on a realistic mixed workload
    N requests, prompt lengths ~U(32, 384), output lengths ~U(16, 256).
    Every system must produce exactly the requested number of tokens per request
    (EOS is ignored) so the work is identical.
      * HF sequential      : generate() one request at a time (batch = 1)
      * HF static batching : generate() on padded batches of size `--hf-batch`
                             -> each batch runs until its LONGEST request finishes
      * mini-vllm          : paged KV cache + continuous batching
    Reported: useful output tokens / second (padding and overshoot don't count).

Experiment 2 - prefix caching
    Many requests that share a long system prompt; prefix caching on vs off.

Experiment 3 - KV memory waste (fragmentation) for the same workload
    How many KV slots each allocation strategy reserves vs actually uses.
"""
import argparse
import json
import os
import random
import statistics
import time

import torch

from minivllm import LLMEngine, SamplingParams


def make_workload(n, seed, vocab_lo=1000, vocab_hi=30000, p_range=(32, 384), o_range=(16, 256)):
    rng = random.Random(seed)
    reqs = []
    for _ in range(n):
        p = rng.randint(*p_range)
        reqs.append(([rng.randint(vocab_lo, vocab_hi) for _ in range(p)], rng.randint(*o_range)))
    return reqs


def sync():
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------- HF baselines
def hf_sequential(model, reqs):
    sync(); t0 = time.perf_counter()
    for prompt, out_len in reqs:
        ids = torch.tensor([prompt], device="cuda")
        model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, repetition_penalty=1.0,
                       max_new_tokens=out_len, min_new_tokens=out_len, pad_token_id=0)
    sync()
    return time.perf_counter() - t0


def hf_static_batched(model, reqs, batch_size):
    """Classic static batching. Left-pad prompts; the batch generates max(out_len) tokens."""
    sync(); t0 = time.perf_counter()
    computed_tokens = 0
    for i in range(0, len(reqs), batch_size):
        batch = reqs[i:i + batch_size]
        L = max(len(p) for p, _ in batch)
        M = max(o for _, o in batch)
        ids = torch.zeros(len(batch), L, dtype=torch.long)
        am = torch.zeros(len(batch), L, dtype=torch.long)
        for j, (p, _) in enumerate(batch):
            ids[j, L - len(p):] = torch.tensor(p)
            am[j, L - len(p):] = 1
        model.generate(ids.cuda(), attention_mask=am.cuda(), do_sample=False, repetition_penalty=1.0,
                       max_new_tokens=M, min_new_tokens=M, pad_token_id=0)
        computed_tokens += len(batch) * M
    sync()
    return time.perf_counter() - t0, computed_tokens


# ----------------------------------------------------------------------------- mini-vllm
def run_engine(eng, reqs):
    eng.reset_stats()
    params = [SamplingParams(max_tokens=o, ignore_eos=True) for _, o in reqs]
    seqs = eng.generate([p for p, _ in reqs], params)
    assert all(s.num_output_tokens == o for s, (_, o) in zip(seqs, reqs))
    ttft = [s.first_token_time - s.arrival_time for s in seqs]
    lat = [s.finish_time - s.arrival_time for s in seqs]
    return eng.last_elapsed, ttft, lat


def free_engine(eng):
    del eng.kv_cache, eng.model
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- experiment 3
def fragmentation_report(reqs, block_size, max_model_len, hf_batch, frag_samples):
    used = sum(len(p) + o for p, o in reqs)               # slots that hold real K/V at the end
    # (a) contiguous pre-allocation for the max model length (FasterTransformer / early servers)
    prealloc_max = len(reqs) * max_model_len
    # (b) contiguous pre-allocation for prompt + max_tokens (best case for contiguous)
    prealloc_req = used
    # (c) HF static batching: every row padded to longest prompt + longest output in its batch
    hf = 0
    for i in range(0, len(reqs), hf_batch):
        b = reqs[i:i + hf_batch]
        hf += len(b) * (max(len(p) for p, _ in b) + max(o for _, o in b))
    # (d) paged: only the last block of each sequence is partly empty
    paged = sum(-(-(len(p) + o) // block_size) * block_size for p, o in reqs)
    rows = [
        ("contiguous, reserve max_model_len", prealloc_max),
        ("HF static batch (padding to longest)", hf),
        (f"paged, block_size={block_size}", paged),
        ("ideal (exact)", used),
    ]
    live = None
    if frag_samples:
        waste = [1 - s["used_tokens"] / s["allocated_slots"] for s in frag_samples if s["allocated_slots"]]
        live = statistics.mean(waste)
    return rows, used, live, prealloc_req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--num-requests", type=int, default=128)
    ap.add_argument("--hf-batch", type=int, default=16)
    ap.add_argument("--hf-seq-requests", type=int, default=12, help="subset for the slow batch=1 baseline")
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = {"gpu": torch.cuda.get_device_name(), "model": a.model}

    reqs = make_workload(a.num_requests, seed=0)
    useful = sum(o for _, o in reqs)
    print(f"workload: {len(reqs)} requests, {sum(len(p) for p, _ in reqs)} prompt tokens, "
          f"{useful} output tokens")

    # ---------------- mini-vllm
    eng = LLMEngine(a.model, max_num_seqs=a.max_num_seqs, verbose=True)
    run_engine(eng, make_workload(8, seed=1))                     # warmup (cuda init, allocator)
    el, ttft, lat = run_engine(eng, reqs)
    st = dict(eng.stats)
    results["minivllm"] = {"elapsed_s": el, "tok_per_s": useful / el,
                           "mean_ttft_s": statistics.mean(ttft), "mean_latency_s": statistics.mean(lat),
                           "steps": st["steps"], "prefill_steps": st["prefill_steps"],
                           "decode_steps": st["decode_steps"], "peak_blocks": st["peak_blocks"],
                           "num_blocks": eng.num_gpu_blocks,
                           "preemptions": eng.scheduler.num_preemptions,
                           "avg_decode_batch": st["decode_tokens"] / max(1, st["decode_steps"])}
    print(f"mini-vllm          : {el:7.2f}s  {useful / el:8.1f} tok/s  "
          f"(avg decode batch {results['minivllm']['avg_decode_batch']:.1f}, "
          f"peak {st['peak_blocks']}/{eng.num_gpu_blocks} blocks)")
    frag = fragmentation_report(reqs, eng.block_size, 32768, a.hf_batch, st["frag_samples"])

    # same engine, CUDA graphs off: shows how launch-bound eager decode is
    graphs, eng.graphs = eng.graphs, None
    el_e, _, _ = run_engine(eng, reqs)
    eng.graphs = graphs
    results["minivllm_eager"] = {"elapsed_s": el_e, "tok_per_s": useful / el_e}
    print(f"mini-vllm (eager)  : {el_e:7.2f}s  {useful / el_e:8.1f} tok/s")

    # ---------------- Experiment 2: prefix caching
    rng = random.Random(7)
    system = [rng.randint(1000, 30000) for _ in range(1024)]
    pref_reqs = [(system + [rng.randint(1000, 30000) for _ in range(rng.randint(8, 32))], 32)
                 for _ in range(64)]
    pc = {}
    for flag in (False, True):
        eng.block_manager.enable_prefix_caching = flag
        # flush the cache so "on" starts cold too (first request must compute the prefix)
        bm = eng.block_manager
        bm.hash_to_block.clear(); bm.block_hash = [None] * bm.num_blocks
        el2, ttft2, _ = run_engine(eng, pref_reqs)
        pc[flag] = {"elapsed_s": el2, "tok_per_s": 64 * 32 / el2, "mean_ttft_s": statistics.mean(ttft2),
                    "prefill_tokens_computed": eng.stats["prefill_tokens"],
                    "hit_rate": bm.prompt_tokens_hit / max(1, bm.prompt_tokens_seen)}
        print(f"prefix caching {'ON ' if flag else 'OFF'}: {el2:6.2f}s  "
              f"prefill computed {eng.stats['prefill_tokens']:6d} tok  "
              f"mean TTFT {statistics.mean(ttft2) * 1000:7.1f} ms  hit rate {pc[flag]['hit_rate']:.0%}")
    eng.block_manager.enable_prefix_caching = True
    results["prefix_caching"] = {"off": pc[False], "on": pc[True]}
    model_dir = eng.model_dir
    free_engine(eng); del eng

    # ---------------- HF baselines
    if not a.skip_hf:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).cuda().eval()
        hf_static_batched(hf, make_workload(4, seed=1), 4)          # warmup
        torch.cuda.reset_peak_memory_stats()
        el_b, computed = hf_static_batched(hf, reqs, a.hf_batch)
        results["hf_static"] = {"elapsed_s": el_b, "tok_per_s": useful / el_b, "batch_size": a.hf_batch,
                                "computed_tokens": computed, "wasted_fraction": 1 - useful / computed,
                                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        print(f"HF static batch={a.hf_batch:<3}: {el_b:7.2f}s  {useful / el_b:8.1f} tok/s  "
              f"({1 - useful / computed:.0%} of generated tokens were overshoot)")

        sub = reqs[:a.hf_seq_requests]
        sub_useful = sum(o for _, o in sub)
        el_s = hf_sequential(hf, sub)
        results["hf_sequential"] = {"elapsed_s": el_s, "tok_per_s": sub_useful / el_s,
                                    "num_requests": len(sub)}
        print(f"HF sequential      : {el_s:7.2f}s  {sub_useful / el_s:8.1f} tok/s  (on {len(sub)} requests)")
        del hf; torch.cuda.empty_cache()

    # ---------------- report
    rows, used, live, _ = frag
    results["fragmentation"] = {"rows": rows, "used_slots": used, "paged_live_waste": live}
    lines = ["# mini-vllm benchmark", "", f"GPU: {results['gpu']}  ", f"Model: {a.model}  ",
             f"Workload: {len(reqs)} requests, prompts 32-384 tok, outputs 16-256 tok "
             f"({useful} output tokens), greedy, EOS ignored", "",
             "## 1. Throughput", "", "| system | time (s) | output tok/s | speed-up vs HF seq |",
             "|---|---:|---:|---:|"]
    base = results.get("hf_sequential", {}).get("tok_per_s")
    def row(name, r):
        su = f"{r['tok_per_s'] / base:.1f}x" if base else "-"
        lines.append(f"| {name} | {r['elapsed_s']:.1f} | {r['tok_per_s']:.0f} | {su} |")
    if "hf_sequential" in results:
        row(f"HF generate, batch=1 ({results['hf_sequential']['num_requests']} req subset)",
            results["hf_sequential"])
    if "hf_static" in results:
        row(f"HF generate, static batch={a.hf_batch}", results["hf_static"])
    row("mini-vllm, eager (paged + continuous batching)", results["minivllm_eager"])
    row("mini-vllm + CUDA graphs for decode", results["minivllm"])
    m = results["minivllm"]
    lines += ["", f"mini-vllm: {m['steps']} steps ({m['prefill_steps']} prefill / {m['decode_steps']} decode), "
              f"avg decode batch {m['avg_decode_batch']:.1f}, peak {m['peak_blocks']}/{m['num_blocks']} blocks, "
              f"{m['preemptions']} preemptions, mean TTFT {m['mean_ttft_s']:.2f}s"]
    if "hf_static" in results:
        lines.append(f"HF static: {results['hf_static']['wasted_fraction']:.0%} of generated tokens "
                     f"were overshoot past a request's own length")
    p = results["prefix_caching"]
    lines += ["", "## 2. Prefix caching (64 requests sharing a 1024-token prefix, 32 output tok each)", "",
              "| | time (s) | prefill tokens computed | mean TTFT (ms) | hit rate |", "|---|---:|---:|---:|---:|"]
    for k in ("off", "on"):
        r = p[k]
        lines.append(f"| {k} | {r['elapsed_s']:.2f} | {r['prefill_tokens_computed']} | "
                     f"{r['mean_ttft_s'] * 1000:.0f} | {r['hit_rate']:.0%} |")
    lines += ["", "## 3. KV-cache slots reserved for the same workload", "",
              "| allocation scheme | slots reserved | waste |", "|---|---:|---:|"]
    for name, n in rows:
        lines.append(f"| {name} | {n:,} | {1 - used / n:.1%} |")
    if live is not None:
        lines.append(f"\nMeasured live internal fragmentation of the paged cache during the run: {live:.1%}")
    md = "\n".join(lines)
    print("\n" + md)
    open(os.path.join(a.out, "benchmark.md"), "w").write(md + "\n")
    json.dump(results, open(os.path.join(a.out, "benchmark.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
