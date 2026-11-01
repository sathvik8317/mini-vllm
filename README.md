# mini-vllm

A small LLM inference engine written from scratch in about 1,000 lines of PyTorch to learn how
servers like [vLLM](https://github.com/vllm-project/vllm) work. It serves
**Qwen2.5-0.5B-Instruct** and includes:

- a **paged KV cache** (PagedAttention-style block tables)
- **continuous batching** with preemption
- **automatic prefix caching** (hash-chained blocks, LRU eviction)
- **CUDA-graph decode**
- an OpenAI-style HTTP server with streaming

It does **not** use HuggingFace's modeling code. The Qwen2 transformer is written by hand so that
attention can read K/V straight out of the paged cache. Greedy outputs are tested to match HF
`generate()` token for token.

On a 4 GB laptop GPU it produces **9× the throughput of batched HF `generate()`** and **68× that
of one-request-at-a-time `generate()`**:

| system (RTX 3050 Laptop, 4 GB) | output tok/s | vs HF batch=1 |
|---|---:|---:|
| HF `generate()`, batch = 1 | 17 | 1.0× |
| HF `generate()`, static batch = 16 | 132 | 7.6× |
| mini-vllm, eager | 834 | 48× |
| **mini-vllm + CUDA graphs** | **1,185** | **68×** |

<sub>128 requests, prompts 32–384 tokens, outputs 16–256 tokens, greedy, fp16. Full results in
[`results/benchmark.md`](results/benchmark.md).</sub>

---

## Contents

- [Quick start](#quick-start)
- [The concepts](#the-concepts)
  - [1. The KV cache](#1-the-kv-cache)
  - [2. Memory fragmentation, and why paging fixes it](#2-memory-fragmentation-and-why-paging-fixes-it)
  - [3. PagedAttention](#3-pagedattention)
  - [4. Batching schedulers: static vs continuous](#4-batching-schedulers-static-vs-continuous)
  - [5. Prefix caching](#5-prefix-caching)
  - [6. CUDA graphs](#6-cuda-graphs-the-cpu-is-the-bottleneck-now)
- [Benchmarks](#benchmarks)
- [Making it fast: what the profiler showed](#making-it-fast-what-the-profiler-showed)
- [Code map](#code-map)
- [Limitations and next steps](#limitations-and-next-steps)

---

## Quick start

Requires Python 3.10+ and an NVIDIA GPU. It was developed on Windows 11 with an RTX 3050
4 GB, CUDA 12.8 and torch 2.11.

```bash
git clone https://github.com/<you>/mini-vllm.git
cd mini-vllm
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate

# Install the CUDA build of torch first (pick the index URL that matches your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[server,dev]"
```

**Offline generation**

```python
from minivllm import LLMEngine, SamplingParams

engine = LLMEngine("Qwen/Qwen2.5-0.5B-Instruct")      # downloads ~1 GB on first run
seqs = engine.generate(["The capital of France is", "def fibonacci(n):"],
                       SamplingParams(max_tokens=64))
for s in seqs:
    print(engine.decode(s))
```

**Server**

```bash
python -m minivllm.server --port 8000          # or: minivllm-serve

curl -X POST localhost:8000/v1/completions -H "content-type: application/json" \
     -d '{"prompt": "Explain the KV cache.", "chat": true, "max_tokens": 64, "stream": true}'
curl localhost:8000/stats                       # running/waiting, free blocks, prefix hit rate

python examples/client_demo.py -n 32            # 32 concurrent requests share one batch
```

**Tests and benchmark**

```bash
pytest -m "not gpu"            # allocator / scheduler logic, no GPU needed
pytest                         # also checks token-for-token parity with HF generate()
python benchmarks/benchmark.py # writes results/benchmark.md (~5 min)
```

---

## The concepts

### 1. The KV cache

A decoder generates one token at a time, and each new token attends to **every previous
token**. In layer *l*, attention needs a key and a value vector for every past position.
Those don't change once computed, so recomputing them each step (O(n²) work) is wasted. They
are computed once, stored, and reused. That store is the **KV cache**.

Inference therefore has two phases with very different costs:

| phase | input | what it does | bound by |
|---|---|---|---|
| **prefill** | the whole prompt (hundreds of tokens) | computes K/V for all prompt tokens in parallel | compute (big matmuls) |
| **decode** | 1 new token per sequence | appends 1 K/V per layer, attends over the cache | memory bandwidth |

In decode, every step reads all the model weights (~1 GB) to produce **one** token per sequence.
The weights cost the same whether 1 or 100 sequences share the step, so **throughput comes from
batching many sequences together**. How many fit is limited by KV-cache memory, which is why the
rest of this project is about managing that memory.

Size per token for Qwen2.5-0.5B (24 layers, 2 KV heads, head dim 64, fp16):

```
2 (K and V) × 24 layers × 2 kv_heads × 64 dims × 2 bytes = 12 KiB per token
```

Qwen uses **grouped-query attention (GQA)**: 14 query heads share 2 KV heads. Without GQA the cache
would need 84 KiB per token, 7× more. On this GPU, 0.9 GB of cache holds about 79k tokens.

### 2. Memory fragmentation, and why paging fixes it

The simplest approach gives each request a **contiguous** KV buffer. You don't know in advance
how long the output will be, so you reserve the maximum. That wastes memory in three ways:

- **Over-reservation**: a request that stops after 40 tokens still holds space for 32k.
- **Internal fragmentation**: reserved slots inside an allocation that are never used.
- **External fragmentation**: free memory gets split into gaps between live buffers. Even when
  the total free memory is enough, no single gap may be large enough for a new request.

HF `generate()` with a batch has a related problem. Every row is padded to the longest prompt and
keeps running until the **longest** output finishes.

**Paging** fixes all of this the same way an OS fixes it for RAM. The cache is one big pool of
fixed-size **blocks** of 16 tokens each. A sequence owns a *list* of blocks, its **block
table**, and the blocks don't need to be next to each other:

```
logical view (seq A, 37 tokens)      block table        physical pool
[tok 0-15][tok 16-31][tok 32-36 ...] -> [7, 2, 9]  ->   | 0 | 1 | 2:A | 3 | ... | 7:A | 8 | 9:A |
```

- Any free block can serve any sequence, so **external fragmentation disappears**.
- Blocks are allocated **on demand** as the sequence grows, so nothing is over-reserved.
- Only the last block can be partly empty, so waste is **less than one block per sequence**.

Measured on the benchmark workload (`results/benchmark.md`):

| allocation scheme | KV slots reserved | waste |
|---|---:|---:|
| contiguous, reserve max length (32k) | 4,194,304 | 98.9% |
| HF static batching (pad to longest) | 75,696 | 41.1% |
| **paged, 16-token blocks** | **45,568** | **2.1%** |
| ideal | 44,608 | 0% |

The same GPU memory holds about 1.7× more live sequences than with padded batching. More sequences
per step means more throughput.

### 3. PagedAttention

Once the cache is paged, attention has to follow the block table to find each sequence's K/V.
Each layer does three things ([`model.py`](minivllm/model.py)):

```python
# 1. paged write: put this step's new K/V into their slots (slot = block * 16 + offset)
k_cache.index_copy_(1, slot_mapping, k)
# 2. paged read: gather each sequence's blocks through its block table
kg = k_cache.index_select(1, block_tables.flatten()).view(nkv, B, nb * 16, hd)
# 3. masked attention (causal + "only this sequence's real tokens")
```

The engine flattens tokens from all sequences in the step into one batch. It builds a
`slot_mapping`, a padded `block_tables` tensor and a mask (`AttnMeta`), and the **same code path
handles both prefill and decode**. A prefill whose prefix is cached simply has fewer queries
than keys.

> Real PagedAttention kernels (vLLM, FlashInfer, FlashAttention-varlen) read the blocks *in
> place* inside the attention kernel. Step 2 here is an explicit gather, so it costs one extra
> copy. That keeps the code readable and runs anywhere torch runs; FlashAttention and Triton
> aren't easily available on Windows. The algorithm and the memory management are the same.

### 4. Batching schedulers: static vs continuous

**Static batching** (HF `generate()` on a batch) forms a batch, runs it until *every* member is
done, and then forms the next one:

```
static:      [A A A A A A A A]  B finished at step 2, but its slot sits idle;
             [B B . . . . . .]  C, waiting in the queue, can't start
             [C C C C . . . .]  until A is done
continuous:  [A A A A A A A A]
             [B B D D D D E E]  <- D joins the moment B leaves
             [C C C C F F F F]
```

In the benchmark, **43% of the tokens HF computed were wasted** on finished requests waiting for
the longest one in their batch.

**Continuous batching** (also called iteration-level scheduling, from the
[Orca paper](https://www.usenix.org/conference/osdi22/presentation/yu)) decides the batch again
on *every forward step*. The scheduler is in [`scheduler.py`](minivllm/scheduler.py):

1. **Admit**: if requests are waiting and their blocks fit, run a **prefill step** for as many
   as the token budget (`max_num_batched_tokens`) allows. Prefix-cache hits don't count against
   the budget.
2. Otherwise run a **decode step**: every running sequence produces one token. A sequence that
   just filled its last block gets a new one.
3. **Preempt** if the pool runs out mid-decode. Evict the most recently admitted sequence, free
   its blocks, and put it at the front of the queue. It is later recomputed from its prompt plus
   output so far ("recompute" preemption; vLLM can also swap blocks to CPU). Its blocks keep
   their prefix hashes, so recomputing is often cheap.
4. Finished sequences leave right away and return their blocks.

In the server ([`minivllm/server.py`](minivllm/server.py)), one engine thread runs `step()` in a
loop. HTTP handlers just queue requests, so a request that arrives while 50 others are decoding
joins at the next iteration.

### 5. Prefix caching

Chat traffic repeats itself: the same system prompt, the same few-shot examples, the same
conversation history. Their K/V are identical, so they don't need recomputing.
[`block_manager.py`](minivllm/block_manager.py) does what vLLM calls *automatic prefix caching*:

- Each **full** block gets a hash, `hash(parent_block_hash, its 16 token ids)`. Including the
  parent hash makes it identify the *whole prefix* up to that block, not just those 16 tokens.
- When a request arrives, its leading full blocks are looked up. Hits are shared by
  **reference count**, with no copy and no recompute. Only the rest of the prompt is computed.
- When a sequence finishes, its hashed blocks **stay in the cache** at the back of an LRU free
  list. They are evicted (and their hash dropped) only when a fresh block is actually needed.
- Copy-on-write isn't needed: only *full* blocks are shared, and a sequence only writes into its
  last, partially filled, private block.

64 requests sharing a 1,024-token system prompt:

| | prefill tokens computed | mean time to first token | total time |
|---|---:|---:|---:|
| prefix caching off | 66,724 | 6,679 ms | 14.8 s |
| **prefix caching on** | **2,212** (97% hit) | **528 ms** | **2.1 s** |

### 6. CUDA graphs: the CPU is the bottleneck now

A decode step for a 0.5B model launches about **1,300 small GPU kernels**. On this laptop the GPU
finishes the work in about 18 ms, but Python and the CUDA driver take about **67 ms just to
launch** them (around 19 µs each). The GPU sits idle waiting on the CPU.

A **CUDA graph** records the whole forward pass once and replays it with **one launch**. Replay
needs fixed shapes and fixed memory addresses, so [`DecodeGraphs`](minivllm/engine.py):

- writes inputs into persistent buffers before each replay,
- pads the batch up to a bucket size (1, 2, 4, 8, 16, 24, 32, 48, 64, …) and the block-table width
  up to a block bucket,
- sends padding rows' K/V writes to a reserved **scratch block** so they can't corrupt real
  sequences,
- captures one graph per (batch bucket, block bucket) on first use, all sharing one memory pool.

Result: about 55 ms down to **about 22 ms per decode step**, and +42% end-to-end throughput
(834 to 1,185 tok/s). Prefill still runs eagerly because its big matmuls amortize launch cost.

---

## Benchmarks

`python benchmarks/benchmark.py` runs three experiments. Every system must emit **exactly** the
requested number of tokens per request (EOS ignored), so all systems do the same useful work.
HF's built-in `repetition_penalty=1.05` from Qwen's generation config is turned off so the
comparison (and the parity test) is like-for-like.

| system | time (s) | output tok/s |
|---|---:|---:|
| HF `generate()`, batch = 1 (12-request subset) | 108.4 | 17 |
| HF `generate()`, static batch = 16 | 131.5 | 132 |
| mini-vllm, eager | 20.9 | 834 |
| mini-vllm + CUDA graphs | 14.7 | 1,185 |

mini-vllm ran 269 engine steps (14 prefill, 255 decode) with an average decode batch of 67.8
sequences. The KV pool peaked at 1,919 of 4,937 blocks, with no preemptions.

Why mini-vllm wins, from largest effect to smallest:

1. **Continuous batching**: no wasted padding or overshoot, and new requests join immediately.
2. **Paged memory**: more sequences fit, so batches are bigger.
3. **CUDA graphs**: decode stops being launch-bound.
4. Fused QKV and gate/up matmuls, and logits computed only for the last token of each sequence.

Correctness: [`tests/test_parity.py`](tests/test_parity.py) checks that greedy outputs match HF
`generate()` in three cases: a normal batch, prefix-cached reuse, and a cache small enough to
force 7 preemptions and recomputes. All runs match HF for all 48 tokens, except that a recomputed
sequence occasionally drifts late in its output from fp16 rounding (different batch sizes pick
different matmul kernels). The test therefore requires the first 32 tokens to match exactly.

---

## Making it fast: what the profiler showed

The first working version was correct but managed only **304 tok/s**, at **220 ms per decode
step**. This is how `torch.profiler` brought it down to 22 ms:

| step | decode step (B=64) | what the profiler showed | fix |
|---|---:|---|---|
| 0 | ~100 ms GPU | 77 of 91 ms in `_scaled_dot_product_attention_math`: SDPA with `enable_gqa=True` **and** a mask falls back to an fp32 path that `repeat_interleave`s K/V 7× | Fold GQA into the matmul: the 7 query heads sharing a KV head become 7 rows of one `[g·Q, d] @ [d, L]` product, so each KV head is read once |
| 1 | 21 ms GPU, ~45 ms wall | CPU-bound: kernel launches take longer than the GPU work | CUDA graphs for decode |
| 2 | 35 ms (graph) | gather of blocks 8.5 ms (slow generic index kernel) plus a 7.7 ms hidden `.contiguous()` copy, because the gathered K was `[B, L, kv, d]` and the matmul wanted `[kv, B, L, d]` | Change the cache layout to `[kv_head, block, slot, d]`: gathering whole blocks with `index_select` copies contiguous 2 KiB chunks and is already in matmul layout |
| 3 | **22 ms** | - | - |

Two things were also wrong in the HF comparison at first. HF's Qwen generation config applies
`repetition_penalty=1.05` even with `do_sample=False`, so outputs diverged after a few tokens
until it was set to 1.0. And the weights were briefly materialized in fp32 on the GPU while
loading, which left only 0.25 GB for the KV cache. Casting while still on the `meta` device fixed
that.

---

## Code map

```
minivllm/
  sequence.py       Sequence (one request: tokens, block table, #computed tokens), SamplingParams
  block_manager.py  paged allocator: block pool, ref counts, prefix-cache hashing, LRU eviction
  scheduler.py      continuous batching: admit / decode / preempt, token budget
  model.py          hand-written Qwen2 / Llama decoder with paged attention, weight loader
  engine.py         LLMEngine.step(): schedule -> build metadata -> forward -> sample -> bookkeeping
                    DecodeGraphs: bucketed CUDA-graph capture / replay for decode
  server.py         FastAPI server, OpenAI-style /v1/completions (+ SSE streaming), /stats
benchmarks/
  benchmark.py      throughput vs HF, prefix caching on/off, fragmentation report
examples/
  offline_generate.py   batch generation in a few lines
  client_demo.py        concurrent HTTP load to watch continuous batching
tests/
  test_block_manager.py prefix sharing, LRU eviction, preemption (CPU only)
  test_parity.py        token-for-token parity with HF generate() (GPU)
results/
  benchmark.md / .json  numbers quoted in this README
```

The life of one engine step (`LLMEngine.step`):

```
scheduler.schedule()                 -> which sequences run; prefill or decode
  block_manager.try_allocate(seq)       new request: reuse cached prefix blocks, allocate the rest
  block_manager.append_slot(seq)        decode: new block when the last one is full (or preempt)
engine._build(seqs) / DecodeGraphs   -> flat token ids, positions, slot_mapping, block_tables, mask
model(ids, meta, kv_cache)           -> per layer: write K/V to slots, gather blocks, attention
engine._sample(logits)               -> greedy / temperature + top-p
bookkeeping                          -> commit newly full blocks to the prefix cache,
                                        append tokens, finish on EOS / max_tokens, free blocks
```

---

## Limitations and next steps

This project is for learning, not production. Known gaps, roughly in order of payoff:

- **A real PagedAttention kernel** (Triton or CUDA) that reads blocks in place instead of
  gathering them.
- **Chunked prefill**: long prompts currently stall decodes for a whole step. Splitting them into
  chunks mixed into decode steps keeps time between tokens smooth.
- **Batched sampling on the GPU** with per-request top-k and penalties. Only greedy and
  temperature + top-p are implemented.
- **Swap-based preemption** (to CPU memory) as an alternative to recompute.
- Tensor parallelism, quantized weights and KV cache (int8/fp8), speculative decoding.
- Only the Qwen2/Llama architecture family is supported, tested on Qwen2.5-0.5B-Instruct.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*
  (SOSP 2023) - [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*
  (OSDI 2022) - continuous batching
- [vLLM](https://github.com/vllm-project/vllm) and
  [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), another compact re-implementation worth reading
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)

## License

MIT, see [LICENSE](LICENSE).
