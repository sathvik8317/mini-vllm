"""LLMEngine: glues scheduler + block manager + model together.

One call to `step()` = one forward pass over whatever the scheduler picked:

    seqs, is_prefill = scheduler.schedule()     # who runs this iteration
    meta = build_metadata(seqs)                 # flat tokens + block tables + mask
    logits = model(tokens, meta, kv_cache)      # writes new K/V into the paged cache
    next_tokens = sample(logits)
    bookkeeping: commit full blocks to the prefix cache, append tokens, finish seqs
"""
import os
import time

import torch

from .block_manager import BlockManager
from .model import AttnMeta, load_model
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence, Status


def resolve_model_dir(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


class LLMEngine:
    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", block_size=16, num_gpu_blocks=None,
                 kv_cache_memory_gb=None, max_num_seqs=128, max_num_batched_tokens=2048,
                 enable_prefix_caching=True, dtype=torch.float16, device="cuda", verbose=False,
                 enforce_eager=False, cuda_graph_max_blocks=128):
        from transformers import AutoTokenizer
        self.model_dir = resolve_model_dir(model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model, cfg = load_model(self.model_dir, dtype, device)
        self.device, self.dtype, self.verbose = device, dtype, verbose

        L, nkv = cfg["num_hidden_layers"], cfg["num_key_value_heads"]
        hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        self.bytes_per_token = 2 * L * nkv * hd * torch.finfo(dtype).bits // 8   # K and V, all layers
        if num_gpu_blocks is None:
            if kv_cache_memory_gb is None:
                free, _ = torch.cuda.mem_get_info()
                # leave room for activations / the gathered K,V of a full prefill
                kv_cache_memory_gb = max(0.25, (free - 1100 * 2**20) / 2**30)
            num_gpu_blocks = int(kv_cache_memory_gb * 2**30 // (self.bytes_per_token * block_size))
        self.block_size, self.num_gpu_blocks = block_size, num_gpu_blocks
        # ONE contiguous allocation for the whole cache, carved into blocks.
        # +1 = a scratch block never handed out: CUDA-graph padding rows write their K/V there.
        self.kv_cache = torch.zeros(L, 2, nkv, num_gpu_blocks + 1, block_size, hd, dtype=dtype, device=device)
        self.scratch_slot = num_gpu_blocks * block_size

        self.block_manager = BlockManager(num_gpu_blocks, block_size, enable_prefix_caching)
        self.scheduler = Scheduler(self.block_manager, max_num_seqs, max_num_batched_tokens)
        self.graphs = None if enforce_eager else DecodeGraphs(self, max_num_seqs, cuda_graph_max_blocks)

        gc = cfg.get("eos_token_id")
        eos = set(gc if isinstance(gc, list) else [gc])
        gen_cfg = os.path.join(self.model_dir, "generation_config.json")
        if os.path.exists(gen_cfg):
            import json
            g = json.load(open(gen_cfg)).get("eos_token_id")
            eos |= set(g if isinstance(g, list) else [g])
        self.eos_ids = {e for e in eos if e is not None}

        self.stats = {"steps": 0, "prefill_steps": 0, "decode_steps": 0,
                      "prefill_tokens": 0, "decode_tokens": 0, "peak_blocks": 0,
                      "frag_samples": []}
        if verbose:
            print(f"[engine] KV cache: {num_gpu_blocks} blocks x {block_size} tok "
                  f"= {num_gpu_blocks * block_size} tokens, "
                  f"{self.kv_cache.numel() * self.kv_cache.element_size() / 2**30:.2f} GiB "
                  f"({self.bytes_per_token / 1024:.1f} KiB/token)")

    # ------------------------------------------------------------------ requests
    def add_request(self, prompt, params: SamplingParams | None = None) -> Sequence:
        params = params or SamplingParams()
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        max_ctx = self.num_gpu_blocks * self.block_size
        if len(ids) + params.max_tokens > max_ctx:
            raise ValueError(f"request needs {len(ids) + params.max_tokens} tokens; cache holds {max_ctx}")
        seq = Sequence(ids, params)
        self.scheduler.add(seq)
        return seq

    def has_unfinished(self):
        return self.scheduler.has_unfinished()

    # ------------------------------------------------------------------ one iteration
    def _build(self, seqs):
        bs = self.block_size
        tokens, positions, slots, pad_index, last_idx = [], [], [], [], []
        q_lens, ctx_lens = [], []
        max_q = max(len(s) - s.num_computed_tokens for s in seqs)
        for b, s in enumerate(seqs):
            start, end = s.num_computed_tokens, len(s)
            tokens += s.token_ids[start:end]
            pos = range(start, end)
            positions += pos
            bt = s.block_table
            slots += [bt[p // bs] * bs + p % bs for p in pos]
            pad_index += range(b * max_q, b * max_q + (end - start))
            last_idx.append(len(tokens) - 1)
            q_lens.append(end - start)
            ctx_lens.append(end)

        nb = max(len(s.block_table) for s in seqs)
        tables = [s.block_table + [0] * (nb - len(s.block_table)) for s in seqs]

        dev = self.device
        T = lambda x: torch.tensor(x, dtype=torch.long).to(dev, non_blocking=True)
        q_lens_t, ctx_t = T(q_lens), T(ctx_lens)
        # query i of seq b sits at absolute position ctx_len - q_len + i
        qi = torch.arange(max_q, device=dev)
        qpos = (ctx_t - q_lens_t)[:, None] + qi[None, :]                  # [B, max_q]
        qpos = torch.where(qi[None, :] < q_lens_t[:, None], qpos, 0)       # padded queries -> see key 0 only
        kpos = torch.arange(nb * bs, device=dev)
        mask = (kpos[None, None, :] <= qpos[:, :, None]) & (kpos[None, None, :] < ctx_t[:, None, None])
        meta = AttnMeta(positions=T(positions), slot_mapping=T(slots), block_tables=T(tables),
                        pad_index=T(pad_index), mask=mask[:, None], batch_size=len(seqs),
                        max_q=max_q, last_token_index=T(last_idx))
        return T(tokens), meta

    def _sample(self, logits, seqs):
        temps = [s.params.temperature for s in seqs]
        if all(t == 0 for t in temps):
            return logits.argmax(-1).tolist()
        t = torch.tensor([max(x, 1e-5) for x in temps], device=logits.device)[:, None]
        probs = torch.softmax(logits / t, -1)
        top_p = torch.tensor([s.params.top_p for s in seqs], device=logits.device)[:, None]
        sp, si = probs.sort(-1, descending=True)
        sp[(sp.cumsum(-1) - sp) > top_p] = 0                       # nucleus filter
        choice = si.gather(-1, torch.multinomial(sp, 1)).squeeze(-1)
        greedy = logits.argmax(-1)
        is_greedy = torch.tensor([x == 0 for x in temps], device=logits.device)
        return torch.where(is_greedy, greedy, choice).tolist()

    def step(self) -> list[Sequence]:
        """Run one iteration. Returns sequences that finished during it."""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []
        if not is_prefill and self.graphs is not None and self.graphs.can_run(seqs):
            logits = self.graphs.run(seqs)
            n_new = len(seqs)
        else:
            input_ids, meta = self._build(seqs)
            logits = self.model(input_ids, meta, self.kv_cache)
            n_new = int(input_ids.numel())
        next_tokens = self._sample(logits, seqs)

        st = self.stats
        st["steps"] += 1
        if is_prefill:
            st["prefill_steps"] += 1
            st["prefill_tokens"] += n_new
        else:
            st["decode_steps"] += 1
            st["decode_tokens"] += n_new

        now = time.perf_counter()
        finished = []
        for s, tok in zip(seqs, next_tokens):
            s.num_computed_tokens = len(s)
            self.block_manager.commit(s)
            s.token_ids.append(tok)
            if s.first_token_time is None:
                s.first_token_time = now
            p = s.params
            if (not p.ignore_eos and tok in self.eos_ids) or s.num_output_tokens >= p.max_tokens:
                s.finish_reason = "stop" if tok in self.eos_ids and not p.ignore_eos else "length"
                s.finish_time = now
                self.scheduler.finish(s)
                finished.append(s)

        used = self.num_gpu_blocks - self.block_manager.num_free_blocks()
        st["peak_blocks"] = max(st["peak_blocks"], used)
        if st["steps"] % 8 == 0 and self.scheduler.running:
            st["frag_samples"].append(self.block_manager.usage(self.scheduler.running))
        return finished

    # ------------------------------------------------------------------ offline API
    def generate(self, prompts, params=None, progress=False):
        """Batch API: submit everything, run continuous batching to completion."""
        if not isinstance(params, list):
            params = [params or SamplingParams()] * len(prompts)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        t0 = time.perf_counter()
        done = 0
        while self.has_unfinished():
            done += len(self.step())
            if progress and self.stats["steps"] % 50 == 0:
                print(f"  step {self.stats['steps']}: {done}/{len(seqs)} done, "
                      f"running={len(self.scheduler.running)} waiting={len(self.scheduler.waiting)}")
        torch.cuda.synchronize()
        self.last_elapsed = time.perf_counter() - t0
        return seqs

    def decode(self, seq: Sequence) -> str:
        return self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = [] if isinstance(self.stats[k], list) else 0
        bm = self.block_manager
        bm.prompt_tokens_seen = bm.prompt_tokens_hit = bm.num_evictions = 0
        self.scheduler.num_preemptions = 0


class DecodeGraphs:
    """CUDA graphs for decode steps.

    A decode step for a 0.5B model launches ~300 tiny kernels; Python + launch overhead
    (~40 ms on this laptop) dwarfs the ~20 ms of actual GPU work. A CUDA graph records
    the whole forward once and replays it with a single launch.

    Graphs need static shapes and static memory addresses, so:
      * inputs live in persistent buffers that we overwrite before each replay,
      * the batch is padded up to a bucket size (1, 2, 4, 8, 16, 24, 32, ...),
      * the block-table width is padded up to a bucket (4, 8, 16, ... blocks),
      * padding rows write K/V to a dedicated scratch slot and attend only to key 0.
    One graph per (batch bucket, block bucket), captured lazily on first use,
    all sharing one memory pool. vLLM does the same (it captures them all at startup).
    """

    def __init__(self, eng, max_bs, max_blocks):
        self.eng, self.bs = eng, eng.block_size
        self.batch_buckets = [b for b in [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256]
                              if b < max_bs] + [max_bs]
        self.block_buckets = [n for n in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256] if n < max_blocks] + [max_blocks]
        self.max_blocks = max_blocks
        dev, long = eng.device, torch.long
        self.ids = torch.zeros(max_bs, dtype=long, device=dev)
        self.pos = torch.zeros(max_bs, dtype=long, device=dev)
        self.slots = torch.full((max_bs,), eng.scratch_slot, dtype=long, device=dev)
        self.tables = torch.zeros(max_bs, max_blocks, dtype=long, device=dev)
        self.mask = torch.zeros(max_bs, 1, 1, max_blocks * self.bs, dtype=torch.bool, device=dev)
        self.mask[:, ..., 0] = True
        self.last = torch.arange(max_bs, device=dev)
        self.kpos = torch.arange(max_blocks * self.bs, device=dev)
        self.graphs, self.outputs, self.pool = {}, {}, None

    def can_run(self, seqs):
        return max(len(s.block_table) for s in seqs) <= self.max_blocks

    def _meta(self, B, NB):
        return AttnMeta(positions=self.pos[:B], slot_mapping=self.slots[:B],
                        block_tables=self.tables[:B, :NB], pad_index=None,
                        mask=self.mask[:B, :, :, :NB * self.bs], batch_size=B, max_q=1,
                        last_token_index=self.last[:B])

    def _capture(self, B, NB):
        model, kv = self.eng.model, self.eng.kv_cache
        meta = self._meta(B, NB)
        self.slots.fill_(self.eng.scratch_slot)        # warmup/capture must not touch real blocks
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model(self.ids[:B], meta, kv)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self.outputs[(B, NB)] = model(self.ids[:B], meta, kv)
        self.pool = g.pool()
        self.graphs[(B, NB)] = g

    def run(self, seqs):
        n = len(seqs)
        B = next(b for b in self.batch_buckets if b >= n)
        NB = next(x for x in self.block_buckets if x >= max(len(s.block_table) for s in seqs))
        if (B, NB) not in self.graphs:
            self._capture(B, NB)
        bs = self.bs
        ids, pos, slots, ctx = [], [], [], []
        rows = []
        for s in seqs:
            p = len(s) - 1
            ids.append(s.token_ids[p]); pos.append(p)
            slots.append(s.block_table[p // bs] * bs + p % bs)
            ctx.append(p + 1)
            rows.append(s.block_table + [0] * (NB - len(s.block_table)))
        pad = B - n
        dev = self.eng.device
        T = lambda x: torch.tensor(x, dtype=torch.long).to(dev, non_blocking=True)
        self.ids[:B].copy_(T(ids + [0] * pad))
        self.pos[:B].copy_(T(pos + [0] * pad))
        self.slots[:B].copy_(T(slots + [self.eng.scratch_slot] * pad))
        self.tables[:B, :NB].copy_(T(rows + [[0] * NB] * pad))
        ctx_t = T(ctx + [1] * pad)
        self.mask[:B, 0, 0, :NB * bs].copy_(self.kpos[None, :NB * bs] < ctx_t[:, None])
        self.graphs[(B, NB)].replay()
        return self.outputs[(B, NB)][:n]
