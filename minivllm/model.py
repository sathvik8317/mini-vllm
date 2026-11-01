"""Qwen2 / Llama-style decoder written from scratch so attention can read a *paged* KV cache.

HF's modeling code assumes each sequence's K/V is one contiguous tensor that grows
by concatenation (`past_key_values`). Here every layer instead owns a slice of a
single pre-allocated pool:

    kv_cache[layer] : [2, num_kv_heads, num_blocks, block_size, head_dim]

and a forward pass works on a *flat* batch of tokens from many sequences at once
(no padding on the token dimension). Per step the engine builds an `AttnMeta`:

    positions     [T]       absolute position of each new token (for RoPE)
    slot_mapping  [T]       flat cache slot where each new token's K/V is written
    block_tables  [B, nb]   physical block ids of each sequence (padded with 0)
    pad_index     [T]       where each flat token goes in a padded [B * max_q] layout
    mask          [B,1,max_q,max_ctx]  which cached keys each query may attend to

Attention per layer =
    1. write the new K/V into their slots           (the "paged" write)
    2. gather each sequence's blocks via its table  (the "paged" read)
    3. masked SDPA

A real PagedAttention kernel (vLLM / FlashInfer / FlashAttention-varlen) fuses
step 2 into the attention kernel so it reads the blocks in place. Doing the
gather in PyTorch costs a memory copy but is the same algorithm. That keeps
this file readable and lets it run on Windows, where those kernels are unavailable.
The same code path serves prefill (many queries per sequence, part of the keys
possibly coming from the prefix cache) and decode (one query per sequence).
"""
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


@dataclass
class AttnMeta:
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    pad_index: torch.Tensor
    mask: torch.Tensor
    batch_size: int
    max_q: int
    last_token_index: torch.Tensor   # [B] index into the flat T dim of each seq's last token


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class Rotary(nn.Module):
    def __init__(self, head_dim, theta, max_pos):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, positions):            # x: [T, heads, D]
        cos = self.cos[positions].unsqueeze(1).to(x.dtype)
        sin = self.sin[positions].unsqueeze(1).to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, cfg, rotary):
        super().__init__()
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or cfg["hidden_size"] // self.nh
        h = cfg["hidden_size"]
        bias = cfg.get("attention_bias", cfg.get("model_type") == "qwen2")
        # q, k, v fused into one matmul (fewer kernel launches - matters a lot for small models)
        self.qkv = nn.Linear(h, (self.nh + 2 * self.nkv) * self.hd, bias=bias)
        self.o = nn.Linear(self.nh * self.hd, h, bias=False)
        self.rotary = rotary
        self.scale = self.hd ** -0.5

    def forward(self, x, meta: AttnMeta, kv_cache):
        T = x.shape[0]
        q, k, v = self.qkv(x).split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], -1)
        q = self.rotary(q.view(T, self.nh, self.hd), meta.positions)
        k = self.rotary(k.view(T, self.nkv, self.hd), meta.positions)
        v = v.view(T, self.nkv, self.hd)

        k_cache, v_cache = kv_cache[0], kv_cache[1]          # [nkv, num_blocks, bs, hd]
        nblk, bs = k_cache.shape[1:3]
        # 1. paged write: scatter new K/V into their (non-contiguous) slots
        k_cache.view(self.nkv, nblk * bs, self.hd).index_copy_(1, meta.slot_mapping, k.transpose(0, 1))
        v_cache.view(self.nkv, nblk * bs, self.hd).index_copy_(1, meta.slot_mapping, v.transpose(0, 1))

        # 2. paged read: gather each sequence's blocks through its block table.
        #    Layout [nkv, block, slot, hd] means each gathered block is one contiguous
        #    2 KiB chunk, and the result is already [nkv, B, L, hd] - no transpose copy.
        B, nb = meta.block_tables.shape
        flat = meta.block_tables.reshape(-1)
        kg = k_cache.index_select(1, flat).view(self.nkv, B, nb * bs, self.hd)
        vg = v_cache.index_select(1, flat).view(self.nkv, B, nb * bs, self.hd)

        # queries: flat [T] -> padded [B, max_q]
        Q = meta.max_q
        q = q * self.scale                                       # pre-scale: keeps fp16 scores small
        if Q == 1:
            qp = q.view(B, 1, self.nh, self.hd)
        else:
            qp = q.new_zeros(B * Q, self.nh, self.hd)
            qp[meta.pad_index] = q
            qp = qp.view(B, Q, self.nh, self.hd)

        # 3. attention with GQA folded into the matmul.
        #    g = nh / nkv query heads (7 for Qwen2.5-0.5B) share one KV head, so we stack
        #    them along the row dimension: [nkv, B, g*Q, hd] @ [nkv, B, hd, L].
        #    Each KV head is read ONCE - no repeat_interleave copies of the cache.
        #    (torch SDPA with enable_gqa + a mask falls back to a slow fp32 "math" path.)
        g = self.nh // self.nkv
        qg = qp.view(B, Q, self.nkv, g, self.hd).permute(2, 0, 3, 1, 4).reshape(self.nkv, B, g * Q, self.hd)
        scores = torch.matmul(qg, kg.transpose(-1, -2))          # [nkv, B, g*Q, L]
        L = scores.shape[-1]
        scores = scores.view(self.nkv, B, g, Q, L).float()
        scores.masked_fill_(~meta.mask.view(1, B, 1, Q, L), float("-inf"))
        probs = scores.softmax(-1).to(q.dtype).view(self.nkv, B, g * Q, L)
        out = torch.matmul(probs, vg)                             # [nkv, B, g*Q, hd]
        out = out.view(self.nkv, B, g, Q, self.hd).permute(1, 3, 0, 2, 4).reshape(B * Q, self.nh * self.hd)
        if Q != 1:
            out = out[meta.pad_index]
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_up = nn.Linear(h, 2 * i, bias=False)
        self.down = nn.Linear(i, h, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Layer(nn.Module):
    def __init__(self, cfg, rotary):
        super().__init__()
        eps = cfg["rms_norm_eps"]
        self.ln1 = RMSNorm(cfg["hidden_size"], eps)
        self.attn = Attention(cfg, rotary)
        self.ln2 = RMSNorm(cfg["hidden_size"], eps)
        self.mlp = MLP(cfg)

    def forward(self, x, meta, kv_cache):
        x = x + self.attn(self.ln1(x), meta, kv_cache)
        return x + self.mlp(self.ln2(x))


class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        nh = cfg["num_attention_heads"]
        hd = cfg.get("head_dim") or cfg["hidden_size"] // nh
        self.rotary = Rotary(hd, cfg.get("rope_theta", 10000.0), cfg["max_position_embeddings"])
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Layer(cfg, self.rotary) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", False):
            self.lm_head.weight = self.embed.weight

    @torch.inference_mode()
    def forward(self, input_ids, meta: AttnMeta, kv_cache):
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, meta, kv_cache[i])
        # only the last token of each sequence needs logits: [B, vocab], not [T, vocab]
        x = self.norm(x[meta.last_token_index])
        return self.lm_head(x).float()


def load_model(model_dir: str, dtype=torch.float16, device="cuda"):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    with torch.device("meta"):
        model = DecoderLM(cfg)

    raw = {}
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith(".safetensors"):
            raw.update(load_file(os.path.join(model_dir, fn)))

    sd = {}
    for k, w in raw.items():
        k = k.removeprefix("model.")
        k = (k.replace("embed_tokens", "embed").replace("self_attn", "attn")
              .replace("input_layernorm", "ln1").replace("post_attention_layernorm", "ln2")
              .replace("o_proj", "o").replace("down_proj", "down"))
        sd[k] = w
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        for suffix in ("weight", "bias"):
            names = [f"{p}attn.{n}_proj.{suffix}" for n in "qkv"]
            if names[0] in sd:
                sd[f"{p}attn.qkv.{suffix}"] = torch.cat([sd.pop(n) for n in names])
        sd[f"{p}mlp.gate_up.weight"] = torch.cat([sd.pop(f"{p}mlp.gate_proj.weight"),
                                                  sd.pop(f"{p}mlp.up_proj.weight")])
    if cfg.get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)

    model.to(dtype)                 # cast while still on meta: avoids a transient fp32 copy on the GPU
    model.to_empty(device=device)
    missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    missing = [m for m in missing if not m.startswith("lm_head") or not cfg.get("tie_word_embeddings")]
    assert not missing and not unexpected, (missing, unexpected)
    if cfg.get("tie_word_embeddings", False):
        model.lm_head.weight = model.embed.weight
    # buffers were created on meta; rebuild rotary tables on the real device
    nh = cfg["num_attention_heads"]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // nh
    rot = Rotary(hd, cfg.get("rope_theta", 10000.0), cfg["max_position_embeddings"]).to(device)
    model.rotary.cos, model.rotary.sin = rot.cos, rot.sin     # keep RoPE tables in fp32
    torch.cuda.empty_cache()
    return model.eval(), cfg
