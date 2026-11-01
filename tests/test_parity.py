"""Greedy outputs of mini-vllm must match HF generate() token-for-token (modulo fp16 noise).

Also exercises the paged/prefix paths: the second run reuses cached prefix blocks,
and a tiny cache forces preemption + recompute - outputs must not change.
"""
import pytest
import torch
from transformers import AutoModelForCausalLM
from minivllm import LLMEngine, SamplingParams

SHARED = ("You are a helpful assistant running inside a tiny inference engine built for learning. "
          "Answer briefly and accurately, and never make up facts you are unsure about. Now continue: ")
PROMPTS = [SHARED + p for p in [
    "The capital of France is",
    "Write a haiku about GPU memory:",
    "def fibonacci(n):",
    "Explain the KV cache in one sentence. The KV cache",
    "1, 2, 3, 4, 5,",
]] + ["Unshared short prompt:"]
N = 48


def prefix_match(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_matches_hf_generate():
    eng = LLMEngine(verbose=True)
    tok = eng.tokenizer
    hf = AutoModelForCausalLM.from_pretrained(eng.model_dir, dtype=torch.float16).cuda().eval()
    ref = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        out = hf.generate(ids, max_new_tokens=N, do_sample=False, repetition_penalty=1.0)
        ref.append(out[0, ids.shape[1]:].tolist())
    del hf; torch.cuda.empty_cache()

    sp = SamplingParams(max_tokens=N, ignore_eos=True)
    def check(label, seqs):
        ok = 0
        for s, r in zip(seqs, ref):
            m = prefix_match(s.output_token_ids, r[:N])
            ok += m >= min(len(r), 32)          # fp16 kernels differ; first 32 tokens must agree
            print(f"  [{label}] match {m:2d}/{len(r)}  {eng.decode(s)[:60]!r}")
        assert ok == len(PROMPTS), f"{label}: only {ok}/{len(PROMPTS)} matched"

    check("batched", eng.generate(PROMPTS, sp))
    check("prefix-cached", eng.generate(PROMPTS, sp))
    hits = eng.block_manager.prompt_tokens_hit
    print("  prefix hits:", hits, "tokens")
    assert hits > 0, "prefix cache was never exercised"

    small = LLMEngine(num_gpu_blocks=12, block_size=16)     # 192 slots for 5 x ~60 tokens
    seqs = small.generate(PROMPTS, sp)
    print("  preemptions with tiny cache:", small.scheduler.num_preemptions)
    eng = small
    check("preempted", seqs)
    print("PARITY OK")


if __name__ == "__main__":
    test_matches_hf_generate()
