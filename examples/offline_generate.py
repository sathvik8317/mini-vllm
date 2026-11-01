"""Minimal offline usage: submit a batch of prompts, let continuous batching run them."""
from minivllm import LLMEngine, SamplingParams

engine = LLMEngine("Qwen/Qwen2.5-0.5B-Instruct", verbose=True)
tok = engine.tokenizer

questions = ["What is a KV cache?", "Why does paging reduce memory fragmentation?",
             "What is continuous batching?", "Write a haiku about GPUs."]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True) for q in questions]

for q, seq in zip(questions, engine.generate(prompts, SamplingParams(max_tokens=96))):
    print(f"\n### {q}\n{engine.decode(seq)}")

bm = engine.block_manager
print(f"\nsteps={engine.stats['steps']}  prefix-cache hit rate="
      f"{bm.prompt_tokens_hit / max(1, bm.prompt_tokens_seen):.0%}")
