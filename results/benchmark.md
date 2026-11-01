# mini-vllm benchmark

GPU: NVIDIA GeForce RTX 3050 Laptop GPU  
Model: Qwen/Qwen2.5-0.5B-Instruct  
Workload: 128 requests, prompts 32-384 tok, outputs 16-256 tok (17422 output tokens), greedy, EOS ignored

## 1. Throughput

| system | time (s) | output tok/s | speed-up vs HF seq |
|---|---:|---:|---:|
| HF generate, batch=1 (12 req subset) | 108.4 | 17 | 1.0x |
| HF generate, static batch=16 | 131.5 | 132 | 7.6x |
| mini-vllm, eager (paged + continuous batching) | 20.9 | 834 | 48.0x |
| mini-vllm + CUDA graphs for decode | 14.7 | 1185 | 68.3x |

mini-vllm: 269 steps (14 prefill / 255 decode), avg decode batch 67.8, peak 1919/4937 blocks, 0 preemptions, mean TTFT 2.19s
HF static: 43% of generated tokens were overshoot past a request's own length

## 2. Prefix caching (64 requests sharing a 1024-token prefix, 32 output tok each)

| | time (s) | prefill tokens computed | mean TTFT (ms) | hit rate |
|---|---:|---:|---:|---:|
| off | 14.80 | 66724 | 6679 | 0% |
| on | 2.12 | 2212 | 528 | 97% |

## 3. KV-cache slots reserved for the same workload

| allocation scheme | slots reserved | waste |
|---|---:|---:|
| contiguous, reserve max_model_len | 4,194,304 | 98.9% |
| HF static batch (padding to longest) | 75,696 | 41.1% |
| paged, block_size=16 | 45,568 | 2.1% |
| ideal (exact) | 44,608 | 0.0% |

Measured live internal fragmentation of the paged cache during the run: 2.3%
