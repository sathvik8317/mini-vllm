"""OpenAI-style inference server on top of LLMEngine.

The engine is single-threaded and owns the GPU. HTTP handlers never touch it.
They put requests on a thread-safe inbox, and one background thread runs:

    loop:
        drain inbox -> engine.add_request(...)      # new requests join *mid-flight*
        engine.step()                               # one batched forward pass
        push newly generated tokens to each request's asyncio queue

That loop is continuous batching: a request arriving while 50 others are
decoding joins at the very next iteration instead of waiting for them to finish.

Run:   python -m minivllm.server --port 8000      (or: minivllm-serve)
Query: curl -X POST localhost:8000/v1/completions -H "content-type: application/json" \
            -d '{"prompt": "Hello", "max_tokens": 32}'
"""
import argparse
import asyncio
import json
import queue
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from minivllm import LLMEngine, SamplingParams


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False
    chat: bool = False          # wrap the prompt in the model's chat template


class EngineWorker:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.inbox: queue.Queue = queue.Queue()
        self.subs = {}                       # seq.id -> (asyncio loop, asyncio.Queue, n_sent)
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, prompt, params, loop):
        q = asyncio.Queue()
        self.inbox.put((prompt, params, loop, q))
        return q

    def _run(self):
        eng = self.engine
        while True:
            # block when idle, otherwise just drain whatever arrived
            try:
                block = not eng.has_unfinished()
                while True:
                    prompt, params, loop, q = self.inbox.get(block=block, timeout=None if block else 0)
                    block = False
                    try:
                        seq = eng.add_request(prompt, params)
                        self.subs[seq.id] = [loop, q, 0, seq]
                    except ValueError as e:
                        loop.call_soon_threadsafe(q.put_nowait, {"error": str(e)})
            except queue.Empty:
                pass

            finished = {s.id for s in eng.step()}
            for sid, sub in list(self.subs.items()):
                loop, q, sent, seq = sub
                new = seq.output_token_ids[sent:]
                done = sid in finished
                if new or done:
                    sub[2] = len(seq.output_token_ids)
                    msg = {"tokens": new, "done": done, "finish_reason": seq.finish_reason}
                    loop.call_soon_threadsafe(q.put_nowait, msg)
                if done:
                    del self.subs[sid]


def build_app(engine: LLMEngine) -> FastAPI:
    app = FastAPI(title="mini-vllm")
    worker = EngineWorker(engine)
    tok = engine.tokenizer

    def make_prompt(r: CompletionRequest):
        if r.chat:
            return tok.apply_chat_template([{"role": "user", "content": r.prompt}],
                                           tokenize=False, add_generation_prompt=True)
        return r.prompt

    @app.post("/v1/completions")
    async def completions(r: CompletionRequest):
        params = SamplingParams(temperature=r.temperature, top_p=r.top_p,
                                max_tokens=r.max_tokens, ignore_eos=r.ignore_eos)
        q = worker.submit(make_prompt(r), params, asyncio.get_running_loop())
        rid, t0 = f"cmpl-{uuid.uuid4().hex[:12]}", time.perf_counter()

        if r.stream:
            async def gen():
                ids, text_so_far = [], ""
                while True:
                    msg = await q.get()
                    if "error" in msg:
                        yield f"data: {json.dumps(msg)}\n\n"; break
                    ids += msg["tokens"]
                    text = tok.decode(ids, skip_special_tokens=True)
                    delta, text_so_far = text[len(text_so_far):], text
                    chunk = {"id": rid, "object": "text_completion",
                             "choices": [{"index": 0, "text": delta,
                                          "finish_reason": msg["finish_reason"] if msg["done"] else None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if msg["done"]:
                        break
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        ids, ttft = [], None
        while True:
            msg = await q.get()
            if "error" in msg:
                return JSONResponse(msg, status_code=400)
            if ttft is None:
                ttft = time.perf_counter() - t0
            ids += msg["tokens"]
            if msg["done"]:
                break
        return {"id": rid, "object": "text_completion",
                "choices": [{"index": 0, "text": tok.decode(ids, skip_special_tokens=True),
                             "finish_reason": msg["finish_reason"]}],
                "usage": {"completion_tokens": len(ids)},
                "timing": {"ttft_s": ttft, "total_s": time.perf_counter() - t0}}

    @app.get("/stats")
    async def stats():
        bm, sch = engine.block_manager, engine.scheduler
        return {"running": len(sch.running), "waiting": len(sch.waiting),
                "blocks_free": bm.num_free_blocks(), "blocks_total": bm.num_blocks,
                "prefix_hit_rate": bm.prompt_tokens_hit / max(1, bm.prompt_tokens_seen),
                "preemptions": sch.num_preemptions,
                **{k: v for k, v in engine.stats.items() if k != "frag_samples"}}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--no-prefix-caching", action="store_true")
    a = ap.parse_args()
    eng = LLMEngine(a.model, max_num_seqs=a.max_num_seqs,
                    enable_prefix_caching=not a.no_prefix_caching, verbose=True)
    uvicorn.run(build_app(eng), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
