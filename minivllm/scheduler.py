"""Continuous (iteration-level) batching scheduler.

Static batching (HF generate on a batch) forms a batch, runs it until the
*longest* request finishes, then forms the next batch. Short requests sit idle
as padding, and new requests wait for the whole batch.

Continuous batching re-decides the batch on EVERY forward step:
  - finished sequences leave immediately and free their KV blocks,
  - waiting sequences join as soon as there are free blocks.

Each step is one of two kinds (vLLM v0-style, prefill-prioritised):
  1. PREFILL step: if requests are waiting and their blocks fit, admit as many as
     the token budget allows and compute their prompts (minus prefix-cache hits).
  2. DECODE step: otherwise, every running sequence produces one token.

If a decode needs a new block and the pool is empty, we PREEMPT the most recently
admitted sequence: free its blocks and put it back at the front of the waiting
queue. It will later be re-prefilled from its prompt + output so far
("recompute" preemption; vLLM can also swap blocks to CPU instead).
Its freed blocks keep their prefix-cache hashes, so the recompute is often cheap.

A natural next step is "chunked prefill": mixing a slice of a long prompt into
decode steps so decodes never stall behind a big prefill.
"""
from collections import deque

from .sequence import Status


class Scheduler:
    def __init__(self, block_manager, max_num_seqs=128, max_num_batched_tokens=2048):
        self.bm = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque = deque()
        self.running: list = []
        self.num_preemptions = 0

    def add(self, seq):
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self):
        # ---- 1. try to admit new sequences (prefill)
        scheduled, budget = [], self.max_num_batched_tokens
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            # Budget counts only tokens we must compute (prefix-cache hits are free).
            # A sequence longer than the whole budget may still run alone.
            if scheduled and self.bm.num_uncached_tokens(seq) > budget:
                break
            if not self.bm.try_allocate(seq):
                break
            budget -= len(seq) - seq.num_computed_tokens
            seq.status = Status.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)
        if scheduled:
            return scheduled, True

        # ---- 2. decode every running sequence, preempting if blocks run out
        scheduled, i = [], 0
        while i < len(self.running):
            seq = self.running[i]
            preempted_self = False
            while not self.bm.can_append(seq):
                victim = self.running.pop()          # newest admitted
                self._preempt(victim)
                if victim is seq:
                    preempted_self = True
                    break
            if preempted_self:
                break
            self.bm.append_slot(seq)
            scheduled.append(seq)
            i += 1
        return scheduled, False

    def _preempt(self, seq):
        self.bm.free_seq(seq)
        seq.status = Status.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def finish(self, seq):
        seq.status = Status.FINISHED
        self.running.remove(seq)
        self.bm.free_seq(seq)
