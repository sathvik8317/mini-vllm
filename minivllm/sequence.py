"""A Sequence is one request moving through the engine.

The important bookkeeping fields:

  token_ids            prompt + generated tokens so far
  num_computed_tokens  how many of token_ids already have their K/V in the cache.
                       After a step, the newly sampled token is appended but NOT yet
                       computed - its K/V gets written during the *next* step.
  block_table          logical block i -> physical block id in the KV cache.
                       Token at position p lives at slot  block_table[p // B] * B + p % B
  block_hashes         prefix-cache hashes of the blocks that are full and committed.
"""
import itertools
import time
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class SamplingParams:
    temperature: float = 0.0      # 0 => greedy
    top_p: float = 1.0
    max_tokens: int = 128
    ignore_eos: bool = False      # used by benchmarks so every system emits the same #tokens


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


_ids = itertools.count()


class Sequence:
    def __init__(self, prompt_token_ids, params: SamplingParams):
        self.id = next(_ids)
        self.token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.params = params
        self.status = Status.WAITING

        self.block_table: list[int] = []
        self.block_hashes: list[int] = []
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0     # prompt tokens served from the prefix cache (stats)
        self.num_preemptions = 0

        self.arrival_time = time.perf_counter()
        self.first_token_time = None
        self.finish_time = None
        self.finish_reason = None

    def __len__(self):
        return len(self.token_ids)

    @property
    def output_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_output_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens
