"""Pure-Python tests of the paged allocator + prefix cache (no GPU needed)."""
from minivllm.block_manager import BlockManager
from minivllm.scheduler import Scheduler
from minivllm.sequence import Sequence, SamplingParams


def mk(tokens):
    return Sequence(tokens, SamplingParams(max_tokens=8))


def run_prefill(bm, s):
    assert bm.try_allocate(s)
    s.num_computed_tokens = len(s)
    bm.commit(s)


def test_prefix_sharing_and_lru():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = mk(list(range(10)))                  # 3 blocks: [0-3][4-7][8,9]
    run_prefill(bm, a)
    assert len(a.block_hashes) == 2          # only the 2 full blocks are cacheable
    b = mk(list(range(8)) + [99, 98, 97])    # shares the first 2 blocks
    assert bm.try_allocate(b)
    assert b.block_table[:2] == a.block_table[:2] and b.num_computed_tokens == 8
    assert bm.ref_counts[a.block_table[0]] == 2
    bm.free_seq(a); bm.free_seq(b)
    assert bm.num_free_blocks() == 8
    c = mk(list(range(9)))                   # hits again from the free (cached) list
    assert bm.try_allocate(c) and c.num_computed_tokens == 8


def test_never_matches_whole_prompt():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = mk(list(range(8)))
    run_prefill(bm, a); bm.free_seq(a)
    b = mk(list(range(8)))                   # identical prompt: must still compute 1 block
    bm.try_allocate(b)
    assert b.num_computed_tokens == 4


def test_eviction_when_full():
    bm = BlockManager(num_blocks=4, block_size=4)
    a = mk(list(range(16))); run_prefill(bm, a); bm.free_seq(a)
    b = mk(list(range(100, 116)))            # needs all 4 blocks -> evicts a's cached blocks
    assert bm.try_allocate(b) and bm.num_evictions == 4
    assert bm.hash_to_block == {}


def test_preemption():
    bm = BlockManager(num_blocks=3, block_size=4)
    sch = Scheduler(bm, max_num_seqs=8, max_num_batched_tokens=100)
    s1, s2 = mk([1, 2, 3, 4, 5]), mk([6, 7, 8])  # 2 blocks + 1 block = pool full
    sch.add(s1); sch.add(s2)
    seqs, pre = sch.schedule()
    assert pre and len(seqs) == 2
    for s in seqs:
        s.num_computed_tokens = len(s); s.token_ids.append(0)
    # s2 now has 4 tokens (fits); s1 has 6 (fits). Grow s2 to need a new block.
    s2.num_computed_tokens = len(s2); s2.token_ids.append(0)
    seqs, pre = sch.schedule()
    assert not pre and seqs == [s1] and s2 in sch.waiting and sch.num_preemptions == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
