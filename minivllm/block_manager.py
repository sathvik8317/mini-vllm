"""Paged KV-cache allocator with automatic prefix caching.

The GPU KV cache is one big pool of fixed-size *blocks* (e.g. 16 tokens each).
A sequence owns a list of block ids (its block table) - they do NOT need to be
contiguous, exactly like virtual-memory pages. Consequences:

  * No external fragmentation: any free block can serve any sequence.
  * Internal fragmentation is bounded: only the last block of each sequence can
    be partially empty, so waste is < block_size tokens per sequence.
  * Blocks can be shared between sequences (ref_count > 1). We use this for
    prefix caching: identical prompt prefixes map to the same physical blocks.

Prefix caching (same idea as vLLM's "automatic prefix caching"):
  A *full* block is identified by  hash(parent_block_hash, tokens_in_block).
  Chaining the parent hash means the hash identifies the whole prefix up to and
  including this block, not just these 16 tokens.
  When a sequence frees its blocks, blocks that carry a hash are NOT wiped - they
  go to the back of an LRU free list and stay findable in `hash_to_block`. Only
  when the allocator actually needs a fresh block does it evict the
  least-recently-used one (and drop its hash).

Why no copy-on-write is needed: only *full* blocks are ever shared, and a
sequence only ever writes into its last, partially-filled block, which is
private. (vLLM needs CoW for beam search / parallel sampling forks; we don't.)
"""
from collections import OrderedDict


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.ref_counts = [0] * num_blocks
        self.block_hash: list[int | None] = [None] * num_blocks
        # Free blocks in eviction order: front = evict first (LRU), back = most recently freed.
        self.free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.hash_to_block: dict[int, int] = {}

        # stats
        self.prompt_tokens_seen = 0
        self.prompt_tokens_hit = 0
        self.num_evictions = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def hash_block(parent: int | None, tokens) -> int:
        # Python's int/tuple hash is deterministic within a process. A production
        # system would use a cryptographic hash to make collisions practically impossible.
        return hash((parent, tuple(tokens)))

    def num_free_blocks(self) -> int:
        return len(self.free)

    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _pop_free(self) -> int:
        b, _ = self.free.popitem(last=False)          # LRU end
        h = self.block_hash[b]
        if h is not None:                               # evict it from the prefix cache
            if self.hash_to_block.get(h) == b:
                del self.hash_to_block[h]
            self.block_hash[b] = None
            self.num_evictions += 1
        self.ref_counts[b] = 1
        return b

    def _match_prefix(self, seq):
        """Longest run of leading full blocks already present in the cache."""
        if not self.enable_prefix_caching:
            return [], []
        bs = self.block_size
        # Never match the whole sequence: we must compute >= 1 token to get logits.
        max_blocks = (len(seq) - 1) // bs
        parent, blocks, hashes = None, [], []
        for i in range(max_blocks):
            h = self.hash_block(parent, seq.token_ids[i * bs:(i + 1) * bs])
            b = self.hash_to_block.get(h)
            if b is None:
                break
            blocks.append(b)
            hashes.append(h)
            parent = h
        return blocks, hashes

    # ------------------------------------------------------------------ API
    def num_uncached_tokens(self, seq) -> int:
        return len(seq) - len(self._match_prefix(seq)[0]) * self.block_size

    def try_allocate(self, seq) -> bool:
        """Give a waiting sequence blocks for all its tokens (prefill). False if it doesn't fit."""
        hit_blocks, hit_hashes = self._match_prefix(seq)
        total = self._blocks_for(len(seq))
        # Cached blocks that someone is using cost nothing; cached blocks sitting in the
        # free list (ref 0) must be pulled out of it, so they count against free space.
        already_in_use = sum(1 for b in hit_blocks if self.ref_counts[b] > 0)
        if total - already_in_use > len(self.free):
            return False

        for b in hit_blocks:
            if self.ref_counts[b] == 0:
                del self.free[b]
            self.ref_counts[b] += 1
        new_blocks = [self._pop_free() for _ in range(total - len(hit_blocks))]

        seq.block_table = hit_blocks + new_blocks
        seq.block_hashes = hit_hashes
        seq.num_computed_tokens = len(hit_blocks) * self.block_size
        if seq.num_preemptions == 0:
            seq.num_cached_tokens = seq.num_computed_tokens
            self.prompt_tokens_seen += len(seq)
            self.prompt_tokens_hit += seq.num_computed_tokens
        return True

    def can_append(self, seq) -> bool:
        """Decode writes K/V for position len(seq)-1. Is there a slot for it?"""
        needs_block = len(seq) > len(seq.block_table) * self.block_size
        return not needs_block or len(self.free) > 0

    def append_slot(self, seq):
        if len(seq) > len(seq.block_table) * self.block_size:
            seq.block_table.append(self._pop_free())

    def commit(self, seq):
        """After a forward pass: register blocks that just became full in the prefix cache.
        Done *after* the forward so nobody can hit a block whose K/V isn't written yet."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        n_full = seq.num_computed_tokens // bs
        while len(seq.block_hashes) < n_full:
            i = len(seq.block_hashes)
            parent = seq.block_hashes[-1] if seq.block_hashes else None
            h = self.hash_block(parent, seq.token_ids[i * bs:(i + 1) * bs])
            seq.block_hashes.append(h)
            b = seq.block_table[i]
            if h not in self.hash_to_block:   # an identical block may exist already; keep the first
                self.hash_to_block[h] = b
                self.block_hash[b] = h

    def free_seq(self, seq):
        # Reverse order: a prefix's *tail* blocks end up evicted before its head blocks,
        # since the head is what other requests are most likely to share.
        for b in reversed(seq.block_table):
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                self.free[b] = None
                if self.block_hash[b] is None:          # nothing worth caching: reuse first
                    self.free.move_to_end(b, last=False)
        seq.block_table = []
        seq.block_hashes = []
        seq.num_computed_tokens = 0

    # ------------------------------------------------------------------ stats
    def usage(self, running) -> dict:
        used_tokens = sum(s.num_computed_tokens for s in running)
        # shared blocks are counted once - that's the point of sharing
        distinct = {b for s in running for b in s.block_table}
        return {
            "blocks_in_use": len(distinct),
            "blocks_free": len(self.free),
            "allocated_slots": len(distinct) * self.block_size,
            "used_tokens": used_tokens,
        }
