from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import hashlib

from mini_vllm.core.block import Block, BlockStatus

def compute_block_hash(
    parent_hash: Optional[int],
    token_ids: Tuple[int, ...],
) -> int:

    if parent_hash is None:
        hash_input = f"ROOT:{token_ids}"
    else:
        hash_input = f"{parent_hash}:{token_ids}"

    hash_bytes = hashlib.sha256(hash_input.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder='big')

@dataclass
class CachedBlock:

    block: Block
    block_hash: int
    last_access_time: float = 0.0
    content_hash: Optional[int] = None

@dataclass
class PrefixCacheMetrics:

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_lookups: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.hits / self.total_lookups

    def record_hit(self) -> None:
        self.hits += 1
        self.total_lookups += 1

    def record_miss(self) -> None:
        self.misses += 1
        self.total_lookups += 1

    def record_eviction(self) -> None:
        self.evictions += 1

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_lookups = 0

class PrefixCache:

    def __init__(
        self,
        max_cached_blocks: int = 0,
        enable_caching: bool = True,
    ):
        self.max_cached_blocks = max_cached_blocks
        self.enable_caching = enable_caching

        self._hash_table: Dict[int, CachedBlock] = {}

        self._lru_order: OrderedDict[int, None] = OrderedDict()

        self.metrics = PrefixCacheMetrics()

        self._access_counter = 0

    def compute_hash(
        self,
        parent_hash: Optional[int],
        token_ids: Tuple[int, ...],
    ) -> int:
        return compute_block_hash(parent_hash, token_ids)

    def lookup(self, block_hash: int) -> Optional[Block]:
        if not self.enable_caching:
            self.metrics.record_miss()
            return None

        if block_hash in self._hash_table:
            cached = self._hash_table[block_hash]

            self._touch(block_hash)

            self._access_counter += 1
            cached.last_access_time = self._access_counter

            self.metrics.record_hit()
            return cached.block

        self.metrics.record_miss()
        return None

    def insert(
        self,
        block_hash: int,
        block: Block,
        content_hash: Optional[int] = None,
    ) -> bool:
        if not self.enable_caching:
            return False

        if block_hash in self._hash_table:
            return False

        if self.max_cached_blocks > 0 and len(self._hash_table) >= self.max_cached_blocks:
            self._evict_lru()

        self._access_counter += 1
        cached = CachedBlock(
            block=block,
            block_hash=block_hash,
            last_access_time=self._access_counter,
            content_hash=content_hash,
        )

        self._hash_table[block_hash] = cached
        self._lru_order[block_hash] = None

        block.status = BlockStatus.CACHED
        block.block_hash = block_hash

        return True

    def remove(self, block_hash: int) -> Optional[Block]:
        if block_hash not in self._hash_table:
            return None

        cached = self._hash_table.pop(block_hash)
        if block_hash in self._lru_order:
            del self._lru_order[block_hash]

        cached.block.block_hash = None
        cached.block.status = BlockStatus.FREE

        return cached.block

    def _touch(self, block_hash: int) -> None:
        if block_hash in self._lru_order:
            self._lru_order.move_to_end(block_hash)

    def _evict_lru(self) -> Optional[Block]:

        for block_hash in list(self._lru_order.keys()):
            cached = self._hash_table.get(block_hash)
            if cached and cached.block.ref_count == 0:
                self.metrics.record_eviction()
                return self.remove(block_hash)

        return None

    def evict_blocks(self, num_blocks: int) -> List[Block]:
        evicted = []
        for _ in range(num_blocks):
            block = self._evict_lru()
            if block is None:
                break
            evicted.append(block)
        return evicted

    def get_cached_block_ids(self) -> Set[int]:
        return {cached.block.block_id for cached in self._hash_table.values()}

    def clear(self) -> List[Block]:
        blocks = [cached.block for cached in self._hash_table.values()]
        self._hash_table.clear()
        self._lru_order.clear()
        return blocks

    @property
    def num_cached(self) -> int:
        return len(self._hash_table)

    @property
    def num_evictable(self) -> int:
        return sum(
            1 for cached in self._hash_table.values()
            if cached.block.ref_count == 0
        )

    def get_stats(self) -> Dict:
        return {
            "num_cached": self.num_cached,
            "num_evictable": self.num_evictable,
            "max_cached": self.max_cached_blocks,
            "hit_rate": self.metrics.hit_rate,
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "evictions": self.metrics.evictions,
            "total_lookups": self.metrics.total_lookups,
        }

    def __repr__(self) -> str:
        return (
            f"PrefixCache(cached={self.num_cached}, "
            f"hit_rate={self.metrics.hit_rate:.2%})"
        )

class PrefixCacheManager:

    def __init__(
        self,
        cache: PrefixCache,
        block_size: int,
    ):
        self.cache = cache
        self.block_size = block_size

        self._seq_hash_chains: Dict[str, List[int]] = {}

    def compute_hash_chain(
        self,
        token_ids: List[int],
    ) -> List[int]:
        hashes = []
        parent_hash = None

        for i in range(0, len(token_ids), self.block_size):
            block_tokens = tuple(token_ids[i:i + self.block_size])
            if len(block_tokens) == self.block_size:

                block_hash = self.cache.compute_hash(parent_hash, block_tokens)
                hashes.append(block_hash)
                parent_hash = block_hash

        return hashes

    def find_cached_prefix_length(
        self,
        token_ids: List[int],
    ) -> Tuple[int, List[Block]]:
        hash_chain = self.compute_hash_chain(token_ids)
        cached_blocks = []
        cached_tokens = 0

        for block_hash in hash_chain:
            block = self.cache.lookup(block_hash)
            if block is None:
                break
            cached_blocks.append(block)
            cached_tokens += self.block_size

        return cached_tokens, cached_blocks

    def register_sequence(
        self,
        seq_id: str,
        token_ids: List[int],
        blocks: List[Block],
    ) -> None:
        hash_chain = self.compute_hash_chain(token_ids)

        for i, (block_hash, block) in enumerate(zip(hash_chain, blocks)):

            self.cache.insert(block_hash, block)

        self._seq_hash_chains[seq_id] = hash_chain

    def unregister_sequence(self, seq_id: str) -> None:
        if seq_id in self._seq_hash_chains:
            del self._seq_hash_chains[seq_id]

    def get_sequence_hashes(self, seq_id: str) -> Optional[List[int]]:
        return self._seq_hash_chains.get(seq_id)
