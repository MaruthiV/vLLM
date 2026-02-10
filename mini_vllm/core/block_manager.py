from typing import Dict, List, Optional, Set, Tuple
from collections import deque
from dataclasses import dataclass
import time

from mini_vllm.core.block import (
    Block,
    BlockTable,
    BlockStatus,
    compute_num_blocks,
)
from mini_vllm.core.prefix_cache import PrefixCache, PrefixCacheManager
from mini_vllm.config import CacheConfig

class BlockAllocator:

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.blocks: Dict[int, Block] = {
            i: Block(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        }

        self._free_blocks: deque[int] = deque(range(num_blocks))

        self._allocated_blocks: Set[int] = set()

    def allocate(self) -> Optional[Block]:
        if not self._free_blocks:
            return None

        block_id = self._free_blocks.pop()
        block = self.blocks[block_id]

        block.ref_count = 1
        block.status = BlockStatus.ALLOCATED
        block.computed_tokens = 0
        block.block_hash = None
        block.token_ids = tuple()
        block.last_access_time = time.time()

        self._allocated_blocks.add(block_id)
        return block

    def free(self, block: Block) -> None:
        block.decr_ref()

        if block.ref_count == 0:
            block.status = BlockStatus.FREE
            block.block_hash = None
            block.token_ids = tuple()
            self._allocated_blocks.discard(block.block_id)
            self._free_blocks.append(block.block_id)

    def fork(self, block: Block) -> Block:
        block.incr_ref()
        return block

    def cow_copy(self, src_block: Block) -> Optional[Block]:
        new_block = self.allocate()
        if new_block is None:
            return None

        new_block.computed_tokens = src_block.computed_tokens
        new_block.token_ids = src_block.token_ids

        return new_block

    def get_block(self, block_id: int) -> Block:
        return self.blocks[block_id]

    def get_num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def get_num_allocated_blocks(self) -> int:
        return len(self._allocated_blocks)

    def get_utilization(self) -> float:
        if self.num_blocks == 0:
            return 0.0
        return len(self._allocated_blocks) / self.num_blocks

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(total={self.num_blocks}, "
            f"free={self.get_num_free_blocks()}, "
            f"allocated={self.get_num_allocated_blocks()})"
        )

@dataclass
class AllocStatus:

    can_allocate: bool

    num_required_blocks: int

    num_free_blocks: int

class BlockSpaceManager:

    def __init__(self, config: CacheConfig, num_gpu_blocks: int):
        self.block_size = config.block_size
        self.enable_prefix_caching = config.enable_prefix_caching

        self.gpu_allocator = BlockAllocator(
            num_blocks=num_gpu_blocks,
            block_size=self.block_size,
        )

        self.block_tables: Dict[str, BlockTable] = {}

        self._seq_to_blocks: Dict[str, List[Block]] = {}

        self._seq_token_ids: Dict[str, List[int]] = {}

        self.prefix_cache: Optional[PrefixCache] = None
        self.prefix_cache_manager: Optional[PrefixCacheManager] = None

        if self.enable_prefix_caching:

            max_cached = int(num_gpu_blocks * 0.2)
            self.prefix_cache = PrefixCache(
                max_cached_blocks=max_cached,
                enable_caching=True,
            )
            self.prefix_cache_manager = PrefixCacheManager(
                cache=self.prefix_cache,
                block_size=self.block_size,
            )

    def can_allocate(self, num_tokens: int) -> AllocStatus:
        num_required = compute_num_blocks(num_tokens, self.block_size)
        num_free = self.gpu_allocator.get_num_free_blocks()

        return AllocStatus(
            can_allocate=num_free >= num_required,
            num_required_blocks=num_required,
            num_free_blocks=num_free,
        )

    def allocate(
        self,
        seq_id: str,
        num_tokens: int,
        token_ids: Optional[List[int]] = None,
    ) -> BlockTable:
        num_blocks_needed = compute_num_blocks(num_tokens, self.block_size)

        cached_blocks: List[Block] = []
        cached_tokens = 0

        if (self.enable_prefix_caching and
            self.prefix_cache_manager is not None and
            token_ids is not None):
            cached_tokens, cached_blocks = self.prefix_cache_manager.find_cached_prefix_length(
                token_ids
            )

            for block in cached_blocks:
                block.incr_ref()

        num_cached_blocks = len(cached_blocks)
        num_new_blocks_needed = num_blocks_needed - num_cached_blocks

        available = self.gpu_allocator.get_num_free_blocks()

        if available < num_new_blocks_needed and self.prefix_cache is not None:
            needed = num_new_blocks_needed - available
            evicted = self.prefix_cache.evict_blocks(needed)
            for block in evicted:
                self.gpu_allocator._free_blocks.append(block.block_id)

        if self.gpu_allocator.get_num_free_blocks() < num_new_blocks_needed:

            for block in cached_blocks:
                block.decr_ref()
            raise MemoryError(
                f"Cannot allocate {num_new_blocks_needed} blocks for sequence {seq_id}. "
                f"Only {self.gpu_allocator.get_num_free_blocks()} free blocks available."
            )

        block_table = BlockTable(
            sequence_id=seq_id,
            block_size=self.block_size,
        )

        allocated_blocks: List[Block] = list(cached_blocks)
        for block in cached_blocks:
            block_table.append_block(block.block_id)

        for _ in range(num_new_blocks_needed):
            block = self.gpu_allocator.allocate()
            if block is None:

                for b in allocated_blocks:
                    if b not in cached_blocks:
                        self.gpu_allocator.free(b)
                    else:
                        b.decr_ref()
                raise MemoryError(f"Block allocation failed for sequence {seq_id}")

            block_table.append_block(block.block_id)
            allocated_blocks.append(block)

        tokens_remaining = num_tokens
        for block in allocated_blocks:
            tokens_in_block = min(tokens_remaining, self.block_size)
            block.computed_tokens = tokens_in_block
            tokens_remaining -= tokens_in_block
            if tokens_remaining <= 0:
                break

        self.block_tables[seq_id] = block_table
        self._seq_to_blocks[seq_id] = allocated_blocks
        if token_ids is not None:
            self._seq_token_ids[seq_id] = list(token_ids)

        return block_table

    def free(self, seq_id: str, cache_blocks: bool = True) -> None:
        if seq_id not in self.block_tables:
            return

        if (cache_blocks and
            self.enable_prefix_caching and
            self.prefix_cache_manager is not None and
            seq_id in self._seq_token_ids):

            token_ids = self._seq_token_ids[seq_id]
            blocks = self._seq_to_blocks.get(seq_id, [])

            self.prefix_cache_manager.register_sequence(seq_id, token_ids, blocks)
            self.prefix_cache_manager.unregister_sequence(seq_id)

        if seq_id in self._seq_to_blocks:
            for block in self._seq_to_blocks[seq_id]:

                if block.status == BlockStatus.CACHED and block.ref_count > 1:
                    block.decr_ref()
                else:
                    self.gpu_allocator.free(block)
            del self._seq_to_blocks[seq_id]

        if seq_id in self._seq_token_ids:
            del self._seq_token_ids[seq_id]

        del self.block_tables[seq_id]

    def can_append_slot(self, seq_id: str, num_tokens_after: int) -> bool:
        if seq_id not in self.block_tables:
            return False

        block_table = self.block_tables[seq_id]
        current_capacity = block_table.token_capacity()

        if num_tokens_after <= current_capacity:

            return True

        new_blocks_needed = compute_num_blocks(num_tokens_after, self.block_size) - block_table.num_blocks()
        return self.gpu_allocator.get_num_free_blocks() >= new_blocks_needed

    def append_slot(
        self,
        seq_id: str,
        num_new_tokens: int = 1,
        new_token_ids: Optional[List[int]] = None,
    ) -> List[int]:
        if seq_id not in self.block_tables:
            raise KeyError(f"Sequence {seq_id} not found")

        block_table = self.block_tables[seq_id]
        blocks = self._seq_to_blocks[seq_id]

        slot_mapping = []

        if new_token_ids is not None and seq_id in self._seq_token_ids:
            self._seq_token_ids[seq_id].extend(new_token_ids)

        for _ in range(num_new_tokens):

            if blocks:
                last_block = blocks[-1]
                if not last_block.is_full():

                    slot = last_block.computed_tokens
                    flat_slot = last_block.block_id * self.block_size + slot
                    slot_mapping.append(flat_slot)
                    last_block.computed_tokens += 1
                    continue

            if blocks and blocks[-1].is_shared():
                new_block = self._handle_cow(seq_id, len(blocks) - 1)
                if new_block is None:
                    raise MemoryError(f"CoW allocation failed for sequence {seq_id}")

            new_block = self.gpu_allocator.allocate()
            if new_block is None:
                raise MemoryError(f"Block allocation failed for sequence {seq_id}")

            block_table.append_block(new_block.block_id)
            blocks.append(new_block)

            slot = new_block.computed_tokens
            flat_slot = new_block.block_id * self.block_size + slot
            slot_mapping.append(flat_slot)
            new_block.computed_tokens += 1

        return slot_mapping

    def _handle_cow(self, seq_id: str, block_idx: int) -> Optional[Block]:
        blocks = self._seq_to_blocks[seq_id]
        old_block = blocks[block_idx]

        if not old_block.is_shared():
            return old_block

        new_block = self.gpu_allocator.cow_copy(old_block)
        if new_block is None:
            return None

        self.gpu_allocator.free(old_block)
        blocks[block_idx] = new_block
        self.block_tables[seq_id].set_physical_block(block_idx, new_block.block_id)

        return new_block

    def fork_sequence(self, src_seq_id: str, dst_seq_id: str) -> BlockTable:
        if src_seq_id not in self.block_tables:
            raise KeyError(f"Source sequence {src_seq_id} not found")

        src_table = self.block_tables[src_seq_id]
        src_blocks = self._seq_to_blocks[src_seq_id]

        dst_table = src_table.copy()
        dst_table.sequence_id = dst_seq_id

        dst_blocks = []
        for block in src_blocks:
            forked = self.gpu_allocator.fork(block)
            dst_blocks.append(forked)

        self.block_tables[dst_seq_id] = dst_table
        self._seq_to_blocks[dst_seq_id] = dst_blocks

        return dst_table

    def get_block_table(self, seq_id: str) -> Optional[BlockTable]:
        return self.block_tables.get(seq_id)

    def get_block_table_tensor_data(self, seq_id: str) -> List[int]:
        table = self.block_tables.get(seq_id)
        if table is None:
            return []
        return table.get_all_physical_blocks()

    def get_num_free_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_total_blocks(self) -> int:
        return self.gpu_allocator.num_blocks

    def get_utilization(self) -> float:
        return self.gpu_allocator.get_utilization()

    def get_num_sequences(self) -> int:
        return len(self.block_tables)

    def get_prefix_cache_stats(self) -> Optional[Dict]:
        if self.prefix_cache is None:
            return None
        return self.prefix_cache.get_stats()

    def get_prefix_cache_hit_rate(self) -> float:
        if self.prefix_cache is None:
            return 0.0
        return self.prefix_cache.metrics.hit_rate

    def reset_prefix_cache_metrics(self) -> None:
        if self.prefix_cache is not None:
            self.prefix_cache.metrics.reset()

    def __repr__(self) -> str:
        cache_info = ""
        if self.prefix_cache is not None:
            cache_info = f", cache_hit_rate={self.get_prefix_cache_hit_rate():.1%}"
        return (
            f"BlockSpaceManager("
            f"sequences={len(self.block_tables)}, "
            f"utilization={self.get_utilization():.1%}, "
            f"free={self.get_num_free_blocks()}/{self.get_num_total_blocks()}"
            f"{cache_info})"
        )
