from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import time

class BlockStatus(Enum):

    FREE = "free"

    ALLOCATED = "allocated"

    CACHED = "cached"

@dataclass
class Block:

    block_id: int

    block_size: int

    ref_count: int = 0

    status: BlockStatus = BlockStatus.FREE

    block_hash: Optional[int] = None

    token_ids: Tuple[int, ...] = field(default_factory=tuple)

    computed_tokens: int = 0

    last_access_time: float = field(default_factory=time.time)

    def is_full(self) -> bool:
        return self.computed_tokens >= self.block_size

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.computed_tokens

    def incr_ref(self) -> None:
        self.ref_count += 1
        self.last_access_time = time.time()

    def decr_ref(self) -> None:
        self.ref_count -= 1

    def is_shared(self) -> bool:
        return self.ref_count > 1

    def can_evict(self) -> bool:
        return self.ref_count == 0 and self.status == BlockStatus.CACHED

    def __repr__(self) -> str:
        return (
            f"Block(id={self.block_id}, "
            f"refs={self.ref_count}, "
            f"tokens={self.computed_tokens}/{self.block_size}, "
            f"status={self.status.value})"
        )

@dataclass
class BlockTable:

    sequence_id: str

    block_size: int

    logical_to_physical: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.logical_to_physical)

    def num_blocks(self) -> int:
        return len(self.logical_to_physical)

    def append_block(self, physical_block_id: int) -> None:
        self.logical_to_physical.append(physical_block_id)

    def get_physical_block(self, logical_idx: int) -> int:
        return self.logical_to_physical[logical_idx]

    def set_physical_block(self, logical_idx: int, physical_block_id: int) -> None:
        self.logical_to_physical[logical_idx] = physical_block_id

    def get_last_block_id(self) -> Optional[int]:
        if not self.logical_to_physical:
            return None
        return self.logical_to_physical[-1]

    def get_all_physical_blocks(self) -> List[int]:
        return self.logical_to_physical.copy()

    def token_capacity(self) -> int:
        return len(self.logical_to_physical) * self.block_size

    def copy(self) -> "BlockTable":
        return BlockTable(
            sequence_id=self.sequence_id,
            block_size=self.block_size,
            logical_to_physical=self.logical_to_physical.copy(),
        )

    def get_block_index_for_token(self, token_position: int) -> int:
        return token_position // self.block_size

    def get_slot_in_block(self, token_position: int) -> int:
        return token_position % self.block_size

    def get_physical_slot(self, token_position: int) -> Tuple[int, int]:
        logical_idx = self.get_block_index_for_token(token_position)
        physical_id = self.get_physical_block(logical_idx)
        slot = self.get_slot_in_block(token_position)
        return physical_id, slot

    def __repr__(self) -> str:
        return (
            f"BlockTable(seq={self.sequence_id}, "
            f"blocks={len(self.logical_to_physical)}, "
            f"mapping={self.logical_to_physical})"
        )

def compute_num_blocks(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size

def compute_slot_mapping(
    block_table: BlockTable,
    start_pos: int,
    num_tokens: int,
) -> List[int]:
    slot_mapping = []
    for i in range(num_tokens):
        pos = start_pos + i
        physical_id, slot = block_table.get_physical_slot(pos)
        flat_slot = physical_id * block_table.block_size + slot
        slot_mapping.append(flat_slot)
    return slot_mapping
