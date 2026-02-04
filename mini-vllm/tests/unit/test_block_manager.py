import pytest

from mini_vllm.core.block import Block, BlockTable, BlockStatus, compute_num_blocks
from mini_vllm.core.block_manager import BlockAllocator, BlockSpaceManager
from mini_vllm.config import CacheConfig

class TestBlock:

    def test_block_creation(self):
        block = Block(block_id=0, block_size=16)
        assert block.block_id == 0
        assert block.block_size == 16
        assert block.ref_count == 0
        assert block.status == BlockStatus.FREE

    def test_block_ref_counting(self):
        block = Block(block_id=0, block_size=16)
        block.incr_ref()
        assert block.ref_count == 1
        assert not block.is_shared()

        block.incr_ref()
        assert block.ref_count == 2
        assert block.is_shared()

        block.decr_ref()
        assert block.ref_count == 1

    def test_block_is_full(self):
        block = Block(block_id=0, block_size=16)
        block.computed_tokens = 8
        assert not block.is_full()

        block.computed_tokens = 16
        assert block.is_full()

class TestBlockTable:

    def test_block_table_creation(self):
        table = BlockTable(sequence_id="seq-1", block_size=16)
        assert table.sequence_id == "seq-1"
        assert table.block_size == 16
        assert table.num_blocks() == 0

    def test_append_and_get_block(self):
        table = BlockTable(sequence_id="seq-1", block_size=16)
        table.append_block(5)
        table.append_block(12)

        assert table.num_blocks() == 2
        assert table.get_physical_block(0) == 5
        assert table.get_physical_block(1) == 12

    def test_get_physical_slot(self):
        table = BlockTable(sequence_id="seq-1", block_size=16)
        table.append_block(5)
        table.append_block(12)

        block_id, slot = table.get_physical_slot(0)
        assert block_id == 5
        assert slot == 0

        block_id, slot = table.get_physical_slot(15)
        assert block_id == 5
        assert slot == 15

        block_id, slot = table.get_physical_slot(16)
        assert block_id == 12
        assert slot == 0

    def test_token_capacity(self):
        table = BlockTable(sequence_id="seq-1", block_size=16)
        assert table.token_capacity() == 0

        table.append_block(5)
        assert table.token_capacity() == 16

        table.append_block(12)
        assert table.token_capacity() == 32

class TestBlockAllocator:

    def test_allocator_creation(self):
        allocator = BlockAllocator(num_blocks=100, block_size=16)
        assert allocator.num_blocks == 100
        assert allocator.get_num_free_blocks() == 100
        assert allocator.get_num_allocated_blocks() == 0

    def test_allocate_and_free(self):
        allocator = BlockAllocator(num_blocks=10, block_size=16)

        block = allocator.allocate()
        assert block is not None
        assert block.ref_count == 1
        assert block.status == BlockStatus.ALLOCATED
        assert allocator.get_num_free_blocks() == 9

        allocator.free(block)
        assert block.ref_count == 0
        assert block.status == BlockStatus.FREE
        assert allocator.get_num_free_blocks() == 10

    def test_allocate_all_blocks(self):
        allocator = BlockAllocator(num_blocks=5, block_size=16)

        blocks = []
        for _ in range(5):
            block = allocator.allocate()
            assert block is not None
            blocks.append(block)

        assert allocator.get_num_free_blocks() == 0

        block = allocator.allocate()
        assert block is None

    def test_fork_increases_ref_count(self):
        allocator = BlockAllocator(num_blocks=10, block_size=16)

        block = allocator.allocate()
        assert block.ref_count == 1

        forked = allocator.fork(block)
        assert forked is block
        assert block.ref_count == 2

    def test_cow_copy(self):
        allocator = BlockAllocator(num_blocks=10, block_size=16)

        src_block = allocator.allocate()
        src_block.computed_tokens = 8

        new_block = allocator.cow_copy(src_block)
        assert new_block is not None
        assert new_block.block_id != src_block.block_id
        assert new_block.computed_tokens == 8
        assert allocator.get_num_allocated_blocks() == 2

class TestBlockSpaceManager:

    @pytest.fixture
    def manager(self):
        config = CacheConfig(block_size=16)
        return BlockSpaceManager(config, num_gpu_blocks=100)

    def test_manager_creation(self, manager):
        assert manager.get_num_free_blocks() == 100
        assert manager.get_num_sequences() == 0

    def test_can_allocate(self, manager):

        status = manager.can_allocate(50)
        assert status.can_allocate
        assert status.num_required_blocks == 4

    def test_allocate_sequence(self, manager):
        block_table = manager.allocate("seq-1", num_tokens=50)

        assert block_table is not None
        assert block_table.num_blocks() == 4
        assert manager.get_num_sequences() == 1
        assert manager.get_num_free_blocks() == 96

    def test_free_sequence(self, manager):
        manager.allocate("seq-1", num_tokens=50)
        assert manager.get_num_free_blocks() == 96

        manager.free("seq-1")
        assert manager.get_num_free_blocks() == 100
        assert manager.get_num_sequences() == 0

    def test_append_slot(self, manager):
        manager.allocate("seq-1", num_tokens=16)
        initial_blocks = manager.get_block_table("seq-1").num_blocks()
        assert initial_blocks == 1

        slot_mapping = manager.append_slot("seq-1")
        assert len(slot_mapping) == 1

    def test_fork_sequence(self, manager):
        manager.allocate("seq-1", num_tokens=50)

        dst_table = manager.fork_sequence("seq-1", "seq-2")

        assert manager.get_num_sequences() == 2
        assert dst_table.num_blocks() == 4

        src_table = manager.get_block_table("seq-1")
        assert src_table.get_all_physical_blocks() == dst_table.get_all_physical_blocks()

class TestComputeNumBlocks:

    def test_exact_fit(self):
        assert compute_num_blocks(16, 16) == 1
        assert compute_num_blocks(32, 16) == 2

    def test_partial_fill(self):
        assert compute_num_blocks(17, 16) == 2
        assert compute_num_blocks(1, 16) == 1
        assert compute_num_blocks(50, 16) == 4

    def test_zero_tokens(self):
        assert compute_num_blocks(0, 16) == 0
