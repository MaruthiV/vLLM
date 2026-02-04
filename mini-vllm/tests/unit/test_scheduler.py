import pytest
from unittest.mock import MagicMock, patch

from mini_vllm.core.scheduler import (
    Scheduler,
    SchedulerOutput,
    SchedulingBudget,
    PreemptionMode,
)
from mini_vllm.core.block_manager import BlockSpaceManager
from mini_vllm.config import SchedulerConfig, CacheConfig
from mini_vllm.engine.request import Request, RequestStatus
from mini_vllm.sampling.sampling_params import SamplingParams

def make_request(
    request_id: str,
    num_prompt_tokens: int = 10,
    num_output_tokens: int = 0,
) -> Request:
    return Request(
        request_id=request_id,
        prompt="test prompt",
        prompt_token_ids=list(range(num_prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=100),
        output_token_ids=list(range(num_output_tokens)),
    )

class TestSchedulingBudget:

    def test_budget_creation(self):
        budget = SchedulingBudget(token_budget=4096, max_num_seqs=256)
        assert budget.token_budget == 4096
        assert budget.max_num_seqs == 256
        assert budget.num_batched_tokens == 0
        assert budget.num_batched_seqs == 0

    def test_can_schedule(self):
        budget = SchedulingBudget(token_budget=100, max_num_seqs=10)

        assert budget.can_schedule(50)
        assert budget.can_schedule(100)
        assert not budget.can_schedule(101)

        budget.num_batched_seqs = 9
        assert budget.can_schedule(10, num_seqs=1)
        assert not budget.can_schedule(10, num_seqs=2)

    def test_add_prefill(self):
        budget = SchedulingBudget(token_budget=100, max_num_seqs=10)

        budget.add_prefill(50)
        assert budget.num_batched_tokens == 50
        assert budget.num_batched_seqs == 1
        assert budget.num_prefill_tokens == 50
        assert budget.num_prefill_seqs == 1

    def test_add_decode(self):
        budget = SchedulingBudget(token_budget=100, max_num_seqs=10)

        budget.add_decode()
        assert budget.num_batched_tokens == 1
        assert budget.num_batched_seqs == 1
        assert budget.num_decode_tokens == 1
        assert budget.num_decode_seqs == 1

class TestScheduler:

    @pytest.fixture
    def scheduler(self):
        scheduler_config = SchedulerConfig(
            max_num_seqs=256,
            max_num_batched_tokens=4096,
            chunked_prefill_enabled=True,
            chunk_size=512,
        )
        cache_config = CacheConfig(block_size=16)

        block_manager = MagicMock(spec=BlockSpaceManager)
        block_manager.can_allocate.return_value = MagicMock(can_allocate=True)
        block_manager.can_append_slot.return_value = True

        return Scheduler(scheduler_config, cache_config, block_manager)

    def test_add_request(self, scheduler):
        request = make_request("req-1", num_prompt_tokens=50)

        scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 1
        assert scheduler.get_num_running() == 0
        assert request.status == RequestStatus.WAITING

    def test_schedule_new_request(self, scheduler):
        request = make_request("req-1", num_prompt_tokens=50)
        scheduler.add_request(request)

        output = scheduler.schedule()

        assert len(output.scheduled_requests) == 1
        assert output.scheduled_requests[0] == request
        assert output.num_prefill_tokens > 0
        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 1

    def test_schedule_multiple_requests(self, scheduler):
        requests = [make_request(f"req-{i}", num_prompt_tokens=20) for i in range(5)]
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()

        assert len(output.scheduled_requests) == 5
        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 5

    def test_token_budget_limit(self, scheduler):
        scheduler.config.max_num_batched_tokens = 100

        for i in range(10):
            request = make_request(f"req-{i}", num_prompt_tokens=50)
            scheduler.add_request(request)

        output = scheduler.schedule()

        assert output.num_scheduled_tokens <= 100

    def test_max_seqs_limit(self, scheduler):
        scheduler.config.max_num_seqs = 3

        for i in range(10):
            request = make_request(f"req-{i}", num_prompt_tokens=10)
            scheduler.add_request(request)

        output = scheduler.schedule()

        assert len(output.scheduled_requests) <= 3

    def test_finish_request(self, scheduler):
        request = make_request("req-1", num_prompt_tokens=50)
        scheduler.add_request(request)
        scheduler.schedule()

        assert scheduler.get_num_running() == 1

        scheduler.finish_request("req-1")

        assert scheduler.get_num_running() == 0
        assert not scheduler.has_unfinished_requests()

    def test_abort_request_waiting(self, scheduler):
        request = make_request("req-1", num_prompt_tokens=50)
        scheduler.add_request(request)

        aborted = scheduler.abort_request("req-1")

        assert aborted is not None
        assert scheduler.get_num_waiting() == 0

    def test_abort_request_not_found(self, scheduler):
        aborted = scheduler.abort_request("nonexistent")
        assert aborted is None

    def test_has_unfinished_requests(self, scheduler):
        assert not scheduler.has_unfinished_requests()

        request = make_request("req-1", num_prompt_tokens=50)
        scheduler.add_request(request)

        assert scheduler.has_unfinished_requests()

    def test_chunked_prefill(self, scheduler):
        scheduler.config.chunk_size = 100

        request = make_request("req-1", num_prompt_tokens=500)
        scheduler.add_request(request)

        output = scheduler.schedule()

        assert output.num_prefill_tokens <= 100

class TestSchedulerOutput:

    def test_empty_output(self):
        output = SchedulerOutput()
        assert output.is_empty
        assert output.num_scheduled_tokens == 0
        assert output.num_scheduled_requests == 0

    def test_output_with_requests(self):
        output = SchedulerOutput(
            scheduled_requests=[make_request("req-1")],
            num_prefill_tokens=50,
            num_decode_tokens=10,
        )
        assert not output.is_empty
        assert output.num_scheduled_tokens == 60
        assert output.num_scheduled_requests == 1
