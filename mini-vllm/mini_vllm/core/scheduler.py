from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import deque
from enum import Enum
import time

from mini_vllm.config import SchedulerConfig, CacheConfig
from mini_vllm.engine.request import Request, RequestStatus
from mini_vllm.core.block_manager import BlockSpaceManager

class PreemptionMode(Enum):

    RECOMPUTE = "recompute"

    SWAP = "swap"

@dataclass
class SchedulerOutput:

    scheduled_requests: List[Request] = field(default_factory=list)

    preempted_requests: List[Request] = field(default_factory=list)

    finished_requests: List[Request] = field(default_factory=list)

    num_prefill_tokens: int = 0

    num_decode_tokens: int = 0

    blocks_to_copy: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def num_scheduled_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def num_scheduled_requests(self) -> int:
        return len(self.scheduled_requests)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0

@dataclass
class SchedulingBudget:

    token_budget: int

    max_num_seqs: int

    num_batched_tokens: int = 0
    num_batched_seqs: int = 0

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefill_seqs: int = 0
    num_decode_seqs: int = 0

    def can_schedule(self, num_tokens: int, num_seqs: int = 1) -> bool:
        return (
            self.num_batched_tokens + num_tokens <= self.token_budget
            and self.num_batched_seqs + num_seqs <= self.max_num_seqs
        )

    def add_prefill(self, num_tokens: int) -> None:
        self.num_batched_tokens += num_tokens
        self.num_batched_seqs += 1
        self.num_prefill_tokens += num_tokens
        self.num_prefill_seqs += 1

    def add_decode(self) -> None:
        self.num_batched_tokens += 1
        self.num_batched_seqs += 1
        self.num_decode_tokens += 1
        self.num_decode_seqs += 1

    @property
    def remaining_tokens(self) -> int:
        return self.token_budget - self.num_batched_tokens

    @property
    def remaining_seqs(self) -> int:
        return self.max_num_seqs - self.num_batched_seqs

class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        block_manager: BlockSpaceManager,
    ):
        self.config = scheduler_config
        self.cache_config = cache_config
        self.block_manager = block_manager

        self.waiting: deque[Request] = deque()
        self.running: List[Request] = []
        self.preempted: deque[Request] = deque()

        self.preemption_mode = PreemptionMode.RECOMPUTE

        self._request_id_to_request: Dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        self._request_id_to_request[request.request_id] = request

    def abort_request(self, request_id: str) -> Optional[Request]:
        request = self._request_id_to_request.get(request_id)
        if request is None:
            return None

        if request in self.waiting:
            self.waiting.remove(request)
        elif request in self.running:
            self.running.remove(request)
            self.block_manager.free(request_id)
        elif request in self.preempted:
            self.preempted.remove(request)

        request.status = RequestStatus.FINISHED
        del self._request_id_to_request[request_id]
        return request

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        budget = SchedulingBudget(
            token_budget=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
        )

        self._schedule_running(output, budget)

        self._schedule_preempted(output, budget)

        self._schedule_waiting(output, budget)

        output.num_prefill_tokens = budget.num_prefill_tokens
        output.num_decode_tokens = budget.num_decode_tokens

        return output

    def _schedule_running(
        self,
        output: SchedulerOutput,
        budget: SchedulingBudget,
    ) -> None:
        running_scheduled = []

        for request in self.running:

            if not budget.can_schedule(num_tokens=1):

                continue

            if not self.block_manager.can_append_slot(
                request.request_id,
                request.num_total_tokens + 1
            ):

                self._preempt_request(request, output)
                continue

            budget.add_decode()
            running_scheduled.append(request)
            output.scheduled_requests.append(request)

        self.running = running_scheduled

    def _schedule_preempted(
        self,
        output: SchedulerOutput,
        budget: SchedulingBudget,
    ) -> None:
        while self.preempted and budget.remaining_seqs > 0:
            request = self.preempted[0]

            if self.preemption_mode == PreemptionMode.RECOMPUTE:

                alloc_status = self.block_manager.can_allocate(
                    request.num_prompt_tokens
                )
                if not alloc_status.can_allocate:
                    break

                if self.config.chunked_prefill_enabled:
                    tokens_to_process = min(
                        request.num_prompt_tokens,
                        self.config.chunk_size,
                        budget.remaining_tokens,
                    )
                else:
                    tokens_to_process = min(
                        request.num_prompt_tokens,
                        budget.remaining_tokens,
                    )

                if tokens_to_process <= 0:
                    break

                self.preempted.popleft()
                self.block_manager.allocate(
                    request.request_id,
                    request.num_prompt_tokens
                )
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = 0

                budget.add_prefill(tokens_to_process)
                self.running.append(request)
                output.scheduled_requests.append(request)

    def _schedule_waiting(
        self,
        output: SchedulerOutput,
        budget: SchedulingBudget,
    ) -> None:
        while self.waiting and budget.remaining_seqs > 0:
            request = self.waiting[0]

            alloc_status = self.block_manager.can_allocate(
                request.num_prompt_tokens
            )
            if not alloc_status.can_allocate:

                break

            if self.config.chunked_prefill_enabled:
                tokens_to_process = min(
                    request.num_prompt_tokens - request.num_computed_tokens,
                    self.config.chunk_size,
                    budget.remaining_tokens,
                )
            else:
                tokens_to_process = min(
                    request.num_prompt_tokens - request.num_computed_tokens,
                    budget.remaining_tokens,
                )

            if tokens_to_process <= 0:
                break

            self.waiting.popleft()

            try:
                self.block_manager.allocate(
                    request.request_id,
                    request.num_prompt_tokens
                )
            except MemoryError:

                self.waiting.appendleft(request)
                break

            request.status = RequestStatus.RUNNING
            if request.first_scheduled_time is None:
                request.first_scheduled_time = time.time()

            budget.add_prefill(tokens_to_process)
            self.running.append(request)
            output.scheduled_requests.append(request)

    def _preempt_request(
        self,
        request: Request,
        output: SchedulerOutput,
    ) -> None:
        request.status = RequestStatus.PREEMPTED

        self.block_manager.free(request.request_id)

        if self.preemption_mode == PreemptionMode.RECOMPUTE:
            request.num_computed_tokens = 0

        self.preempted.appendleft(request)
        output.preempted_requests.append(request)

    def finish_request(self, request_id: str) -> Optional[Request]:
        request = self._request_id_to_request.get(request_id)
        if request is None:
            return None

        if request in self.running:
            self.running.remove(request)

        self.block_manager.free(request_id)

        request.status = RequestStatus.FINISHED
        del self._request_id_to_request[request_id]

        return request

    def update_computed_tokens(
        self,
        request: Request,
        num_tokens: int,
    ) -> None:
        request.num_computed_tokens += num_tokens

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running or self.preempted)

    def get_num_waiting(self) -> int:
        return len(self.waiting)

    def get_num_running(self) -> int:
        return len(self.running)

    def get_num_preempted(self) -> int:
        return len(self.preempted)

    def get_num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.preempted)

    def __repr__(self) -> str:
        return (
            f"Scheduler("
            f"waiting={len(self.waiting)}, "
            f"running={len(self.running)}, "
            f"preempted={len(self.preempted)})"
        )
