from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import time

from mini_vllm.sampling.sampling_params import SamplingParams

class RequestStatus(Enum):

    WAITING = "waiting"

    RUNNING = "running"

    PREEMPTED = "preempted"

    FINISHED = "finished"

@dataclass
class Request:

    request_id: str

    prompt: str

    prompt_token_ids: List[int]

    sampling_params: SamplingParams

    arrival_time: float = field(default_factory=time.time)

    status: RequestStatus = RequestStatus.WAITING

    output_token_ids: List[int] = field(default_factory=list)

    block_table: List[int] = field(default_factory=list)

    num_computed_tokens: int = 0

    first_scheduled_time: Optional[float] = None

    first_token_time: Optional[float] = None

    finish_time: Optional[float] = None

    num_cached_tokens: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_num_uncomputed_tokens(self) -> int:
        return self.num_total_tokens - self.num_computed_tokens

    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens

    def is_decode(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    def get_num_new_tokens(self) -> int:
        if self.is_prefill():
            return self.get_num_uncomputed_tokens()
        else:
            return 1

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

        if len(self.output_token_ids) == 1:
            self.first_token_time = time.time()

    def mark_finished(self, finish_reason: str = "stop") -> None:
        self.status = RequestStatus.FINISHED
        self.finish_time = time.time()

    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    def should_stop(self, eos_token_id: Optional[int] = None) -> Optional[str]:
        params = self.sampling_params

        if self.num_output_tokens >= params.max_tokens:
            return "length"

        if self.num_output_tokens < params.min_tokens:
            return None

        if not params.ignore_eos and eos_token_id is not None:
            if self.output_token_ids and self.output_token_ids[-1] == eos_token_id:
                return "stop"

        if params.stop_token_ids and self.output_token_ids:
            if self.output_token_ids[-1] in params.stop_token_ids:
                return "stop"

        return None

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id}, "
            f"status={self.status.value}, "
            f"prompt_tokens={self.num_prompt_tokens}, "
            f"output_tokens={self.num_output_tokens}, "
            f"computed={self.num_computed_tokens})"
        )
