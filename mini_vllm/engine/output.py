from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class CompletionOutput:

    index: int

    text: str

    token_ids: List[int]

    cumulative_logprob: float = 0.0

    logprobs: Optional[List[Dict[int, float]]] = None

    finish_reason: Optional[str] = None

@dataclass
class RequestMetrics:

    arrival_time: float

    first_scheduled_time: Optional[float] = None

    first_token_time: Optional[float] = None

    finish_time: Optional[float] = None

    @property
    def time_to_first_token(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def total_time(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def generation_time(self) -> Optional[float]:
        if self.first_token_time is None or self.finish_time is None:
            return None
        return self.finish_time - self.first_token_time

@dataclass
class RequestOutput:

    request_id: str

    prompt: str

    prompt_token_ids: List[int]

    outputs: List[CompletionOutput]

    finished: bool

    metrics: Optional[RequestMetrics] = None

    @property
    def output(self) -> Optional[CompletionOutput]:
        if not self.outputs:
            return None
        return self.outputs[0]

    @property
    def generated_text(self) -> str:
        if not self.outputs:
            return ""
        return self.outputs[0].text

    @property
    def generated_token_ids(self) -> List[int]:
        if not self.outputs:
            return []
        return self.outputs[0].token_ids

@dataclass
class EmbeddingOutput:

    request_id: str
    embedding: List[float]
    finished: bool = True
