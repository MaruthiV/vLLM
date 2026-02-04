from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class SamplingParams:

    temperature: float = 1.0

    top_p: float = 1.0

    top_k: int = -1

    min_p: float = 0.0

    max_tokens: int = 256

    min_tokens: int = 0

    stop: Optional[List[str]] = None

    stop_token_ids: Optional[List[int]] = None

    include_stop_str_in_output: bool = False

    ignore_eos: bool = False

    presence_penalty: float = 0.0

    frequency_penalty: float = 0.0

    repetition_penalty: float = 1.0

    n: int = 1

    best_of: Optional[int] = None

    use_beam_search: bool = False

    logprobs: Optional[int] = None

    prompt_logprobs: Optional[int] = None

    guided_json: Optional[Union[str, dict]] = None

    guided_regex: Optional[str] = None

    guided_grammar: Optional[str] = None

    seed: Optional[int] = None

    skip_special_tokens: bool = True

    spaces_between_special_tokens: bool = True

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")

        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")

        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disabled) or positive, got {self.top_k}")

        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")

        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be non-negative, got {self.min_tokens}")

        if self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) cannot exceed max_tokens ({self.max_tokens})"
            )

        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}"
            )

        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}"
            )

        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be positive, got {self.repetition_penalty}"
            )

        if self.n < 1:
            raise ValueError(f"n must be positive, got {self.n}")

        if self.best_of is not None and self.best_of < self.n:
            raise ValueError(
                f"best_of ({self.best_of}) must be >= n ({self.n})"
            )

        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}")

        guided_options = [
            self.guided_json is not None,
            self.guided_regex is not None,
            self.guided_grammar is not None,
        ]
        if sum(guided_options) > 1:
            raise ValueError("Only one guided decoding option can be specified at a time")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0 or self.top_k == 1

    @property
    def has_guided_decoding(self) -> bool:
        return any([
            self.guided_json is not None,
            self.guided_regex is not None,
            self.guided_grammar is not None,
        ])

    def clone(self) -> "SamplingParams":
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            stop=self.stop.copy() if self.stop else None,
            stop_token_ids=self.stop_token_ids.copy() if self.stop_token_ids else None,
            include_stop_str_in_output=self.include_stop_str_in_output,
            ignore_eos=self.ignore_eos,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            n=self.n,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs,
            guided_json=self.guided_json,
            guided_regex=self.guided_regex,
            guided_grammar=self.guided_grammar,
            seed=self.seed,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )
