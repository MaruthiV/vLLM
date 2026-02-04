from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterator
import time
import torch
import torch.nn.functional as F

from mini_vllm.config import VllmConfig, SpeculativeConfig
from mini_vllm.sampling.sampling_params import SamplingParams
from mini_vllm.speculative.draft_model import DraftModelRunner, DraftOutput
from mini_vllm.speculative.rejection_sampler import (
    RejectionSampler,
    RejectionSamplerOutput,
    SpeculativeVerifier,
)

@dataclass
class SpeculativeOutput:

    new_token_ids: List[int]
    num_draft_tokens: int
    num_accepted: int
    bonus_token_id: Optional[int]
    iteration_time: float

    @property
    def tokens_per_iteration(self) -> int:
        return len(self.new_token_ids)

    @property
    def acceptance_rate(self) -> float:
        if self.num_draft_tokens == 0:
            return 0.0
        return self.num_accepted / self.num_draft_tokens

@dataclass
class SpeculativeMetrics:

    total_iterations: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_generated_tokens: int = 0
    total_time: float = 0.0
    draft_time: float = 0.0
    verify_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_iteration(self) -> float:
        if self.total_iterations == 0:
            return 0.0
        return self.total_generated_tokens / self.total_iterations

    @property
    def speedup_factor(self) -> float:
        return self.tokens_per_iteration

    def update(self, output: SpeculativeOutput) -> None:
        self.total_iterations += 1
        self.total_draft_tokens += output.num_draft_tokens
        self.total_accepted_tokens += output.num_accepted
        self.total_generated_tokens += output.tokens_per_iteration
        self.total_time += output.iteration_time

    def to_dict(self) -> dict:
        return {
            "total_iterations": self.total_iterations,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_iteration": self.tokens_per_iteration,
            "speedup_factor": self.speedup_factor,
            "total_time": self.total_time,
        }

class SpeculativeDecodeWorker:

    def __init__(
        self,
        target_model: torch.nn.Module,
        draft_runner: DraftModelRunner,
        device: torch.device,
        num_speculative_tokens: int = 5,
    ):
        self.target_model = target_model
        self.draft_runner = draft_runner
        self.device = device
        self.num_speculative_tokens = num_speculative_tokens

        self.rejection_sampler = RejectionSampler()
        self.verifier = SpeculativeVerifier(self.rejection_sampler)

        self.metrics = SpeculativeMetrics()

    def _run_target_model(
        self,
        input_ids: torch.Tensor,
        num_positions: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.target_model(input_ids)

            logits = outputs.logits[0, -num_positions:, :]
            return logits

    def _speculative_step(
        self,
        input_ids: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> SpeculativeOutput:
        start_time = time.time()
        K = self.num_speculative_tokens

        draft_start = time.time()
        draft_output = self.draft_runner.speculate(
            input_ids=input_ids,
            num_speculative_tokens=K,
            sampling_params=sampling_params,
        )
        draft_time = time.time() - draft_start

        draft_tokens_tensor = torch.tensor(
            [draft_output.draft_token_ids],
            device=self.device,
            dtype=input_ids.dtype,
        )
        extended_ids = torch.cat([input_ids, draft_tokens_tensor], dim=1)

        verify_start = time.time()
        target_logits = self._run_target_model(
            extended_ids,
            num_positions=K + 1,
        )
        verify_time = time.time() - verify_start

        target_logits_for_draft = target_logits[:K]

        verification_output = self.verifier.verify(
            draft_token_ids=draft_output.draft_token_ids,
            draft_probs=draft_output.draft_probs,
            target_logits=target_logits_for_draft,
            temperature=sampling_params.temperature or 1.0,
        )

        new_token_ids = list(verification_output.accepted_token_ids)

        if verification_output.num_accepted == K:
            bonus_logits = target_logits[K]
            if sampling_params.temperature and sampling_params.temperature > 0:
                bonus_probs = F.softmax(
                    bonus_logits / sampling_params.temperature, dim=-1
                )
                bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
            else:
                bonus_token = torch.argmax(bonus_logits).item()
            new_token_ids.append(bonus_token)
            bonus_token_id = bonus_token
        else:
            bonus_token_id = verification_output.bonus_token_id
            if bonus_token_id is not None:
                new_token_ids.append(bonus_token_id)

        iteration_time = time.time() - start_time

        output = SpeculativeOutput(
            new_token_ids=new_token_ids,
            num_draft_tokens=K,
            num_accepted=verification_output.num_accepted,
            bonus_token_id=bonus_token_id,
            iteration_time=iteration_time,
        )

        self.metrics.update(output)
        self.metrics.draft_time += draft_time
        self.metrics.verify_time += verify_time

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> Iterator[SpeculativeOutput]:
        current_ids = input_ids.to(self.device)
        generated_tokens = 0
        max_tokens = sampling_params.max_tokens or 100

        stop_token_ids = set()
        if hasattr(self.target_model.config, 'eos_token_id'):
            eos = self.target_model.config.eos_token_id
            if isinstance(eos, int):
                stop_token_ids.add(eos)
            elif isinstance(eos, list):
                stop_token_ids.update(eos)

        while generated_tokens < max_tokens:

            output = self._speculative_step(current_ids, sampling_params)

            tokens_to_add = []
            for token_id in output.new_token_ids:
                if token_id in stop_token_ids:
                    if tokens_to_add:

                        partial_output = SpeculativeOutput(
                            new_token_ids=tokens_to_add,
                            num_draft_tokens=output.num_draft_tokens,
                            num_accepted=len(tokens_to_add),
                            bonus_token_id=None,
                            iteration_time=output.iteration_time,
                        )
                        yield partial_output
                    return
                tokens_to_add.append(token_id)
                generated_tokens += 1
                if generated_tokens >= max_tokens:
                    break

            if not tokens_to_add:
                break

            if len(tokens_to_add) < len(output.new_token_ids):
                output = SpeculativeOutput(
                    new_token_ids=tokens_to_add,
                    num_draft_tokens=output.num_draft_tokens,
                    num_accepted=min(output.num_accepted, len(tokens_to_add)),
                    bonus_token_id=output.bonus_token_id if len(tokens_to_add) > output.num_accepted else None,
                    iteration_time=output.iteration_time,
                )

            new_tokens_tensor = torch.tensor(
                [tokens_to_add],
                device=self.device,
                dtype=current_ids.dtype,
            )
            current_ids = torch.cat([current_ids, new_tokens_tensor], dim=1)

            yield output

    def generate_to_completion(
        self,
        input_ids: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], SpeculativeMetrics]:
        all_tokens: List[int] = []

        for output in self.generate(input_ids, sampling_params):
            all_tokens.extend(output.new_token_ids)

        return all_tokens, self.metrics

    def reset_metrics(self) -> None:
        self.metrics = SpeculativeMetrics()
        self.verifier.reset_metrics()

    def get_metrics(self) -> dict:
        return {
            **self.metrics.to_dict(),
            "verifier_metrics": self.verifier.get_metrics(),
        }

def create_speculative_worker(
    target_model: torch.nn.Module,
    draft_model_name: str,
    device: torch.device,
    num_speculative_tokens: int = 5,
    draft_dtype: str = "float16",
    target_tokenizer=None,
) -> SpeculativeDecodeWorker:
    from mini_vllm.speculative.draft_model import create_draft_model_runner

    draft_runner = create_draft_model_runner(
        draft_model=draft_model_name,
        device=device,
        num_speculative_tokens=num_speculative_tokens,
        dtype=draft_dtype,
        target_tokenizer=target_tokenizer,
    )

    return SpeculativeDecodeWorker(
        target_model=target_model,
        draft_runner=draft_runner,
        device=device,
        num_speculative_tokens=num_speculative_tokens,
    )
