from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import torch.nn.functional as F

from mini_vllm.sampling.sampling_params import SamplingParams

@dataclass
class SamplerOutput:

    token_ids: List[int]

    logprobs: Optional[List[Optional[Dict[int, float]]]] = None

class Sampler:

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def forward(
        self,
        logits: torch.Tensor,
        sampling_params_list: List[SamplingParams],
        output_token_ids: Optional[List[List[int]]] = None,
    ) -> SamplerOutput:
        batch_size = logits.shape[0]
        assert len(sampling_params_list) == batch_size

        token_ids = []
        all_logprobs = []

        for i, params in enumerate(sampling_params_list):
            seq_logits = logits[i].clone()

            if output_token_ids is not None and len(output_token_ids) > i:
                seq_logits = self._apply_repetition_penalties(
                    seq_logits,
                    output_token_ids[i],
                    params,
                )

            if params.temperature > 0:
                seq_logits = seq_logits / params.temperature

            if params.min_p > 0:
                seq_logits = self._apply_min_p(seq_logits, params.min_p)

            if params.top_k > 0:
                seq_logits = self._apply_top_k(seq_logits, params.top_k)

            if params.top_p < 1.0:
                seq_logits = self._apply_top_p(seq_logits, params.top_p)

            if params.is_greedy:
                token_id = seq_logits.argmax().item()
            else:
                probs = F.softmax(seq_logits, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()

            token_ids.append(token_id)

            if params.logprobs is not None:
                log_probs = F.log_softmax(seq_logits, dim=-1)
                top_logprobs = self._get_top_logprobs(log_probs, params.logprobs)
                all_logprobs.append(top_logprobs)
            else:
                all_logprobs.append(None)

        return SamplerOutput(
            token_ids=token_ids,
            logprobs=all_logprobs if any(lp is not None for lp in all_logprobs) else None,
        )

    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return logits
        return logits / temperature

    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.shape[-1]:
            return logits

        top_k_values, _ = torch.topk(logits, top_k)
        min_top_k = top_k_values[-1]

        logits = torch.where(
            logits < min_top_k,
            torch.full_like(logits, float("-inf")),
            logits,
        )
        return logits

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return logits

    def _apply_min_p(self, logits: torch.Tensor, min_p: float) -> torch.Tensor:
        if min_p <= 0:
            return logits

        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max()
        threshold = min_p * max_prob

        logits = torch.where(
            probs < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )
        return logits

    def _apply_repetition_penalties(
        self,
        logits: torch.Tensor,
        output_tokens: List[int],
        params: SamplingParams,
    ) -> torch.Tensor:
        if not output_tokens:
            return logits

        token_counts: Dict[int, int] = {}
        for token_id in output_tokens:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

        for token_id, count in token_counts.items():
            if token_id >= logits.shape[-1]:
                continue

            if params.repetition_penalty != 1.0:
                if logits[token_id] > 0:
                    logits[token_id] /= params.repetition_penalty
                else:
                    logits[token_id] *= params.repetition_penalty

            if params.presence_penalty != 0:
                logits[token_id] -= params.presence_penalty

            if params.frequency_penalty != 0:
                logits[token_id] -= params.frequency_penalty * count

        return logits

    def _get_top_logprobs(
        self,
        log_probs: torch.Tensor,
        num_logprobs: int,
    ) -> Dict[int, float]:
        top_values, top_indices = torch.topk(log_probs, min(num_logprobs, log_probs.shape[-1]))
        return {
            idx.item(): val.item()
            for idx, val in zip(top_indices, top_values)
        }

def create_sampler(vocab_size: int) -> Sampler:
    return Sampler(vocab_size)
