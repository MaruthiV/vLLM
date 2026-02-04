from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

@dataclass
class RejectionSamplerOutput:

    accepted_token_ids: List[int]
    num_accepted: int
    bonus_token_id: Optional[int]
    total_tokens: int
    acceptance_mask: List[bool]

    @property
    def all_token_ids(self) -> List[int]:
        tokens = list(self.accepted_token_ids)
        if self.bonus_token_id is not None:
            tokens.append(self.bonus_token_id)
        return tokens

class RejectionSampler:

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def sample(
        self,
        draft_token_ids: List[int],
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        temperature: float = 1.0,
    ) -> RejectionSamplerOutput:
        K = len(draft_token_ids)
        device = draft_probs.device

        if temperature != 1.0 and temperature > 0:

            target_probs = target_probs ** (1.0 / temperature)
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)

        accepted_token_ids: List[int] = []
        acceptance_mask: List[bool] = []
        num_accepted = 0
        bonus_token_id = None

        for i in range(K):
            draft_token = draft_token_ids[i]

            q = draft_probs[i]
            p = target_probs[i]

            q_draft = q[draft_token].item()
            p_draft = p[draft_token].item()

            if q_draft > 0:
                acceptance_prob = min(1.0, p_draft / q_draft)
            else:

                acceptance_prob = 1.0 if p_draft > 0 else 0.0

            r = torch.rand(1, device=device).item()

            if r < acceptance_prob:

                accepted_token_ids.append(draft_token)
                acceptance_mask.append(True)
                num_accepted += 1
            else:

                acceptance_mask.append(False)
                bonus_token_id = self._sample_residual(p, q, device)
                break

        if num_accepted == K and bonus_token_id is None:

            pass

        total_tokens = num_accepted + (1 if bonus_token_id is not None else 0)

        return RejectionSamplerOutput(
            accepted_token_ids=accepted_token_ids,
            num_accepted=num_accepted,
            bonus_token_id=bonus_token_id,
            total_tokens=total_tokens,
            acceptance_mask=acceptance_mask,
        )

    def _sample_residual(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        device: torch.device,
    ) -> int:

        residual = torch.clamp(p - q, min=0)

        residual_sum = residual.sum()

        if residual_sum > 0:
            residual_probs = residual / residual_sum

            token_id = torch.multinomial(residual_probs, num_samples=1).item()
        else:

            token_id = torch.multinomial(p, num_samples=1).item()

        return token_id

    def compute_acceptance_rate(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        draft_token_ids: List[int],
    ) -> float:
        K = len(draft_token_ids)
        total_acceptance = 0.0

        for i in range(K):
            draft_token = draft_token_ids[i]
            q = draft_probs[i, draft_token].item()
            p = target_probs[i, draft_token].item()

            if q > 0:
                acceptance_prob = min(1.0, p / q)
            else:
                acceptance_prob = 1.0 if p > 0 else 0.0

            total_acceptance += acceptance_prob

        return total_acceptance / K if K > 0 else 0.0

class SpeculativeVerifier:

    def __init__(self, rejection_sampler: Optional[RejectionSampler] = None):
        self.rejection_sampler = rejection_sampler or RejectionSampler()

        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.num_verification_rounds = 0

    def verify(
        self,
        draft_token_ids: List[int],
        draft_probs: torch.Tensor,
        target_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> RejectionSamplerOutput:

        if temperature > 0:
            target_probs = F.softmax(target_logits / temperature, dim=-1)
        else:

            target_probs = torch.zeros_like(target_logits)
            argmax_indices = target_logits.argmax(dim=-1)
            for i, idx in enumerate(argmax_indices):
                target_probs[i, idx] = 1.0

        output = self.rejection_sampler.sample(
            draft_token_ids=draft_token_ids,
            draft_probs=draft_probs,
            target_probs=target_probs,
            temperature=temperature,
        )

        self.total_draft_tokens += len(draft_token_ids)
        self.total_accepted_tokens += output.num_accepted
        self.num_verification_rounds += 1

        return output

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def average_accepted_per_round(self) -> float:
        if self.num_verification_rounds == 0:
            return 0.0
        return self.total_accepted_tokens / self.num_verification_rounds

    def reset_metrics(self) -> None:
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.num_verification_rounds = 0

    def get_metrics(self) -> dict:
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "num_verification_rounds": self.num_verification_rounds,
            "acceptance_rate": self.acceptance_rate,
            "avg_accepted_per_round": self.average_accepted_per_round,
        }
