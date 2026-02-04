from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from mini_vllm.config import VllmConfig, ModelConfig, SpeculativeConfig
from mini_vllm.model.loader import load_model_and_tokenizer
from mini_vllm.sampling.sampling_params import SamplingParams

@dataclass
class DraftOutput:

    draft_token_ids: List[int]
    draft_probs: torch.Tensor
    draft_logits: torch.Tensor

    @property
    def num_draft_tokens(self) -> int:
        return len(self.draft_token_ids)

class DraftModelRunner:

    def __init__(
        self,
        spec_config: SpeculativeConfig,
        device: torch.device,
        target_tokenizer=None,
    ):
        self.spec_config = spec_config
        self.device = device
        self.num_speculative_tokens = spec_config.num_speculative_tokens

        print(f"Loading draft model: {spec_config.draft_model}")
        model_config = ModelConfig(
            model_name_or_path=spec_config.draft_model,
            dtype=spec_config.draft_model_dtype,
        )

        self.model, self.tokenizer = load_model_and_tokenizer(model_config)
        self.model = self.model.to(device)
        self.model.eval()

        if target_tokenizer is not None:
            self._validate_vocab_compatibility(target_tokenizer)

        self.vocab_size = self.model.config.vocab_size
        print(f"Draft model loaded: vocab_size={self.vocab_size}")

    def _validate_vocab_compatibility(self, target_tokenizer) -> None:
        draft_vocab_size = len(self.tokenizer)
        target_vocab_size = len(target_tokenizer)

        if draft_vocab_size != target_vocab_size:
            print(
                f"Warning: Vocab size mismatch - draft={draft_vocab_size}, "
                f"target={target_vocab_size}. Speculative decoding may not work correctly."
            )

    @torch.no_grad()
    def speculate(
        self,
        input_ids: torch.Tensor,
        num_speculative_tokens: Optional[int] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> DraftOutput:
        if num_speculative_tokens is None:
            num_speculative_tokens = self.num_speculative_tokens

        input_ids = input_ids.to(self.device)

        if sampling_params is None:
            temperature = 1.0
        else:
            temperature = sampling_params.temperature or 1.0

        draft_token_ids: List[int] = []
        draft_probs_list: List[torch.Tensor] = []
        draft_logits_list: List[torch.Tensor] = []

        current_ids = input_ids

        for _ in range(num_speculative_tokens):

            outputs = self.model(current_ids)
            logits = outputs.logits[:, -1, :]

            if temperature != 1.0:
                scaled_logits = logits / temperature
            else:
                scaled_logits = logits

            probs = F.softmax(scaled_logits, dim=-1)

            if sampling_params is not None and sampling_params.temperature > 0:

                next_token = torch.multinomial(probs, num_samples=1)
            else:

                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            next_token_id = next_token.item()
            draft_token_ids.append(next_token_id)
            draft_probs_list.append(probs.squeeze(0))
            draft_logits_list.append(logits.squeeze(0))

            current_ids = torch.cat([current_ids, next_token], dim=1)

        draft_probs = torch.stack(draft_probs_list, dim=0)
        draft_logits = torch.stack(draft_logits_list, dim=0)

        return DraftOutput(
            draft_token_ids=draft_token_ids,
            draft_probs=draft_probs,
            draft_logits=draft_logits,
        )

    @torch.no_grad()
    def speculate_batch(
        self,
        input_ids_list: List[torch.Tensor],
        num_speculative_tokens: Optional[int] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[DraftOutput]:
        return [
            self.speculate(input_ids, num_speculative_tokens, sampling_params)
            for input_ids in input_ids_list
        ]

    def get_model_info(self) -> dict:
        return {
            "model_name": self.spec_config.draft_model,
            "vocab_size": self.vocab_size,
            "dtype": self.spec_config.draft_model_dtype,
            "num_speculative_tokens": self.num_speculative_tokens,
        }

def create_draft_model_runner(
    draft_model: str,
    device: torch.device,
    num_speculative_tokens: int = 5,
    dtype: str = "float16",
    target_tokenizer=None,
) -> DraftModelRunner:
    spec_config = SpeculativeConfig(
        enabled=True,
        draft_model=draft_model,
        num_speculative_tokens=num_speculative_tokens,
        draft_model_dtype=dtype,
    )
    return DraftModelRunner(spec_config, device, target_tokenizer)
