from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union
import json
import torch

from mini_vllm.guided.fsm import GuidedFSM, OutlinesGuidedFSM, create_guided_fsm
from mini_vllm.guided.grammar import Grammar

@dataclass
class GuidedDecodingConfig:

    json_schema: Optional[Dict[str, Any]] = None

    regex_pattern: Optional[str] = None

    choice: Optional[List[str]] = None

    use_outlines: bool = True

class GuidedLogitsProcessor:

    MASK_VALUE = -1e10

    def __init__(
        self,
        schema: Union[str, Dict[str, Any]],
        tokenizer,
        use_outlines: bool = True,
    ):
        if isinstance(schema, str):
            schema = json.loads(schema)

        self.schema = schema
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self.fsm = create_guided_fsm(schema, tokenizer, use_outlines)

        self._mask_cache: Dict[int, torch.Tensor] = {}

        self.generated_tokens: List[int] = []

    def __call__(
        self,
        logits: torch.Tensor,
        generated_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:

        valid_tokens = self.fsm.get_valid_tokens()

        if not valid_tokens:

            return logits

        mask = self._get_mask(valid_tokens, logits.device)

        if logits.dim() == 1:
            masked_logits = logits + mask
        else:

            masked_logits = logits + mask.unsqueeze(0)

        return masked_logits

    def _get_mask(
        self,
        valid_tokens: Set[int],
        device: torch.device,
    ) -> torch.Tensor:

        cache_key = hash(frozenset(valid_tokens))

        if cache_key not in self._mask_cache:

            mask = torch.full(
                (self.vocab_size,),
                self.MASK_VALUE,
                device=device,
            )
            for token_id in valid_tokens:
                if 0 <= token_id < self.vocab_size:
                    mask[token_id] = 0.0

            self._mask_cache[cache_key] = mask

        return self._mask_cache[cache_key].to(device)

    def advance(self, token_id: int) -> bool:
        self.generated_tokens.append(token_id)
        return self.fsm.advance(token_id)

    def is_complete(self) -> bool:
        return self.fsm.is_complete()

    def reset(self) -> None:
        self.fsm.reset()
        self.generated_tokens = []
        self._mask_cache.clear()

    def get_generated_text(self) -> str:
        return self.tokenizer.decode(self.generated_tokens)

    def validate_output(self) -> bool:
        try:
            text = self.get_generated_text()
            data = json.loads(text)

            return True
        except (json.JSONDecodeError, ValueError):
            return False

class ChoiceLogitsProcessor:

    MASK_VALUE = -1e10

    def __init__(
        self,
        choices: List[str],
        tokenizer,
    ):
        self.choices = choices
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self.choice_token_ids = [
            tokenizer.encode(choice, add_special_tokens=False)
            for choice in choices
        ]

        self.generated_tokens: List[int] = []

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        pos = len(self.generated_tokens)

        valid_tokens = set()
        for choice_ids in self.choice_token_ids:
            if pos < len(choice_ids):

                if self._is_prefix_match(choice_ids, self.generated_tokens):
                    valid_tokens.add(choice_ids[pos])

        if not valid_tokens:
            return logits

        mask = torch.full((self.vocab_size,), self.MASK_VALUE, device=logits.device)
        for token_id in valid_tokens:
            mask[token_id] = 0.0

        if logits.dim() == 1:
            return logits + mask
        return logits + mask.unsqueeze(0)

    def _is_prefix_match(
        self,
        choice_ids: List[int],
        generated: List[int],
    ) -> bool:
        if len(generated) > len(choice_ids):
            return False
        return choice_ids[:len(generated)] == generated

    def advance(self, token_id: int) -> bool:
        self.generated_tokens.append(token_id)
        return True

    def is_complete(self) -> bool:
        for choice_ids in self.choice_token_ids:
            if self.generated_tokens == choice_ids:
                return True
        return False

    def reset(self) -> None:
        self.generated_tokens = []

class RegexLogitsProcessor:

    MASK_VALUE = -1e10

    def __init__(
        self,
        pattern: str,
        tokenizer,
    ):
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self._fsm = None

        self._init_outlines()

        self.generated_tokens: List[int] = []

    def _init_outlines(self) -> None:
        try:
            from outlines.fsm.regex import RegexFSM
            from outlines.models.transformers import TransformerTokenizer

            outlines_tokenizer = TransformerTokenizer(self.tokenizer)
            self._fsm = RegexFSM(self.pattern, outlines_tokenizer)
            self._current_state = self._fsm.first_state
            print(f"Regex FSM initialized for pattern: {self.pattern[:50]}...")

        except ImportError:
            print("Warning: outlines not installed. Regex constraint disabled.")
        except Exception as e:
            print(f"Warning: Failed to create regex FSM: {e}")

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self._fsm is None:
            return logits

        try:
            allowed = self._fsm.allowed_token_ids(self._current_state)
            valid_tokens = set(allowed)
        except Exception:
            return logits

        if not valid_tokens:
            return logits

        mask = torch.full((self.vocab_size,), self.MASK_VALUE, device=logits.device)
        for token_id in valid_tokens:
            if 0 <= token_id < self.vocab_size:
                mask[token_id] = 0.0

        if logits.dim() == 1:
            return logits + mask
        return logits + mask.unsqueeze(0)

    def advance(self, token_id: int) -> bool:
        self.generated_tokens.append(token_id)
        if self._fsm is None:
            return True

        try:
            self._current_state = self._fsm.next_state(
                self._current_state, token_id
            )
            return True
        except Exception:
            return False

    def is_complete(self) -> bool:
        if self._fsm is None:
            return False
        try:
            return self._fsm.is_final_state(self._current_state)
        except Exception:
            return False

    def reset(self) -> None:
        self.generated_tokens = []
        if self._fsm is not None:
            self._current_state = self._fsm.first_state

def create_guided_processor(
    config: GuidedDecodingConfig,
    tokenizer,
) -> Optional[Union[GuidedLogitsProcessor, ChoiceLogitsProcessor, RegexLogitsProcessor]]:
    if config.json_schema is not None:
        return GuidedLogitsProcessor(
            schema=config.json_schema,
            tokenizer=tokenizer,
            use_outlines=config.use_outlines,
        )
    elif config.choice is not None:
        return ChoiceLogitsProcessor(
            choices=config.choice,
            tokenizer=tokenizer,
        )
    elif config.regex_pattern is not None:
        return RegexLogitsProcessor(
            pattern=config.regex_pattern,
            tokenizer=tokenizer,
        )

    return None
