from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import re

from mini_vllm.guided.grammar import Grammar, GrammarRule

@dataclass
class FSMState:

    state_id: int
    is_final: bool = False
    transitions: Dict[str, int] = field(default_factory=dict)
    valid_tokens: Optional[Set[int]] = None

class GuidedFSM:

    def __init__(
        self,
        states: Dict[int, FSMState],
        initial_state: int,
        vocabulary: Dict[str, int],
    ):
        self.states = states
        self.initial_state = initial_state
        self.vocabulary = vocabulary
        self.current_state = initial_state

        self.id_to_token = {v: k for k, v in vocabulary.items()}

    @classmethod
    def from_grammar(
        cls,
        grammar: Grammar,
        tokenizer,
    ) -> "GuidedFSM":

        vocabulary = tokenizer.get_vocab()

        states = cls._build_json_fsm_states(grammar, vocabulary)
        initial_state = 0

        return cls(states, initial_state, vocabulary)

    @classmethod
    def _build_json_fsm_states(
        cls,
        grammar: Grammar,
        vocabulary: Dict[str, int],
    ) -> Dict[int, FSMState]:
        states = {}

        states[0] = FSMState(
            state_id=0,
            is_final=False,
            valid_tokens=cls._find_tokens_starting_with(vocabulary, ['{', '[', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f', 'n']),
        )

        states[1] = FSMState(
            state_id=1,
            is_final=False,
            valid_tokens=cls._find_tokens_starting_with(vocabulary, ['"', '}', ' ', '\n', '\t']),
        )

        states[2] = FSMState(
            state_id=2,
            is_final=False,
            valid_tokens=cls._find_tokens_containing(vocabulary, ':'),
        )

        states[3] = FSMState(
            state_id=3,
            is_final=False,
            valid_tokens=cls._find_tokens_starting_with(vocabulary, ['{', '[', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f', 'n', ' ']),
        )

        states[4] = FSMState(
            state_id=4,
            is_final=False,
            valid_tokens=cls._find_tokens_starting_with(vocabulary, [',', '}', ']', ' ', '\n']),
        )

        states[5] = FSMState(
            state_id=5,
            is_final=True,
            valid_tokens=set(),
        )

        return states

    @staticmethod
    def _find_tokens_starting_with(
        vocabulary: Dict[str, int],
        prefixes: List[str],
    ) -> Set[int]:
        valid = set()
        for token, token_id in vocabulary.items():

            decoded = token.replace('Ġ', ' ').replace('▁', ' ').replace('Ċ', '\n')
            for prefix in prefixes:
                if decoded.startswith(prefix) or token.startswith(prefix):
                    valid.add(token_id)
                    break
        return valid

    @staticmethod
    def _find_tokens_containing(
        vocabulary: Dict[str, int],
        char: str,
    ) -> Set[int]:
        valid = set()
        for token, token_id in vocabulary.items():
            decoded = token.replace('Ġ', ' ').replace('▁', ' ').replace('Ċ', '\n')
            if char in decoded or char in token:
                valid.add(token_id)
        return valid

    def get_valid_tokens(self) -> Set[int]:
        state = self.states.get(self.current_state)
        if state is None or state.valid_tokens is None:

            return set(self.vocabulary.values())
        return state.valid_tokens

    def advance(self, token_id: int) -> bool:
        token = self.id_to_token.get(token_id, "")

        current = self.states.get(self.current_state)
        if current is None:
            return False

        if current.valid_tokens and token_id not in current.valid_tokens:
            return False

        decoded = token.replace('Ġ', ' ').replace('▁', ' ').replace('Ċ', '\n')

        if self.current_state == 0:
            if '{' in decoded:
                self.current_state = 1
            elif '[' in decoded:
                self.current_state = 3
        elif self.current_state == 1:
            if '"' in decoded:
                self.current_state = 2
            elif '}' in decoded:
                self.current_state = 5
        elif self.current_state == 2:
            if ':' in decoded:
                self.current_state = 3
        elif self.current_state == 3:

            self.current_state = 4
        elif self.current_state == 4:
            if ',' in decoded:
                self.current_state = 1
            elif '}' in decoded or ']' in decoded:
                self.current_state = 5

        return True

    def is_complete(self) -> bool:
        state = self.states.get(self.current_state)
        return state is not None and state.is_final

    def reset(self) -> None:
        self.current_state = self.initial_state

class OutlinesGuidedFSM:

    def __init__(
        self,
        schema: Dict[str, Any],
        tokenizer,
    ):
        self.schema = schema
        self.tokenizer = tokenizer
        self._fsm = None
        self._guide = None

        self._init_outlines()

    def _init_outlines(self) -> None:
        try:
            from outlines.fsm.json_schema import build_regex_from_schema
            from outlines.fsm.regex import RegexFSM
            from outlines.models.transformers import TransformerTokenizer

            regex_str = build_regex_from_schema(json.dumps(self.schema))

            outlines_tokenizer = TransformerTokenizer(self.tokenizer)

            self._fsm = RegexFSM(regex_str, outlines_tokenizer)
            self._current_state = self._fsm.first_state

            print("Outlines FSM initialized successfully")

        except ImportError:
            print("Warning: outlines not installed. Using fallback FSM.")
            self._fsm = None
        except Exception as e:
            print(f"Warning: Failed to initialize outlines FSM: {e}")
            self._fsm = None

    @property
    def is_available(self) -> bool:
        return self._fsm is not None

    def get_valid_tokens(self) -> Set[int]:
        if self._fsm is None:
            return set(range(self.tokenizer.vocab_size))

        try:

            allowed = self._fsm.allowed_token_ids(self._current_state)
            return set(allowed)
        except Exception:
            return set(range(self.tokenizer.vocab_size))

    def advance(self, token_id: int) -> bool:
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
        if self._fsm is not None:
            self._current_state = self._fsm.first_state

def create_guided_fsm(
    schema: Union[str, Dict[str, Any]],
    tokenizer,
    use_outlines: bool = True,
) -> Union[GuidedFSM, OutlinesGuidedFSM]:
    if isinstance(schema, str):
        schema = json.loads(schema)

    if use_outlines:
        fsm = OutlinesGuidedFSM(schema, tokenizer)
        if fsm.is_available:
            return fsm

    from mini_vllm.guided.grammar import parse_json_schema
    grammar = parse_json_schema(schema)
    return GuidedFSM.from_grammar(grammar, tokenizer)
