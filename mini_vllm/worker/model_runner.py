from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
from transformers import PreTrainedModel

from mini_vllm.config import VllmConfig
from mini_vllm.model.loader import load_model_and_tokenizer, get_num_layers
from mini_vllm.worker.cache_engine import CacheEngine, profile_and_allocate_cache
from mini_vllm.core.block_manager import BlockSpaceManager
from mini_vllm.engine.request import Request
from mini_vllm.sampling.sampler import Sampler, SamplerOutput

@dataclass
class ModelInput:

    input_ids: torch.Tensor

    positions: torch.Tensor

    slot_mapping: torch.Tensor

    block_tables: Optional[torch.Tensor] = None

    context_lens: Optional[torch.Tensor] = None

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    tokens_per_request: List[int] = field(default_factory=list)

    is_prefill_per_request: List[bool] = field(default_factory=list)

    @property
    def is_prefill(self) -> bool:
        return self.num_decode_tokens == 0

    @property
    def is_decode(self) -> bool:
        return self.num_prefill_tokens == 0

    @property
    def is_mixed(self) -> bool:
        return self.num_prefill_tokens > 0 and self.num_decode_tokens > 0

    @property
    def batch_size(self) -> int:
        return len(self.tokens_per_request)

@dataclass
class ModelOutput:

    logits: torch.Tensor

    hidden_states: Optional[torch.Tensor] = None

class ModelRunner:

    def __init__(
        self,
        config: VllmConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or config.get_device()

        print(f"Loading model: {config.model.model_name_or_path}")
        self.model, self.tokenizer = load_model_and_tokenizer(config, self.device)
        print(f"Model loaded on {self.device}")

        self.vocab_size = self.model.config.vocab_size
        self.num_layers = get_num_layers(config)

        print("Allocating KV cache...")
        self.cache_engine, self.num_blocks = profile_and_allocate_cache(
            config, self.device
        )
        print(f"Allocated {self.num_blocks} blocks ({self.cache_engine.get_memory_usage_mb():.1f} MB)")

        self.block_manager = BlockSpaceManager(
            config.cache,
            num_gpu_blocks=self.num_blocks,
        )

        self.sampler = Sampler(self.vocab_size)

        self._request_kv_cache: Dict[str, Any] = {}

        self._past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    def prepare_model_input(
        self,
        requests: List[Request],
        chunked_prefill_size: Optional[int] = None,
    ) -> ModelInput:
        input_ids_list = []
        positions_list = []
        slot_mapping_list = []
        block_table_list = []
        context_lens_list = []
        tokens_per_request = []
        is_prefill_per_request = []
        max_blocks = 0

        num_prefill_tokens = 0
        num_decode_tokens = 0

        for request in requests:
            block_table = self.block_manager.get_block_table(request.request_id)
            if block_table is None:
                continue

            block_ids = block_table.get_all_physical_blocks()
            max_blocks = max(max_blocks, len(block_ids))
            block_table_list.append(block_ids)

            if request.is_prefill():

                start_pos = request.num_computed_tokens
                remaining = request.num_prompt_tokens - start_pos

                if chunked_prefill_size is not None:
                    num_tokens = min(remaining, chunked_prefill_size)
                else:
                    num_tokens = remaining

                tokens_to_process = request.prompt_token_ids[start_pos:start_pos + num_tokens]
                num_prefill_tokens += len(tokens_to_process)
                is_prefill_per_request.append(True)
            else:

                start_pos = request.num_total_tokens - 1
                tokens_to_process = [request.all_token_ids[-1]]
                num_decode_tokens += 1
                is_prefill_per_request.append(False)

            tokens_per_request.append(len(tokens_to_process))
            input_ids_list.extend(tokens_to_process)

            for i, _ in enumerate(tokens_to_process):
                positions_list.append(start_pos + i)

            for i, _ in enumerate(tokens_to_process):
                pos = start_pos + i
                physical_block, slot = block_table.get_physical_slot(pos)
                flat_slot = physical_block * self.config.cache.block_size + slot
                slot_mapping_list.append(flat_slot)

            context_lens_list.append(start_pos + len(tokens_to_process))

        padded_block_tables = []
        for bt in block_table_list:
            padded = bt + [0] * (max_blocks - len(bt))
            padded_block_tables.append(padded)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions_list, dtype=torch.long, device=self.device)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=self.device)

        block_tables = None
        context_lens = None
        if padded_block_tables:
            block_tables = torch.tensor(padded_block_tables, dtype=torch.long, device=self.device)
            context_lens = torch.tensor(context_lens_list, dtype=torch.long, device=self.device)

        return ModelInput(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            tokens_per_request=tokens_per_request,
            is_prefill_per_request=is_prefill_per_request,
        )

    @torch.no_grad()
    def execute_model(
        self,
        model_input: ModelInput,
        requests: List[Request],
    ) -> ModelOutput:
        batch_size = len(requests)

        if batch_size == 0:
            return ModelOutput(
                logits=torch.empty(0, self.vocab_size, device=self.device)
            )

        all_logits = []

        for i, request in enumerate(requests):

            start_idx = sum(model_input.tokens_per_request[:i])
            num_tokens = model_input.tokens_per_request[i]
            end_idx = start_idx + num_tokens

            req_input_ids = model_input.input_ids[start_idx:end_idx]
            req_positions = model_input.positions[start_idx:end_idx]

            logits = self._execute_single_request(
                request,
                req_input_ids,
                req_positions,
                is_prefill=model_input.is_prefill_per_request[i],
            )
            all_logits.append(logits)

        combined_logits = torch.cat(all_logits, dim=0)
        return ModelOutput(logits=combined_logits)

    def _execute_single_request(
        self,
        request: Request,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:

        cache_key = request.request_id
        past_kv = self._request_kv_cache.get(cache_key)

        input_ids = input_ids.unsqueeze(0)
        positions = positions.unsqueeze(0)

        if is_prefill or past_kv is None:

            outputs = self.model(
                input_ids=input_ids,
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            )
            self._request_kv_cache[cache_key] = outputs.past_key_values
        else:

            outputs = self.model(
                input_ids=input_ids,
                position_ids=positions,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            self._request_kv_cache[cache_key] = outputs.past_key_values

        return outputs.logits[:, -1, :]

    def sample(
        self,
        model_output: ModelOutput,
        requests: List[Request],
    ) -> SamplerOutput:
        sampling_params_list = [r.sampling_params for r in requests]
        output_tokens_list = [r.output_token_ids for r in requests]

        return self.sampler.forward(
            model_output.logits,
            sampling_params_list,
            output_tokens_list,
        )

    def allocate_request(self, request: Request) -> bool:
        try:
            num_tokens = request.num_prompt_tokens
            self.block_manager.allocate(request.request_id, num_tokens)
            return True
        except MemoryError:
            return False

    def free_request(self, request: Request) -> None:
        self.block_manager.free(request.request_id)

        if request.request_id in self._request_kv_cache:
            del self._request_kv_cache[request.request_id]

        self._past_key_values = None

    def append_slot(self, request: Request) -> bool:
        try:
            self.block_manager.append_slot(request.request_id)
            return True
        except MemoryError:
            return False

    def can_allocate(self, num_tokens: int) -> bool:
        status = self.block_manager.can_allocate(num_tokens)
        return status.can_allocate

    def get_num_free_blocks(self) -> int:
        return self.block_manager.get_num_free_blocks()

    def get_cache_utilization(self) -> float:
        return self.block_manager.get_utilization()

    def __repr__(self) -> str:
        return (
            f"ModelRunner("
            f"model={self.config.model.model_name_or_path}, "
            f"device={self.device}, "
            f"blocks={self.num_blocks}, "
            f"utilization={self.get_cache_utilization():.1%})"
        )
