from typing import List, Optional, Tuple, Dict
import torch

from mini_vllm.config import VllmConfig, CacheConfig
from mini_vllm.model.loader import (
    get_num_layers,
    get_num_kv_heads,
    get_head_dim,
)

class CacheEngine:

    def __init__(
        self,
        config: VllmConfig,
        num_blocks: int,
        device: torch.device,
    ):
        self.config = config
        self.num_blocks = num_blocks
        self.block_size = config.cache.block_size
        self.device = device

        self.num_layers = get_num_layers(config)
        self.num_kv_heads = get_num_kv_heads(config)
        self.head_dim = get_head_dim(config)

        self.dtype = config.model.get_torch_dtype()
        if self.dtype == "auto":
            self.dtype = torch.float16

        self.gpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._allocate_cache()

    def _allocate_cache(self) -> None:

        cache_shape = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )

        for _ in range(self.num_layers):

            key_cache = torch.zeros(
                cache_shape,
                dtype=self.dtype,
                device=self.device,
            )
            value_cache = torch.zeros(
                cache_shape,
                dtype=self.dtype,
                device=self.device,
            )
            self.gpu_cache.append((key_cache, value_cache))

    def get_cache_block(
        self,
        layer_idx: int,
        block_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self.gpu_cache[layer_idx]
        return key_cache[block_id], value_cache[block_id]

    def write_kv(
        self,
        layer_idx: int,
        block_id: int,
        slot_in_block: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        key_cache, value_cache = self.gpu_cache[layer_idx]

        if key.dim() == 3:
            key = key.squeeze(0)
            value = value.squeeze(0)

        key_cache[block_id, slot_in_block] = key
        value_cache[block_id, slot_in_block] = value

    def write_kv_batch(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        key_cache, value_cache = self.gpu_cache[layer_idx]

        block_ids = slot_mapping // self.block_size
        slots_in_block = slot_mapping % self.block_size

        for i, (bid, slot) in enumerate(zip(block_ids, slots_in_block)):
            key_cache[bid, slot] = keys[i]
            value_cache[bid, slot] = values[i]

    def read_kv_blocks(
        self,
        layer_idx: int,
        block_ids: List[int],
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self.gpu_cache[layer_idx]

        if not block_ids:
            return (
                torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype),
                torch.empty(0, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype),
            )

        block_tensors_k = key_cache[block_ids]
        block_tensors_v = value_cache[block_ids]

        flat_k = block_tensors_k.reshape(-1, self.num_kv_heads, self.head_dim)
        flat_v = block_tensors_v.reshape(-1, self.num_kv_heads, self.head_dim)

        return flat_k[:num_tokens], flat_v[:num_tokens]

    def copy_blocks(
        self,
        src_block_id: int,
        dst_block_id: int,
    ) -> None:
        for layer_idx in range(self.num_layers):
            key_cache, value_cache = self.gpu_cache[layer_idx]
            key_cache[dst_block_id].copy_(key_cache[src_block_id])
            value_cache[dst_block_id].copy_(value_cache[src_block_id])

    def copy_blocks_batch(
        self,
        block_mapping: List[Tuple[int, int]],
    ) -> None:
        if not block_mapping:
            return

        for src_id, dst_id in block_mapping:
            self.copy_blocks(src_id, dst_id)

    def clear_block(self, block_id: int) -> None:
        for layer_idx in range(self.num_layers):
            key_cache, value_cache = self.gpu_cache[layer_idx]
            key_cache[block_id].zero_()
            value_cache[block_id].zero_()

    def get_memory_usage(self) -> int:
        total = 0
        for key_cache, value_cache in self.gpu_cache:
            total += key_cache.numel() * key_cache.element_size()
            total += value_cache.numel() * value_cache.element_size()
        return total

    def get_memory_usage_mb(self) -> float:
        return self.get_memory_usage() / (1024 * 1024)

    def get_cache_shape(self) -> Dict[str, int]:
        return {
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }

    def __repr__(self) -> str:
        return (
            f"CacheEngine("
            f"layers={self.num_layers}, "
            f"blocks={self.num_blocks}, "
            f"block_size={self.block_size}, "
            f"memory={self.get_memory_usage_mb():.1f}MB)"
        )

def estimate_num_blocks(
    config: VllmConfig,
    available_memory_bytes: int,
) -> int:
    num_layers = get_num_layers(config)
    num_kv_heads = get_num_kv_heads(config)
    head_dim = get_head_dim(config)
    block_size = config.cache.block_size

    dtype = config.model.get_torch_dtype()
    if dtype == torch.float16 or dtype == torch.bfloat16:
        element_size = 2
    elif dtype == torch.float32:
        element_size = 4
    else:
        element_size = 2

    bytes_per_block = (
        2 * num_layers * block_size * num_kv_heads * head_dim * element_size
    )

    num_blocks = available_memory_bytes // bytes_per_block
    return max(1, num_blocks)

def profile_and_allocate_cache(
    config: VllmConfig,
    device: torch.device,
) -> Tuple[CacheEngine, int]:
    if device.type == "cuda":

        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        available = total_memory - reserved_memory

        usable = int(available * config.cache.gpu_memory_utilization)

    elif device.type == "mps":

        usable = int(8 * 1024 * 1024 * 1024 * config.cache.gpu_memory_utilization)

    else:

        usable = int(2 * 1024 * 1024 * 1024 * config.cache.gpu_memory_utilization)

    if config.cache.num_gpu_blocks is not None:
        num_blocks = config.cache.num_gpu_blocks
    else:
        num_blocks = estimate_num_blocks(config, usable)

    cache_engine = CacheEngine(config, num_blocks, device)

    return cache_engine, num_blocks
