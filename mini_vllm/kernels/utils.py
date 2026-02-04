import torch
from typing import Dict, List, Optional, Tuple
import time

def create_paged_kv_cache(
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = (num_blocks, num_kv_heads, block_size, head_dim)
    key_cache = torch.zeros(shape, dtype=dtype, device=device)
    value_cache = torch.zeros(shape, dtype=dtype, device=device)
    return key_cache, value_cache

def create_block_tables(
    batch_size: int,
    max_num_blocks_per_seq: int,
    context_lens: torch.Tensor,
    block_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    block_counter = 0
    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        num_blocks = (ctx_len + block_size - 1) // block_size

        for i in range(num_blocks):
            block_tables[b, i] = block_counter
            block_counter += 1

    return block_tables

def fill_kv_cache_random(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> None:
    batch_size = block_tables.shape[0]

    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        num_blocks = (ctx_len + block_size - 1) // block_size

        for block_idx in range(num_blocks):
            physical_block = block_tables[b, block_idx].item()

            start_pos = block_idx * block_size
            end_pos = min(start_pos + block_size, ctx_len)
            num_tokens = end_pos - start_pos

            key_cache[physical_block, :, :num_tokens, :] = torch.randn_like(
                key_cache[physical_block, :, :num_tokens, :]
            )
            value_cache[physical_block, :, :num_tokens, :] = torch.randn_like(
                value_cache[physical_block, :, :num_tokens, :]
            )

def naive_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    attn_weights = torch.matmul(
        query.unsqueeze(2),
        keys.transpose(-2, -1)
    ) * scale

    attn_probs = torch.softmax(attn_weights, dim=-1)

    output = torch.matmul(attn_probs, values)

    return output.squeeze(2)

def gather_kv_from_paged_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_tables.shape[0]
    num_kv_heads = key_cache.shape[1]
    head_dim = key_cache.shape[3]
    max_ctx_len = context_lens.max().item()

    keys = torch.zeros(
        (batch_size, num_kv_heads, max_ctx_len, head_dim),
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    values = torch.zeros_like(keys)

    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        num_blocks = (ctx_len + block_size - 1) // block_size

        pos = 0
        for block_idx in range(num_blocks):
            physical_block = block_tables[b, block_idx].item()

            start_pos = block_idx * block_size
            end_pos = min(start_pos + block_size, ctx_len)
            num_tokens = end_pos - start_pos

            keys[b, :, pos:pos+num_tokens, :] = key_cache[physical_block, :, :num_tokens, :]
            values[b, :, pos:pos+num_tokens, :] = value_cache[physical_block, :, :num_tokens, :]

            pos += num_tokens

    return keys, values

class KernelBenchmark:

    def __init__(
        self,
        warmup_iters: int = 10,
        benchmark_iters: int = 100,
    ):
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.results: Dict[str, Dict] = {}

    def benchmark_function(
        self,
        name: str,
        fn,
        *args,
        **kwargs,
    ) -> Dict:

        for _ in range(self.warmup_iters):
            fn(*args, **kwargs)

        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.benchmark_iters)]

        for i in range(self.benchmark_iters):
            start_events[i].record()
            fn(*args, **kwargs)
            end_events[i].record()

        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times = sorted(times)

        trim = int(len(times) * 0.1)
        if trim > 0:
            times = times[trim:-trim]

        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        result = {
            "name": name,
            "mean_ms": mean_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "iterations": self.benchmark_iters,
        }

        self.results[name] = result
        return result

    def compare(self, name1: str, name2: str) -> float:
        if name1 not in self.results or name2 not in self.results:
            return 0.0

        t1 = self.results[name1]["mean_ms"]
        t2 = self.results[name2]["mean_ms"]

        if t2 == 0:
            return float("inf")

        return t1 / t2

    def print_results(self) -> None:
        print("\nBenchmark Results:")
        print("-" * 60)
        print(f"{'Name':<30} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 60)

        for name, result in self.results.items():
            print(
                f"{name:<30} "
                f"{result['mean_ms']:<12.3f} "
                f"{result['min_ms']:<12.3f} "
                f"{result['max_ms']:<12.3f}"
            )

        print("-" * 60)

def verify_correctness(
    triton_output: torch.Tensor,
    reference_output: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> Tuple[bool, float]:
    diff = torch.abs(triton_output - reference_output)
    max_diff = diff.max().item()

    is_close = torch.allclose(triton_output, reference_output, rtol=rtol, atol=atol)

    return is_close, max_diff

def get_optimal_num_warps(head_dim: int) -> int:
    if head_dim <= 64:
        return 4
    elif head_dim <= 128:
        return 8
    else:
        return 8

def get_optimal_num_stages(head_dim: int) -> int:
    if head_dim <= 64:
        return 3
    else:
        return 2
