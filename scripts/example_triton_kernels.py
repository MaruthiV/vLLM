import sys
import time
sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This script requires a GPU.")
    print("Please run on Colab Pro or a machine with CUDA support.")
    sys.exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

from mini_vllm.kernels import (
    paged_attention_forward,
    paged_attention_forward_pytorch,
    is_triton_available,
    create_paged_kv_cache,
    create_block_tables,
    fill_kv_cache_random,
    verify_correctness,
    KernelBenchmark,
)

def create_test_inputs(
    batch_size: int = 4,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    max_context_len: int = 2048,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):

    context_lens = torch.randint(
        low=block_size,
        high=max_context_len,
        size=(batch_size,),
        dtype=torch.int32,
        device=device,
    )

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    total_blocks = batch_size * max_num_blocks_per_seq

    query = torch.randn(
        (batch_size, num_heads, head_dim),
        dtype=dtype,
        device=device,
    )

    key_cache = torch.randn(
        (total_blocks, num_kv_heads, block_size, head_dim),
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn(
        (total_blocks, num_kv_heads, block_size, head_dim),
        dtype=dtype,
        device=device,
    )

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

    return query, key_cache, value_cache, block_tables, context_lens

def test_correctness():
    print("=" * 70)
    print("Part 1: Correctness Test")
    print("=" * 70)
    print()

    if not is_triton_available():
        print("WARNING: Triton not available, skipping kernel test")
        return False

    query, key_cache, value_cache, block_tables, context_lens = create_test_inputs(
        batch_size=4,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=16,
        max_context_len=512,
    )

    print(f"Query shape: {query.shape}")
    print(f"Key cache shape: {key_cache.shape}")
    print(f"Block tables shape: {block_tables.shape}")
    print(f"Context lens: {context_lens.tolist()}")
    print()

    print("Running Triton kernel...")
    output_triton = paged_attention_forward(
        query, key_cache, value_cache, block_tables, context_lens
    )

    print("Running PyTorch reference...")
    output_pytorch = paged_attention_forward_pytorch(
        query, key_cache, value_cache, block_tables, context_lens
    )

    is_correct, max_diff = verify_correctness(output_triton, output_pytorch)

    print()
    print(f"Max difference: {max_diff:.6f}")
    print(f"Correctness: {'PASS' if is_correct else 'FAIL'}")
    print()

    return is_correct

def benchmark_kernels():
    print("=" * 70)
    print("Part 2: Performance Benchmark")
    print("=" * 70)
    print()

    if not is_triton_available():
        print("WARNING: Triton not available, using PyTorch only")

    configs = [
        {'batch_size': 1, 'max_context_len': 512, 'num_heads': 32},
        {'batch_size': 4, 'max_context_len': 512, 'num_heads': 32},
        {'batch_size': 8, 'max_context_len': 1024, 'num_heads': 32},
        {'batch_size': 16, 'max_context_len': 2048, 'num_heads': 32},
        {'batch_size': 32, 'max_context_len': 4096, 'num_heads': 32},
    ]

    benchmark = KernelBenchmark(warmup_iters=10, benchmark_iters=100)
    results = []

    for cfg in configs:
        print(f"Config: batch={cfg['batch_size']}, seq_len={cfg['max_context_len']}")

        query, key_cache, value_cache, block_tables, context_lens = create_test_inputs(
            batch_size=cfg['batch_size'],
            num_heads=cfg['num_heads'],
            max_context_len=cfg['max_context_len'],
        )

        if is_triton_available():
            triton_result = benchmark.benchmark_function(
                f"triton_b{cfg['batch_size']}_l{cfg['max_context_len']}",
                paged_attention_forward,
                query, key_cache, value_cache, block_tables, context_lens,
            )
        else:
            triton_result = {'mean_ms': float('inf')}

        pytorch_result = benchmark.benchmark_function(
            f"pytorch_b{cfg['batch_size']}_l{cfg['max_context_len']}",
            paged_attention_forward_pytorch,
            query, key_cache, value_cache, block_tables, context_lens,
        )

        speedup = pytorch_result['mean_ms'] / triton_result['mean_ms'] if triton_result['mean_ms'] > 0 else 0

        results.append({
            'config': cfg,
            'triton_ms': triton_result['mean_ms'],
            'pytorch_ms': pytorch_result['mean_ms'],
            'speedup': speedup,
        })

        print(f"  Triton: {triton_result['mean_ms']:.3f}ms")
        print(f"  PyTorch: {pytorch_result['mean_ms']:.3f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        print()

    return results

def analyze_memory():
    print("=" * 70)
    print("Part 3: Memory Analysis")
    print("=" * 70)
    print()

    def calc_paged_memory(batch_size, max_seq_len, num_kv_heads, head_dim, block_size, actual_lens):
        total_blocks = sum((l + block_size - 1) // block_size for l in actual_lens)
        bytes_per_kv = 2 * num_kv_heads * block_size * head_dim * 2
        return total_blocks * bytes_per_kv

    def calc_dense_memory(batch_size, max_seq_len, num_kv_heads, head_dim):
        bytes_per_kv = 2 * num_kv_heads * max_seq_len * head_dim * 2
        return batch_size * bytes_per_kv

    batch_size = 8
    max_seq_len = 4096
    num_kv_heads = 8
    head_dim = 128
    block_size = 16

    scenarios = [
        ("All short (512)", [512] * batch_size),
        ("All medium (1024)", [1024] * batch_size),
        ("All long (2048)", [2048] * batch_size),
        ("Mixed lengths", [256, 512, 1024, 2048, 256, 512, 1024, 2048]),
        ("All max (4096)", [4096] * batch_size),
    ]

    print("Memory Comparison: Paged vs Dense KV Cache")
    print("-" * 70)

    for name, lens in scenarios:
        paged_mem = calc_paged_memory(batch_size, max_seq_len, num_kv_heads, head_dim, block_size, lens)
        dense_mem = calc_dense_memory(batch_size, max_seq_len, num_kv_heads, head_dim)
        savings = (1 - paged_mem / dense_mem) * 100

        print(f"{name:20s} | Paged: {paged_mem/1e6:6.1f}MB | "
              f"Dense: {dense_mem/1e6:6.1f}MB | Savings: {savings:5.1f}%")

    print()

def main():
    print("=" * 70)
    print("Mini vLLM - Phase 8: Triton Paged Attention Kernels")
    print("=" * 70)
    print()

    print(f"Triton available: {is_triton_available()}")
    print()

    is_correct = test_correctness()

    if not is_correct:
        print("WARNING: Correctness test failed!")
        print()

    results = benchmark_kernels()

    analyze_memory()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    if is_triton_available() and results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"Average speedup: {avg_speedup:.2f}x")
        print()

    print("Key benefits of Triton paged attention:")
    print("  1. Handles non-contiguous KV cache natively")
    print("  2. Uses online softmax for numerical stability")
    print("  3. Supports GQA/MQA (grouped/multi-query attention)")
    print("  4. Memory efficient with block-based allocation")
    print()

if __name__ == "__main__":
    main()
