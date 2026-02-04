import argparse
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

@dataclass
class SequenceInfo:
    seq_id: str
    prompt_len: int
    max_output_len: int
    current_len: int = 0
    is_complete: bool = False

@dataclass
class MemorySnapshot:
    step: int
    paged_memory_mb: float
    dense_memory_mb: float
    paged_utilization: float
    dense_utilization: float
    num_active_seqs: int
    total_tokens: int

def calculate_paged_memory(
    sequences: List[SequenceInfo],
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> Tuple[float, float]:
    total_blocks = 0
    total_tokens = 0

    for seq in sequences:
        if not seq.is_complete:

            num_blocks = (seq.current_len + block_size - 1) // block_size
            total_blocks += num_blocks
            total_tokens += seq.current_len

    bytes_per_block = 2 * num_kv_heads * block_size * head_dim * dtype_bytes
    total_bytes = total_blocks * bytes_per_block
    memory_mb = total_bytes / (1024 * 1024)

    allocated_slots = total_blocks * block_size
    utilization = total_tokens / allocated_slots if allocated_slots > 0 else 0

    return memory_mb, utilization

def calculate_dense_memory(
    sequences: List[SequenceInfo],
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> Tuple[float, float]:
    num_active = sum(1 for seq in sequences if not seq.is_complete)
    total_tokens = sum(seq.current_len for seq in sequences if not seq.is_complete)

    bytes_per_seq = 2 * num_kv_heads * max_seq_len * head_dim * dtype_bytes
    total_bytes = num_active * bytes_per_seq
    memory_mb = total_bytes / (1024 * 1024)

    allocated_slots = num_active * max_seq_len
    utilization = total_tokens / allocated_slots if allocated_slots > 0 else 0

    return memory_mb, utilization

def simulate_workload(
    num_sequences: int,
    prompt_len_range: Tuple[int, int],
    output_len_range: Tuple[int, int],
    max_concurrent: int,
    block_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> List[MemorySnapshot]:
    snapshots = []
    sequences: List[SequenceInfo] = []
    completed_count = 0
    step = 0
    seq_counter = 0

    for _ in range(min(max_concurrent, num_sequences)):
        prompt_len = random.randint(*prompt_len_range)
        output_len = random.randint(*output_len_range)
        sequences.append(SequenceInfo(
            seq_id=f"seq-{seq_counter}",
            prompt_len=prompt_len,
            max_output_len=output_len,
            current_len=prompt_len,
        ))
        seq_counter += 1

    while completed_count < num_sequences:

        paged_mem, paged_util = calculate_paged_memory(
            sequences, block_size, num_kv_heads, head_dim
        )
        dense_mem, dense_util = calculate_dense_memory(
            sequences, max_seq_len, num_kv_heads, head_dim
        )

        active_seqs = [s for s in sequences if not s.is_complete]
        total_tokens = sum(s.current_len for s in active_seqs)

        snapshots.append(MemorySnapshot(
            step=step,
            paged_memory_mb=paged_mem,
            dense_memory_mb=dense_mem,
            paged_utilization=paged_util,
            dense_utilization=dense_util,
            num_active_seqs=len(active_seqs),
            total_tokens=total_tokens,
        ))

        newly_completed = []
        for seq in sequences:
            if not seq.is_complete:
                seq.current_len += 1

                generated = seq.current_len - seq.prompt_len
                if generated >= seq.max_output_len:
                    seq.is_complete = True
                    newly_completed.append(seq)
                    completed_count += 1

        for _ in newly_completed:
            if seq_counter < num_sequences:
                prompt_len = random.randint(*prompt_len_range)
                output_len = random.randint(*output_len_range)
                sequences.append(SequenceInfo(
                    seq_id=f"seq-{seq_counter}",
                    prompt_len=prompt_len,
                    max_output_len=output_len,
                    current_len=prompt_len,
                ))
                seq_counter += 1

        step += 1

        if step > 10000:
            break

    return snapshots

def analyze_memory_savings(snapshots: List[MemorySnapshot]) -> Dict:
    if not snapshots:
        return {}

    paged_total = sum(s.paged_memory_mb for s in snapshots)
    dense_total = sum(s.dense_memory_mb for s in snapshots)

    paged_peak = max(s.paged_memory_mb for s in snapshots)
    dense_peak = max(s.dense_memory_mb for s in snapshots)

    avg_paged_util = sum(s.paged_utilization for s in snapshots) / len(snapshots)
    avg_dense_util = sum(s.dense_utilization for s in snapshots) / len(snapshots)

    savings_total = (1 - paged_total / dense_total) * 100 if dense_total > 0 else 0
    savings_peak = (1 - paged_peak / dense_peak) * 100 if dense_peak > 0 else 0

    return {
        "paged_peak_mb": paged_peak,
        "dense_peak_mb": dense_peak,
        "peak_savings_pct": savings_peak,
        "avg_paged_utilization": avg_paged_util,
        "avg_dense_utilization": avg_dense_util,
        "total_steps": len(snapshots),
    }

def print_analysis(analysis: Dict):
    print("\n" + "=" * 60)
    print("Memory Analysis Results")
    print("=" * 60)

    print(f"\nPeak Memory Usage:")
    print(f"  Paged allocation: {analysis['paged_peak_mb']:.1f} MB")
    print(f"  Dense allocation: {analysis['dense_peak_mb']:.1f} MB")
    print(f"  Peak savings: {analysis['peak_savings_pct']:.1f}%")

    print(f"\nAverage Utilization:")
    print(f"  Paged: {analysis['avg_paged_utilization']:.1%}")
    print(f"  Dense: {analysis['avg_dense_utilization']:.1%}")

    print(f"\nSimulation steps: {analysis['total_steps']}")

def create_visualization(
    snapshots: List[MemorySnapshot],
    output_path: Optional[str] = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = [s.step for s in snapshots]
    paged_mem = [s.paged_memory_mb for s in snapshots]
    dense_mem = [s.dense_memory_mb for s in snapshots]
    paged_util = [s.paged_utilization * 100 for s in snapshots]
    dense_util = [s.dense_utilization * 100 for s in snapshots]
    active_seqs = [s.num_active_seqs for s in snapshots]

    ax1 = axes[0, 0]
    ax1.plot(steps, paged_mem, label='Paged KV Cache', color='steelblue', linewidth=2)
    ax1.plot(steps, dense_mem, label='Dense KV Cache', color='coral', linewidth=2)
    ax1.fill_between(steps, paged_mem, alpha=0.3, color='steelblue')
    ax1.fill_between(steps, dense_mem, alpha=0.3, color='coral')
    ax1.set_xlabel('Decode Step')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('KV Cache Memory Usage Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    savings = [(d - p) / d * 100 if d > 0 else 0 for p, d in zip(paged_mem, dense_mem)]
    ax2.plot(steps, savings, color='green', linewidth=2)
    ax2.fill_between(steps, savings, alpha=0.3, color='green')
    ax2.set_xlabel('Decode Step')
    ax2.set_ylabel('Memory Savings (%)')
    ax2.set_title('Memory Savings from Paged Allocation')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    ax3 = axes[1, 0]
    ax3.plot(steps, paged_util, label='Paged', color='steelblue', linewidth=2)
    ax3.plot(steps, dense_util, label='Dense', color='coral', linewidth=2)
    ax3.set_xlabel('Decode Step')
    ax3.set_ylabel('Utilization (%)')
    ax3.set_title('Memory Utilization (Actual Tokens / Allocated Slots)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    ax4 = axes[1, 1]
    ax4.plot(steps, active_seqs, color='purple', linewidth=2)
    ax4.fill_between(steps, active_seqs, alpha=0.3, color='purple')
    ax4.set_xlabel('Decode Step')
    ax4.set_ylabel('Number of Sequences')
    ax4.set_title('Active Sequences Over Time')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Paged vs Dense KV Cache Memory Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {output_path}")
    else:
        plt.show()

def run_scenario_comparison():
    print("\n" + "=" * 60)
    print("Scenario Comparison")
    print("=" * 60)

    scenarios = [
        {
            "name": "Uniform short sequences",
            "prompt_len_range": (64, 128),
            "output_len_range": (32, 64),
        },
        {
            "name": "Uniform long sequences",
            "prompt_len_range": (512, 1024),
            "output_len_range": (256, 512),
        },
        {
            "name": "Mixed lengths (realistic)",
            "prompt_len_range": (64, 512),
            "output_len_range": (32, 256),
        },
        {
            "name": "High variance",
            "prompt_len_range": (32, 1024),
            "output_len_range": (16, 512),
        },
    ]

    results = []

    for scenario in scenarios:
        snapshots = simulate_workload(
            num_sequences=50,
            prompt_len_range=scenario["prompt_len_range"],
            output_len_range=scenario["output_len_range"],
            max_concurrent=8,
            block_size=16,
            max_seq_len=2048,
            num_kv_heads=8,
            head_dim=128,
        )

        analysis = analyze_memory_savings(snapshots)
        analysis["scenario"] = scenario["name"]
        results.append(analysis)

    print(f"\n{'Scenario':<30} {'Peak Paged':<12} {'Peak Dense':<12} {'Savings':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['scenario']:<30} {r['paged_peak_mb']:<12.1f} "
              f"{r['dense_peak_mb']:<12.1f} {r['peak_savings_pct']:<10.1f}%")

    return results

def main():
    parser = argparse.ArgumentParser(description="KV Cache Memory Benchmark")
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("Mini-vLLM KV Cache Memory Benchmark")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Sequences: {args.num_sequences}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print(f"  Block size: {args.block_size}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  KV heads: {args.num_kv_heads}")
    print(f"  Head dim: {args.head_dim}")

    print("\nRunning simulation...")
    snapshots = simulate_workload(
        num_sequences=args.num_sequences,
        prompt_len_range=(64, 512),
        output_len_range=(32, 256),
        max_concurrent=args.max_concurrent,
        block_size=args.block_size,
        max_seq_len=args.max_seq_len,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )

    analysis = analyze_memory_savings(snapshots)
    print_analysis(analysis)

    create_visualization(snapshots, args.output)

    run_scenario_comparison()

    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
Paged KV Cache Benefits:
1. Memory proportional to actual sequence length, not max length
2. Higher utilization (less wasted memory)
3. Enables serving more concurrent requests
4. Greater savings with variable-length sequences
