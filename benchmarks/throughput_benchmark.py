import argparse
import csv
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json

sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

import torch

@dataclass
class BenchmarkResult:

    model: str
    batch_size: int
    input_len: int
    output_len: int

    total_time_s: float
    ttft_s: float
    tpot_ms: float
    throughput_tok_s: float

    total_tokens: int
    backend: str

@dataclass
class BenchmarkConfig:

    model: str
    batch_sizes: List[int]
    input_lengths: List[int]
    output_length: int
    num_warmup: int
    num_iterations: int
    dtype: str
    device: str

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def generate_prompts(batch_size: int, input_len: int, tokenizer) -> List[str]:
    base_text = "This is a test prompt for benchmarking the inference engine. " * 20
    tokens = tokenizer.encode(base_text)

    if len(tokens) > input_len:
        tokens = tokens[:input_len]
    else:
        while len(tokens) < input_len:
            tokens = tokens + tokens
        tokens = tokens[:input_len]

    prompt = tokenizer.decode(tokens)
    return [prompt] * batch_size

def benchmark_mini_vllm(
    config: BenchmarkConfig,
    batch_size: int,
    input_len: int,
) -> BenchmarkResult:
    from mini_vllm import LLM, SamplingParams

    llm = LLM(
        model=config.model,
        dtype=config.dtype,
        gpu_memory_utilization=0.8,
    )

    prompts = generate_prompts(batch_size, input_len, llm.tokenizer)
    sampling_params = SamplingParams(
        max_tokens=config.output_length,
        temperature=0.0,
    )

    for _ in range(config.num_warmup):
        _ = llm.generate(prompts[:1], sampling_params)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    total_times = []
    ttfts = []

    for _ in range(config.num_iterations):
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()

        total_times.append(end_time - start_time)

        if outputs and outputs[0].metrics:
            ttfts.append(outputs[0].metrics.time_to_first_token)
        else:
            ttfts.append(0.0)

    avg_total_time = sum(total_times) / len(total_times)
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

    total_output_tokens = batch_size * config.output_length
    throughput = total_output_tokens / avg_total_time
    tpot = (avg_total_time - avg_ttft) / config.output_length * 1000

    return BenchmarkResult(
        model=config.model,
        batch_size=batch_size,
        input_len=input_len,
        output_len=config.output_length,
        total_time_s=avg_total_time,
        ttft_s=avg_ttft,
        tpot_ms=tpot,
        throughput_tok_s=throughput,
        total_tokens=total_output_tokens,
        backend="mini-vllm",
    )

def benchmark_huggingface(
    config: BenchmarkConfig,
    batch_size: int,
    input_len: int,
) -> BenchmarkResult:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=getattr(torch, config.dtype),
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    prompts = generate_prompts(batch_size, input_len, tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=input_len,
    ).to(device)

    with torch.no_grad():
        for _ in range(config.num_warmup):
            _ = model.generate(
                inputs.input_ids[:1],
                attention_mask=inputs.attention_mask[:1],
                max_new_tokens=config.output_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    total_times = []

    for _ in range(config.num_iterations):
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=config.output_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()

        total_times.append(end_time - start_time)

    avg_total_time = sum(total_times) / len(total_times)
    total_output_tokens = batch_size * config.output_length
    throughput = total_output_tokens / avg_total_time
    tpot = avg_total_time / config.output_length * 1000

    return BenchmarkResult(
        model=config.model,
        batch_size=batch_size,
        input_len=input_len,
        output_len=config.output_length,
        total_time_s=avg_total_time,
        ttft_s=0.0,
        tpot_ms=tpot,
        throughput_tok_s=throughput,
        total_tokens=total_output_tokens,
        backend="huggingface",
    )

def run_benchmarks(config: BenchmarkConfig, include_hf: bool = True) -> List[BenchmarkResult]:
    results = []

    for batch_size in config.batch_sizes:
        for input_len in config.input_lengths:
            print(f"\nBenchmarking: batch_size={batch_size}, input_len={input_len}")

            try:
                print("  Running mini-vllm...")
                result = benchmark_mini_vllm(config, batch_size, input_len)
                results.append(result)
                print(f"    Throughput: {result.throughput_tok_s:.1f} tok/s")
            except Exception as e:
                print(f"    Error: {e}")

            if include_hf:
                try:
                    print("  Running HuggingFace...")
                    result = benchmark_huggingface(config, batch_size, input_len)
                    results.append(result)
                    print(f"    Throughput: {result.throughput_tok_s:.1f} tok/s")
                except Exception as e:
                    print(f"    Error: {e}")

    return results

def save_results(results: List[BenchmarkResult], output_path: str):
    if not results:
        return

    fieldnames = list(asdict(results[0]).keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    print(f"\nResults saved to {output_path}")

def print_summary(results: List[BenchmarkResult]):
    if not results:
        print("No results to summarize.")
        return

    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)

    mini_vllm_results = [r for r in results if r.backend == "mini-vllm"]
    hf_results = [r for r in results if r.backend == "huggingface"]

    print(f"\n{'Config':<25} {'mini-vllm (tok/s)':<20} {'HF (tok/s)':<20} {'Speedup':<10}")
    print("-" * 80)

    for mv_result in mini_vllm_results:
        config_key = f"bs={mv_result.batch_size}, in={mv_result.input_len}"

        hf_result = next(
            (r for r in hf_results
             if r.batch_size == mv_result.batch_size and r.input_len == mv_result.input_len),
            None
        )

        hf_throughput = hf_result.throughput_tok_s if hf_result else 0
        speedup = mv_result.throughput_tok_s / hf_throughput if hf_throughput > 0 else 0

        print(f"{config_key:<25} {mv_result.throughput_tok_s:<20.1f} {hf_throughput:<20.1f} {speedup:<10.2f}x")

    if mini_vllm_results and hf_results:
        avg_mv = sum(r.throughput_tok_s for r in mini_vllm_results) / len(mini_vllm_results)
        avg_hf = sum(r.throughput_tok_s for r in hf_results) / len(hf_results)
        avg_speedup = avg_mv / avg_hf if avg_hf > 0 else 0

        print("-" * 80)
        print(f"{'Average':<25} {avg_mv:<20.1f} {avg_hf:<20.1f} {avg_speedup:<10.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Mini-vLLM Throughput Benchmark")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace baseline")

    args = parser.parse_args()

    print("=" * 80)
    print("Mini-vLLM Throughput Benchmark")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Input lengths: {args.input_lengths}")
    print(f"Output length: {args.output_length}")
    print(f"Device: {get_device()}")

    config = BenchmarkConfig(
        model=args.model,
        batch_sizes=args.batch_sizes,
        input_lengths=args.input_lengths,
        output_length=args.output_length,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        dtype=args.dtype,
        device=get_device(),
    )

    results = run_benchmarks(config, include_hf=not args.skip_hf)

    print_summary(results)

    if args.output:
        save_results(results, args.output)

if __name__ == "__main__":
    main()
