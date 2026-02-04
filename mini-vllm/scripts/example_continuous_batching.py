import sys
import time
sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

from mini_vllm import LLM, SamplingParams
from mini_vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig

def main():
    print("=" * 70)
    print("Mini vLLM - Phase 3: Continuous Batching Demo")
    print("=" * 70)
    print()

    model_config = ModelConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.8,
    )

    scheduler_config = SchedulerConfig(
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        chunked_prefill_enabled=True,
        chunk_size=256,
        enable_preemption=True,
    )

    config = VllmConfig(
        model=model_config,
        cache=cache_config,
        scheduler=scheduler_config,
    )

    print("Loading model with continuous batching scheduler...")
    print(f"  Max sequences: {scheduler_config.max_num_seqs}")
    print(f"  Token budget: {scheduler_config.max_num_batched_tokens}")
    print(f"  Chunk size: {scheduler_config.chunk_size}")
    print()

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        gpu_memory_utilization=0.8,
    )
    print(f"Engine ready: {llm.engine}")
    print()

    print("=" * 70)
    print("Example 1: Batched Generation (Multiple Concurrent Requests)")
    print("=" * 70)

    prompts = [
        "What is machine learning?",
        "Explain neural networks briefly:",
        "The future of AI is",
        "Python is a programming language that",
        "Deep learning differs from ML because",
    ]

    print(f"Submitting {len(prompts)} prompts for concurrent processing...")
    print()

    start_time = time.time()
    params = SamplingParams(max_tokens=50, temperature=0.7)
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start_time

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"[Request {i+1}]")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {output.generated_text[:100]}...")
        if output.metrics:
            print(f"  TTFT: {output.metrics.time_to_first_token:.3f}s")
        print()

    total_tokens = sum(len(o.generated_token_ids) for o in outputs)
    print(f"Batch Statistics:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens/elapsed:.1f} tokens/sec")
    print()

    print("=" * 70)
    print("Example 2: Scheduler Queue Dynamics")
    print("=" * 70)

    engine = llm.engine

    prompts = [
        "Short prompt one",
        "Short prompt two",
        "Short prompt three",
    ]

    for i, prompt in enumerate(prompts):
        engine.add_request(f"demo-{i}", prompt, SamplingParams(max_tokens=20))
        stats = engine.get_scheduler_stats()
        print(f"After adding request {i+1}: {stats}")

    print()
    print("Processing requests...")

    step = 0
    while engine.has_unfinished_requests():
        outputs = engine.step()
        step += 1
        stats = engine.get_scheduler_stats()
        cache = engine.get_cache_stats()

        if step % 5 == 0:
            print(f"  Step {step}: scheduler={stats}, cache_util={cache['utilization']:.1%}")

    print(f"Completed in {step} steps")
    print()

    print("=" * 70)
    print("Example 3: Chunked Prefill (Long Prompt)")
    print("=" * 70)

    long_prompt = """
    The following is a comprehensive overview of transformer architecture
    and its applications in modern natural language processing. Transformers
    were introduced in the landmark paper "Attention Is All You Need" and
    have since revolutionized the field of deep learning. The key innovation
    is the self-attention mechanism, which allows the model to weigh the
    importance of different parts of the input when producing each part of
    the output. This has proven to be highly effective for tasks such as
    machine translation, text summarization, and question answering.

    Unlike recurrent neural networks, transformers can process all positions
    in the sequence simultaneously, making them highly parallelizable and
    efficient for training on modern hardware. The architecture consists of
    an encoder and decoder, each made up of multiple layers of self-attention
    and feed-forward neural networks.

    Based on this context, briefly summarize the key benefits of transformers:
