import sys
sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

from mini_vllm import LLM, SamplingParams
from mini_vllm.config import VllmConfig, ModelConfig, CacheConfig

def main():
    print("=" * 60)
    print("Mini vLLM - Phase 2: Paged KV Cache Demo")
    print("=" * 60)
    print()

    model_config = ModelConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
    )

    config = VllmConfig(
        model=model_config,
        cache=cache_config,
    )

    print("Loading model with paged KV cache...")
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        gpu_memory_utilization=0.8,
    )
    print(f"Model loaded: {llm.engine}")
    print()

    print("Initial Cache Stats:")
    stats = llm.engine.get_cache_stats()
    if stats:
        print(f"  Total blocks: {stats['total_blocks']}")
        print(f"  Free blocks: {stats['free_blocks']}")
        print(f"  Utilization: {stats['utilization']:.1%}")
        print(f"  Cache memory: {stats['cache_memory_mb']:.1f} MB")
    print()

    print("=" * 60)
    print("Example 1: Single generation with paged KV cache")
    print("=" * 60)

    prompt = "The key advantages of paged memory management are"
    params = SamplingParams(max_tokens=50, temperature=0.7)

    print(f"Prompt: {prompt}")
    outputs = llm.generate(prompt, params)
    print(f"Generated: {outputs[0].generated_text}")
    print()

    print("Cache stats after generation:")
    stats = llm.engine.get_cache_stats()
    if stats:
        print(f"  Utilization: {stats['utilization']:.1%}")
        print(f"  Free blocks: {stats['free_blocks']}")
    print()

    print("=" * 60)
    print("Example 2: Multiple generations (cache reuse)")
    print("=" * 60)

    prompts = [
        "Explain transformers in one sentence:",
        "What is attention in neural networks?",
        "The future of AI is",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Generation {i}]")
        print(f"Prompt: {prompt}")

        outputs = llm.generate(prompt, SamplingParams(max_tokens=30))
        print(f"Output: {outputs[0].generated_text}")

        stats = llm.engine.get_cache_stats()
        if stats:
            print(f"Cache utilization: {stats['utilization']:.1%}")

    print()
    print("=" * 60)
    print("Example 3: Memory efficiency with longer context")
    print("=" * 60)

    long_prompt = """The following is a detailed technical explanation of how
paged attention works in large language model serving systems.
Traditional attention mechanisms store key-value pairs contiguously
in memory, which leads to significant memory fragmentation and waste.
The paged attention approach, inspired by operating system virtual memory,
divides the KV cache into fixed-size blocks that can be stored
non-contiguously in GPU memory.

Based on this context, explain the main benefits:"""

    print(f"Prompt length: {len(long_prompt)} characters")
    params = SamplingParams(max_tokens=100, temperature=0.7)

    outputs = llm.generate(long_prompt, params)
    print(f"\nGenerated ({len(outputs[0].generated_token_ids)} tokens):")
    print(outputs[0].generated_text)

    print()
    print("=" * 60)
    print("Final Cache Statistics")
    print("=" * 60)
    stats = llm.engine.get_cache_stats()
    if stats:
        print(f"  Total blocks: {stats['total_blocks']}")
        print(f"  Free blocks: {stats['free_blocks']}")
        print(f"  Peak utilization: {stats['utilization']:.1%}")
        print(f"  Total cache memory: {stats['cache_memory_mb']:.1f} MB")

    print()
    print("Done! Paged KV cache enables efficient memory management.")

if __name__ == "__main__":
    main()
