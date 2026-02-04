import sys
import time
sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

from mini_vllm import LLM, SamplingParams
from mini_vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig
from mini_vllm.core.prefix_cache import PrefixCache, compute_block_hash

def main():
    print("=" * 70)
    print("Mini vLLM - Phase 5: Prefix Caching Demo")
    print("=" * 70)
    print()

    print("Part 1: Block Hash Computation")
    print("-" * 40)

    tokens_block1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    tokens_block2 = (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)

    hash1 = compute_block_hash(None, tokens_block1)
    hash2 = compute_block_hash(hash1, tokens_block2)

    print(f"Block 1 tokens: {tokens_block1[:4]}...{tokens_block1[-4:]}")
    print(f"Block 1 hash: {hash1}")
    print()
    print(f"Block 2 tokens: {tokens_block2[:4]}...{tokens_block2[-4:]}")
    print(f"Block 2 hash (with parent): {hash2}")
    print()

    hash1_repeat = compute_block_hash(None, tokens_block1)
    print(f"Hash is deterministic: {hash1 == hash1_repeat}")
    print()

    print("=" * 70)
    print("Part 2: PrefixCache Operations")
    print("-" * 40)

    from mini_vllm.core.block import Block

    cache = PrefixCache(max_cached_blocks=100, enable_caching=True)

    block1 = Block(block_id=0, block_size=16)
    block2 = Block(block_id=1, block_size=16)

    cache.insert(hash1, block1)
    print(f"Inserted block 1 with hash {hash1}")
    print(f"Cache stats: {cache.get_stats()}")
    print()

    found = cache.lookup(hash1)
    print(f"Lookup hash {hash1}: {'HIT' if found else 'MISS'}")
    print(f"Cache stats: {cache.get_stats()}")
    print()

    fake_hash = 12345
    found = cache.lookup(fake_hash)
    print(f"Lookup fake hash: {'HIT' if found else 'MISS'}")
    print(f"Cache stats: {cache.get_stats()}")
    print()

    print("=" * 70)
    print("Part 3: End-to-End Prefix Caching with LLM")
    print("-" * 40)

    model_config = ModelConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        prefix_cache_ratio=0.3,
    )

    scheduler_config = SchedulerConfig(
        max_num_seqs=32,
        max_num_batched_tokens=2048,
    )

    config = VllmConfig(
        model=model_config,
        cache=cache_config,
        scheduler=scheduler_config,
    )

    print("Loading model with prefix caching enabled...")
    print(f"  Prefix cache ratio: {cache_config.prefix_cache_ratio:.0%}")
    print()

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        gpu_memory_utilization=0.8,
    )
    print(f"Engine ready: {llm.engine}")
    print()

    system_prompt = """You are a helpful AI assistant. You provide accurate,
    concise, and helpful responses to user questions. Always be polite and
    professional in your responses. If you don't know something, admit it
    rather than making up information."""

    user_messages = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is deep learning?",
        "How does gradient descent work?",
        "What is backpropagation?",
    ]

    print("=" * 70)
    print("Sending 5 requests with the SAME system prompt prefix")
    print("=" * 70)
    print(f"System prompt length: ~{len(system_prompt.split())} words")
    print()

    prompts = [
        f"### System:\n{system_prompt}\n\n### User:\n{msg}\n\n### Assistant:\n"
        for msg in user_messages
    ]

    params = SamplingParams(max_tokens=50, temperature=0.7)

    start_time = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start_time

    for i, (msg, output) in enumerate(zip(user_messages, outputs)):
        print(f"[Request {i+1}] {msg}")
        print(f"  Response: {output.generated_text[:80]}...")
        print()

    total_tokens = sum(len(o.generated_token_ids) for o in outputs)
    print(f"Batch Statistics:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens/elapsed:.1f} tokens/sec")
    print()

    cache_stats = llm.engine.get_cache_stats()
    print(f"Cache Statistics:")
    print(f"  Total blocks: {cache_stats['total_blocks']}")
    print(f"  Free blocks: {cache_stats['free_blocks']}")
    print(f"  Utilization: {cache_stats['utilization']:.1%}")
    print()

    print("=" * 70)
    print("Part 4: Memory Savings Analysis")
    print("-" * 40)

    num_requests = len(prompts)
    prompt_tokens_per_request = 100
    block_size = 16

    blocks_without_caching = num_requests * (prompt_tokens_per_request // block_size)

    prefix_blocks = prompt_tokens_per_request // block_size
    unique_blocks_per_request = 1
    blocks_with_caching = prefix_blocks + (num_requests * unique_blocks_per_request)

    savings = 1 - (blocks_with_caching / blocks_without_caching)

    print(f"Theoretical analysis (for {num_requests} requests with shared prefix):")
    print(f"  Without prefix caching: ~{blocks_without_caching} block computations")
    print(f"  With prefix caching: ~{blocks_with_caching} block computations")
    print(f"  Computation savings: ~{savings:.0%}")
    print()

    print("=" * 70)
    print("Part 5: Different Prefixes (No Sharing)")
    print("-" * 40)

    different_prompts = [
        "### System:\nYou are a math tutor.\n\n### User:\nWhat is 2+2?\n\n### Assistant:\n",
        "### System:\nYou are a poet.\n\n### User:\nWrite a haiku.\n\n### Assistant:\n",
        "### System:\nYou are a chef.\n\n### User:\nRecipe for pasta?\n\n### Assistant:\n",
    ]

    print(f"Sending {len(different_prompts)} requests with DIFFERENT prefixes")
    print("(No cache hits expected)")
    print()

    start_time = time.time()
    outputs = llm.generate(different_prompts, params)
    elapsed = time.time() - start_time

    for i, output in enumerate(outputs):
        print(f"[Request {i+1}] {output.generated_text[:60]}...")

    print()
    print(f"Time for different prefixes: {elapsed:.2f}s")

    print()
    print("=" * 70)
    print("Summary: Prefix Caching Benefits")
    print("=" * 70)
    print("""
Prefix caching automatically detects and reuses KV cache blocks when
multiple requests share the same prefix (e.g., system prompt).

Benefits:
1. Memory efficiency: Shared prefixes stored once
2. Computation savings: KV values computed once, reused
3. Automatic: No user intervention needed
4. LRU eviction: Least-used cached blocks freed first

Use cases:
- Multiple user queries with same system prompt
- RAG applications with shared context
- Multi-turn conversations with shared history
- Batch processing with common instructions
