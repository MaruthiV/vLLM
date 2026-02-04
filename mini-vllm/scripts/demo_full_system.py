import sys
import time
import json
from typing import List, Dict

sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")

def print_step(step_num: int, description: str):
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)

def demo_basic_generation():
    print_header("Demo 1: Basic Text Generation")

    from mini_vllm import LLM, SamplingParams

    print("Loading model...")
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        gpu_memory_utilization=0.8,
    )
    print(f"Model loaded: {llm.model_name}\n")

    prompt = "What is machine learning? Explain in one sentence."
    print(f"Prompt: {prompt}\n")

    params = SamplingParams(max_tokens=50, temperature=0.7)

    start = time.time()
    outputs = llm.generate(prompt, params)
    elapsed = time.time() - start

    print(f"Response: {outputs[0].generated_text}")
    print(f"\nGeneration time: {elapsed:.2f}s")
    print(f"Tokens generated: {len(outputs[0].generated_token_ids)}")
    print(f"Tokens/sec: {len(outputs[0].generated_token_ids)/elapsed:.1f}")

    return llm

def demo_batch_generation(llm):
    print_header("Demo 2: Continuous Batching")

    from mini_vllm import SamplingParams

    prompts = [
        "What is Python?",
        "Explain neural networks:",
        "The capital of France is",
        "Write a haiku about coding:",
        "What is 2+2?",
    ]

    print(f"Submitting {len(prompts)} prompts for concurrent processing...\n")
    for i, p in enumerate(prompts):
        print(f"  [{i+1}] {p}")

    params = SamplingParams(max_tokens=30, temperature=0.7)

    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    print("\nResponses:")
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n[{i+1}] {prompt}")
        print(f"    → {output.generated_text[:60]}...")

    total_tokens = sum(len(o.generated_token_ids) for o in outputs)
    print(f"\nBatch Statistics:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens/elapsed:.1f} tokens/sec")

def demo_kv_cache_stats(llm):
    print_header("Demo 3: Paged KV Cache")

    stats = llm.engine.get_cache_stats()

    print("KV Cache Configuration:")
    print(f"  Total blocks: {stats['total_blocks']}")
    print(f"  Free blocks: {stats['free_blocks']}")
    print(f"  Utilization: {stats['utilization']:.1%}")
    print(f"  Cache memory: {stats['cache_memory_mb']:.1f} MB")

    if 'prefix_cache' in stats:
        pc = stats['prefix_cache']
        print(f"\nPrefix Cache:")
        print(f"  Cached blocks: {pc.get('num_cached', 0)}")
        print(f"  Hit rate: {pc.get('hit_rate', 0):.1%}")

    print("\nKey benefits of paged KV cache:")
    print("  ✓ Memory proportional to actual sequence length")
    print("  ✓ No external fragmentation")
    print("  ✓ Just-in-time block allocation")
    print("  ✓ Copy-on-Write for shared prefixes")

def demo_prefix_caching(llm):
    print_header("Demo 4: Prefix Caching")

    from mini_vllm import SamplingParams

    system_prompt = """You are a helpful AI assistant. You provide accurate,
    concise, and helpful responses. Always be professional."""

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    prompts = [
        f"System: {system_prompt}\n\nUser: {q}\n\nAssistant:"
        for q in questions
    ]

    print(f"System prompt: {system_prompt[:50]}...")
    print(f"\nSending {len(questions)} requests with SAME system prompt prefix:")
    for q in questions:
        print(f"  - {q}")

    params = SamplingParams(max_tokens=30, temperature=0.7)

    print("\n[First pass - populating cache]")
    start = time.time()
    _ = llm.generate(prompts[:1], params)
    first_time = time.time() - start
    print(f"  Time: {first_time:.3f}s")

    print("\n[Second pass - using cached prefix]")
    start = time.time()
    _ = llm.generate(prompts[1:2], params)
    second_time = time.time() - start
    print(f"  Time: {second_time:.3f}s")

    if second_time < first_time:
        speedup = first_time / second_time
        print(f"\n  → {speedup:.1f}x faster with cached prefix!")

def demo_guided_decoding():
    print_header("Demo 5: Guided Decoding (Structured Output)")

    from mini_vllm.guided import (
        GuidedLogitsProcessor,
        ChoiceLogitsProcessor,
        parse_json_schema,
    )

    print("Example 1: JSON Schema Constraint")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age"],
    }
    print(f"Schema: {json.dumps(schema, indent=2)}")

    grammar = parse_json_schema(schema)
    sample_output = '{"name": "Alice", "age": 30, "city": "NYC"}'
    is_valid = grammar.validate_json(sample_output)
    print(f"\nSample output: {sample_output}")
    print(f"Valid against schema: {is_valid}")

    print("\n" + "-" * 40)
    print("Example 2: Multiple Choice Constraint")
    choices = ["positive", "negative", "neutral"]
    print(f"Choices: {choices}")
    print("→ LLM output constrained to these exact options")

    print("\nGuided decoding ensures:")
    print("  ✓ Output always matches schema")
    print("  ✓ No post-processing validation needed")
    print("  ✓ Works with any JSON schema")

def demo_streaming():
    print_header("Demo 6: Streaming Responses")

    import asyncio
    from mini_vllm.engine.async_llm_engine import AsyncLLMEngine
    from mini_vllm.config import VllmConfig, ModelConfig

    async def stream_demo():
        config = VllmConfig(
            model=ModelConfig(
                model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                dtype="float16",
            ),
        )

        engine = AsyncLLMEngine.from_config(config)
        await engine.start()

        prompt = "Count from 1 to 5:"
        print(f"Prompt: {prompt}")
        print("Streaming: ", end="", flush=True)

        from mini_vllm.sampling.sampling_params import SamplingParams
        params = SamplingParams(max_tokens=30, temperature=0.7)

        token_count = 0
        async for output in engine.generate(prompt, params):

            if output.generated_text:
                new_text = output.generated_text[token_count:]
                print(new_text, end="", flush=True)
                token_count = len(output.generated_text)

        print("\n")
        await engine.stop()

    asyncio.run(stream_demo())
    print("Streaming enables low-latency user experience!")

def demo_api_usage():
    print_header("Demo 7: OpenAI-Compatible API")

    print("Server command:")
    print("  python scripts/run_server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print()

    print("curl example:")
    print('''  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
      "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "messages": [{"role": "user", "content": "Hello!"}],
      "max_tokens": 50
    }' ''')
    print()

    print("Python OpenAI client:")
    print('''  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
  response = client.chat.completions.create(
      model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      messages=[{"role": "user", "content": "Hello!"}],
  )
  print(response.choices[0].message.content)''')
    print()

    print("API Endpoints:")
    print("  POST /v1/chat/completions  - Chat completions (streaming supported)")
    print("  GET  /v1/models            - List available models")
    print("  GET  /health               - Health check")

def demo_summary():
    print_header("Mini-vLLM Feature Summary")

    features = [
        ("Paged KV Cache", "Memory-efficient block-based allocation"),
        ("Continuous Batching", "Dynamic request scheduling for high throughput"),
        ("Prefix Caching", "Automatic caching of shared prefixes"),
        ("Speculative Decoding", "Draft-verify for faster generation"),
        ("Guided Decoding", "Guaranteed valid structured outputs"),
        ("Triton Kernels", "GPU-optimized paged attention"),
        ("OpenAI API", "Drop-in compatible HTTP server"),
        ("Streaming", "Low-latency token-by-token responses"),
    ]

    print("Core Features:")
    for name, desc in features:
        print(f"  ✓ {name:20s} - {desc}")

    print("\nUse Cases:")
    print("  • High-throughput inference serving")
    print("  • Structured data extraction (JSON)")
    print("  • Chatbots with streaming responses")
    print("  • Batch processing workloads")

    print("\n" + "=" * 70)
    print(" Thank you for exploring mini-vLLM!")
    print(" GitHub: https://github.com/yourusername/mini-vllm")
    print("=" * 70)

def main():
    print("\n" + "=" * 70)
    print(" 🚀 Mini-vLLM: Full System Demo")
    print("=" * 70)
    print("\nThis demo showcases all major features of mini-vLLM,")
    print("a portfolio project implementing production LLM serving techniques.\n")

    input("Press Enter to start the demo...")

    llm = demo_basic_generation()
    input("\nPress Enter for next demo...")

    demo_batch_generation(llm)
    input("\nPress Enter for next demo...")

    demo_kv_cache_stats(llm)
    input("\nPress Enter for next demo...")

    demo_prefix_caching(llm)
    input("\nPress Enter for next demo...")

    demo_guided_decoding()
    input("\nPress Enter for next demo...")

    demo_streaming()
    input("\nPress Enter for next demo...")

    demo_api_usage()
    input("\nPress Enter for summary...")

    demo_summary()

if __name__ == "__main__":
    main()
