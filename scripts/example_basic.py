import sys
sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

from mini_vllm import LLM, SamplingParams

def main():

    print("Loading model...")
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
    )
    print("Model loaded!\n")

    print("=" * 50)
    print("Example 1: Simple generation")
    print("=" * 50)

    prompt = "The capital of France is"
    params = SamplingParams(
        max_tokens=20,
        temperature=0.7,
    )

    outputs = llm.generate(prompt, params)
    print(f"Prompt: {prompt}")
    print(f"Generated: {outputs[0].generated_text}\n")

    print("=" * 50)
    print("Example 2: Multiple prompts")
    print("=" * 50)

    prompts = [
        "Hello, my name is",
        "The best programming language is",
        "In the year 2050,",
    ]
    params = SamplingParams(
        max_tokens=30,
        temperature=0.8,
        top_p=0.9,
    )

    outputs = llm.generate(prompts, params)
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generated: {output.generated_text}")

    print("\n" + "=" * 50)
    print("Example 3: Chat completion")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]
    params = SamplingParams(max_tokens=100, temperature=0.7)

    output = llm.chat(messages, params)
    print(f"Response: {output.generated_text}")

    print("\n" + "=" * 50)
    print("Example 4: Generation with metrics")
    print("=" * 50)

    prompt = "Write a short poem about coding:"
    params = SamplingParams(max_tokens=50, temperature=0.9)

    outputs = llm.generate(prompt, params)
    output = outputs[0]

    print(f"Prompt: {prompt}")
    print(f"Generated: {output.generated_text}")
    if output.metrics:
        print(f"\nMetrics:")
        print(f"  Time to first token: {output.metrics.time_to_first_token:.3f}s")
        print(f"  Total time: {output.metrics.total_time:.3f}s")
        print(f"  Tokens generated: {len(output.generated_token_ids)}")
        if output.metrics.total_time and len(output.generated_token_ids) > 0:
            tokens_per_sec = len(output.generated_token_ids) / output.metrics.total_time
            print(f"  Tokens/second: {tokens_per_sec:.2f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
