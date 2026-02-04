import sys
import time
sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

import torch
from mini_vllm.config import VllmConfig, ModelConfig, SpeculativeConfig
from mini_vllm.model.loader import load_model_and_tokenizer
from mini_vllm.sampling.sampling_params import SamplingParams
from mini_vllm.speculative import (
    DraftModelRunner,
    RejectionSampler,
    SpeculativeDecodeWorker,
    create_speculative_worker,
)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def demo_rejection_sampling():
    print("=" * 70)
    print("Part 1: Rejection Sampling Algorithm")
    print("=" * 70)
    print()

    vocab_size = 100
    K = 5

    draft_probs = torch.softmax(torch.randn(K, vocab_size) * 0.8, dim=-1)

    target_probs = torch.softmax(torch.randn(K, vocab_size) * 1.2, dim=-1)

    draft_token_ids = [torch.multinomial(draft_probs[i], 1).item() for i in range(K)]

    print(f"Draft tokens: {draft_token_ids}")
    print()

    sampler = RejectionSampler()
    output = sampler.sample(
        draft_token_ids=draft_token_ids,
        draft_probs=draft_probs,
        target_probs=target_probs,
    )

    print(f"Rejection sampling results:")
    print(f"  Accepted tokens: {output.accepted_token_ids}")
    print(f"  Num accepted: {output.num_accepted} / {K}")
    print(f"  Bonus token: {output.bonus_token_id}")
    print(f"  Acceptance mask: {output.acceptance_mask}")
    print(f"  Total new tokens: {output.total_tokens}")
    print()

    expected_rate = sampler.compute_acceptance_rate(
        draft_probs, target_probs, draft_token_ids
    )
    print(f"  Expected acceptance rate: {expected_rate:.1%}")
    print()

    print("Algorithm explanation:")
    print("  For each draft token x at position i:")
    print("    - Get q(x) = draft_prob[i, x]")
    print("    - Get p(x) = target_prob[i, x]")
    print("    - Accept with probability min(1, p(x)/q(x))")
    print("    - If rejected, sample from residual: max(0, p - q)")
    print()

def demo_draft_model():
    print("=" * 70)
    print("Part 2: Draft Model Speculation")
    print("=" * 70)
    print()

    device = get_device()
    print(f"Using device: {device}")
    print()

    spec_config = SpeculativeConfig(
        enabled=True,
        draft_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_speculative_tokens=5,
        draft_model_dtype="float16",
    )

    print(f"Loading draft model: {spec_config.draft_model}")
    draft_runner = DraftModelRunner(spec_config, device)
    print(f"Draft model info: {draft_runner.get_model_info()}")
    print()

    prompt = "The quick brown fox"
    input_ids = draft_runner.tokenizer.encode(prompt, return_tensors="pt")
    print(f"Prompt: '{prompt}'")
    print(f"Input IDs: {input_ids[0].tolist()}")
    print()

    print(f"Generating {spec_config.num_speculative_tokens} draft tokens...")
    start_time = time.time()
    draft_output = draft_runner.speculate(
        input_ids=input_ids,
        num_speculative_tokens=5,
    )
    elapsed = time.time() - start_time

    print(f"Draft tokens: {draft_output.draft_token_ids}")
    print(f"Decoded: {draft_runner.tokenizer.decode(draft_output.draft_token_ids)}")
    print(f"Draft probs shape: {draft_output.draft_probs.shape}")
    print(f"Time: {elapsed:.3f}s")
    print()

    return draft_runner

def demo_speculative_decoding(draft_runner=None):
    print("=" * 70)
    print("Part 3: Full Speculative Decoding")
    print("=" * 70)
    print()

    device = get_device()

    print("Loading target model (TinyLlama for demo)...")
    target_config = ModelConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
    )
    target_model, tokenizer = load_model_and_tokenizer(target_config)
    target_model = target_model.to(device)
    target_model.eval()
    print("Target model loaded.")
    print()

    if draft_runner is None:
        spec_config = SpeculativeConfig(
            enabled=True,
            draft_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            num_speculative_tokens=5,
            draft_model_dtype="float16",
        )
        draft_runner = DraftModelRunner(spec_config, device)

    worker = SpeculativeDecodeWorker(
        target_model=target_model,
        draft_runner=draft_runner,
        device=device,
        num_speculative_tokens=5,
    )

    prompt = "### User:\nExplain machine learning in one sentence.\n\n### Assistant:\n"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"Prompt: {prompt[:50]}...")
    print(f"Input length: {input_ids.shape[1]} tokens")
    print()

    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.7,
    )

    print("Generating with speculative decoding...")
    print("-" * 40)

    generated_tokens = []
    start_time = time.time()

    for output in worker.generate(input_ids, sampling_params):
        generated_tokens.extend(output.new_token_ids)
        decoded = tokenizer.decode(output.new_token_ids)
        print(f"  Iteration: +{output.tokens_per_iteration} tokens "
              f"(accepted {output.num_accepted}/5) -> '{decoded}'")

    elapsed = time.time() - start_time

    print("-" * 40)
    print()

    full_text = tokenizer.decode(generated_tokens)
    print(f"Generated text: {full_text}")
    print()

    metrics = worker.get_metrics()
    print("Speculative Decoding Metrics:")
    print(f"  Total iterations: {metrics['total_iterations']}")
    print(f"  Total tokens generated: {metrics['total_generated_tokens']}")
    print(f"  Acceptance rate: {metrics['acceptance_rate']:.1%}")
    print(f"  Avg tokens/iteration: {metrics['tokens_per_iteration']:.2f}")
    print(f"  Estimated speedup: {metrics['speedup_factor']:.2f}x")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Tokens/sec: {metrics['total_generated_tokens']/elapsed:.1f}")
    print()

    return worker, tokenizer

def compare_with_standard_decoding():
    print("=" * 70)
    print("Part 4: Comparison with Standard Decoding")
    print("=" * 70)
    print()

    device = get_device()

    config = ModelConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
    )
    model, tokenizer = load_model_and_tokenizer(config)
    model = model.to(device)
    model.eval()

    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    max_tokens = 30

    print("Standard autoregressive decoding...")
    start_time = time.time()

    current_ids = input_ids
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(current_ids)
            next_token = torch.argmax(outputs.logits[0, -1, :])
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    standard_time = time.time() - start_time
    standard_text = tokenizer.decode(current_ids[0, input_ids.shape[1]:])

    print(f"  Generated: {standard_text[:60]}...")
    print(f"  Time: {standard_time:.2f}s")
    print(f"  Tokens/sec: {max_tokens/standard_time:.1f}")
    print()

    print("Speculative decoding (K=5)...")
    spec_config = SpeculativeConfig(
        enabled=True,
        draft_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_speculative_tokens=5,
    )
    draft_runner = DraftModelRunner(spec_config, device)
    worker = SpeculativeDecodeWorker(
        target_model=model,
        draft_runner=draft_runner,
        device=device,
        num_speculative_tokens=5,
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    start_time = time.time()
    tokens, metrics = worker.generate_to_completion(input_ids, params)
    spec_time = time.time() - start_time
    spec_text = tokenizer.decode(tokens)

    print(f"  Generated: {spec_text[:60]}...")
    print(f"  Time: {spec_time:.2f}s")
    print(f"  Tokens/sec: {len(tokens)/spec_time:.1f}")
    print(f"  Acceptance rate: {metrics.acceptance_rate:.1%}")
    print()

    print("-" * 40)
    print("Summary:")
    print(f"  Standard decoding: {standard_time:.2f}s ({max_tokens/standard_time:.1f} tok/s)")
    print(f"  Speculative decoding: {spec_time:.2f}s ({len(tokens)/spec_time:.1f} tok/s)")

    if spec_time < standard_time:
        speedup = standard_time / spec_time
        print(f"  Speedup: {speedup:.2f}x faster with speculative decoding!")
    else:
        print(f"  Note: Same model as draft/target, so minimal speedup expected.")
        print(f"  Real speedup comes from using a smaller draft model.")
    print()

def main():
    print("=" * 70)
    print("Mini vLLM - Phase 6: Speculative Decoding Demo")
    print("=" * 70)
    print()

    print("Speculative decoding accelerates inference by:")
    print("  1. Draft model generates K tokens speculatively (fast)")
    print("  2. Target model verifies all K+1 positions in parallel")
    print("  3. Rejection sampling preserves target distribution")
    print("  4. Accept up to K+1 tokens per iteration (vs 1 normally)")
    print()

    demo_rejection_sampling()

    print("\n" + "=" * 70)
    draft_runner = demo_draft_model()

    print("\n" + "=" * 70)
    demo_speculative_decoding(draft_runner)

    print("\n" + "=" * 70)
    compare_with_standard_decoding()

    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
Speculative decoding provides speedup when:
  1. Draft model is significantly faster than target (e.g., 1B vs 7B)
  2. Acceptance rate is high (draft and target agree often)
  3. K (num_speculative_tokens) is tuned appropriately

Typical speedups:
  - Same model family (e.g., LLaMA 7B + 1B): 1.5-2.5x
  - Well-matched models: up to 3x
  - Greedy decoding: higher acceptance than sampling

Note: This demo uses the same model as draft and target for simplicity.
In production, use a smaller draft model for real speedup.
