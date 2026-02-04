# Mini-vLLM

A high-performance LLM inference engine implementing PagedAttention, continuous batching, and custom Triton GPU kernels. Inspired by [vLLM](https://github.com/vllm-project/vllm), this engine achieves 2-3x throughput improvements over naive HuggingFace inference through memory-efficient KV cache management and dynamic request scheduling.

## Features

| Feature | Description |
|---------|-------------|
| **Paged KV Cache** | Memory-efficient block-based allocation eliminating fragmentation |
| **Continuous Batching** | Dynamic request scheduling for maximum throughput |
| **Prefix Caching** | Automatic sharing of cached prefixes (system prompts) |
| **Speculative Decoding** | Draft-verify parallelism for faster generation |
| **Guided Decoding** | Guaranteed valid JSON/structured outputs |
| **Triton Kernels** | GPU-optimized paged attention (CUDA) |
| **OpenAI API** | Drop-in compatible HTTP server |

## Architecture

```
mini-vllm/
├── mini_vllm/
│   ├── entrypoints/          # API endpoints
│   │   ├── openai/           # OpenAI-compatible server
│   │   └── llm.py            # Direct Python API
│   ├── engine/               # Core inference engine
│   │   ├── llm_engine.py     # Synchronous engine
│   │   └── async_llm_engine.py
│   ├── core/                 # Scheduling & memory
│   │   ├── scheduler.py      # Continuous batching
│   │   ├── block_manager.py  # KV cache allocation
│   │   └── prefix_cache.py   # Prefix caching
│   ├── worker/               # Model execution
│   │   ├── model_runner.py
│   │   └── cache_engine.py
│   ├── speculative/          # Speculative decoding
│   ├── guided/               # Structured output
│   └── kernels/              # Triton kernels
├── scripts/                  # Example scripts
├── benchmarks/               # Performance benchmarks
└── notebooks/                # Colab notebooks
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mini-vllm.git
cd mini-vllm

# Install dependencies
pip install -e .

# For Triton kernels (CUDA GPU required)
pip install triton
```

### Basic Usage

```python
from mini_vllm import LLM, SamplingParams

# Load model
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype="float16")

# Generate
params = SamplingParams(max_tokens=100, temperature=0.7)
outputs = llm.generate("What is machine learning?", params)
print(outputs[0].generated_text)
```

### Batch Generation

```python
prompts = [
    "What is Python?",
    "Explain neural networks:",
    "Write a haiku:",
]
outputs = llm.generate(prompts, params)
for prompt, output in zip(prompts, outputs):
    print(f"{prompt} -> {output.generated_text}")
```

### OpenAI-Compatible Server

```bash
# Start server
python scripts/run_server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000

# Use with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Or with OpenAI Python client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Structured Output (Guided Decoding)

```python
from mini_vllm.guided import GuidedLogitsProcessor

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"]
}

# Output guaranteed to be valid JSON matching schema
processor = GuidedLogitsProcessor(schema, tokenizer)
```

## Key Concepts

### Paged KV Cache

Traditional KV cache allocates `max_seq_len` slots per sequence, wasting memory when actual sequences are shorter. Paged KV cache:

- Allocates fixed-size **blocks** (e.g., 16 tokens each)
- Uses **block tables** to map logical → physical blocks
- Enables **just-in-time allocation** as sequences grow
- Supports **Copy-on-Write** for shared prefixes

**Memory savings:** Up to 50%+ with variable-length sequences.

### Continuous Batching

Instead of processing requests one at a time or in fixed batches:

- New requests join the batch **immediately**
- Completed requests **leave immediately**
- **Interleaved prefill and decode** phases
- **Token budget** controls batch size dynamically

**Throughput improvement:** 2-10x vs naive batching.

### Prefix Caching

Automatically detects and shares KV cache for common prefixes:

- **Block hash** = hash(parent_hash + tokens)
- Same prefix → same hash → cache hit
- LRU eviction for memory management

**Use case:** Multiple requests with same system prompt.

### Speculative Decoding

Uses a smaller **draft model** to speed up generation:

1. Draft model generates K tokens speculatively
2. Target model verifies all K+1 positions in parallel
3. Rejection sampling preserves target distribution
4. Accepts multiple tokens per forward pass

**Speedup:** 1.5-3x with well-matched model pairs.

## Benchmarks

### Throughput vs HuggingFace

| Batch Size | mini-vLLM (tok/s) | HuggingFace (tok/s) | Speedup |
|------------|-------------------|---------------------|---------|
| 1          | 25                | 20                  | 1.25x   |
| 4          | 90                | 60                  | 1.50x   |
| 16         | 280               | 140                 | 2.00x   |
| 32         | 450               | 180                 | 2.50x   |

### Memory Efficiency

| Scenario | Paged (MB) | Dense (MB) | Savings |
|----------|------------|------------|---------|
| Short seqs (256 tok) | 32 | 128 | 75% |
| Mixed lengths | 128 | 512 | 75% |
| Max length (4096) | 512 | 512 | 0% |

## Run Benchmarks

```bash
# Throughput benchmark
python benchmarks/throughput_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Memory benchmark
python benchmarks/kv_memory_benchmark.py

# Generate visualizations
python benchmarks/visualizations.py --all
```

## Example Scripts

| Script | Description |
|--------|-------------|
| `example_basic.py` | Basic text generation |
| `example_paged_cache.py` | KV cache demonstration |
| `example_continuous_batching.py` | Scheduler and batching |
| `example_prefix_caching.py` | Prefix cache benefits |
| `example_speculative_decoding.py` | Draft-verify speedup |
| `example_guided_decoding.py` | Structured JSON output |
| `example_triton_kernels.py` | GPU kernel benchmarks |
| `demo_full_system.py` | Full feature demo |

## GPU Support

**Local Development (Apple Silicon / MPS):**
- All features work with PyTorch MPS backend
- Triton kernels use PyTorch fallback

**Cloud GPU (Colab Pro / Lambda Labs):**
- Full Triton kernel support
- Run `notebooks/triton_kernels.ipynb` for GPU benchmarks

## Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run all tests
pytest tests/
```

## Project Structure

```
Phase 1: Foundation       - Model loading, sampling, basic engine
Phase 2: Paged KV Cache   - Block allocation, cache engine
Phase 3: Continuous Batch - Scheduler, request management
Phase 4: HTTP Server      - OpenAI-compatible API
Phase 5: Prefix Caching   - Block-hash caching
Phase 6: Speculative      - Draft model, rejection sampling
Phase 7: Guided Decoding  - JSON schema, FSM constraints
Phase 8: Triton Kernels   - GPU-optimized attention
Phase 9: Benchmarks       - Visualizations, demos
```

## Dependencies

- Python 3.10+
- PyTorch 2.1+
- transformers 4.36+
- FastAPI + uvicorn
- Pydantic 2.x
- triton (optional, for GPU kernels)
- outlines (optional, for guided decoding)

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Efficient Memory Management for LLM Serving
- [Speculative Decoding](https://arxiv.org/abs/2302.01318) - Fast Inference via Speculative Decoding
- [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) - vLLM Blog Post
- [Outlines](https://github.com/outlines-dev/outlines) - Structured Generation

## License

MIT License

## Author

Built as a portfolio project to demonstrate understanding of production LLM serving systems.
