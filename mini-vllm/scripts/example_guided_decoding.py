import sys
import time
import json
sys.path.insert(0, "/Users/maruthi/Documents/dev/vLLM/mini-vllm")

import torch
from mini_vllm.config import ModelConfig
from mini_vllm.model.loader import load_model_and_tokenizer
from mini_vllm.sampling.sampling_params import SamplingParams
from mini_vllm.guided import (
    Grammar,
    JSONSchemaParser,
    GuidedLogitsProcessor,
    ChoiceLogitsProcessor,
    GuidedDecodingConfig,
    create_guided_processor,
    parse_json_schema,
)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def demo_json_schema_parsing():
    print("=" * 70)
    print("Part 1: JSON Schema Parsing")
    print("=" * 70)
    print()

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
            "is_active": {"type": "boolean"},
        },
        "required": ["name", "age"],
    }

    print("JSON Schema:")
    print(json.dumps(schema, indent=2))
    print()

    grammar = parse_json_schema(schema)
    print(f"Parsed grammar with {len(grammar.rules)} rules:")
    for name, rule in list(grammar.rules.items())[:5]:
        print(f"  {name}: {rule.pattern[:50]}...")
    print()

    test_cases = [
        '{"name": "Alice", "age": 30}',
        '{"name": "Bob", "age": 25, "is_active": true}',
        '{"age": 20}',
        'not json at all',
    ]

    print("Validation tests:")
    for test in test_cases:
        is_valid = grammar.validate_json(test)
        status = "VALID" if is_valid else "INVALID"
        print(f"  [{status}] {test[:40]}...")
    print()

def demo_choice_constraint():
    print("=" * 70)
    print("Part 2: Multiple Choice Selection")
    print("=" * 70)
    print()

    device = get_device()
    print(f"Using device: {device}")

    config = ModelConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
    )
    model, tokenizer = load_model_and_tokenizer(config)
    model = model.to(device)
    model.eval()

    choices = ["positive", "negative", "neutral"]
    print(f"Choices: {choices}")

    processor = ChoiceLogitsProcessor(choices, tokenizer)

    prompt = """Classify the sentiment of the following text.
Text: "I love this product! It's amazing!"
Sentiment: """

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {prompt}")
    print(f"Generating constrained to choices...")

    generated_tokens = []
    max_new_tokens = 10

    with torch.no_grad():
        current_ids = input_ids

        for _ in range(max_new_tokens):
            outputs = model(current_ids)
            logits = outputs.logits[0, -1, :]

            masked_logits = processor(logits)

            next_token = torch.argmax(masked_logits).item()
            processor.advance(next_token)
            generated_tokens.append(next_token)

            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

            if processor.is_complete():
                break

    result = tokenizer.decode(generated_tokens)
    print(f"Generated: '{result}'")
    print(f"Is valid choice: {result.strip() in choices}")
    print()

def demo_json_generation():
    print("=" * 70)
    print("Part 3: JSON Schema-Constrained Generation")
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

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        },
        "required": ["name", "age"],
    }

    print("JSON Schema:")
    print(json.dumps(schema, indent=2))
    print()

    try:
        processor = GuidedLogitsProcessor(schema, tokenizer, use_outlines=True)
        print("Using outlines-backed FSM for guidance")
    except Exception as e:
        print(f"Outlines not available, using basic FSM: {e}")
        processor = GuidedLogitsProcessor(schema, tokenizer, use_outlines=False)

    prompt = """Generate a JSON object for a person.
JSON: """

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt: {prompt}")
    print("Generating JSON...")

    generated_tokens = []
    max_new_tokens = 50

    with torch.no_grad():
        current_ids = input_ids

        for i in range(max_new_tokens):
            outputs = model(current_ids)
            logits = outputs.logits[0, -1, :]

            masked_logits = processor(logits)

            probs = torch.softmax(masked_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            processor.advance(next_token)
            generated_tokens.append(next_token)

            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

            try:
                text = tokenizer.decode(generated_tokens)
                json.loads(text)
                break
            except json.JSONDecodeError:
                continue

    result = tokenizer.decode(generated_tokens)
    print(f"\nGenerated: {result}")

    try:
        parsed = json.loads(result)
        print(f"Parsed JSON: {parsed}")
        print("Valid JSON: True")
    except json.JSONDecodeError as e:
        print(f"Valid JSON: False - {e}")
    print()

def demo_guided_config():
    print("=" * 70)
    print("Part 4: GuidedDecodingConfig API")
    print("=" * 70)
    print()

    configs = [

        GuidedDecodingConfig(
            json_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                }
            }
        ),

        GuidedDecodingConfig(
            choice=["yes", "no", "maybe"]
        ),

        GuidedDecodingConfig(
            regex_pattern=r"[A-Z][a-z]+ [A-Z][a-z]+"
        ),
    ]

    print("Configuration examples:")
    for i, cfg in enumerate(configs, 1):
        print(f"\n{i}. Config:")
        if cfg.json_schema:
            print(f"   Type: JSON Schema")
            print(f"   Schema: {json.dumps(cfg.json_schema)[:60]}...")
        elif cfg.choice:
            print(f"   Type: Multiple Choice")
            print(f"   Choices: {cfg.choice}")
        elif cfg.regex_pattern:
            print(f"   Type: Regex Pattern")
            print(f"   Pattern: {cfg.regex_pattern}")
    print()

def demo_practical_example():
    print("=" * 70)
    print("Part 5: Practical Example - Structured Entity Extraction")
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

    entity_schema = {
        "type": "object",
        "properties": {
            "entity_type": {
                "type": "string",
                "enum": ["PERSON", "ORGANIZATION", "LOCATION", "DATE"]
            },
            "text": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["entity_type", "text"],
    }

    print("Entity extraction schema:")
    print(json.dumps(entity_schema, indent=2))
    print()

    text = "Apple Inc. announced today that Tim Cook will visit Tokyo next month."
    prompt = f"""Extract the main entity from this text as JSON.
Text: "{text}"
Entity: """

    print(f"Input text: {text}")
    print(f"Extracting entity...")

    expected_output = {
        "entity_type": "ORGANIZATION",
        "text": "Apple Inc.",
        "confidence": 0.95
    }

    print(f"\nExpected output format:")
    print(json.dumps(expected_output, indent=2))
    print()

    grammar = parse_json_schema(entity_schema)
    is_valid = grammar.validate_json(json.dumps(expected_output))
    print(f"Output validates against schema: {is_valid}")
    print()

def main():
    print("=" * 70)
    print("Mini vLLM - Phase 7: Guided Decoding Demo")
    print("=" * 70)
    print()

    print("Guided decoding ensures LLM outputs are valid structured data by:")
    print("  1. Converting JSON Schema to grammar rules")
    print("  2. Building FSM from grammar for token validation")
    print("  3. Masking invalid tokens before sampling")
    print("  4. Advancing FSM state after each token")
    print()

    demo_json_schema_parsing()

    print("\n" + "=" * 70)
    demo_choice_constraint()

    print("\n" + "=" * 70)
    demo_json_generation()

    print("\n" + "=" * 70)
    demo_guided_config()

    print("\n" + "=" * 70)
    demo_practical_example()

    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
Guided decoding benefits:
  1. Guaranteed valid output structure
  2. No post-processing validation needed
  3. Reduced token waste on invalid paths
  4. Works with any JSON Schema

Supported constraint types:
  - JSON Schema (objects, arrays, primitives, enums)
  - Multiple choice (list of allowed outputs)
  - Regex patterns (with outlines library)

For best results with complex schemas, install outlines:
    pip install outlines

The outlines library provides production-quality FSM compilation
that handles edge cases and complex nested schemas.
