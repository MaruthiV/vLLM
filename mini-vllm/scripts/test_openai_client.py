import argparse
import asyncio
import json
import sys
import time
from typing import Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

def test_health(base_url: str) -> bool:
    print("\n" + "=" * 60)
    print("Testing: GET /health")
    print("=" * 60)

    if not HAS_REQUESTS:
        print("  [SKIP] requests library not installed")
        return True

    try:
        url = base_url.replace("/v1", "/health")
        response = requests.get(url, timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def test_list_models(base_url: str) -> bool:
    print("\n" + "=" * 60)
    print("Testing: GET /v1/models")
    print("=" * 60)

    if not HAS_REQUESTS:
        print("  [SKIP] requests library not installed")
        return True

    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        print(f"  Status: {response.status_code}")
        data = response.json()
        print(f"  Models: {[m['id'] for m in data.get('data', [])]}")
        return response.status_code == 200
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def test_chat_completion_sync(base_url: str, model: str) -> bool:
    print("\n" + "=" * 60)
    print("Testing: POST /v1/chat/completions (non-streaming)")
    print("=" * 60)

    if not HAS_REQUESTS:
        print("  [SKIP] requests library not installed")
        return True

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer briefly."}
            ],
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": False,
        }

        print(f"  Request: {json.dumps(payload, indent=2)}")

        start = time.time()
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=60,
        )
        elapsed = time.time() - start

        print(f"  Status: {response.status_code}")
        print(f"  Time: {elapsed:.2f}s")

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(f"  Response: {content}")
            print(f"  Usage: {usage}")
            return True
        else:
            print(f"  Error: {response.text}")
            return False

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def test_chat_completion_streaming(base_url: str, model: str) -> bool:
    print("\n" + "=" * 60)
    print("Testing: POST /v1/chat/completions (streaming)")
    print("=" * 60)

    if not HAS_REQUESTS:
        print("  [SKIP] requests library not installed")
        return True

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": True,
        }

        print(f"  Request: streaming enabled")

        start = time.time()
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=60,
            stream=True,
        )

        if response.status_code != 200:
            print(f"  Status: {response.status_code}")
            print(f"  Error: {response.text}")
            return False

        print("  Streaming response:")
        print("  ", end="")

        full_content = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_content += content
                    except json.JSONDecodeError:
                        pass

        elapsed = time.time() - start
        print()
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Full response: {full_content}")
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def test_with_openai_client(base_url: str, model: str) -> bool:
    print("\n" + "=" * 60)
    print("Testing: OpenAI Python Client")
    print("=" * 60)

    if not HAS_OPENAI:
        print("  [SKIP] openai library not installed")
        print("  Install with: pip install openai")
        return True

    try:
        client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
        )

        print("  Non-streaming request...")
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello in one word."}
            ],
            max_tokens=20,
            stream=False,
        )
        elapsed = time.time() - start
        print(f"  Response: {response.choices[0].message.content}")
        print(f"  Time: {elapsed:.2f}s")

        print("\n  Streaming request...")
        print("  ", end="")
        start = time.time()
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Count 1, 2, 3."}
            ],
            max_tokens=30,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.2f}s")

        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

async def test_concurrent_requests(base_url: str, model: str) -> bool:
    print("\n" + "=" * 60)
    print("Testing: Concurrent Requests")
    print("=" * 60)

    if not HAS_AIOHTTP:
        print("  [SKIP] aiohttp library not installed")
        return True

    async def make_request(session: aiohttp.ClientSession, idx: int) -> dict:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": f"What is {idx} + {idx}? Answer with just the number."}
            ],
            "max_tokens": 20,
            "temperature": 0.0,
            "stream": False,
        }

        start = time.time()
        async with session.post(f"{base_url}/chat/completions", json=payload) as resp:
            result = await resp.json()
            elapsed = time.time() - start
            return {
                "idx": idx,
                "time": elapsed,
                "response": result["choices"][0]["message"]["content"] if resp.status == 200 else "error",
            }

    try:
        num_requests = 3
        print(f"  Sending {num_requests} concurrent requests...")

        async with aiohttp.ClientSession() as session:
            start = time.time()
            tasks = [make_request(session, i) for i in range(1, num_requests + 1)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start

        print(f"\n  Results:")
        for r in results:
            print(f"    Request {r['idx']}: {r['response'][:50]}... ({r['time']:.2f}s)")

        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Requests/sec: {num_requests / total_time:.2f}")

        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Mini vLLM OpenAI API")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name to use in requests",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Mini vLLM OpenAI API Test Client")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")

    results = []

    results.append(("Health Check", test_health(args.base_url)))
    results.append(("List Models", test_list_models(args.base_url)))
    results.append(("Chat (non-streaming)", test_chat_completion_sync(args.base_url, args.model)))
    results.append(("Chat (streaming)", test_chat_completion_streaming(args.base_url, args.model)))
    results.append(("OpenAI Client", test_with_openai_client(args.base_url, args.model)))

    results.append(("Concurrent Requests", asyncio.run(test_concurrent_requests(args.base_url, args.model))))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")
        if success:
            passed += 1
        else:
            failed += 1

    print()
    print(f"  Total: {passed}/{len(results)} passed")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
