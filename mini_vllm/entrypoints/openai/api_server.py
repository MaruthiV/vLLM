import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from mini_vllm import __version__
from mini_vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig
from mini_vllm.engine.async_llm_engine import AsyncLLMEngine
from mini_vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelList,
    ModelCard,
    HealthResponse,
    ErrorResponse,
    create_error_response,
)
from mini_vllm.entrypoints.openai.serving_chat import OpenAIChatServing

engine: Optional[AsyncLLMEngine] = None
chat_serving: Optional[OpenAIChatServing] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, chat_serving

    if engine is not None:
        await engine.start()
        print(f"Server ready. Model: {engine.get_model_name()}")

    yield

    if engine is not None:
        await engine.stop()
        print("Server stopped.")

app = FastAPI(
    title="Mini vLLM API",
    description="OpenAI-compatible API server for mini-vllm",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            message=str(exc.detail),
            err_type="invalid_request_error",
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            message=str(exc),
            err_type="internal_error",
        ).model_dump(),
    )

@app.get("/health")
async def health() -> HealthResponse:
    if engine is None:
        return HealthResponse(status="unhealthy", version=__version__)

    return HealthResponse(
        status="healthy",
        model=engine.get_model_name(),
        version=__version__,
    )

@app.get("/v1/models")
async def list_models() -> ModelList:
    if engine is None:
        return ModelList(data=[])

    model_card = ModelCard(
        id=engine.get_model_name(),
        created=int(time.time()),
        owned_by="mini-vllm",
    )
    return ModelList(data=[model_card])

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelCard:
    if engine is None:
        raise HTTPException(status_code=404, detail="Model not found")

    if model_id != engine.get_model_name():
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelCard(
        id=model_id,
        created=int(time.time()),
        owned_by="mini-vllm",
    )

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    if engine is None or chat_serving is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        result = await chat_serving.create_chat_completion(request)

        if request.stream:

            return EventSourceResponse(
                result,
                media_type="text/event-stream",
            )
        else:

            return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def init_engine(
    model: str,
    dtype: str = "float16",
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 4096,
    trust_remote_code: bool = False,
) -> None:
    global engine, chat_serving

    model_config = ModelConfig(
        model_name_or_path=model,
        dtype=dtype,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )

    cache_config = CacheConfig(
        gpu_memory_utilization=gpu_memory_utilization,
    )

    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    config = VllmConfig(
        model=model_config,
        cache=cache_config,
        scheduler=scheduler_config,
    )

    engine = AsyncLLMEngine.from_config(config)
    chat_serving = OpenAIChatServing(engine)

    print(f"Initialized engine with model: {model}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mini vLLM OpenAI-compatible API server"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory for KV cache (0.0-1.0)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum concurrent sequences",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per batch",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    init_engine(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API docs available at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )

if __name__ == "__main__":
    main()
