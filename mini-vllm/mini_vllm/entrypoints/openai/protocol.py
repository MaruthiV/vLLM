from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import time

class ChatMessage(BaseModel):

    role: Literal["system", "user", "assistant", "tool"]

    content: Optional[str] = None

    name: Optional[str] = None

    tool_calls: Optional[List[dict]] = None

    tool_call_id: Optional[str] = None

class DeltaMessage(BaseModel):

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None

class ChatCompletionRequest(BaseModel):

    model: str

    messages: List[ChatMessage]

    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)

    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)

    n: Optional[int] = Field(default=1, ge=1)

    max_tokens: Optional[int] = Field(default=None, ge=1)

    stop: Optional[Union[str, List[str]]] = None

    stream: Optional[bool] = False

    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

    logit_bias: Optional[Dict[str, float]] = None

    user: Optional[str] = None

    response_format: Optional[dict] = None

    seed: Optional[int] = None

    top_k: Optional[int] = Field(default=-1)

    min_p: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

    repetition_penalty: Optional[float] = Field(default=1.0, gt=0.0)

    skip_special_tokens: Optional[bool] = True

class UsageInfo(BaseModel):

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

class ChatCompletionChoice(BaseModel):

    index: int

    message: ChatMessage

    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None

    logprobs: Optional[dict] = None

class ChatCompletionResponse(BaseModel):

    id: str

    object: Literal["chat.completion"] = "chat.completion"

    created: int = Field(default_factory=lambda: int(time.time()))

    model: str

    choices: List[ChatCompletionChoice]

    usage: Optional[UsageInfo] = None

    system_fingerprint: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):

    index: int

    delta: DeltaMessage

    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None

    logprobs: Optional[dict] = None

class ChatCompletionStreamResponse(BaseModel):

    id: str

    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"

    created: int = Field(default_factory=lambda: int(time.time()))

    model: str

    choices: List[ChatCompletionStreamChoice]

    system_fingerprint: Optional[str] = None

class ModelCard(BaseModel):

    id: str

    object: Literal["model"] = "model"

    created: int = Field(default_factory=lambda: int(time.time()))

    owned_by: str = "mini-vllm"

    root: Optional[str] = None

    parent: Optional[str] = None

    permission: List[dict] = Field(default_factory=list)

class ModelList(BaseModel):

    object: Literal["list"] = "list"
    data: List[ModelCard]

class ErrorResponse(BaseModel):

    object: Literal["error"] = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None

class HealthResponse(BaseModel):

    status: Literal["healthy", "unhealthy"]
    model: Optional[str] = None
    version: str = "0.1.0"

def create_error_response(
    message: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> ErrorResponse:
    return ErrorResponse(
        message=message,
        type=err_type,
        param=param,
        code=code,
    )

def random_completion_id() -> str:
    import secrets
    return f"chatcmpl-{secrets.token_hex(12)}"
