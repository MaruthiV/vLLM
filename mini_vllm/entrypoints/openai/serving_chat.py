from typing import AsyncIterator, List, Optional, Union
import time

from mini_vllm.engine.async_llm_engine import AsyncLLMEngine
from mini_vllm.engine.output import RequestOutput
from mini_vllm.sampling.sampling_params import SamplingParams
from mini_vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionChoice,
    ChatCompletionStreamChoice,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
    ErrorResponse,
    create_error_response,
    random_completion_id,
)

class OpenAIChatServing:

    def __init__(self, engine: AsyncLLMEngine):
        self.engine = engine
        self.model_name = engine.get_model_name()
        self.tokenizer = engine.get_tokenizer()

    def _create_sampling_params(
        self,
        request: ChatCompletionRequest,
    ) -> SamplingParams:

        stop = None
        if request.stop:
            if isinstance(request.stop, str):
                stop = [request.stop]
            else:
                stop = list(request.stop)

        max_tokens = request.max_tokens
        if max_tokens is None:
            max_tokens = 2048

        return SamplingParams(
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            top_k=request.top_k or -1,
            min_p=request.min_p or 0.0,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=request.presence_penalty or 0.0,
            frequency_penalty=request.frequency_penalty or 0.0,
            repetition_penalty=request.repetition_penalty or 1.0,
            n=request.n or 1,
            seed=request.seed,
            skip_special_tokens=request.skip_special_tokens,
        )

    def _apply_chat_template(
        self,
        messages: List[ChatMessage],
    ) -> str:

        message_dicts = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in messages
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    message_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception:
                pass

        return self._format_messages_fallback(message_dicts)

    def _format_messages_fallback(self, messages: List[dict]) -> str:
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_parts.append(f"### System:\n{content}\n")
            elif role == "user":
                formatted_parts.append(f"### User:\n{content}\n")
            elif role == "assistant":
                formatted_parts.append(f"### Assistant:\n{content}\n")

        formatted_parts.append("### Assistant:\n")

        return "\n".join(formatted_parts)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, AsyncIterator[str]]:

        completion_id = random_completion_id()
        created_time = int(time.time())

        prompt = self._apply_chat_template(request.messages)

        sampling_params = self._create_sampling_params(request)

        if request.stream:
            return self._stream_chat_completion(
                prompt=prompt,
                sampling_params=sampling_params,
                request=request,
                completion_id=completion_id,
                created_time=created_time,
            )
        else:
            return await self._complete_chat_completion(
                prompt=prompt,
                sampling_params=sampling_params,
                request=request,
                completion_id=completion_id,
                created_time=created_time,
            )

    async def _complete_chat_completion(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request: ChatCompletionRequest,
        completion_id: str,
        created_time: int,
    ) -> ChatCompletionResponse:

        output = await self.engine.generate_to_completion(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(output.generated_token_ids)

        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content=output.generated_text,
            ),
            finish_reason=output.output.finish_reason if output.output else "stop",
        )

        return ChatCompletionResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _stream_chat_completion(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request: ChatCompletionRequest,
        completion_id: str,
        created_time: int,
    ) -> AsyncIterator[str]:

        first_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        previous_text = ""

        async for output in self.engine.generate(prompt, sampling_params):
            current_text = output.generated_text
            new_text = current_text[len(previous_text):]

            if new_text:
                chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(content=new_text),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                previous_text = current_text

            if output.finished:
                finish_reason = output.output.finish_reason if output.output else "stop"
                final_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=DeltaMessage(),
                            finish_reason=finish_reason,
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                break

        yield "data: [DONE]\n\n"

def create_chat_serving(engine: AsyncLLMEngine) -> OpenAIChatServing:
    return OpenAIChatServing(engine)
