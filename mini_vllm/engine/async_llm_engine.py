import asyncio
from typing import AsyncIterator, Dict, List, Optional, Set
from dataclasses import dataclass
import time

from mini_vllm.config import VllmConfig
from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.engine.output import RequestOutput
from mini_vllm.sampling.sampling_params import SamplingParams

@dataclass
class AsyncRequest:

    request_id: str
    arrival_time: float
    result_queue: asyncio.Queue
    finished: bool = False
    aborted: bool = False

class AsyncLLMEngine:

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.config = engine.config

        self._requests: Dict[str, AsyncRequest] = {}
        self._request_counter = 0

        self._background_loop: Optional[asyncio.Task] = None
        self._is_running = False
        self._stop_event = asyncio.Event()

        self._loop_interval = 0.001

    @classmethod
    def from_config(cls, config: VllmConfig) -> "AsyncLLMEngine":
        engine = LLMEngine(config)
        return cls(engine)

    @classmethod
    def from_model(
        cls,
        model: str,
        dtype: str = "float16",
        **kwargs,
    ) -> "AsyncLLMEngine":
        config = VllmConfig.from_model_name(model, dtype=dtype, **kwargs)
        return cls.from_config(config)

    async def start(self) -> None:
        if self._is_running:
            return

        self._is_running = True
        self._stop_event.clear()
        self._background_loop = asyncio.create_task(self._run_engine_loop())

    async def stop(self) -> None:
        if not self._is_running:
            return

        self._is_running = False
        self._stop_event.set()

        if self._background_loop:
            await self._background_loop
            self._background_loop = None

    async def _run_engine_loop(self) -> None:
        while self._is_running:

            if self._stop_event.is_set():
                break

            if self.engine.has_unfinished_requests():
                await self._process_step()
            else:

                await asyncio.sleep(self._loop_interval)

    async def _process_step(self) -> None:

        outputs = self.engine.step()

        for output in outputs:
            request_id = output.request_id
            if request_id in self._requests:
                async_req = self._requests[request_id]

                await async_req.result_queue.put(output)

                if output.finished:
                    async_req.finished = True

        await asyncio.sleep(0)

    async def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> str:
        if request_id is None:
            request_id = f"async-req-{self._request_counter}"
            self._request_counter += 1

        async_req = AsyncRequest(
            request_id=request_id,
            arrival_time=time.time(),
            result_queue=asyncio.Queue(),
        )
        self._requests[request_id] = async_req

        self.engine.add_request(request_id, prompt, sampling_params)

        return request_id

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[RequestOutput]:
        request_id = await self.add_request(prompt, sampling_params, request_id)
        async_req = self._requests[request_id]

        try:
            while not async_req.finished and not async_req.aborted:
                try:

                    output = await asyncio.wait_for(
                        async_req.result_queue.get(),
                        timeout=30.0,
                    )
                    yield output

                    if output.finished:
                        break
                except asyncio.TimeoutError:

                    if not self.engine.has_unfinished_requests():
                        break
        finally:

            if request_id in self._requests:
                del self._requests[request_id]

    async def generate_to_completion(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> RequestOutput:
        final_output = None
        async for output in self.generate(prompt, sampling_params, request_id):
            final_output = output
        return final_output

    async def abort_request(self, request_id: str) -> bool:
        if request_id in self._requests:
            self._requests[request_id].aborted = True
            self.engine.abort_request(request_id)
            del self._requests[request_id]
            return True
        return False

    def get_model_name(self) -> str:
        return self.config.model.model_name_or_path

    def get_tokenizer(self):
        return self.engine.tokenizer

    @property
    def is_running(self) -> bool:
        return self._is_running

    def __repr__(self) -> str:
        return (
            f"AsyncLLMEngine("
            f"model={self.config.model.model_name_or_path}, "
            f"running={self._is_running}, "
            f"pending={len(self._requests)})"
        )

async def create_async_engine(
    model: str,
    dtype: str = "float16",
    **kwargs,
) -> AsyncLLMEngine:
    engine = AsyncLLMEngine.from_model(model, dtype=dtype, **kwargs)
    await engine.start()
    return engine
