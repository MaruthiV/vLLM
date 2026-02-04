from typing import List, Optional, Iterator, Union, Dict
import time
import torch

from mini_vllm.config import VllmConfig
from mini_vllm.model.loader import load_model_and_tokenizer, get_model_max_length
from mini_vllm.sampling.sampling_params import SamplingParams
from mini_vllm.sampling.sampler import Sampler, SamplerOutput
from mini_vllm.engine.request import Request, RequestStatus
from mini_vllm.engine.output import RequestOutput, CompletionOutput, RequestMetrics
from mini_vllm.worker.model_runner import ModelRunner
from mini_vllm.core.block_manager import BlockSpaceManager
from mini_vllm.core.scheduler import Scheduler, SchedulerOutput

class LLMEngine:

    def __init__(self, config: VllmConfig, use_scheduler: bool = True):
        self.config = config
        self.use_scheduler = use_scheduler
        config.validate()

        torch.manual_seed(config.seed)

        self.device = config.get_device()
        print(f"Using device: {self.device}")

        self.model_runner = ModelRunner(config, self.device)
        self.model = self.model_runner.model
        self.tokenizer = self.model_runner.tokenizer
        self.vocab_size = self.model_runner.vocab_size
        self.sampler = self.model_runner.sampler

        self.max_model_len = get_model_max_length(config)
        self.eos_token_id = self.tokenizer.eos_token_id

        if use_scheduler:
            self.scheduler = Scheduler(
                scheduler_config=config.scheduler,
                cache_config=config.cache,
                block_manager=self.model_runner.block_manager,
            )
        else:
            self.scheduler = None

        self._request_counter = 0
        self._request_outputs: Dict[str, RequestOutput] = {}

    def add_request(
        self,
        request_id: Optional[str],
        prompt: str,
        sampling_params: SamplingParams,
    ) -> str:
        if request_id is None:
            request_id = f"req-{self._request_counter}"
            self._request_counter += 1

        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if len(prompt_token_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_token_ids)} tokens, "
                f"exceeding max_model_len of {self.max_model_len}"
            )

        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        if self.use_scheduler and self.scheduler is not None:
            self.scheduler.add_request(request)
        else:

            if not self.model_runner.allocate_request(request):
                raise MemoryError(
                    f"Cannot allocate KV cache for request {request_id}. "
                    f"Free blocks: {self.model_runner.get_num_free_blocks()}"
                )

        return request_id

    def step(self) -> List[RequestOutput]:
        if self.use_scheduler and self.scheduler is not None:
            return self._step_with_scheduler()
        else:
            return self._step_legacy()

    def _step_with_scheduler(self) -> List[RequestOutput]:

        scheduler_output = self.scheduler.schedule()

        if scheduler_output.is_empty:
            return []

        scheduled_requests = scheduler_output.scheduled_requests
        outputs = []

        for request in scheduled_requests:
            if not request.is_prefill():
                try:
                    self.model_runner.block_manager.append_slot(request.request_id)
                except MemoryError:

                    continue

        chunked_size = None
        if self.config.scheduler.chunked_prefill_enabled:
            chunked_size = self.config.scheduler.chunk_size

        model_input = self.model_runner.prepare_model_input(
            scheduled_requests,
            chunked_prefill_size=chunked_size,
        )

        model_output = self.model_runner.execute_model(model_input, scheduled_requests)

        sampler_output = self.model_runner.sample(model_output, scheduled_requests)

        for i, request in enumerate(scheduled_requests):

            tokens_processed = model_input.tokens_per_request[i]
            request.num_computed_tokens += tokens_processed

            if not request.is_prefill():
                token_id = sampler_output.token_ids[i]
                request.append_output_token(token_id)

            finish_reason = request.should_stop(self.eos_token_id)

            output = self._build_request_output(request, finish_reason)
            outputs.append(output)

            if finish_reason:
                request.mark_finished(finish_reason)
                self.scheduler.finish_request(request.request_id)
                self.model_runner.free_request(request)

        return outputs

    def _step_legacy(self) -> List[RequestOutput]:

        outputs = []

        running_requests = [
            req for req in self.model_runner.block_manager.block_tables.keys()
        ]

        return outputs

    def _build_request_output(
        self,
        request: Request,
        finish_reason: Optional[str],
    ) -> RequestOutput:

        generated_text = self.tokenizer.decode(
            request.output_token_ids,
            skip_special_tokens=request.sampling_params.skip_special_tokens,
        )

        if finish_reason == "stop" and not request.sampling_params.include_stop_str_in_output:

            if request.sampling_params.stop:
                for stop_str in request.sampling_params.stop:
                    if generated_text.endswith(stop_str):
                        generated_text = generated_text[:-len(stop_str)]
                        break

        completion = CompletionOutput(
            index=0,
            text=generated_text,
            token_ids=request.output_token_ids.copy(),
            finish_reason=finish_reason,
        )

        metrics = None
        if finish_reason:
            metrics = RequestMetrics(
                arrival_time=request.arrival_time,
                first_scheduled_time=request.first_scheduled_time,
                first_token_time=request.first_token_time,
                finish_time=request.finish_time,
            )

        return RequestOutput(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            outputs=[completion],
            finished=finish_reason is not None,
            metrics=metrics,
        )

    def run_to_completion(self) -> List[RequestOutput]:
        outputs = []
        while self.has_unfinished_requests():
            step_outputs = self.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return outputs

    def has_unfinished_requests(self) -> bool:
        if self.use_scheduler and self.scheduler is not None:
            return self.scheduler.has_unfinished_requests()
        return False

    def generate(
        self,
        prompt: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Union[RequestOutput, List[RequestOutput]]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(prompt, str):
            prompts = [prompt]
            single = True
        else:
            prompts = prompt
            single = False

        for p in prompts:
            self.add_request(None, p, sampling_params)

        outputs = self.run_to_completion()

        if single:
            return outputs[0]
        return outputs

    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Iterator[RequestOutput]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        request_id = self.add_request(None, prompt, sampling_params)

        while self.has_unfinished_requests():
            outputs = self.step()
            for output in outputs:
                if output.request_id == request_id:
                    yield output

    def abort_request(self, request_id: str) -> bool:
        if self.use_scheduler and self.scheduler is not None:
            request = self.scheduler.abort_request(request_id)
            if request is not None:
                self.model_runner.free_request(request)
                return True
        return False

    def get_num_pending_requests(self) -> int:
        if self.use_scheduler and self.scheduler is not None:
            return self.scheduler.get_num_unfinished()
        return 0

    def get_scheduler_stats(self) -> dict:
        if not self.use_scheduler or self.scheduler is None:
            return {}

        return {
            "waiting": self.scheduler.get_num_waiting(),
            "running": self.scheduler.get_num_running(),
            "preempted": self.scheduler.get_num_preempted(),
        }

    def get_cache_stats(self) -> dict:
        stats = {
            "total_blocks": self.model_runner.num_blocks,
            "free_blocks": self.model_runner.get_num_free_blocks(),
            "utilization": self.model_runner.get_cache_utilization(),
            "cache_memory_mb": self.model_runner.cache_engine.get_memory_usage_mb(),
        }

        if (self.use_scheduler and
            self.scheduler is not None and
            self.scheduler.block_manager.prefix_cache is not None):
            prefix_stats = self.scheduler.block_manager.get_prefix_cache_stats()
            if prefix_stats:
                stats["prefix_cache"] = prefix_stats

        return stats

    def __repr__(self) -> str:
        if self.use_scheduler and self.scheduler is not None:
            return (
                f"LLMEngine("
                f"model={self.config.model.model_name_or_path}, "
                f"device={self.device}, "
                f"scheduler={self.scheduler}, "
                f"cache_util={self.model_runner.get_cache_utilization():.1%})"
            )
        return (
            f"LLMEngine("
            f"model={self.config.model.model_name_or_path}, "
            f"device={self.device})"
        )
