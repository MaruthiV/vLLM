from typing import List, Optional, Union

from mini_vllm.config import VllmConfig, ModelConfig
from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.engine.output import RequestOutput
from mini_vllm.sampling.sampling_params import SamplingParams

class LLM:

    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        device: str = "auto",
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        seed: int = 42,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):

        model_config = ModelConfig(
            model_name_or_path=model,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )

        self.config = VllmConfig(
            model=model_config,
            device=device,
            seed=seed,
        )
        self.config.cache.gpu_memory_utilization = gpu_memory_utilization

        self.engine = LLMEngine(self.config)

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(prompts, str):
            prompts = [prompts]

        for prompt in prompts:
            self.engine.add_request(None, prompt, sampling_params)

        return self.engine.run_to_completion()

    def chat(
        self,
        messages: List[dict],
        sampling_params: Optional[SamplingParams] = None,
    ) -> RequestOutput:

        if hasattr(self.engine.tokenizer, "apply_chat_template"):
            prompt = self.engine.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:

            prompt = self._format_messages_basic(messages)

        outputs = self.generate(prompt, sampling_params)
        return outputs[0]

    def _format_messages_basic(self, messages: List[dict]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        formatted.append("Assistant:")
        return "\n".join(formatted)

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def model(self):
        return self.engine.model
