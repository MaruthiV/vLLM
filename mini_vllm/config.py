from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class ModelConfig:

    model_name_or_path: str

    dtype: str = "float16"

    max_model_len: Optional[int] = None

    trust_remote_code: bool = False

    revision: Optional[str] = None

    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        return dtype_map.get(self.dtype, torch.float16)

@dataclass
class CacheConfig:

    block_size: int = 16

    num_gpu_blocks: Optional[int] = None

    num_cpu_blocks: int = 0

    gpu_memory_utilization: float = 0.9

    enable_prefix_caching: bool = True

    prefix_cache_ratio: float = 0.2

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        if not 0.0 <= self.prefix_cache_ratio <= 1.0:
            raise ValueError(
                f"prefix_cache_ratio must be in [0, 1], got {self.prefix_cache_ratio}"
            )

@dataclass
class SchedulerConfig:

    max_num_seqs: int = 256

    max_num_batched_tokens: int = 4096

    max_paddings: int = 256

    chunked_prefill_enabled: bool = True

    chunk_size: int = 512

    enable_preemption: bool = True

    delay_factor: float = 0.0

    def validate(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")
        if self.max_num_batched_tokens <= 0:
            raise ValueError(
                f"max_num_batched_tokens must be positive, got {self.max_num_batched_tokens}"
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

@dataclass
class SpeculativeConfig:

    enabled: bool = False

    draft_model: Optional[str] = None

    num_speculative_tokens: int = 5

    draft_model_dtype: str = "float16"

    def validate(self) -> None:
        if self.enabled and self.draft_model is None:
            raise ValueError("draft_model must be specified when speculative decoding is enabled")
        if self.num_speculative_tokens <= 0:
            raise ValueError(
                f"num_speculative_tokens must be positive, got {self.num_speculative_tokens}"
            )

@dataclass
class GuidedDecodingConfig:

    backend: str = "outlines"

    whitespace_pattern: Optional[str] = None

    def validate(self) -> None:
        valid_backends = {"outlines", "lm-format-enforcer"}
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got {self.backend}")

@dataclass
class ParallelConfig:

    tensor_parallel_size: int = 1

    pipeline_parallel_size: int = 1

    def validate(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )

@dataclass
class VllmConfig:

    model: ModelConfig

    cache: CacheConfig = field(default_factory=CacheConfig)

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)

    guided: GuidedDecodingConfig = field(default_factory=GuidedDecodingConfig)

    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    device: str = "auto"

    seed: int = 42

    def validate(self) -> None:
        self.cache.validate()
        self.scheduler.validate()
        self.speculative.validate()
        self.guided.validate()
        self.parallel.validate()

    def get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        dtype: str = "float16",
        **kwargs,
    ) -> "VllmConfig":
        model_config = ModelConfig(
            model_name_or_path=model_name,
            dtype=dtype,
        )
        return cls(model=model_config, **kwargs)
