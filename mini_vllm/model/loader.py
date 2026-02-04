from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from mini_vllm.config import ModelConfig, VllmConfig

def get_model_config(model_config: ModelConfig) -> AutoConfig:
    return AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )

def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer

def load_model(
    config: VllmConfig,
    device: Optional[torch.device] = None,
) -> PreTrainedModel:
    if device is None:
        device = config.get_device()

    model_config = config.model
    torch_dtype = model_config.get_torch_dtype()

    if device.type == "cuda":
        device_map = "auto"
    elif device.type == "mps":

        device_map = None
    else:
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )

    if device_map is None:
        model = model.to(device)

    model.eval()

    return model

def load_model_and_tokenizer(
    config: VllmConfig,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(config.model)
    model = load_model(config, device)
    return model, tokenizer

def get_model_max_length(config: VllmConfig) -> int:
    if config.model.max_model_len is not None:
        return config.model.max_model_len

    hf_config = get_model_config(config.model)

    for attr in ["max_position_embeddings", "n_positions", "max_seq_len", "seq_length"]:
        if hasattr(hf_config, attr):
            return getattr(hf_config, attr)

    return 2048

def get_num_layers(config: VllmConfig) -> int:
    hf_config = get_model_config(config.model)

    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(hf_config, attr):
            return getattr(hf_config, attr)

    raise ValueError("Could not determine number of layers from model config")

def get_num_kv_heads(config: VllmConfig) -> int:
    hf_config = get_model_config(config.model)

    if hasattr(hf_config, "num_key_value_heads"):
        return hf_config.num_key_value_heads

    if hasattr(hf_config, "num_attention_heads"):
        return hf_config.num_attention_heads

    if hasattr(hf_config, "n_head"):
        return hf_config.n_head

    raise ValueError("Could not determine number of KV heads from model config")

def get_head_dim(config: VllmConfig) -> int:
    hf_config = get_model_config(config.model)

    if hasattr(hf_config, "head_dim"):
        return hf_config.head_dim

    if hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
        return hf_config.hidden_size // hf_config.num_attention_heads

    raise ValueError("Could not determine head dimension from model config")

def estimate_kv_cache_size_per_token(config: VllmConfig) -> int:
    num_layers = get_num_layers(config)
    num_kv_heads = get_num_kv_heads(config)
    head_dim = get_head_dim(config)

    dtype = config.model.get_torch_dtype()
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        dtype_size = 2

    return 2 * num_layers * num_kv_heads * head_dim * dtype_size
