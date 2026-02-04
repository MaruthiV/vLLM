from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import math

from mini_vllm.worker.cache_engine import CacheEngine

class PagedAttention:

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or (1.0 / math.sqrt(head_dim))

        self.num_queries_per_kv = num_heads // num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_engine: CacheEngine,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        is_prefill: bool = False,
    ) -> torch.Tensor:

        cache_engine.write_kv_batch(layer_idx, slot_mapping, key, value)

        if is_prefill:

            return self._prefill_attention(
                query, key, value,
                cache_engine, layer_idx,
                block_tables, context_lens,
            )
        else:

            return self._decode_attention(
                query,
                cache_engine, layer_idx,
                block_tables, context_lens,
                max_context_len,
            )

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_engine: CacheEngine,
        layer_idx: int,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if self.num_queries_per_kv > 1:
            key = self._repeat_kv(key, self.num_queries_per_kv)
            value = self._repeat_kv(value, self.num_queries_per_kv)

        q = query.transpose(0, 1).unsqueeze(0)
        k = key.transpose(0, 1).unsqueeze(0)
        v = value.transpose(0, 1).unsqueeze(0)

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scale,
        )

        output = output.squeeze(0).transpose(0, 1)

        return output

    def _decode_attention(
        self,
        query: torch.Tensor,
        cache_engine: CacheEngine,
        layer_idx: int,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        device = query.device
        dtype = query.dtype

        outputs = []

        for seq_idx in range(batch_size):
            seq_query = query[seq_idx:seq_idx+1]
            seq_context_len = context_lens[seq_idx].item()
            seq_block_table = block_tables[seq_idx]

            num_blocks_needed = (seq_context_len + cache_engine.block_size - 1) // cache_engine.block_size
            block_ids = seq_block_table[:num_blocks_needed].tolist()

            cached_k, cached_v = cache_engine.read_kv_blocks(
                layer_idx, block_ids, seq_context_len
            )

            if self.num_queries_per_kv > 1:
                cached_k = self._repeat_kv(cached_k, self.num_queries_per_kv)
                cached_v = self._repeat_kv(cached_v, self.num_queries_per_kv)

            q = seq_query.unsqueeze(2)

            k = cached_k.transpose(0, 1).unsqueeze(0)
            v = cached_v.transpose(0, 1).unsqueeze(0)

            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )

            output = output.squeeze(2)
            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        seq_len, num_kv_heads, head_dim = x.shape
        x = x.unsqueeze(2)
        x = x.expand(seq_len, num_kv_heads, n_rep, head_dim)
        x = x.reshape(seq_len, num_kv_heads * n_rep, head_dim)
        return x

class PagedAttentionMetadata:

    def __init__(
        self,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        num_prefill_tokens: int,
        num_decode_tokens: int,
    ):
        self.slot_mapping = slot_mapping
        self.block_tables = block_tables
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.num_prefill_tokens = num_prefill_tokens
        self.num_decode_tokens = num_decode_tokens

    @property
    def is_prefill_only(self) -> bool:
        return self.num_decode_tokens == 0

    @property
    def is_decode_only(self) -> bool:
        return self.num_prefill_tokens == 0

    @property
    def has_mixed_batch(self) -> bool:
        return self.num_prefill_tokens > 0 and self.num_decode_tokens > 0

def create_attention_metadata(
    requests: List,
    block_tables: dict,
    block_size: int,
    device: torch.device,
) -> PagedAttentionMetadata:
    slot_mapping_list = []
    block_table_list = []
    context_lens_list = []
    max_context_len = 0
    num_prefill_tokens = 0
    num_decode_tokens = 0
    max_blocks = 0

    for request in requests:

        seq_block_table = block_tables.get(request.request_id, [])
        max_blocks = max(max_blocks, len(seq_block_table))

        context_len = request.num_total_tokens
        context_lens_list.append(context_len)
        max_context_len = max(max_context_len, context_len)

        if request.is_prefill():

            num_new_tokens = request.num_prompt_tokens - request.num_computed_tokens
            start_pos = request.num_computed_tokens
            num_prefill_tokens += num_new_tokens
        else:

            num_new_tokens = 1
            start_pos = context_len - 1
            num_decode_tokens += 1

        for i in range(num_new_tokens):
            pos = start_pos + i
            block_idx = pos // block_size
            slot_in_block = pos % block_size
            if block_idx < len(seq_block_table):
                physical_block = seq_block_table[block_idx]
                flat_slot = physical_block * block_size + slot_in_block
                slot_mapping_list.append(flat_slot)

        block_table_list.append(seq_block_table)

    padded_block_tables = []
    for bt in block_table_list:
        padded = bt + [0] * (max_blocks - len(bt))
        padded_block_tables.append(padded)

    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=device)
    block_tables = torch.tensor(padded_block_tables, dtype=torch.long, device=device)
    context_lens = torch.tensor(context_lens_list, dtype=torch.long, device=device)

    return PagedAttentionMetadata(
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
        max_context_len=max_context_len,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
    )
