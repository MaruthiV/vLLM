import torch
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not available. Using PyTorch fallback.")

if HAS_TRITON:
    @triton.jit
    def _paged_attention_kernel(

        output_ptr,

        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,

        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_num_blocks_per_seq: tl.constexpr,

        stride_qb,
        stride_qh,
        stride_qd,

        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,

        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,

        stride_btb,
        stride_bts,

        stride_ob,
        stride_oh,
        stride_od,

        scale,

        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):

        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        kv_head_idx = head_idx % num_kv_heads

        context_len = tl.load(context_lens_ptr + batch_idx)

        q_offset = batch_idx * stride_qb + head_idx * stride_qh
        q = tl.load(
            query_ptr + q_offset + tl.arange(0, head_dim) * stride_qd,
            mask=tl.arange(0, head_dim) < head_dim,
        )

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([head_dim], dtype=tl.float32)

        num_blocks = (context_len + block_size - 1) // block_size

        for block_idx in range(num_blocks):

            block_table_offset = batch_idx * stride_btb + block_idx * stride_bts
            physical_block_id = tl.load(block_tables_ptr + block_table_offset)

            start_pos = block_idx * block_size
            positions = start_pos + tl.arange(0, block_size)
            valid_mask = positions < context_len

            k_block_offset = (
                physical_block_id * stride_kb +
                kv_head_idx * stride_kh
            )

            qk = tl.zeros([block_size], dtype=tl.float32)

            for d in range(head_dim):
                k_d = tl.load(
                    key_cache_ptr + k_block_offset +
                    tl.arange(0, block_size) * stride_ks + d * stride_kd,
                    mask=valid_mask,
                    other=0.0,
                )
                qk += q[d] * k_d

            qk = qk * scale

            qk = tl.where(valid_mask, qk, float("-inf"))

            m_ij = tl.max(qk, axis=0)
            m_new = tl.maximum(m_i, m_ij)

            exp_qk = tl.exp(qk - m_new)
            exp_sum = tl.sum(exp_qk, axis=0)

            alpha = tl.exp(m_i - m_new)
            l_new = alpha * l_i + exp_sum

            v_block_offset = (
                physical_block_id * stride_vb +
                kv_head_idx * stride_vh
            )

            for d in range(head_dim):
                v_d = tl.load(
                    value_cache_ptr + v_block_offset +
                    tl.arange(0, block_size) * stride_vs + d * stride_vd,
                    mask=valid_mask,
                    other=0.0,
                )

                acc[d] = alpha * acc[d] + tl.sum(exp_qk * v_d, axis=0)

            m_i = m_new
            l_i = l_new

        acc = acc / l_i

        o_offset = batch_idx * stride_ob + head_idx * stride_oh
        tl.store(
            output_ptr + o_offset + tl.arange(0, head_dim) * stride_od,
            acc.to(output_ptr.dtype.element_ty),
            mask=tl.arange(0, head_dim) < head_dim,
        )

    @triton.jit
    def _paged_attention_v2_kernel(

        output_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        tmp_output_ptr,

        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,

        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_num_blocks_per_seq: tl.constexpr,
        num_partitions: tl.constexpr,

        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_btb, stride_bts,
        stride_ob, stride_oh, stride_od,

        scale,

        PARTITION_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        partition_idx = tl.program_id(2)

        kv_head_idx = head_idx % num_kv_heads
        context_len = tl.load(context_lens_ptr + batch_idx)

        start_block = partition_idx * PARTITION_SIZE
        end_block = tl.minimum(start_block + PARTITION_SIZE,
                               (context_len + block_size - 1) // block_size)

        if start_block >= end_block:
            return

        q_offset = batch_idx * stride_qb + head_idx * stride_qh
        q = tl.load(
            query_ptr + q_offset + tl.arange(0, head_dim) * stride_qd,
            mask=tl.arange(0, head_dim) < head_dim,
        )

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([head_dim], dtype=tl.float32)

        for block_idx in range(start_block, end_block):
            block_table_offset = batch_idx * stride_btb + block_idx * stride_bts
            physical_block_id = tl.load(block_tables_ptr + block_table_offset)

            start_pos = block_idx * block_size
            positions = start_pos + tl.arange(0, block_size)
            valid_mask = positions < context_len

            k_block_offset = physical_block_id * stride_kb + kv_head_idx * stride_kh
            qk = tl.zeros([block_size], dtype=tl.float32)

            for d in range(head_dim):
                k_d = tl.load(
                    key_cache_ptr + k_block_offset +
                    tl.arange(0, block_size) * stride_ks + d * stride_kd,
                    mask=valid_mask, other=0.0,
                )
                qk += q[d] * k_d

            qk = qk * scale
            qk = tl.where(valid_mask, qk, float("-inf"))

            m_ij = tl.max(qk, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            exp_qk = tl.exp(qk - m_new)
            exp_sum = tl.sum(exp_qk, axis=0)
            alpha = tl.exp(m_i - m_new)
            l_new = alpha * l_i + exp_sum

            v_block_offset = physical_block_id * stride_vb + kv_head_idx * stride_vh
            for d in range(head_dim):
                v_d = tl.load(
                    value_cache_ptr + v_block_offset +
                    tl.arange(0, block_size) * stride_vs + d * stride_vd,
                    mask=valid_mask, other=0.0,
                )
                acc[d] = alpha * acc[d] + tl.sum(exp_qk * v_d, axis=0)

            m_i = m_new
            l_i = l_new

        partition_offset = (
            batch_idx * num_heads * num_partitions +
            head_idx * num_partitions +
            partition_idx
        )

        tl.store(exp_sums_ptr + partition_offset, l_i)
        tl.store(max_logits_ptr + partition_offset, m_i)

        tmp_offset = partition_offset * head_dim
        tl.store(
            tmp_output_ptr + tmp_offset + tl.arange(0, head_dim),
            acc,
            mask=tl.arange(0, head_dim) < head_dim,
        )

def paged_attention_forward(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: Optional[float] = None,
    max_context_len: Optional[int] = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        return paged_attention_forward_pytorch(
            query, key_cache, value_cache, block_tables, context_lens, scale
        )

    batch_size, num_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, _ = key_cache.shape
    max_num_blocks_per_seq = block_tables.shape[1]

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    output = torch.empty_like(query)

    grid = (batch_size, num_heads)

    _paged_attention_kernel[grid](

        output,

        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,

        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_num_blocks_per_seq,

        query.stride(0), query.stride(1), query.stride(2),

        key_cache.stride(0), key_cache.stride(1),
        key_cache.stride(2), key_cache.stride(3),

        value_cache.stride(0), value_cache.stride(1),
        value_cache.stride(2), value_cache.stride(3),

        block_tables.stride(0), block_tables.stride(1),

        output.stride(0), output.stride(1), output.stride(2),

        scale,

        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=block_size,
    )

    return output

def paged_attention_forward_pytorch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, _ = key_cache.shape

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    output = torch.zeros_like(query)

    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        num_ctx_blocks = (ctx_len + block_size - 1) // block_size

        keys = []
        values = []

        for block_idx in range(num_ctx_blocks):
            physical_block = block_tables[b, block_idx].item()

            start_pos = block_idx * block_size
            end_pos = min(start_pos + block_size, ctx_len)
            num_tokens = end_pos - start_pos

            k = key_cache[physical_block, :, :num_tokens, :]
            v = value_cache[physical_block, :, :num_tokens, :]

            keys.append(k)
            values.append(v)

        if keys:
            k_full = torch.cat(keys, dim=1)
            v_full = torch.cat(values, dim=1)

            num_head_groups = num_heads // num_kv_heads
            if num_head_groups > 1:
                k_full = k_full.repeat_interleave(num_head_groups, dim=0)
                v_full = v_full.repeat_interleave(num_head_groups, dim=0)

            q = query[b]

            attn_weights = torch.einsum('hd,hsd->hs', q, k_full) * scale

            attn_probs = torch.softmax(attn_weights, dim=-1)

            output[b] = torch.einsum('hs,hsd->hd', attn_probs, v_full)

    return output

def is_triton_available() -> bool:
    return HAS_TRITON
