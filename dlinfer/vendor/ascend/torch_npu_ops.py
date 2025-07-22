# Copyright (c) 2024, DeepLink. All rights reserved.
import math
import torch
import torch_npu
import torch.distributed as dist
import numpy as np

from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple
from .utils import SocVersion
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")

__all__ = [
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "prefill_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "get_cache_len",
    "weight_quant_matmul",
    "fused_moe",
    "fused_moe_with_alltoall",
    "linear",
    "dynamic_quant",
    "linear_ascend_w8a8_dynamic",
    "quant_per_tensor",
    "linear_ascend_w8a8",
    "fused_moe_ascend_w8a8",
]


@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    normed_hidden_states, _, added_hidden_states = torch.ops.npu.npu_add_rms_norm(
        hidden_states, residual, weight, epsilon
    )
    return normed_hidden_states, added_hidden_states


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    # rotary pos emb helpers:
    query = query.contiguous().unsqueeze(0)
    key = key.contiguous().unsqueeze(0)
    assert len(query.shape) == 4
    batch, seq_len, _, _ = query.shape
    cos = cos.reshape(batch, seq_len, 1, -1)
    sin = sin.reshape(batch, seq_len, 1, -1)

    def rotate_half_(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_(q, k, cos, sin):
        return (q * cos) + (rotate_half_(q) * sin), (k * cos) + (rotate_half_(k) * sin)

    # ascend ops currently only support dim 128
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        return apply_rotary_pos_emb_(query, key, cos, sin)
    return torch.ops.npu.npu_apply_rotary_pos_emb(query, key, cos, sin, "BSND")


import pickle


def dump_tensor(tensor, name):
    with open(
        f"/mnt/cwai/dev/yaofengchen/dlinfer_dev/demo/tensors/{name}.pkl", "wb"
    ) as f:
        pickle.dump(tensor.cpu(), f)


def load_tensor(name):
    with open(
        f"/mnt/cwai/dev/yaofengchen/dlinfer_dev/demo/tensors/{name}.pkl", "rb"
    ) as f:
        loaded_tensor = pickle.load(f)
    return loaded_tensor


@register_ops(vendor_ops_registry)
def prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    max_q_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    if SocVersion.is_Ascend910B():
        seq_qlen_list = (
            [max_q_seq_len * (i + 1) for i in range(query.shape[0])]
            if q_seq_len is None
            else q_seq_len.cumsum(0).tolist()
        )
        seq_kvlen_list = seq_qlen_list
        if (attn_mask is None or len(attn_mask) == 0) and q_seq_len is None:
            query = query.view(query.shape[0] * query.shape[1], num_q_heads, -1)
            key = key.view(key.shape[0] * key.shape[1], num_kv_heads, -1)
            value = value.view(value.shape[0] * value.shape[1], num_kv_heads, -1)
        # some vl models pass a fp16 mask from lmdeploy in vision part of prefill phase.
        attn_mask_ = (
            None
            if (attn_mask is None or len(attn_mask) == 0)
            else attn_mask[0].to(torch.bool)
        )
        # attn_output = attn_output.contiguous()
        # torch_npu._npu_flash_attention(
        #     query=query,
        #     key=key,
        #     value=value,
        #     mask=attn_mask_.to(query.dtype),
        #     seq_len=q_seq_len.to(torch.int32),
        #     scale_value=scale_value,
        #     num_heads=num_q_heads,
        #     num_kv_heads=num_q_heads,
        #     out=attn_output)
        attn_output[:] = torch.ops.npu.npu_fusion_attention(
            query,
            key,
            value,
            num_q_heads,
            "TND",
            scale=scale_value,
            atten_mask=attn_mask_,
            actual_seq_qlen=seq_qlen_list,
            actual_seq_kvlen=seq_kvlen_list,
        )[0]
    elif SocVersion.is_Ascend310P():
        # Used for Qwen2.5-VL model vision block
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        attn_output[:] = torch.ops.npu.npu_prompt_flash_attention(
            query,
            key,
            value,
            num_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            input_layout="BSND",
            scale_value=scale_value,
        )
    else:
        raise ValueError(
            f"dlinfer doesn't support {SocVersion.device_name()} device currently."
        )
    return attn_output


@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
    k_scales_zeros: Sequence[Optional[Tensor]],
    v_scales_zeros: Sequence[Optional[Tensor]],
    quant_bits: int,
) -> Tuple[Tensor, Tensor]:
    _, head, dim = key.shape
    block_num, block_size = key_cache.shape[:2]
    block_total = block_num * block_size

    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()
    kv_indices = kv_indices.view(-1, 1)

    if quant_bits == 8:

        def quant_int8(x, x_scale, x_offset):
            quantized = (
                ((x / x_scale) - x_offset).round().clamp(-128, 127).to(torch.int8)
            )
            return quantized

        key = quant_int8(key, k_scales_zeros[0], k_scales_zeros[1])
        value = quant_int8(value, v_scales_zeros[0], v_scales_zeros[1])

    is_mla = key.shape[-1] != value.shape[-1]
    if is_mla:
        key_cache_reshaped = key_cache.view(block_total, head, dim)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    else:
        key_cache_reshaped = key_cache.view(block_total, head, dim)
        value_cache_reshaped = value_cache.view(block_total, head, dim)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
        torch.ops.npu.npu_scatter_nd_update_(value_cache_reshaped, kv_indices, value)
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def fill_contiguous_kvcache(
    key_cache: Tensor, value_cache: Tensor, key_state: Tensor, value_state: Tensor
) -> Tuple[Tensor, Tensor]:
    key_cache = torch.cat([key_cache, key_state], dim=1)
    value_cache = torch.cat([value_cache, value_state], dim=1)
    return key_cache, value_cache


@register_ops(vendor_ops_registry)
def get_cache_len(cache: Tensor):
    return cache.shape[1]


@register_ops(vendor_ops_registry)
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Optional[Tensor],
    block_size: int,
    kv_seq_len: Tensor,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    if isinstance(block_table, torch.Tensor) and block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    is_mla = key_cache.shape[-1] != value_cache.shape[-1]
    query = query.contiguous()
    attn_output = attn_output.contiguous()
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(dim)

    if is_mla:
        v_head_size = value_cache.shape[-1]
        torch.ops.atb._npu_paged_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=softmax_scale,
            block_table=block_table,
            context_lens=kv_seq_len,
            mla_vheadsize=v_head_size,
            out=attn_output,
        )
    else:
        bs, _, dim = query.shape
        block_num = key_cache.size(0)
        query = query.view(bs, 1, num_q_heads * dim)
        key_cache = key_cache.view(block_num, block_size, -1)
        value_cache = value_cache.view(block_num, block_size, -1)
        torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
            query,
            key_cache,
            value_cache,
            attn_output.view_as(query),
            padding_mask=None,
            atten_mask=None,
            actual_seq_lengths=kv_seq_len.tolist(),
            antiquant_scale=kv_scales,
            antiquant_offset=kv_zeros,
            block_table=block_table,
            dequant_scale1=None,
            quant_scale1=None,
            dequant_scale2=None,
            quant_scale2=None,
            quant_offset2=None,
            num_heads=num_q_heads,
            scale_value=scale_value,
            input_layout="BSH",
            num_key_value_heads=num_kv_heads,
            block_size=block_size,
            inner_precise=1,
        )
    return attn_output


@register_ops(vendor_ops_registry)
def paged_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    cu_seq_lens_kv: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )

    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    is_mla = key_cache.shape[-1] != value_cache.shape[-1]
    query = query.contiguous()
    attn_output = attn_output.contiguous()
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(dim)

    if is_mla:
        v_head_size = value_cache.shape[-1]
        torch.ops.atb._npu_paged_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=softmax_scale,
            block_table=block_table,
            context_lens=kv_seq_len,
            mla_vheadsize=v_head_size,
            out=attn_output,
        )
    else:
        bs, _, dim = query.shape
        block_num = key_cache.size(0)
        query = query.view(bs, 1, num_q_heads * dim)
        key_cache = key_cache.view(block_num, block_size, -1)
        value_cache = value_cache.view(block_num, block_size, -1)
        torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
            query,
            key_cache,
            value_cache,
            attn_output,
            padding_mask=None,
            atten_mask=attn_mask[0],
            actual_seq_lengths=kv_seq_len.tolist(),
            antiquant_scale=kv_scales,
            antiquant_offset=kv_zeros,
            block_table=block_table,
            dequant_scale1=None,
            quant_scale1=None,
            dequant_scale2=None,
            quant_scale2=None,
            quant_offset2=None,
            num_heads=num_q_heads,
            scale_value=scale_value,
            input_layout="BSH",
            num_key_value_heads=num_kv_heads,
            block_size=block_size,
            inner_precise=1,
        )
    return attn_output


@register_ops(vendor_ops_registry)
def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    hidden_states = hidden_states.contiguous()
    return torch.ops.npu.npu_rms_norm(hidden_states, weight, epsilon)[0]


@register_ops(vendor_ops_registry)
def silu_and_mul(input_tensor: Tensor, dim: int) -> Tensor:
    if SocVersion.is_Ascend910B():
        if isinstance(input_tensor, tuple):
            quantized_x, dynamic_scale, weight_scale = input_tensor
            y, scale = torch.ops.npu.npu_dequant_swiglu_quant(
                x=quantized_x,
                weight_scale=weight_scale,
                activation_scale=dynamic_scale,
                activate_left=True,
                quant_mode=1,
            )
            return (y, scale)
        return torch.ops.npu.npu_swiglu(input_tensor, dim)
    elif SocVersion.is_Ascend310P():
        gate_cache, up_cache = input_tensor.chunk(2, dim)
        return torch.ops.npu.npu_silu(gate_cache) * up_cache


@register_ops(vendor_ops_registry)
def moe_gating_topk_softmax(router_logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    routing_weights = router_logits.new_empty((*router_logits.shape[:-1], topk))
    selected_experts = router_logits.new_empty(
        (*router_logits.shape[:-1], topk), dtype=torch.int32
    )
    selected_idx = torch.empty_like(selected_experts)
    routing_weights, selected_idx = torch.ops.npu_ext.npu_moe_gating_topk_softmax(
        router_logits, None, topk, routing_weights, selected_experts, selected_idx
    )
    return routing_weights, selected_idx.to(torch.int64)


# TODO only for internlm in transformers lib.
# see issue #9 for details
@register_ops(vendor_ops_registry)
def fused_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    mask: Sequence[Optional[Tensor]],
) -> Tensor:
    batch_size = query_states.shape[0]
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    q_seq_len, num_q_heads, _ = query_states.shape
    kv_seq_len, num_kv_heads, _ = value_states.shape
    attn_output = torch.empty_like(query_states)

    for i in range(batch_size):
        if q_seq_len == kv_seq_len:
            # mask must be a square
            if not mask[i : i + 1][0].shape[-1] == mask[i : i + 1][0].shape[-2]:
                min_shape = min(
                    mask[i : i + 1][0].shape[-1], mask[i : i + 1][0].shape[-2]
                )
                square_mask = mask[i : i + 1][0][..., :min_shape, :min_shape]
                square_mask = square_mask.contiguous()
            else:
                square_mask = mask[i : i + 1][0]

            prefill_attention(
                query_states,
                key_states,
                value_states,
                torch.tensor(
                    [kv_seq_len - q_seq_len],
                    dtype=torch.int64,
                    device=query_states.device,
                ),
                torch.tensor(
                    [kv_seq_len], dtype=torch.int64, device=query_states.device
                ),
                q_seq_len,
                num_q_heads,
                num_kv_heads,
                [
                    square_mask,
                ],
                None,
                None,
                attn_output,
            )
        else:
            paged_decode_attention(
                query_states,
                key_states,
                value_states,
                None,
                0,
                torch.tensor(
                    [kv_seq_len], dtype=torch.int64, device=query_states.device
                ),
                kv_seq_len,
                num_q_heads,
                num_kv_heads,
                None,
                None,
                attn_output,
            )
    return attn_output


# Quantification of W4A16 is currently supported and tested.
@register_ops(vendor_ops_registry)
def weight_quant_matmul(
    x: Tensor,
    qweight: Tensor,
    scale: Tensor,
    offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    all_reduce: Optional[bool] = False,
    group_size: Optional[int] = 0,
) -> Tensor:
    offset = None if (offset is None or offset.numel() == 0) else offset
    return torch.ops.npu.npu_weight_quant_batchmatmul(
        x,
        qweight,
        scale,
        antiquant_offset=offset,
        antiquant_group_size=group_size,
        bias=bias,
    )


@register_ops(vendor_ops_registry)
def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
) -> Tensor:
    import pdb

    pdb.set_trace()
    seq_length = hidden_states.size(0)
    num_experts = gate_up_weights.size(0)
    active_num = hidden_states.size(0)
    topk_ids = topk_ids.to(torch.int32)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    # moe init routing
    row_idx = (
        torch.arange(seq_length * topk, dtype=torch.int32, device=hidden_states.device)
        .view((topk, seq_length))
        .transpose(0, 1)
        .contiguous()
    )
    expanded_hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(hidden_states, row_idx, topk_ids, active_num)
    )

    # up sample
    counts = torch.bincount(expanded_expert_idx, minlength=num_experts)
    cumulative_counts = torch.cumsum(counts, dim=0)
    group_list = cumulative_counts.tolist()
    up_proj = torch.ops.npu.npu_grouped_matmul(
        [expanded_hidden_states],
        [weight for weight in gate_up_weights],
        bias=None,
        group_list=group_list,
        split_item=2,
    )[0]

    # activation
    gate_cache = silu_and_mul(up_proj, -1)

    # down sample
    down_proj = torch.ops.npu.npu_grouped_matmul(
        [gate_cache],
        [weight for weight in down_weights],
        bias=None,
        group_list=group_list,
        split_item=2,
    )[0]

    # moe finalize routing
    skip = torch.zeros_like(hidden_states)
    bias = torch.zeros_like(down_proj)
    export_for_source_row = torch.zeros_like(topk_ids)
    moe_output = torch.ops.npu.npu_moe_finalize_routing(
        down_proj,
        skip1=skip,
        skip2=skip,
        bias=bias,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=export_for_source_row,
    )

    return moe_output


@register_ops(vendor_ops_registry)
def fused_moe_with_alltoall(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_experts: int,
    ep_size: int,
    renormalize: bool,
):
    from lmdeploy.utils import get_logger

    logger = get_logger("lmdploy")
    logger.info(
        f"hidden_states.shape: {hidden_states.shape}, w1.shape: {w1.shape}, w2.shape: {w2.shape}"
    )
    logger.info(
        f"topk_weights.shape: {topk_weights.shape}, topk_ids.shape: {topk_ids.shape}"
    )
    logger.info(
        f"topk: {topk}, num_expers: {num_experts}, ep_size: {ep_size}, renormalize: {renormalize}"
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # moe init routing
    seq_length, _ = hidden_states.shape
    local_num_experts = num_experts // ep_size
    row_idx = (
        torch.arange(seq_length * topk, dtype=torch.int32, device=hidden_states.device)
        .view((topk, seq_length))
        .transpose(0, 1)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids.to(torch.int32),
            active_num=seq_length,
        )
    )
    logger.info(
        f"hidden_states.shape: {hidden_states.shape}, expanded_row_idx: {expanded_row_idx}, expanded_expert_idx: {expanded_expert_idx}"
    )

    # dispatch
    global_expert_tokens = torch.bincount(expanded_expert_idx, minlength=num_experts)
    scatter_sizes = global_expert_tokens.view(ep_size, -1).sum(-1)
    logger.info(
        f"global_expert_tokens: {global_expert_tokens}, scatter_sizes: {scatter_sizes}"
    )

    gather_sizes = torch.empty_like(scatter_sizes)
    dist.all_to_all_single(gather_sizes, scatter_sizes)
    scatter_size_list = scatter_sizes.cpu().tolist()
    gather_size_list = gather_sizes.cpu().tolist()

    expanded_expert_idx = expanded_expert_idx % local_num_experts
    original_hidden_states = hidden_states
    hidden_states = original_hidden_states.new_empty(
        (np.sum(np.array(gather_size_list)),) + hidden_states.shape[1:]
    )
    dist.all_to_all_single(
        hidden_states, original_hidden_states, gather_size_list, scatter_size_list
    )
    local_expert_idx = expanded_expert_idx.new_empty(
        (np.sum(np.array(gather_size_list)),) + expanded_expert_idx.shape[1:]
    )
    dist.all_to_all_single(
        local_expert_idx, expanded_expert_idx, gather_size_list, scatter_size_list
    )

    # up sample
    sorted_local_expert_idx, sorted_idx = torch.sort(local_expert_idx)

    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        sorted_local_expert_idx, local_num_experts
    ).to(torch.int64)

    hidden_states = hidden_states[sorted_idx]

    gate_up_out_list = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    # TODO: Remove this in the future.
    # activation
    hidden_states = torch.cat(gate_up_out_list, dim=0)
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

    # down sample
    down_out_list = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    # combine
    hidden_states = torch.cat(down_out_list, dim=0)
    resorted_idx = torch.argsort(sorted_idx)
    hidden_states = hidden_states[resorted_idx]
    dist.all_to_all_single(
        original_hidden_states, hidden_states, scatter_size_list, gather_size_list
    )
    hidden_states = original_hidden_states

    # moe finalize routing
    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


@register_ops(vendor_ops_registry)
def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
    group: Optional[str],
) -> Tensor:
    if all_reduce:
        assert group is None or group == "", "In eager mode, only use default_pg"
        group = torch.distributed.distributed_c10d._world.default_pg
        hcomm_info = group._get_backend(x.device).get_hccl_comm_name(x.device.index)
        out = torch.ops.npu.npu_mm_all_reduce_base(
            x.contiguous(),
            weight.transpose(0, 1),
            hcomm_info,
            reduce_op="sum",
            bias=bias,
        )
    else:
        # on 310p, the weight is transposed to nz format in llm part on graph mode,
        # but in vl part, eager mode is used.
        # we need to reshape it back to nd.
        if (
            len(weight.shape) == 4
            and weight.shape[0] == 1
            and weight.shape[1] * weight.shape[3] == x.shape[-1]
        ):
            weight = weight.permute(0, 2, 1, 3)
            weight = weight.reshape(weight.shape[1], -1)
        out = torch.nn.functional.linear(x, weight, bias)
    return out


@register_ops(vendor_ops_registry)
def transdata(
    hidden_states: Tensor,
    transdata_type: int,
):
    raise NotImplementedError("transdata in eager mode is not implemented yet!")


@register_ops(vendor_ops_registry)
def dynamic_quant(x: Tensor):
    x = x.unsqueeze(0) if len(x.shape) == 1 else x
    quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
    dynamic_scale = (
        dynamic_scale.squeeze() if len(dynamic_scale.size()) != 1 else dynamic_scale
    )
    dynamic_scale = (
        dynamic_scale.unsqueeze(0) if len(dynamic_scale.shape) == 0 else dynamic_scale
    )
    return quantized_x, dynamic_scale


@register_ops(vendor_ops_registry)
def linear_ascend_w8a8_dynamic(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    offset: Tensor,
    bias: Tensor,
    out_dtype: torch.dtype,
):
    weight = weight.transpose(0, 1)
    if isinstance(x, tuple):
        quantized_x, dynamic_scale = x
        x1_shape = quantized_x.shape
        x2_shape = weight.shape
        output_shape = quantized_x.shape
        if len(x2_shape) == 1:
            assert x1_shape[-1] == x2_shape[-1]
        elif len(x2_shape) == 2:
            output_shape = x1_shape[:-1] + x2_shape[-1:]
        else:
            raise NotImplementedError("not implememted output_shape")
        quantized_x = (
            quantized_x.squeeze().unsqueeze(0)
            if len(quantized_x.squeeze().shape) == 1
            else quantized_x.squeeze()
        )
        dynamic_scale = (
            dynamic_scale.squeeze() if len(dynamic_scale.size()) != 1 else dynamic_scale
        )
        dynamic_scale = (
            dynamic_scale.unsqueeze(0)
            if len(dynamic_scale.shape) == 0
            else dynamic_scale
        )
    else:
        x1_shape = x.shape
        x2_shape = weight.shape
        output_shape = x.shape
        if len(x2_shape) == 1:
            assert x1_shape[-1] == x2_shape[-1]
        elif len(x2_shape) == 2:
            output_shape = x1_shape[:-1] + x2_shape[-1:]
        else:
            raise NotImplementedError("not implememted output_shape")
        quantized_x, dynamic_scale = dynamic_quant(x.squeeze())
    pertoken_scale = None if out_dtype is torch.int32 else dynamic_scale
    weight = weight.contiguous()
    weight.data = torch_npu.npu_format_cast(weight.data, 29)
    output = torch.ops.npu.npu_quant_matmul(
        quantized_x,
        weight,
        weight_scale.squeeze(1),
        pertoken_scale=pertoken_scale,
        bias=bias,
        output_dtype=out_dtype,
    ).view(output_shape)

    if out_dtype is torch.int32:
        return (output, dynamic_scale, weight_scale.squeeze().to(torch.float32))
    return output


@register_ops(vendor_ops_registry)
def quant_per_tensor(
    in_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    input_offset: torch.Tensor,
    function=False,
):
    return torch.ops.npu.npu_quantize(
        in_tensor, input_scale, input_offset, torch.qint8, -1, function
    )


@register_ops(vendor_ops_registry)
def linear_ascend_w8a8(
    x: Tensor,
    weight: Tensor,
    input_scale: Tensor,
    input_offset: Tensor,
    quant_bias: Tensor,
    deq_scale: Tensor,
    bias=None,
):
    original_dtype = x.dtype
    if original_dtype != torch.int8:
        x = quant_per_tensor(x, input_scale, input_offset)
    weight = weight.contiguous()
    weight.data = torch_npu.npu_format_cast(weight.data, 29)
    output = torch.ops.npu.npu_quant_matmul(
        x,
        weight,
        deq_scale,
        bias=quant_bias,
        output_dtype=original_dtype,
    )
    return output


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))


def apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_list: torch.Tensor,
    dynamic_scale: torch.Tensor = None,
    group_list_type: int = 1,
) -> torch.Tensor:
    if dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
        # Dispose the original unquantized hidden states
        # to save npu memory because they're no longer used.
        dispose_tensor(unquantized_hidden_states)
    else:
        pertoken_scale = dynamic_scale

    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        scale=[w1_scale],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=w2_scale.dtype,
    )[0]

    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states, swiglu_out_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=w2_scale.dtype,
    )[0]

    return hidden_states


@register_ops(vendor_ops_registry)
def fused_moe_ascend_w8a8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_experts: int = 1,
    ep_size: int = 1,
    renormalize: bool = False,
):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    seq_length, _ = hidden_states.shape

    if ep_size != 1:
        # moe init routing
        local_num_experts = num_experts // ep_size
        row_idx = (
            torch.arange(
                seq_length * topk, dtype=torch.int32, device=hidden_states.device
            )
            .view((topk, seq_length))
            .transpose(0, 1)
            .contiguous()
        )
        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids.to(torch.int32),
                active_num=seq_length,
            )
        )

        # dispatch
        global_expert_tokens = torch.bincount(
            expanded_expert_idx, minlength=num_experts
        )
        scatter_sizes = global_expert_tokens.view(ep_size, -1).sum(-1)

        gather_sizes = torch.empty_like(scatter_sizes)
        dist.all_to_all_single(gather_sizes, scatter_sizes)
        scatter_size_list = scatter_sizes.cpu().tolist()
        gather_size_list = gather_sizes.cpu().tolist()

        expanded_expert_idx = expanded_expert_idx % local_num_experts
        original_hidden_states = hidden_states
        hidden_states = original_hidden_states.new_empty(
            (np.sum(np.array(gather_size_list)),) + hidden_states.shape[1:]
        )
        dist.all_to_all_single(
            hidden_states, original_hidden_states, gather_size_list, scatter_size_list
        )
        local_expert_idx = expanded_expert_idx.new_empty(
            (np.sum(np.array(gather_size_list)),) + expanded_expert_idx.shape[1:]
        )
        dist.all_to_all_single(
            local_expert_idx, expanded_expert_idx, gather_size_list, scatter_size_list
        )

        sorted_local_expert_idx, sorted_idx = torch.sort(local_expert_idx)

        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            sorted_local_expert_idx, local_num_experts
        ).to(torch.int64)

        hidden_states = hidden_states[sorted_idx]
        group_list_type = 0
    else:
        row_idx_len = seq_length * topk
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
            .view(topk, -1)
            .permute(1, 0)
            .contiguous()
        )
        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids.to(torch.int32),
                active_num=seq_length,
            )
        )

        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 0

    # mlp
    hidden_states = apply_mlp(
        hidden_states,
        w1,
        w1_scale.squeeze(),  # 17
        w2,
        w2_scale.squeeze(),
        expert_tokens,  # 16
        group_list_type=group_list_type,
    )

    if ep_size != 1:
        # combine
        resorted_idx = torch.argsort(sorted_idx)
        hidden_states = hidden_states[resorted_idx]
        dist.all_to_all_single(
            original_hidden_states, hidden_states, scatter_size_list, gather_size_list
        )
        hidden_states = original_hidden_states

        # moe finalize routing
        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
    else:
        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states
