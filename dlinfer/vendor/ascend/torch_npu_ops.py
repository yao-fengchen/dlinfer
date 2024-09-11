# Copyright (c) 2024, DeepLink. All rights reserved.
import math
import torch
import torch.nn.functional as F
import torch_npu

import json
from typing import List
from dataclasses import dataclass, asdict

torch.classes.load_library(
    "/data2/yaofengchen/workspaces/dlinfer_dev/atb_models/output/atb_speed/lib/libatb_speed_torch.so"
)
atb_op = torch.classes.OperationTorch.OperationTorch("ATB")
from dlinfer.vendor import vendor_ops_registry
from dlinfer.utils.registry import register_ops
from dlinfer.utils.type_annotation import Tensor, Optional, Sequence, Tuple

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


@dataclass
class RotaryParams:
    rotaryCoeff: int = 2
    name: str = "RopeOperation"


@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    if len(cos.shape) < 4:
        cos = cos.unsqueeze(2)
    if len(sin.shape) < 4:
        sin = sin.unsqueeze(2)
    query = query.contiguous()
    key = key.contiguous()
    torch.cuda.synchronize()
    bsz, seq_len, num_q_heads, head_dim = query.shape
    _, _, num_kv_heads, _ = key.shape
    query = query.view(bsz * seq_len, num_q_heads * head_dim)
    key = key.view(bsz * seq_len, num_kv_heads * head_dim)
    cos = cos.view(bsz * seq_len, head_dim)
    sin = sin.view(bsz * seq_len, head_dim)
    seq_len = torch.tensor([seq_len], dtype=torch.int32).cuda()
    params = RotaryParams(rotaryCoeff=2)
    atb_op.set_op_name("RopeOperation")
    atb_op.set_param(json.dumps(params.__dict__))
    atb_op.execute_out([query, key, cos, sin, seq_len], [query, key])
    ropeQ_atb = query.view(bsz, seq_len, num_q_heads, head_dim)
    ropeK_atb = key.view(bsz, seq_len, num_kv_heads, head_dim)
    torch.cuda.synchronize()
    return ropeQ_atb, ropeK_atb
    # return torch.ops.npu.npu_apply_rotary_pos_emb(query, key, cos, sin, "BSND")


@dataclass
class PagedAttentionPrefillParams:
    headNum: int
    kvHeadNum: int
    qkScale: float = 1.0
    qScale: float = 1.0
    calcType: int = 3
    kernelType: int = 1
    clampType: int = 0
    isTriuMask: int = 1
    maskType: int = 0
    name: str = "SelfAttentionOperation"


def prefill_attention_atb(
    query,
    key,
    value,
    q_seq_len,
    num_q_heads,
    num_kv_heads,
    attn_mask,
    sm_scale,
    attn_output,
):
    seq_len_list = [] if q_seq_len is None else q_seq_len.tolist()
    atb_op.set_op_name("SelfAttentionOperation")
    query_dim, value_dim = query.shape[-1], value.shape[-1]
    value = F.pad(value, (0, query_dim - value_dim))
    atb_output = torch.empty_like(query)
    if attn_mask is not None:
        params = PagedAttentionPrefillParams(
            headNum=num_q_heads,
            kvHeadNum=num_kv_heads,
            qkScale=sm_scale if sm_scale else 1.0 / math.sqrt(query.shape[-1]),
            qScale=1.0,
            calcType=3,  # paged encode
            kernelType=1,  # 1为高精度，0为默认
            isTriuMask=1,  # triu mask
            maskType=1,  # norm mask
        )
        if attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.to(query.dtype)
        if q_seq_len.dtype != torch.int32:
            q_seq_len = q_seq_len.to(torch.int32)
        atb_op.set_param(json.dumps(params.__dict__))
        query = query.view(-1, num_q_heads * query.shape[-1])
        key = key.view(-1, num_kv_heads * key.shape[-1])
        value = value.view(-1, num_kv_heads * value.shape[-1])
        # mask要求f16/bf16, seq_len要求int32
        atb_op.execute_out_with_param(
            [query, key, value, attn_mask, q_seq_len],
            [atb_output],
            json.dumps(
                {
                    "seqLen": seq_len_list,
                    "hasMask": True,
                }
            ),
        )
        torch.cuda.synchronize()
    else:
        params = PagedAttentionPrefillParams(
            headNum=num_q_heads,
            kvHeadNum=num_kv_heads,
            qkScale=sm_scale if sm_scale else 1.0 / math.sqrt(query.shape[-1]),
            qScale=1.0,
            calcType=3,  # paged encode
            maskType=0,  # norm mask
        )
        atb_op.set_param(json.dumps(params.__dict__))
        atb_op.execute_out_with_param(
            [query, key, value],
            [atb_output],
            json.dumps(
                {
                    "seqLen": seq_len_list,
                    "hasMask": False,
                }
            ),
        )
    attn_output[:] = atb_output[..., :value_dim]
    return attn_output


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
    # cann prompt_fa don't support batch query with different seq_len
    seq_len_list = None if q_seq_len is None else q_seq_len.tolist()

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    attn_output = prefill_attention_atb(
        query,
        key,
        value,
        q_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_mask,
        softmax_scale,
        attn_output,
    )
    # attn_qk_scale = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    # atb_self_attention(
    #     query,
    #     key,
    #     value,
    #     q_start_loc,
    #     q_seq_len.to(query.device).to(torch.int32),
    #     num_q_heads,
    #     num_kv_heads,
    #     attn_mask,
    #     attn_qk_scale,
    #     alibi_slopes,
    #     attn_output,
    # )
    # if attn_mask is not None:
    #     batch = q_start_loc.shape[0]
    #     scale_value = 1.0 / math.sqrt(query.shape[-1])
    #     for i in range(batch):
    #         start = q_start_loc[i]
    #         end = start + seq_len_list[i]
    #         single_seqlen = int(seq_len_list[i])
    #         single_q = query[start:end].view(1, single_seqlen, -1)
    #         single_k = key[start:end].reshape(1, single_seqlen, -1)
    #         single_v = value[start:end].reshape(1, single_seqlen, -1)
    #         single_o = attn_output[start:end].view(1, single_seqlen, -1)
    #         actual_seq_lengths = seq_len_list[i : i + 1]
    #         torch.ops.npu_ext.npu_prompt_flash_attention_out(
    #             single_q,
    #             single_k,
    #             single_v,
    #             single_o,
    #             padding_mask=None,
    #             atten_mask=attn_mask[i],
    #             actual_seq_lengths=actual_seq_lengths,
    #             num_heads=num_q_heads,
    #             scale_value=scale_value,
    #             pre_tokens=2147473647,
    #             next_tokens=0,
    #             input_layout="BSH",
    #             num_key_value_heads=num_kv_heads,
    #         )
    # else:
    #     # For now, the value of attn_mask is None only in vit
    #     scale_value = 1.0 / math.sqrt(query.shape[-1] // num_q_heads)
    #     attn_output[:] = torch.ops.npu.npu_prompt_flash_attention(
    #         query,
    #         key,
    #         value,
    #         actual_seq_lengths=seq_len_list,
    #         num_heads=num_q_heads,
    #         scale_value=scale_value,
    #         input_layout="BSH",
    #         num_key_value_heads=num_kv_heads,
    #     )
    return attn_output


@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    head, key_dim = key.shape[1:]
    value_dim = value_cache.shape[-1]
    block_num, block_size = key_cache.shape[:2]
    block_total = block_num * block_size

    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()

    key_cache_reshaped = key_cache.view(block_total, head, key_dim)
    torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    if value_dim:
        value_cache_reshaped = value_cache.view(block_total, head, key_dim)
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


@dataclass
class PagedAttentionParams:
    headNum: int
    qkScale: float
    kvHeadNum: int
    maskType: int = 0


def paged_decode_attentio_atb(
    query,
    key_cache,
    value_cache,
    block_table,
    block_size,
    kv_seq_len,
    num_q_heads,
    num_kv_heads,
    sm_scale,
    attn_output,
):
    _, _, head_dim = query.shape
    total_block, _, _ = key_cache.shape
    query_dim, value_dim = query.shape[-1], value_cache.shape[-1]
    value_cache = F.pad(value_cache, (0, query_dim - value_dim))
    atb_output = torch.empty_like(query)
    key_cache = key_cache.view(
        total_block // block_size, block_size, num_kv_heads, head_dim
    )
    value_cache = value_cache.view(
        total_block // block_size, block_size, num_kv_heads, head_dim
    )
    contextLens = kv_seq_len.tolist()
    params = PagedAttentionParams(
        headNum=num_q_heads,
        qkScale=sm_scale if sm_scale else 1.0 / math.sqrt(head_dim),
        kvHeadNum=num_kv_heads,
    )
    atb_op.set_op_name("PagedAttentionOperation")
    atb_op.set_param(json.dumps(params.__dict__))
    atb_op.execute_out_with_param(
        [query, key_cache, value_cache, block_table, kv_seq_len.to(torch.int32)],
        [atb_output],
        json.dumps({"contextLen": contextLens}),
    )
    torch.cuda.synchronize()  # 不需要同步
    attn_output[:] = atb_output[..., :value_dim]
    return attn_output


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
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support alibi_slopes yet"
        )
    if isinstance(block_table, torch.Tensor) and block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    paged_decode_attentio_atb(
        query,
        key_cache,
        value_cache,
        block_table,
        block_size,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        softmax_scale,
        attn_output,
    )

    # bs, _, dim = query.shape
    # query = query.contiguous()
    # query = query.view(bs, 1, num_q_heads * dim)
    # kv_cache_len = key_cache.shape[0]
    # key_cache = key_cache.view(1, kv_cache_len, -1)
    # value_cache = value_cache.view(1, kv_cache_len, -1)
    # scale_value = 1.0 / math.sqrt(dim)

    # torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
    #     query,
    #     key_cache,
    #     value_cache,
    #     attn_output.view_as(query),
    #     padding_mask=None,
    #     atten_mask=None,
    #     actual_seq_lengths=kv_seq_len.tolist(),
    #     antiquant_scale=None,
    #     antiquant_offset=None,
    #     block_table=block_table,
    #     dequant_scale1=None,
    #     quant_scale1=None,
    #     dequant_scale2=None,
    #     quant_scale2=None,
    #     quant_offset2=None,
    #     num_heads=num_q_heads,
    #     scale_value=scale_value,
    #     input_layout="BSH",
    #     num_key_value_heads=num_kv_heads,
    #     block_size=block_size,
    #     inner_precise=1,
    # )
    return attn_output


@register_ops(vendor_ops_registry)
def paged_prefill_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
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
    if softmax_scale is not None:
        raise RuntimeError(
            "paged_decode_attention does not " "support softmax_scale yet"
        )
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    # cann incre_fa don't support paged_attn when q_seq_len > 1
    batch = q_start_loc.shape[0]
    q_seq_len_list = q_seq_len.tolist()
    kv_seq_len_list = kv_seq_len.tolist()
    scale_value = 1.0 / math.sqrt(query.shape[-1])
    query = query.contiguous()
    for i in range(batch):
        start = q_start_loc[i]
        mask = attn_mask[i]
        for j in range(q_seq_len_list[i]):
            single_q = query[start + j : start + j + 1].view(1, 1, -1)
            single_o = attn_output[start + j : start + j + 1].view(1, 1, -1)
            torch.ops.npu_ext.npu_incre_flash_attention_v4_out(
                single_q,
                key_cache,
                value_cache,
                single_o,
                padding_mask=None,
                atten_mask=mask[j : j + 1],
                actual_seq_lengths=kv_seq_len_list[i : i + 1],
                antiquant_scale=None,
                antiquant_offset=None,
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
def moe_gating_topk_softmax(router_logits: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
    routing_weights = router_logits.new_empty((*router_logits.shape[:-1], topk))
    selected_experts = router_logits.new_empty(
        (*router_logits.shape[:-1], topk), dtype=torch.int32
    )
    selected_idx = torch.empty_like(selected_experts)
    return torch.ops.npu_ext.npu_moe_gating_topk_softmax(
        router_logits, None, topk, routing_weights, selected_experts, selected_idx
    )


# TODO only for internlm on transformers lib.
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
                num_q_heads,
                num_kv_heads,
                mask[i : i + 1],
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
                num_q_heads,
                num_kv_heads,
                None,
                None,
                attn_output,
            )
    return attn_output
