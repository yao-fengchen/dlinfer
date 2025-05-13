import functools
import torch
from dlinfer.graph.dicp.dynamo_bridge.op_transformer import (
    BackendPatternBase,
    PatternMatcherPass,
    register_backend_patterns,
)
from dlinfer.graph.dicp.vendor.AtbGraph.codegen import atb_infer_param as infer_param

atb_pattern_matcher = PatternMatcherPass()

torch_patterns_cls_list_1 = []
register_torch_pattern_1 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_1
)

torch_patterns_cls_list_2 = []
register_torch_pattern_2 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_2
)

torch_patterns_cls_list_3 = []
register_torch_pattern_3 = functools.partial(
    register_backend_patterns, torch_patterns_cls_list_3
)


aten = torch.ops.aten
atb = torch.ops.atb
dlinfer = torch.ops.dlinfer


@register_torch_pattern_1
class TorchLinear(BackendPatternBase):
    @staticmethod
    def pattern(x_input, weight, viewed_input_shape, viewed_output_shape):
        trans_weight = torch.ops.aten.t.default(weight)
        viewed_input = torch.ops.aten.view.default(x_input, viewed_input_shape)
        mm_result = torch.ops.aten.mm.default(viewed_input, trans_weight)
        viewed_mm_result = torch.ops.aten.view.default(mm_result, viewed_output_shape)
        return viewed_mm_result

    @staticmethod
    def replacement(x_input, weight):
        return torch.ops.atb.linear.default(x_input, weight, None, False, True)


@register_torch_pattern_1
class TorchLinearWithBias(BackendPatternBase):
    @staticmethod
    def pattern(bias, x_input, weight, viewed_input_shape, viewed_output_shape):
        trans_weight = torch.ops.aten.t.default(weight)
        viewed_input = torch.ops.aten.view.default(x_input, viewed_input_shape)
        addmm_result = torch.ops.aten.addmm.default(bias, viewed_input, trans_weight)
        viewed_mm_result = torch.ops.aten.view.default(
            addmm_result, viewed_output_shape
        )
        return viewed_mm_result

    @staticmethod
    def replacement(bias, x_input, weight):
        return torch.ops.atb.linear.default(x_input, weight, bias, False, True)


@register_torch_pattern_1
class TorchLInearAllreduce(BackendPatternBase):
    @staticmethod
    def pattern(x, weight, bias, allreduce, lmdeploy_group_type, group):
        linear = torch.ops.dlinfer.linear.default(
            x, weight, bias, allreduce, lmdeploy_group_type
        )
        all_reduce = torch.ops._c10d_functional.all_reduce.default(linear, "sum", group)
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
        copy = torch.ops.aten.copy.default(linear, wait_tensor)
        return copy

    @staticmethod
    def replacement(x, weight, bias, allreduce, lmdeploy_group_type, group):
        return torch.ops.dlinfer.linear.default(x, weight, bias, True, group)


@register_torch_pattern_1
class TorchAllGatherLinear(BackendPatternBase):
    @staticmethod
    def pattern(
        x,
        weight,
        bias,
        dp_gather,
        all_reduce,
        lmdeploy_group_type,
        size,
        group_size,
        group_name,
        slice_dim,
        slice1,
        slice2,
        rank,
        tp_size,
    ):
        new_empty = torch.ops.aten.new_empty.default(x, size, pin_memory=False)
        all_gather_into_tensor = (
            torch.ops._c10d_functional.all_gather_into_tensor.default(
                x, group_size, group_name
            )
        )
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(
            all_gather_into_tensor
        )
        slice_tensor_1 = torch.ops.aten.slice.Tensor(wait_tensor, slice_dim, 0, slice1)
        slice_tensor_2 = torch.ops.aten.slice.Tensor(
            wait_tensor, slice_dim, slice1, slice2
        )
        copy_1 = torch.ops.aten.copy.default(new_empty, slice_tensor_1)
        copy_2 = torch.ops.aten.copy.default(new_empty, slice_tensor_2)
        cat = torch.ops.aten.cat.default([copy_1, copy_2])
        linear = torch.ops.dlinfer.linear.default(
            cat, weight, bias, dp_gather, all_reduce, rank, tp_size, lmdeploy_group_type
        )
        return linear

    @staticmethod
    def replacement(
        x,
        weight,
        bias,
        dp_gather,
        all_reduce,
        lmdeploy_group_type,
        size,
        group_size,
        group_name,
        slice_dim,
        slice1,
        slice2,
        rank,
        tp_size,
    ):
        return torch.ops.dlinfer.linear.default(
            x, weight, bias, dp_gather, all_reduce, rank, tp_size, group_name
        )


@register_torch_pattern_1
class RemoveDoubleTranspose(BackendPatternBase):
    @staticmethod
    def pattern(x, dim1, dim2):
        transpose_1 = torch.ops.aten.transpose.int(x, dim1, dim2)
        transpose_2 = torch.ops.aten.transpose.int(transpose_1, dim1, dim2)
        return transpose_2

    @staticmethod
    def replacement(x, dim1, dim2):
        return x


@register_torch_pattern_1
class TorchAllreduce(BackendPatternBase):
    @staticmethod
    def pattern(x, group):
        all_reduce = torch.ops._c10d_functional.all_reduce.default(x, "sum", group)
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce)
        copy = torch.ops.aten.copy.default(x, wait_tensor)
        return copy

    @staticmethod
    def replacement(x, group):
        return torch.ops.atb.allreduce.default(x, "sum", group)


@register_torch_pattern_1
class TorchInplaceDivTensor(BackendPatternBase):
    @staticmethod
    def pattern(x, other):
        div = torch.ops.aten.div.Tensor(x, other)
        copy = torch.ops.aten.copy_.default(x, div)
        return copy

    @staticmethod
    def replacement(x, other):
        return torch.ops.atb.inplace_div.default(x, other)


@register_torch_pattern_1
class TorchInplaceScatterTensor(BackendPatternBase):
    @staticmethod
    def pattern(x, dim, index, src):
        scatter = torch.ops.aten.scatter.src(x, dim, index, src)
        copy = torch.ops.aten.copy_.default(x, scatter)
        return copy

    @staticmethod
    def replacement(x, dim, index, src):
        return torch.ops.atb.inplace_scatter.default(x, dim, index, src)
