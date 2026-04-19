from collections.abc import Callable

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune, set_autotune_inputs
from torch import autograd

_WARP_ALIGNED_THREADS: tuple[int, ...] = (32, 64, 128, 256)
_MAX_PER_THREAD = 128
_EXTRA_PER_THREAD = 4
_WEIGHT_GRAD_PACK_MAX_COUNT = 512

SparseExpertLinearForwardKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
]
SparseExpertLinearInputBackwardKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
]


def _thread_tile_configs(dim_size: int) -> list[dict[str, int]]:
    if dim_size <= 0:
        return [{"threads": 32, "per_thread": 1}]
    out: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for threads in _WARP_ALIGNED_THREADS:
        min_pt = max(1, (dim_size + threads - 1) // threads)
        hi = min(min_pt + _EXTRA_PER_THREAD, _MAX_PER_THREAD)
        for per_thread in range(min_pt, hi + 1):
            key = (threads, per_thread)
            if key not in seen:
                seen.add(key)
                out.append({"threads": threads, "per_thread": per_thread})
    return out


def _heuristic_tile(dim_size: int) -> dict[str, int]:
    for cfg in _thread_tile_configs(dim_size):
        if cfg["threads"] == 128:
            return cfg
    return _thread_tile_configs(dim_size)[0]


def _forward_configs(*args, **_kwargs):
    if args and isinstance(args[0], tuple):
        _, out_features = args[0]
    else:
        _, out_features = args
    return _thread_tile_configs(out_features)


def _input_backward_configs(*args, **_kwargs):
    if args and isinstance(args[0], tuple):
        (in_features, _) = args[0]
    else:
        in_features, _ = args
    return _thread_tile_configs(in_features)


@autotune(configs=_forward_configs)
@tilelang.jit
def _sparse_expert_linear_forward_factory(
    in_features, out_features, threads, per_thread
):
    batch_size = T.dynamic("batch_size")
    num_experts = T.dynamic("num_experts")

    @T.prim_func
    def kernel(
        input_tensor: T.Tensor((batch_size, in_features), "float32"),
        expert_indices: T.Tensor((batch_size,), "int32"),
        weight: T.Tensor((num_experts, out_features, in_features), "float32"),
        bias: T.Tensor((num_experts, out_features), "float32"),
        output: T.Tensor((batch_size, out_features), "float32"),
    ):
        _shape_capture = (
            batch_size,
            num_experts,
            in_features,
            out_features,
            threads,
            per_thread,
        )
        with T.Kernel(batch_size, threads=threads) as bx:
            tid = T.get_thread_binding(0)
            expert = expert_indices[bx]
            acc = T.alloc_fragment((per_thread,), "float32")

            for p in T.serial(per_thread):
                col = p * threads + tid
                if col < out_features:
                    acc[p] = bias[expert, col]
                else:
                    acc[p] = T.float32(0)

            for k in T.serial(in_features):
                x = input_tensor[bx, k]
                for p in T.serial(per_thread):
                    col = p * threads + tid
                    if col < out_features:
                        acc[p] += weight[expert, col, k] * x

            for p in T.serial(per_thread):
                col = p * threads + tid
                if col < out_features:
                    output[bx, col] = acc[p]

    return kernel


@autotune(configs=_input_backward_configs)
@tilelang.jit
def _sparse_expert_linear_input_backward_factory(
    in_features, out_features, threads, per_thread
):
    batch_size = T.dynamic("batch_size")
    num_experts = T.dynamic("num_experts")

    @T.prim_func
    def kernel(
        expert_indices: T.Tensor((batch_size,), "int32"),
        grad_output: T.Tensor((batch_size, out_features), "float32"),
        weight: T.Tensor((num_experts, out_features, in_features), "float32"),
        grad_input: T.Tensor((batch_size, in_features), "float32"),
    ):
        _shape_capture = (
            batch_size,
            num_experts,
            in_features,
            out_features,
            threads,
            per_thread,
        )
        with T.Kernel(batch_size, threads=threads) as bx:
            tid = T.get_thread_binding(0)
            expert = expert_indices[bx]
            acc = T.alloc_fragment((per_thread,), "float32")

            for p in T.serial(per_thread):
                acc[p] = T.float32(0)

            for out_col in T.serial(out_features):
                go = grad_output[bx, out_col]
                for p in T.serial(per_thread):
                    in_col = p * threads + tid
                    if in_col < in_features:
                        acc[p] += weight[expert, out_col, in_col] * go

            for p in T.serial(per_thread):
                in_col = p * threads + tid
                if in_col < in_features:
                    grad_input[bx, in_col] = acc[p]

    return kernel


class _LazySparseExpertForwardKernel:
    def __init__(self, in_features: int, out_features: int):
        self._in_features = in_features
        self._out_features = out_features
        self._kernel = None
        self._input_backward_kernel_ref: _LazySparseExpertInputBackwardKernel | None = (
            None
        )

    def __call__(
        self,
        input_tensor: torch.Tensor,
        expert_indices: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if self._kernel is None:
            with set_autotune_inputs(
                input_tensor, expert_indices, weight, bias, output
            ):
                self._kernel = _sparse_expert_linear_forward_factory(
                    self._in_features, self._out_features
                )
            bwd = self._input_backward_kernel_ref
            if bwd is not None and bwd._kernel is None:
                bwd._autotune_from_forward(input_tensor, expert_indices, weight)
        self._kernel(input_tensor, expert_indices, weight, bias, output)


class _LazySparseExpertInputBackwardKernel:
    def __init__(self, in_features: int, out_features: int):
        self._in_features = in_features
        self._out_features = out_features
        self._kernel = None

    def _autotune_from_forward(
        self,
        input_tensor: torch.Tensor,
        expert_indices: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        grad_output = torch.randn(
            input_tensor.shape[0],
            self._out_features,
            dtype=torch.float32,
            device=input_tensor.device,
        )
        grad_input = torch.empty_like(input_tensor)
        with set_autotune_inputs(expert_indices, grad_output, weight, grad_input):
            self._kernel = _sparse_expert_linear_input_backward_factory(
                self._in_features, self._out_features
            )

    def __call__(
        self,
        expert_indices: torch.Tensor,
        grad_output: torch.Tensor,
        weight: torch.Tensor,
        grad_input: torch.Tensor,
    ) -> None:
        if self._kernel is None:
            h = _heuristic_tile(self._in_features)
            self._kernel = _sparse_expert_linear_input_backward_factory(
                self._in_features,
                self._out_features,
                h["threads"],
                h["per_thread"],
            )
        self._kernel(expert_indices, grad_output, weight, grad_input)


_forward_kernel_cache: dict[tuple[int, int], _LazySparseExpertForwardKernel] = {}
_input_backward_kernel_cache: dict[
    tuple[int, int], _LazySparseExpertInputBackwardKernel
] = {}


def _get_or_create_input_backward_kernel(
    in_features: int, out_features: int
) -> _LazySparseExpertInputBackwardKernel:
    key = (in_features, out_features)
    if key not in _input_backward_kernel_cache:
        _input_backward_kernel_cache[key] = _LazySparseExpertInputBackwardKernel(
            in_features, out_features
        )
    return _input_backward_kernel_cache[key]


@torch.compiler.disable(recursive=False)
def make_sparse_expert_linear_forward_kernel(
    in_features: int, out_features: int
) -> SparseExpertLinearForwardKernel:
    key = (in_features, out_features)
    if key not in _forward_kernel_cache:
        kernel = _LazySparseExpertForwardKernel(in_features, out_features)
        kernel._input_backward_kernel_ref = _get_or_create_input_backward_kernel(
            in_features, out_features
        )
        _forward_kernel_cache[key] = kernel
    return _forward_kernel_cache[key]


@torch.compiler.disable(recursive=False)
def make_sparse_expert_linear_input_backward_kernel(
    in_features: int, out_features: int
) -> SparseExpertLinearInputBackwardKernel:
    return _get_or_create_input_backward_kernel(in_features, out_features)


def _compute_weight_grad_sorted(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    expert_long = expert_indices.to(torch.int64)
    perm = torch.argsort(expert_long, stable=True)
    sorted_experts = expert_long[perm]
    sorted_input = input_tensor.index_select(0, perm)
    sorted_grad = grad_output.index_select(0, perm)
    active_experts, counts = torch.unique_consecutive(
        sorted_experts, return_counts=True
    )
    grad_weight = grad_output.new_zeros(
        (num_experts, grad_output.shape[1], input_tensor.shape[1])
    )

    if active_experts.numel() == 0:
        return grad_weight

    counts_cpu = counts.tolist()
    active_experts_cpu = active_experts.tolist()
    max_count = max(counts_cpu)
    if max_count <= _WEIGHT_GRAD_PACK_MAX_COUNT:
        num_active = len(active_experts_cpu)
        starts = torch.cumsum(counts, dim=0) - counts
        row_ids = torch.repeat_interleave(
            torch.arange(num_active, device=expert_long.device), counts
        )
        positions = torch.arange(
            sorted_experts.numel(), device=expert_long.device, dtype=torch.int64
        ) - torch.repeat_interleave(starts, counts)
        packed_input = input_tensor.new_zeros(
            (num_active, max_count, input_tensor.shape[1])
        )
        packed_grad = grad_output.new_zeros(
            (num_active, max_count, grad_output.shape[1])
        )
        packed_input[row_ids, positions] = sorted_input
        packed_grad[row_ids, positions] = sorted_grad
        grad_weight[active_experts] = torch.bmm(
            packed_grad.transpose(1, 2), packed_input
        )
        return grad_weight

    start = 0
    for index in range(len(active_experts_cpu)):
        count = counts_cpu[index]
        end = start + count
        expert = active_experts_cpu[index]
        grad_weight[expert] = sorted_grad[start:end].T @ sorted_input[start:end]
        start = end

    return grad_weight


class SparseExpertLinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, expert_indices, weight, bias):
        input_tensor = input_tensor.contiguous()
        expert_indices = expert_indices.to(dtype=torch.int32).contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        output = torch.empty(
            input_tensor.shape[0],
            weight.shape[1],
            dtype=torch.float32,
            device=input_tensor.device,
            requires_grad=True,
        )
        kernel = make_sparse_expert_linear_forward_kernel(
            input_tensor.shape[1], weight.shape[1]
        )
        kernel(input_tensor, expert_indices, weight, bias, output)
        ctx.save_for_backward(input_tensor, expert_indices, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, expert_indices, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.empty_like(input_tensor)
            kernel = make_sparse_expert_linear_input_backward_kernel(
                input_tensor.shape[1], grad_output.shape[1]
            )
            kernel(expert_indices, grad_output, weight, grad_input)

        grad_weight = None
        if ctx.needs_input_grad[2]:
            grad_weight = _compute_weight_grad_sorted(
                input_tensor, grad_output, expert_indices, weight.shape[0]
            )

        grad_bias = None
        if ctx.needs_input_grad[3]:
            grad_bias = torch.zeros(
                weight.shape[0],
                weight.shape[1],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            grad_bias.index_add_(0, expert_indices.to(torch.int64), grad_output)

        return grad_input, None, grad_weight, grad_bias


def _fallback_sparse_expert_linear(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    expert_long = expert_indices.to(torch.int64)
    selected_weight = weight.index_select(0, expert_long)
    selected_bias = bias.index_select(0, expert_long)
    output = torch.bmm(selected_weight, input_tensor.unsqueeze(-1)).squeeze(-1)
    return output + selected_bias


def sparse_expert_linear(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if (
        not input_tensor.is_cuda
        or input_tensor.dtype != torch.float32
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
    ):
        return _fallback_sparse_expert_linear(
            input_tensor, expert_indices, weight, bias
        )
    return SparseExpertLinearFunction.apply(input_tensor, expert_indices, weight, bias)
