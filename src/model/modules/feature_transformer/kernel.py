from collections.abc import Callable

import torch
import tilelang
import tilelang.language as T


SparseInputLinearForwardKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
]
SparseInputLinearBackwardKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
]


def _find_nearest_divisor(value: int, target: int) -> int:
    divisors = []
    for i in range(1, value + 1):
        if value % i == 0:
            divisors.append((i, abs(target - i)))
    divisors.sort(key=lambda x: x[1])
    return divisors[0][0]


_num_threads_forward_cache: dict[int, int] = {}


def _get_num_threads_for_forward(output_size: int) -> int:
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_forward_cache[output_size]


_num_threads_backward_cache: dict[int, int] = {}


def _get_num_threads_for_backward(output_size: int) -> int:
    optimal_num_threads = 512
    if output_size not in _num_threads_backward_cache:
        _num_threads_backward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_backward_cache[output_size]


_flat_batch_index_cache: dict[tuple[int, int, int], torch.Tensor] = {}


def _get_flat_batch_indices(
    batch_size: int, max_active_indices: int, device: torch.device
) -> torch.Tensor:
    key = (
        device.index if device.index is not None else -1,
        batch_size,
        max_active_indices,
    )
    if key not in _flat_batch_index_cache:
        _flat_batch_index_cache[key] = torch.arange(
            batch_size, device=device, dtype=torch.int32
        ).repeat_interleave(max_active_indices)
    return _flat_batch_index_cache[key]


@tilelang.jit
def _sparse_input_linear_forward_factory(
    max_active_indices, output_size, threads, per_thread
):
    batch_size = T.dynamic("batch_size")
    num_inputs = T.dynamic("num_inputs")

    @T.prim_func
    def kernel(
        input_indices: T.Tensor((batch_size, max_active_indices), "int32"),
        input_values: T.Tensor((batch_size, max_active_indices), "float32"),
        weight: T.Tensor((num_inputs, output_size), "float32"),
        bias: T.Tensor((output_size,), "float32"),
        output: T.Tensor((batch_size, output_size), "float32"),
    ):
        _shape_capture = (
            batch_size,
            num_inputs,
            max_active_indices,
            output_size,
            threads,
            per_thread,
        )
        with T.Kernel(batch_size, threads=threads) as bx:
            tid = T.get_thread_binding(0)
            acc = T.alloc_fragment((per_thread,), "float32")

            for p in T.serial(per_thread):
                acc[p] = bias[p * threads + tid]

            for k in T.serial(max_active_indices):
                idx = input_indices[bx, k]
                if idx != -1:
                    val = input_values[bx, k]
                    for p in T.serial(per_thread):
                        acc[p] += weight[idx, p * threads + tid] * val

            for p in T.serial(per_thread):
                output[bx, p * threads + tid] = acc[p]

    return kernel


def _build_sorted_backward_inputs(
    input_indices: torch.Tensor,
    input_values: torch.Tensor,
    flat_batch_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_idx = input_indices.reshape(-1)
    flat_val = input_values.reshape(-1)
    mask = flat_idx != -1

    flat_idx = flat_idx[mask]
    if flat_idx.numel() == 0:
        empty_i32 = torch.empty(0, dtype=torch.int32, device=input_indices.device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=input_indices.device)
        return empty_i32, empty_f32, empty_i32, empty_i32, empty_i32

    flat_val = flat_val[mask]
    flat_bid = flat_batch_indices[mask]

    sorted_idx, perm = torch.sort(flat_idx, stable=True)
    sorted_bid = flat_bid[perm]
    sorted_val = flat_val[perm]

    seg_feat, seg_count = torch.unique_consecutive(sorted_idx, return_counts=True)
    seg_start = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=input_indices.device),
            torch.cumsum(seg_count, dim=0)[:-1],
        ]
    ).to(torch.int32)

    return (
        sorted_bid.to(torch.int32),
        sorted_val.contiguous(),
        seg_feat.to(torch.int32),
        seg_count.to(torch.int32),
        seg_start,
    )


@tilelang.jit
def _sparse_input_linear_backward_factory(output_size, threads, per_thread):
    batch_size = T.dynamic("batch_size")
    nonzero_count = T.dynamic("nonzero_count")
    unique_count = T.dynamic("unique_count")
    num_inputs = T.dynamic("num_inputs")

    @T.prim_func
    def kernel(
        sorted_bidx: T.Tensor((nonzero_count,), "int32"),
        sorted_val: T.Tensor((nonzero_count,), "float32"),
        output_grad: T.Tensor((batch_size, output_size), "float32"),
        seg_start: T.Tensor((unique_count,), "int32"),
        seg_feat: T.Tensor((unique_count,), "int32"),
        seg_count: T.Tensor((unique_count,), "int32"),
        weight_grad: T.Tensor((num_inputs, output_size), "float32"),
    ):
        _shape_capture = (
            batch_size,
            nonzero_count,
            unique_count,
            num_inputs,
            output_size,
            threads,
            per_thread,
        )
        with T.Kernel(unique_count, threads=threads) as bu:
            tid = T.get_thread_binding(0)
            acc = T.alloc_fragment((per_thread,), "float32")

            for p in T.serial(per_thread):
                acc[p] = T.float32(0)

            start = seg_start[bu]
            count = seg_count[bu]
            feat = seg_feat[bu]

            for j in T.serial(count):
                batch_idx = sorted_bidx[start + j]
                value = sorted_val[start + j]
                for p in T.serial(per_thread):
                    acc[p] += output_grad[batch_idx, p * threads + tid] * value

            for p in T.serial(per_thread):
                weight_grad[feat, p * threads + tid] = (
                    weight_grad[feat, p * threads + tid] + acc[p]
                )

    return kernel


class _SortedSparseInputLinearBackwardKernel:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(
        self,
        input_indices: torch.Tensor,
        input_values: torch.Tensor,
        weight_grad: torch.Tensor,
        bias_grad: torch.Tensor,
        output_grad: torch.Tensor,
    ) -> None:
        flat_batch_indices = _get_flat_batch_indices(
            input_indices.shape[0], input_indices.shape[1], input_indices.device
        )
        sorted_bidx, sorted_val, seg_feat, seg_count, seg_start = (
            _build_sorted_backward_inputs(
                input_indices, input_values, flat_batch_indices
            )
        )

        bias_grad.add_(output_grad.sum(dim=0))
        if seg_feat.numel() == 0:
            return

        self.kernel(
            sorted_bidx,
            sorted_val,
            output_grad,
            seg_start,
            seg_feat,
            seg_count,
            weight_grad,
        )


_sparse_input_linear_forward_kernel_cache: dict[
    tuple[int, int, int], SparseInputLinearForwardKernel
] = {}


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_forward_kernel(
    max_active_indices: int, output_size: int
) -> SparseInputLinearForwardKernel:
    num_threads = _get_num_threads_for_forward(output_size)
    per_thread = output_size // num_threads
    key = (max_active_indices, output_size, num_threads)
    if key not in _sparse_input_linear_forward_kernel_cache:
        _sparse_input_linear_forward_kernel_cache[key] = (
            _sparse_input_linear_forward_factory(
                max_active_indices, output_size, num_threads, per_thread
            )
        )
    return _sparse_input_linear_forward_kernel_cache[key]


_sparse_input_linear_backward_kernel_cache: dict[
    tuple[int, int, int], SparseInputLinearBackwardKernel
] = {}


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_backward_kernel(
    max_active_indices: int, output_size: int
) -> SparseInputLinearBackwardKernel:
    num_threads = _get_num_threads_for_backward(output_size)
    per_thread = output_size // num_threads
    key = (max_active_indices, output_size, num_threads)
    if key not in _sparse_input_linear_backward_kernel_cache:
        kernel = _sparse_input_linear_backward_factory(
            output_size, num_threads, per_thread
        )
        _sparse_input_linear_backward_kernel_cache[key] = (
            _SortedSparseInputLinearBackwardKernel(kernel)
        )
    return _sparse_input_linear_backward_kernel_cache[key]
