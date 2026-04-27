from collections.abc import Callable

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune, set_autotune_inputs


SparseInputLinearForwardKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
]
SparseInputLinearBackwardKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
]


def _divisor_threads(output_size: int, lo: int = 32, hi: int = 1024) -> list[dict]:
    """Return autotune configs where threads evenly divides output_size.

    TileLang's layout inference segfaults on bounds-check guards, so we
    only emit configs that guarantee threads * per_thread == output_size.
    """
    return [
        {"threads": t, "per_thread": output_size // t}
        for t in range(lo, min(hi, output_size) + 1)
        if output_size % t == 0
    ]


def _forward_configs(*args, **_kwargs):
    # tilelang 0.1.8 passes (args_tuple, kwargs_tuple) instead of unpacking
    if args and isinstance(args[0], tuple):
        max_active_indices, output_size = args[0]
    else:
        max_active_indices, output_size = args
    _ = max_active_indices
    return _divisor_threads(output_size)


def _backward_configs(*args, **_kwargs):
    if args and isinstance(args[0], tuple):
        (output_size,) = args[0]
    else:
        (output_size,) = args
    return _divisor_threads(output_size)


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


@autotune(configs=_forward_configs, warmup=5, rep=20, timeout=30, skip_check=True)
@tilelang.jit
def _sparse_input_linear_forward_factory(
    max_active_indices, output_size, threads, per_thread
):
    batch_size = T.dynamic("batch_size")
    num_inputs = T.dynamic("num_inputs")

    @T.prim_func
    def kernel(
        input_indices: T.Tensor((batch_size, max_active_indices), "int32"),
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
                    for p in T.serial(per_thread):
                        acc[p] += weight[idx, p * threads + tid]

            for p in T.serial(per_thread):
                output[bx, p * threads + tid] = acc[p]

    return kernel


def _build_sorted_backward_inputs(
    input_indices: torch.Tensor,
    flat_batch_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_idx = input_indices.reshape(-1)
    mask = flat_idx != -1

    flat_idx = flat_idx[mask]
    if flat_idx.numel() == 0:
        empty_i32 = torch.empty(0, dtype=torch.int32, device=input_indices.device)
        return empty_i32, empty_i32, empty_i32, empty_i32

    flat_bid = flat_batch_indices[mask]

    sorted_idx, perm = torch.sort(flat_idx, stable=True)
    sorted_bid = flat_bid[perm]

    seg_feat, seg_count = torch.unique_consecutive(sorted_idx, return_counts=True)
    seg_start = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=input_indices.device),
            torch.cumsum(seg_count, dim=0)[:-1],
        ]
    ).to(torch.int32)

    return (
        sorted_bid.to(torch.int32),
        seg_feat.to(torch.int32),
        seg_count.to(torch.int32),
        seg_start,
    )


@autotune(configs=_backward_configs, warmup=5, rep=20, timeout=30, skip_check=True)
@tilelang.jit
def _sparse_input_linear_backward_factory(output_size, threads, per_thread):
    batch_size = T.dynamic("batch_size")
    nonzero_count = T.dynamic("nonzero_count")
    unique_count = T.dynamic("unique_count")
    num_inputs = T.dynamic("num_inputs")

    @T.prim_func
    def kernel(
        sorted_bidx: T.Tensor((nonzero_count,), "int32"),
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
                for p in T.serial(per_thread):
                    acc[p] += output_grad[batch_idx, p * threads + tid]

            for p in T.serial(per_thread):
                weight_grad[feat, p * threads + tid] = (
                    weight_grad[feat, p * threads + tid] + acc[p]
                )

    return kernel


class _LazyForwardKernel:
    """Autotunes on first real call using actual training tensors."""

    def __init__(self, max_active_indices: int, output_size: int):
        self._max_active_indices = max_active_indices
        self._output_size = output_size
        self._kernel = None
        self._backward_kernel_ref: _LazyBackwardKernel | None = None

    def __call__(
        self,
        input_indices: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if self._kernel is None:
            with set_autotune_inputs(input_indices, weight, bias, output):
                self._kernel = _sparse_input_linear_forward_factory(
                    self._max_active_indices, self._output_size
                )
            # Also autotune the backward kernel now (main thread) since
            # loss.backward() runs on autograd threads where signal.alarm
            # (used by tilelang's timeout) is not available.
            bwd = self._backward_kernel_ref
            if bwd is not None and bwd._kernel is None:
                bwd._autotune_from_forward(input_indices, weight)
        self._kernel(input_indices, weight, bias, output)


class _LazyBackwardKernel:
    """Autotunes on first real call using actual sorted training tensors."""

    def __init__(self, output_size: int):
        self._output_size = output_size
        self._kernel = None

    def _autotune_from_forward(
        self,
        input_indices: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        """Autotune using representative forward-pass tensors (called from main thread)."""
        flat_batch_indices = _get_flat_batch_indices(
            input_indices.shape[0], input_indices.shape[1], input_indices.device
        )
        sorted_bidx, seg_feat, seg_count, seg_start = (
            _build_sorted_backward_inputs(input_indices, flat_batch_indices)
        )
        if seg_feat.numel() == 0:
            return
        weight_grad = torch.zeros_like(weight)
        output_grad = torch.randn(
            input_indices.shape[0], self._output_size,
            dtype=torch.float32, device=input_indices.device,
        )
        with set_autotune_inputs(
            sorted_bidx, output_grad,
            seg_start, seg_feat, seg_count, weight_grad,
        ):
            self._kernel = _sparse_input_linear_backward_factory(
                self._output_size
            )

    def __call__(
        self,
        input_indices: torch.Tensor,
        weight_grad: torch.Tensor,
        bias_grad: torch.Tensor,
        output_grad: torch.Tensor,
    ) -> None:
        flat_batch_indices = _get_flat_batch_indices(
            input_indices.shape[0], input_indices.shape[1], input_indices.device
        )
        sorted_bidx, seg_feat, seg_count, seg_start = (
            _build_sorted_backward_inputs(input_indices, flat_batch_indices)
        )

        bias_grad.add_(output_grad.sum(dim=0))
        if seg_feat.numel() == 0:
            return

        if self._kernel is None:
            # Fallback: compile with best heuristic if not autotuned from forward
            best = _divisor_threads(self._output_size)[-1]
            self._kernel = _sparse_input_linear_backward_factory(
                self._output_size, best["threads"], best["per_thread"]
            )

        self._kernel(
            sorted_bidx, output_grad,
            seg_start, seg_feat, seg_count, weight_grad,
        )


_forward_kernel_cache: dict[tuple[int, int], _LazyForwardKernel] = {}


_backward_kernel_cache: dict[tuple[int, int], _LazyBackwardKernel] = {}


def _get_or_create_backward_kernel(
    max_active_indices: int, output_size: int
) -> _LazyBackwardKernel:
    key = (max_active_indices, output_size)
    if key not in _backward_kernel_cache:
        _backward_kernel_cache[key] = _LazyBackwardKernel(output_size)
    return _backward_kernel_cache[key]


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_forward_kernel(
    max_active_indices: int, output_size: int
) -> SparseInputLinearForwardKernel:
    key = (max_active_indices, output_size)
    if key not in _forward_kernel_cache:
        fwd = _LazyForwardKernel(max_active_indices, output_size)
        fwd._backward_kernel_ref = _get_or_create_backward_kernel(
            max_active_indices, output_size
        )
        _forward_kernel_cache[key] = fwd
    return _forward_kernel_cache[key]


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_backward_kernel(
    max_active_indices: int, output_size: int
) -> SparseInputLinearBackwardKernel:
    return _get_or_create_backward_kernel(max_active_indices, output_size)
