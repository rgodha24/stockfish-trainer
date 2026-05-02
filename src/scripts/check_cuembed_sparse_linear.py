from __future__ import annotations

import argparse
import os
import time

import torch

from src.data import make_sparse_batch_dataset
from src.data.loader import DataloaderSkipConfig
from src.model.modules import get_feature_cls
from src.model.modules.feature_transformer import SparseLinearFunction
from src.model.modules.feature_transformer.cuembed import _load_ext


def sparse_linear_reference(
    flat_indices: torch.Tensor,
    offsets: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    counts = offsets[1:] - offsets[:-1]
    flat_batch = torch.repeat_interleave(
        torch.arange(counts.numel(), device=flat_indices.device, dtype=torch.int64),
        counts.to(torch.int64),
    )
    output = torch.zeros(
        counts.numel(), weight.shape[1], dtype=torch.float32, device=weight.device
    )
    output.index_add_(0, flat_batch, weight[flat_indices.to(torch.int64)])
    return output + bias


def sparse_linear_backward_reference(
    flat_indices: torch.Tensor,
    offsets: torch.Tensor,
    grad_output: torch.Tensor,
    num_features: int,
) -> torch.Tensor:
    counts = offsets[1:] - offsets[:-1]
    flat_batch = torch.repeat_interleave(
        torch.arange(counts.numel(), device=flat_indices.device, dtype=torch.int64),
        counts.to(torch.int64),
    )
    grad_weight = torch.zeros(
        num_features, grad_output.shape[1], dtype=torch.float32, device=grad_output.device
    )
    grad_weight.index_add_(0, flat_indices.to(torch.int64), grad_output[flat_batch])
    return grad_weight


def check_correctness(device: torch.device) -> None:
    torch.manual_seed(1234)
    batch_size = 128
    max_active = 10
    num_features = 257
    output_size = 64

    counts = torch.randint(0, max_active + 1, (batch_size,), device=device)
    offsets = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
    offsets[:1].zero_()
    offsets[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
    flat_indices = torch.randint(
        0, num_features, (int(offsets[-1].item()),), dtype=torch.int32, device=device
    )

    weight = torch.randn(num_features, output_size, dtype=torch.float32, device=device)
    bias = torch.randn(output_size, dtype=torch.float32, device=device)
    grad_output = torch.randn(batch_size, output_size, dtype=torch.float32, device=device)

    weight_cu = weight.detach().clone().requires_grad_(True)
    bias_cu = bias.detach().clone().requires_grad_(True)
    output_cu = SparseLinearFunction.apply(flat_indices, offsets, weight_cu, bias_cu)
    output_cu.backward(grad_output)

    output_ref = sparse_linear_reference(flat_indices, offsets, weight, bias)
    grad_weight_ref = sparse_linear_backward_reference(
        flat_indices, offsets, grad_output, num_features
    )
    bias_grad_ref = grad_output.sum(dim=0)
    torch.cuda.synchronize(device)

    fwd_diff = (output_cu - output_ref).abs().max().item()
    bwd_diff = (weight_cu.grad - grad_weight_ref).abs().max().item()
    bias_diff = (bias_cu.grad - bias_grad_ref).abs().max().item()
    print(
        f"correctness fwd_max_abs={fwd_diff:.6g} "
        f"bwd_max_abs={bwd_diff:.6g} bias_max_abs={bias_diff:.6g}",
        flush=True,
    )
    if fwd_diff > 1.0e-4 or bwd_diff > 1.0e-4 or bias_diff > 1.0e-4:
        raise RuntimeError("cuEmbed sparse linear correctness check failed")


def load_one_batch(
    binpack: str,
    feature_set: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = make_sparse_batch_dataset(
        feature_set=feature_set,
        filenames=[binpack],
        batch_size=batch_size,
        cyclic=False,
        loader_threads=2,
        config=DataloaderSkipConfig(),
        shuffle_buffer_entries=0,
        pin_memory=False,
    )
    batch = next(iter(dataset))
    _, white_indices, white_offsets, black_indices, black_offsets, _, _, _, _ = batch
    return (
        white_indices.to(device, dtype=torch.int32),
        white_offsets.to(device, dtype=torch.int32),
        black_indices.to(device, dtype=torch.int32),
        black_offsets.to(device, dtype=torch.int32),
    )


def time_loop(device: torch.device, warmup: int, measure: int, fn) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started_at = time.perf_counter()
    start_event.record()
    for _ in range(measure):
        fn()
    end_event.record()
    torch.cuda.synchronize(device)
    _ = time.perf_counter() - started_at
    return start_event.elapsed_time(end_event) / measure


def benchmark(
    binpack: str,
    feature_set: str,
    batch_size: int,
    num_features: int,
    output_size: int,
    warmup: int,
    measure: int,
    device: torch.device,
) -> None:
    flat_w, offsets_w, flat_b, offsets_b = load_one_batch(
        binpack, feature_set, batch_size, device
    )
    torch.cuda.synchronize(device)

    ext = _load_ext()
    weight = torch.randn(num_features, output_size, dtype=torch.float32, device=device)
    grad_w = torch.randn(batch_size, output_size, dtype=torch.float32, device=device)
    grad_b = torch.randn(batch_size, output_size, dtype=torch.float32, device=device)

    fwd_ms = time_loop(
        device,
        warmup,
        measure,
        lambda: (
            ext.cuembed_forward(flat_w, offsets_w, weight),
            ext.cuembed_forward(flat_b, offsets_b, weight),
        ),
    )
    bwd_ms = time_loop(
        device,
        warmup,
        measure,
        lambda: (
            ext.cuembed_backward(flat_w, offsets_w, grad_w, num_features),
            ext.cuembed_backward(flat_b, offsets_b, grad_b, num_features),
        ),
    )

    print(
        f"gpu={torch.cuda.get_device_name(device)} batch_size={batch_size} "
        f"num_features={num_features} "
        f"output_size={output_size} nnz_white={flat_w.numel()} nnz_black={flat_b.numel()}",
        flush=True,
    )
    print(
        f"cuembed_direct_fwd_ms={fwd_ms:.3f} "
        f"cuembed_direct_bwd_ms={bwd_ms:.3f} warmup={warmup} measure={measure}",
        flush=True,
    )


def infer_num_features(feature_set: str, output_size: int) -> int:
    try:
        return get_feature_cls(feature_set)(output_size).NUM_INPUTS
    except KeyError:
        return get_feature_cls(feature_set + "^")(output_size).NUM_INPUTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binpack", type=str, default="")
    parser.add_argument("--feature-set", type=str, default="Full_Threats+HalfKAv2_hm")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--num-features", type=int, default=0)
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--measure", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    check_correctness(device)
    if args.binpack:
        num_features = args.num_features
        if num_features == 0:
            num_features = infer_num_features(args.feature_set, args.output_size)
        if not os.path.isfile(args.binpack):
            raise FileNotFoundError(args.binpack)
        benchmark(
            args.binpack,
            args.feature_set,
            args.batch_size,
            num_features,
            args.output_size,
            args.warmup,
            args.measure,
            device,
        )


if __name__ == "__main__":
    main()
