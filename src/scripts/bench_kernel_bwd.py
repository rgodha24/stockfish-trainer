from __future__ import annotations

import argparse
import os
import time

import torch

from src.data import make_sparse_batch_dataset
from src.data.loader import DataloaderSkipConfig
from src.model.modules.feature_transformer.kernel import (
    make_sparse_input_linear_backward_kernel,
)


def _load_one_batch(
    binpack: str,
    feature_set: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ds = make_sparse_batch_dataset(
        feature_set=feature_set,
        filenames=[binpack],
        batch_size=batch_size,
        cyclic=False,
        loader_threads=2,
        config=DataloaderSkipConfig(),
        shuffle_buffer_entries=0,
        pin_memory=False,
    )
    batch = next(iter(ds))
    _, white_indices, black_indices, _, _, _, _ = batch
    return (
        white_indices.to(device, dtype=torch.int32),
        black_indices.to(device, dtype=torch.int32),
    )


def benchmark(
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    num_inputs: int,
    output_size: int,
    device: torch.device,
    warmup: int,
    measure: int,
) -> dict[str, float]:
    max_active_indices = white_indices.shape[1]
    batch_size = white_indices.shape[0]

    weight_grad_w = torch.zeros(
        num_inputs, output_size, dtype=torch.float32, device=device
    )
    bias_grad_w = torch.zeros(output_size, dtype=torch.float32, device=device)
    output_grad_w = torch.randn(
        batch_size, output_size, dtype=torch.float32, device=device
    )

    weight_grad_b = torch.zeros(
        num_inputs, output_size, dtype=torch.float32, device=device
    )
    bias_grad_b = torch.zeros(output_size, dtype=torch.float32, device=device)
    output_grad_b = torch.randn(
        batch_size, output_size, dtype=torch.float32, device=device
    )

    kernel = make_sparse_input_linear_backward_kernel(max_active_indices, output_size)

    # Warmup (triggers autotune + compilation; includes sort/prep)
    for _ in range(warmup):
        kernel(white_indices, weight_grad_w, bias_grad_w, output_grad_w)
        kernel(black_indices, weight_grad_b, bias_grad_b, output_grad_b)
    torch.cuda.synchronize(device)

    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    host_started_at = time.perf_counter()
    start_event.record()
    for _ in range(measure):
        kernel(white_indices, weight_grad_w, bias_grad_w, output_grad_w)
        kernel(black_indices, weight_grad_b, bias_grad_b, output_grad_b)
    end_event.record()
    torch.cuda.synchronize(device)

    host_elapsed = max(time.perf_counter() - host_started_at, 1e-9)
    gpu_elapsed = max(start_event.elapsed_time(end_event) / 1000.0, 1e-9)

    return {
        "host_elapsed_sec": host_elapsed,
        "gpu_elapsed_sec": gpu_elapsed,
        "host_step_ms": host_elapsed * 1000.0 / measure,
        "gpu_step_ms": gpu_elapsed * 1000.0 / measure,
        "host_kops_per_sec": (batch_size * measure) / (host_elapsed * 1000),
        "gpu_kops_per_sec": (batch_size * measure) / (gpu_elapsed * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse input linear backward kernel with real data"
    )
    parser.add_argument("binpack", type=str, help="Path to .binpack file")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--feature_set", type=str, default="Full_Threats+HalfKAv2_hm")
    parser.add_argument("--num_inputs", type=int, default=22528)
    parser.add_argument("--output_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--measure", type=int, default=128)
    args = parser.parse_args()

    if not os.path.isfile(args.binpack):
        raise FileNotFoundError(args.binpack)

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)

    white_indices, black_indices = _load_one_batch(
        args.binpack, args.feature_set, args.batch_size, device
    )
    actual_batch_size = white_indices.shape[0]
    actual_max_active = white_indices.shape[1]

    metrics = benchmark(
        white_indices=white_indices,
        black_indices=black_indices,
        num_inputs=args.num_inputs,
        output_size=args.output_size,
        device=device,
        warmup=args.warmup,
        measure=args.measure,
    )

    print(
        f"gpu={torch.cuda.get_device_name(device)} "
        f"batch_size={actual_batch_size} max_active_indices={actual_max_active} "
        f"num_inputs={args.num_inputs} output_size={args.output_size} "
        f"warmup={args.warmup} measure={args.measure}",
        flush=True,
    )
    print(
        f"host_elapsed_sec={metrics['host_elapsed_sec']:.4f} "
        f"gpu_elapsed_sec={metrics['gpu_elapsed_sec']:.4f} "
        f"host_step_ms={metrics['host_step_ms']:.3f} "
        f"gpu_step_ms={metrics['gpu_step_ms']:.3f}",
        flush=True,
    )
    print(
        f"host_kops/s={metrics['host_kops_per_sec']:.1f} "
        f"gpu_kops/s={metrics['gpu_kops_per_sec']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
