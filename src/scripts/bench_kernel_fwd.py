from __future__ import annotations

import argparse
import time

import torch

from src.model.modules.feature_transformer.kernel import (
    make_sparse_input_linear_forward_kernel,
)


def benchmark(
    batch_size: int,
    max_active_indices: int,
    num_inputs: int,
    output_size: int,
    device: torch.device,
    warmup: int,
    measure: int,
) -> dict[str, float]:
    # Generate realistic sparse indices: half active, rest -1 (padding)
    input_indices = torch.randint(
        0,
        num_inputs,
        (batch_size, max_active_indices),
        dtype=torch.int32,
        device=device,
    )
    mask = torch.rand(batch_size, max_active_indices, device=device) < 0.5
    input_indices = torch.where(mask, input_indices, torch.full_like(input_indices, -1))

    weight = torch.randn(num_inputs, output_size, dtype=torch.float32, device=device)
    bias = torch.randn(output_size, dtype=torch.float32, device=device)
    output = torch.empty(batch_size, output_size, dtype=torch.float32, device=device)

    kernel = make_sparse_input_linear_forward_kernel(max_active_indices, output_size)

    # Warmup (triggers autotune + compilation)
    for _ in range(warmup):
        kernel(input_indices, weight, bias, output)
    torch.cuda.synchronize(device)

    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    host_started_at = time.perf_counter()
    start_event.record()
    for _ in range(measure):
        kernel(input_indices, weight, bias, output)
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
        description="Benchmark sparse input linear forward kernel"
    )
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--max_active_indices", type=int, default=64)
    parser.add_argument("--num_inputs", type=int, default=22528)
    parser.add_argument("--output_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--measure", type=int, default=128)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)

    metrics = benchmark(
        batch_size=args.batch_size,
        max_active_indices=args.max_active_indices,
        num_inputs=args.num_inputs,
        output_size=args.output_size,
        device=device,
        warmup=args.warmup,
        measure=args.measure,
    )

    print(
        f"gpu={torch.cuda.get_device_name(device)} "
        f"batch_size={args.batch_size} max_active_indices={args.max_active_indices} "
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
