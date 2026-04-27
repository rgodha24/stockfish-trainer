from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import tyro

from src.data import Batch, iter_device_batches, make_sparse_batch_dataset
from src.train.common import build_training_state, compute_loss, set_seed
from src.train.config import SingleNodeTrainingConfig


@dataclass(kw_only=True)
class BenchGpuConfig(SingleNodeTrainingConfig):
    preload_batches: int = 4
    warmup_steps: int = 16
    measure_steps: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.preload_batches <= 0:
            raise ValueError("`preload_batches` must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("`warmup_steps` must be non-negative.")
        if self.measure_steps <= 0:
            raise ValueError("`measure_steps` must be positive.")


def resolve_binpack_paths(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    resolved = []
    ignored = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".binpack"):
            resolved.append(path)
        else:
            ignored.append(path)

    if not resolved:
        raise ValueError("No .binpack files found in the provided dataset arguments.")

    return sorted(set(resolved)), ignored


def make_loader(args: BenchGpuConfig, files: list[str]):
    return make_sparse_batch_dataset(
        feature_set=args.features,
        filenames=files,
        batch_size=args.batch_size,
        cyclic=True,
        loader_threads=args.loader_threads,
        config=args.loader_skip_config(),
        shuffle_buffer_entries=args.shuffle_buffer_entries,
        pin_memory=args.pin_memory,
    )


def batch_nbytes(batch: Batch) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in batch)


def preload_device_batches(
    args: BenchGpuConfig,
    files: list[str],
    device: torch.device,
) -> tuple[list[Batch], float]:
    loader = make_loader(args, files)
    device_batches = iter_device_batches(
        loader,
        device,
        queue_size_limit=args.data_loader_queue_size,
    )
    preloaded: list[Batch] = []
    started_at = time.perf_counter()
    try:
        for _ in range(args.preload_batches):
            preloaded.append(next(device_batches))
        torch.cuda.synchronize(device)
    finally:
        close_batches = getattr(device_batches, "close", None)
        if callable(close_batches):
            close_batches()
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    if not preloaded:
        raise RuntimeError("Failed to preload any batches from the dataset.")
    return preloaded, elapsed


def run_train_step(
    model: torch.nn.Module,
    compiled_model: Any,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    args: BenchGpuConfig,
) -> torch.Tensor:
    (
        us,
        white_indices,
        black_indices,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = batch

    optimizer.zero_grad(set_to_none=True)
    scorenet, log_dict = compiled_model(
        us,
        white_indices,
        black_indices,
        psqt_indices,
        layer_stack_indices,
    )
    scorenet = scorenet * model.quantization.nnue2score
    loss = compute_loss(scorenet, outcome, score, args, epoch=0)
    router_loss = log_dict.get("routing/router_loss")
    if router_loss is not None:
        loss = loss + router_loss
    loss.backward()
    optimizer.step()
    return loss


def measured_positions(batches: list[Batch], steps: int) -> int:
    return sum(int(batches[step % len(batches)][0].shape[0]) for step in range(steps))


def benchmark(
    args: BenchGpuConfig,
    model: torch.nn.Module,
    compiled_model: Any,
    optimizer: torch.optim.Optimizer,
    batches: list[Batch],
    device: torch.device,
) -> dict[str, float | int | str]:
    model.train()
    model.set_epoch(0)

    last_loss = torch.zeros((), device=device)
    for step in range(args.warmup_steps):
        last_loss = run_train_step(
            model,
            compiled_model,
            optimizer,
            batches[step % len(batches)],
            args,
        )
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    host_started_at = time.perf_counter()
    start_event.record()
    for step in range(args.measure_steps):
        last_loss = run_train_step(
            model,
            compiled_model,
            optimizer,
            batches[step % len(batches)],
            args,
        )
    end_event.record()
    torch.cuda.synchronize(device)
    host_elapsed = max(time.perf_counter() - host_started_at, 1e-9)
    gpu_elapsed = max(start_event.elapsed_time(end_event) / 1000.0, 1e-9)
    positions = measured_positions(batches, args.measure_steps)

    return {
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "positions": positions,
        "host_elapsed_sec": host_elapsed,
        "gpu_elapsed_sec": gpu_elapsed,
        "host_pos_per_sec": positions / host_elapsed,
        "gpu_pos_per_sec": positions / gpu_elapsed,
        "host_step_ms": host_elapsed * 1000.0 / args.measure_steps,
        "gpu_step_ms": gpu_elapsed * 1000.0 / args.measure_steps,
        "max_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "loss": float(last_loss.detach().item()),
    }


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    args = tyro.cli(BenchGpuConfig)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    files, ignored = resolve_binpack_paths(args.datasets)
    set_seed(args.seed)
    device = torch.device("cuda")
    model, compiled_model, optimizer, _scheduler, _epoch, _global_step = (
        build_training_state(args, device)
    )

    preload_batches, preload_elapsed = preload_device_batches(args, files, device)
    preload_positions = sum(int(batch[0].shape[0]) for batch in preload_batches)
    preload_gib = sum(batch_nbytes(batch) for batch in preload_batches) / (1024**3)
    param_count = sum(parameter.numel() for parameter in model.parameters())

    metrics = benchmark(
        args,
        model,
        compiled_model,
        optimizer,
        preload_batches,
        device,
    )

    print(
        "gpu={} files={} ignored_inputs={} stacks={} batch_size={} compile_backend={} params={} preload_batches={} preload_positions={} preload_gib={:.3f} preload_sec={:.3f}".format(
            torch.cuda.get_device_name(device),
            len(files),
            len(ignored),
            args.stacks,
            args.batch_size,
            args.compile_backend,
            param_count,
            len(preload_batches),
            preload_positions,
            preload_gib,
            preload_elapsed,
        ),
        flush=True,
    )
    print(
        "warmup_steps={} measure_steps={} measured_positions={} host_elapsed_sec={:.3f} gpu_elapsed_sec={:.3f}".format(
            metrics["warmup_steps"],
            metrics["measure_steps"],
            metrics["positions"],
            metrics["host_elapsed_sec"],
            metrics["gpu_elapsed_sec"],
        ),
        flush=True,
    )
    print(
        "host_pos/s={:.0f} gpu_pos/s={:.0f} host_step_ms={:.2f} gpu_step_ms={:.2f} max_memory_gib={:.3f} last_loss={:.6f}".format(
            metrics["host_pos_per_sec"],
            metrics["gpu_pos_per_sec"],
            metrics["host_step_ms"],
            metrics["gpu_step_ms"],
            metrics["max_memory_gib"],
            metrics["loss"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
