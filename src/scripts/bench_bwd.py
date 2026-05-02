from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import tyro

from src.data import Batch, iter_device_batches, make_sparse_batch_dataset
from src.model import NNUEModel
from src.train.common import build_training_state, compute_loss, set_seed
from src.train.config import SingleNodeTrainingConfig
from src.train.distributed import init_training_runtime


@dataclass(kw_only=True)
class BenchBwdConfig(SingleNodeTrainingConfig):
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


def make_loader(args: BenchBwdConfig, files: list[str]):
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


def preload_device_batches(
    args: BenchBwdConfig,
    files: list[str],
    device: torch.device,
) -> list[Batch]:
    loader = make_loader(args, files)
    device_batches = iter_device_batches(
        loader,
        device,
        queue_size_limit=args.data_loader_queue_size,
    )
    preloaded: list[Batch] = []
    try:
        for _ in range(args.preload_batches):
            preloaded.append(next(device_batches))
        torch.cuda.synchronize(device)
    finally:
        close_batches = getattr(device_batches, "close", None)
        if callable(close_batches):
            close_batches()
    if not preloaded:
        raise RuntimeError("Failed to preload any batches from the dataset.")
    return preloaded


def run_forward(
    model: NNUEModel,
    compiled_model: Any,
    batch: Batch,
    args: BenchBwdConfig,
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

    scorenet, log_dict = compiled_model(
        us,
        white_indices,
        black_indices,
        psqt_indices,
        layer_stack_indices,
        score,
    )
    scorenet = scorenet * model.quantization.nnue2score
    loss = compute_loss(scorenet, outcome, score, args, epoch=0)
    router_loss = log_dict.get("routing/router_loss")
    if router_loss is not None:
        loss = loss + router_loss
    return loss


def measured_positions(batches: list[Batch], steps: int) -> int:
    return sum(int(batches[step % len(batches)][0].shape[0]) for step in range(steps))


def benchmark(
    args: BenchBwdConfig,
    model: NNUEModel,
    compiled_model: Any,
    batches: list[Batch],
    device: torch.device,
) -> dict[str, float | int | str]:
    model.train()
    model.set_epoch(0)

    # Warmup: full forward + backward so autotune / compilation happens here
    for step in range(args.warmup_steps):
        loss = run_forward(
            model,
            compiled_model,
            batches[step % len(batches)],
            args,
        )
        loss.backward()
        torch.cuda.synchronize(device)

    # Measure backward only
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    host_started_at = time.perf_counter()
    start_event.record()
    for step in range(args.measure_steps):
        loss = run_forward(
            model,
            compiled_model,
            batches[step % len(batches)],
            args,
        )
        loss.backward()
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
        "loss": float(loss.detach().item()),
    }


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    args = tyro.cli(BenchBwdConfig)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    files, ignored = resolve_binpack_paths(args.datasets)
    set_seed(args.seed)
    device = torch.device("cuda")
    runtime = init_training_runtime(args, allow_distributed=False)
    model, compiled_model, _optimizer, _scheduler, _epoch, _global_step = (
        build_training_state(args, device, runtime)
    )

    preload_batches = preload_device_batches(args, files, device)
    param_count = sum(parameter.numel() for parameter in model.parameters())

    metrics = benchmark(
        args,
        model,
        compiled_model,
        preload_batches,
        device,
    )

    print(
        "gpu={} files={} ignored_inputs={} stacks={} batch_size={} compile_backend={} params={}".format(
            torch.cuda.get_device_name(device),
            len(files),
            len(ignored),
            args.stacks,
            args.batch_size,
            args.compile_backend,
            param_count,
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
