from __future__ import annotations

import os
import random
import shutil
import time
from dataclasses import asdict
from typing import Any, Callable, Iterable

import torch
import wandb

from src.model import ModelConfig, NNUEModel, QuantizationConfig
from src.train.config import BaseTrainingConfig


Batch = tuple[torch.Tensor, ...]
LoaderMetricsFn = Callable[[], dict[str, float | int]]


def ensure_datasets_exist(paths: Iterable[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_batches_for_size(total_positions: int, batch_size: int) -> int:
    if total_positions <= 0:
        return 0
    return max(1, (total_positions + batch_size - 1) // batch_size)


def save_training_checkpoint(
    path: str,
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: BaseTrainingConfig,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_config": asdict(args),
        },
        path,
    )


def compute_loss(
    scorenet: torch.Tensor,
    outcome: torch.Tensor,
    score: torch.Tensor,
    args: BaseTrainingConfig,
    epoch: int,
) -> torch.Tensor:
    start_lambda = args.start_lambda if args.start_lambda is not None else args.lambda_
    end_lambda = args.end_lambda if args.end_lambda is not None else args.lambda_

    q = (scorenet - args.in_offset) / args.in_scaling
    qm = (-scorenet - args.in_offset) / args.in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

    s = (score - args.out_offset) / args.out_scaling
    sm = (-score - args.out_offset) / args.out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

    actual_lambda = start_lambda + (end_lambda - start_lambda) * (
        epoch / args.max_epochs
    )
    pt = pf * actual_lambda + outcome * (1.0 - actual_lambda)

    loss = torch.pow(torch.abs(pt - qf), args.pow_exp)
    if args.qp_asymmetry != 0.0:
        loss = loss * ((qf > pt) * args.qp_asymmetry + 1)

    weights = 1 + (2.0**args.w1 - 1) * torch.pow(
        (pf - 0.5) ** 2 * pf * (1 - pf),
        args.w2,
    )
    return (loss * weights).sum() / weights.sum()


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = batch

    return (
        us.to(device, non_blocking=True),
        them.to(device, non_blocking=True),
        white_indices.to(device, non_blocking=True, dtype=torch.int32),
        white_values.to(device, non_blocking=True),
        black_indices.to(device, non_blocking=True, dtype=torch.int32),
        black_values.to(device, non_blocking=True),
        outcome.to(device, non_blocking=True),
        score.to(device, non_blocking=True),
        psqt_indices.to(device, non_blocking=True, dtype=torch.int64),
        layer_stack_indices.to(device, non_blocking=True, dtype=torch.int64),
    )


def build_training_state(
    args: BaseTrainingConfig,
    device: torch.device,
) -> tuple[
    NNUEModel,
    Any,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    int,
    int,
]:
    model_cfg = ModelConfig(
        L1=args.l1,
        L2=args.l2,
        L3=args.l3,
        layer_stacks=args.layer_stacks,
    )
    model = NNUEModel(args.features, model_cfg, QuantizationConfig()).to(device)

    optimizer = __import__("ranger22").Ranger22(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1.0e-7,
        using_gc=False,
        using_normgc=False,
        weight_decay=0.0,
        num_batches_per_epoch=num_batches_for_size(args.epoch_size, args.batch_size),
        num_epochs=args.max_epochs,
        warmdown_active=False,
        use_warmup=False,
        use_adaptive_gradient_clipping=False,
        softplus=False,
        pnm_momentum_factor=0.0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=args.gamma
    )

    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint is not None:
        start_epoch, global_step = restore_training_checkpoint(
            args.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    setattr(torch._dynamo.config, "cache_size_limit", 32)
    compiled_model = torch.compile(model, backend=args.compile_backend)
    return model, compiled_model, optimizer, scheduler, start_epoch, global_step


def restore_training_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    print(
        f"resumed checkpoint {path} at epoch={start_epoch:03d} global_step={global_step}",
        flush=True,
    )
    return start_epoch, global_step


def _loader_metric_delta(
    start: dict[str, float | int] | None,
    end: dict[str, float | int] | None,
    key: str,
) -> float | int | None:
    if start is None or end is None or key not in start or key not in end:
        return None
    return end[key] - start[key]


def build_loader_metrics(
    start: dict[str, float | int] | None,
    end: dict[str, float | int] | None,
    elapsed: float,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    encoded_entries = _loader_metric_delta(start, end, "encoded_entries")
    received_entries = _loader_metric_delta(start, end, "received_entries")
    received_bytes = _loader_metric_delta(start, end, "received_bytes")
    wait_sec = _loader_metric_delta(start, end, "wait_sec")
    get_sec = _loader_metric_delta(start, end, "get_sec")

    if encoded_entries is not None:
        metrics["loader/encoded_positions_per_sec"] = encoded_entries / max(
            elapsed, 1e-9
        )
    if received_entries is not None:
        metrics["loader/received_positions_per_sec"] = received_entries / max(
            elapsed, 1e-9
        )
    if received_bytes is not None:
        metrics["loader/recv_gib_per_sec"] = (
            received_bytes / max(elapsed, 1e-9) / (1024**3)
        )
    if wait_sec is not None:
        metrics["loader/wait_fraction"] = wait_sec / max(elapsed, 1e-9)
    if get_sec is not None:
        metrics["loader/get_fraction"] = get_sec / max(elapsed, 1e-9)

    if end is not None:
        for key in ("pending_batches", "inflight_calls", "encoded_batches"):
            if key in end:
                metrics[f"loader/{key}"] = end[key]
    return metrics


def run_training(
    args: BaseTrainingConfig,
    train_loader: Iterable[Batch],
    *,
    loader_metrics_fn: LoaderMetricsFn | None = None,
    loader_close_fn: Callable[[], None] | None = None,
) -> None:
    ensure_datasets_exist(args.datasets)
    os.makedirs(args.default_root_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this training path")

    model, compiled_model, optimizer, scheduler, start_epoch, global_step = (
        build_training_state(args, device)
    )

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=asdict(args),
    )

    checkpoint_dir = os.path.join(args.default_root_dir, "checkpoints")
    final_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_start = time.time()
            num_batches = num_batches_for_size(args.epoch_size, args.batch_size)
            processed_batches = 0
            loader_start = (
                loader_metrics_fn() if loader_metrics_fn is not None else None
            )

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= num_batches:
                    break
                if batch_idx == 0:
                    model.clip_input_weights()
                model.clip_weights()

                (
                    us,
                    them,
                    white_indices,
                    white_values,
                    black_indices,
                    black_values,
                    outcome,
                    score,
                    psqt_indices,
                    layer_stack_indices,
                ) = move_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)
                scorenet = (
                    compiled_model(
                        us,
                        them,
                        white_indices,
                        white_values,
                        black_indices,
                        black_values,
                        psqt_indices,
                        layer_stack_indices,
                    )
                    * model.quantization.nnue2score
                )
                loss = compute_loss(scorenet, outcome, score, args, epoch)
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                epoch_loss_sum += loss_value
                global_step += 1
                processed_batches += 1

                if (
                    batch_idx % max(1, num_batches // 4) == 0
                    or batch_idx == num_batches - 1
                ):
                    print(
                        f"epoch={epoch:03d} step={batch_idx + 1}/{num_batches} loss={loss_value:.6f}",
                        flush=True,
                    )

            scheduler.step()
            if processed_batches == 0:
                raise RuntimeError("training loader produced no batches")
            elapsed = max(time.time() - epoch_start, 1e-9)
            lr = optimizer.param_groups[0]["lr"]
            epoch_loss = epoch_loss_sum / processed_batches
            final_epoch = epoch

            metrics: dict[str, Any] = {
                "train/epoch": epoch,
                "train/loss_epoch": epoch_loss,
                "train/lr": lr,
                "train/epoch_time_sec": elapsed,
                "train/it_per_sec": processed_batches / elapsed,
                "train/positions_per_sec": (processed_batches * args.batch_size)
                / elapsed,
            }

            loader_end = loader_metrics_fn() if loader_metrics_fn is not None else None
            metrics.update(build_loader_metrics(loader_start, loader_end, elapsed))

            print(
                f"epoch={epoch:03d} done loss={epoch_loss:.6f} lr={lr:.8g} time={elapsed:.1f}s it/s={processed_batches / elapsed:.1f} pos/s={(processed_batches * args.batch_size) / elapsed:.0f}",
                flush=True,
            )
            if loader_end is not None:
                encoded_pos_per_sec = metrics.get("loader/encoded_positions_per_sec")
                wait_fraction = metrics.get("loader/wait_fraction")
                print(
                    "loader encoded_pos/s={} wait={:.1f}% pending_batches={} inflight={}".format(
                        f"{encoded_pos_per_sec:.0f}"
                        if encoded_pos_per_sec is not None
                        else "n/a",
                        float(wait_fraction or 0.0) * 100.0,
                        int(loader_end.get("pending_batches", 0)),
                        int(loader_end.get("inflight_calls", 0)),
                    ),
                    flush=True,
                )

            wandb.log(metrics, step=global_step)

            if (
                args.checkpoint_every_epochs > 0
                and (epoch + 1) % args.checkpoint_every_epochs == 0
            ):
                last_path = os.path.join(checkpoint_dir, "last.pt")
                epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
                save_training_checkpoint(
                    epoch_path,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                )
                shutil.copyfile(epoch_path, last_path)
                print(f"saved checkpoints {epoch_path} -> {last_path}", flush=True)

        final_path = os.path.join(checkpoint_dir, "final.pt")
        save_training_checkpoint(
            final_path,
            epoch=max(final_epoch, 0),
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )
        print(f"saved final checkpoint {final_path}", flush=True)
    finally:
        if loader_close_fn is not None:
            loader_close_fn()
        run.finish()
