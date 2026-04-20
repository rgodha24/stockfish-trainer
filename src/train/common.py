from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import torch

from src import ranger22
from src.data import Batch, iter_device_batches
from src.model import ModelConfig, NNUEModel, QuantizationConfig
from src.train.config import BaseTrainingConfig
from src.train.log import TrainingLogger

LoaderMetricsFn = Callable[[], dict[str, float | int]]


@dataclass(slots=True)
class TrainBatchSource:
    batches: Iterable[Batch]
    metrics: LoaderMetricsFn
    close: Callable[[], None]


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
        stacks=args.stacks,
        num_experts=args.num_experts,
        router_features=args.router_features,
        aux_loss_alpha=args.aux_loss_alpha,
        z_loss_alpha=args.z_loss_alpha,
        gumbel_tau_start=args.gumbel_tau_start,
        gumbel_tau_end=args.gumbel_tau_end,
        gumbel_anneal_fraction=args.gumbel_anneal_fraction,
    )
    model = NNUEModel(args.features, model_cfg, QuantizationConfig()).to(device)

    # Separate param groups: router gets lower LR
    router_params = set()
    if args.stacks == "moe":
        router_params = set(model.layer_stacks.router.parameters())
    main_params = [p for p in model.parameters() if p not in router_params]
    param_groups = [{"params": main_params}]
    if router_params:
        param_groups.append(
            {"params": list(router_params), "lr": args.lr * args.router_lr_multiplier}
        )

    optimizer = ranger22.Ranger22(
        param_groups,
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


def run_training(
    args: BaseTrainingConfig,
    source: TrainBatchSource,
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

    logger = TrainingLogger(args)

    num_batches = num_batches_for_size(args.epoch_size, args.batch_size)
    final_epoch = start_epoch - 1
    device_batches = iter_device_batches(
        source.batches,
        device,
        queue_size_limit=args.data_loader_queue_size,
    )

    try:
        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            # Update tau annealing progress for MoE
            if args.stacks == "moe":
                progress = epoch / args.max_epochs
                model.layer_stacks.set_training_progress(progress)
            epoch_loss_sum = torch.zeros((), device=device)
            epoch_start = time.time()
            processed_batches = 0
            logger.start_epoch(source.metrics() if source.metrics is not None else None)

            for batch_idx, batch in enumerate(device_batches):
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
                ) = batch

                optimizer.zero_grad(set_to_none=True)
                scorenet, log_dict = compiled_model(
                    us,
                    them,
                    white_indices,
                    white_values,
                    black_indices,
                    black_values,
                    psqt_indices,
                    layer_stack_indices,
                )
                router_loss = logger.on_batch(log_dict, scorenet)
                scorenet = scorenet * model.quantization.nnue2score
                loss = compute_loss(scorenet, outcome, score, args, epoch)
                loss = loss + router_loss
                loss.backward()
                optimizer.step()

                epoch_loss_sum.add_(loss.detach())
                global_step += 1
                processed_batches += 1
                logger.log_step(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                    loss=loss,
                )

            scheduler.step()
            if processed_batches == 0:
                raise RuntimeError("training loader produced no batches")
            elapsed = max(time.time() - epoch_start, 1e-9)
            lr = optimizer.param_groups[0]["lr"]
            epoch_loss = float(epoch_loss_sum.item()) / processed_batches
            final_epoch = epoch
            logger.finish_epoch(
                epoch=epoch,
                epoch_loss=epoch_loss,
                lr=lr,
                elapsed=elapsed,
                processed_batches=processed_batches,
                global_step=global_step,
                loader_end=source.metrics() if source.metrics is not None else None,
            )

            if (
                args.checkpoint_every_epochs > 0
                and (epoch + 1) % args.checkpoint_every_epochs == 0
            ):
                logger.save_periodic_checkpoint(
                    epoch=epoch,
                    save_fn=lambda path: save_training_checkpoint(
                        path,
                        epoch=epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                    ),
                )

        logger.save_final_checkpoint(
            save_fn=lambda path: save_training_checkpoint(
                path,
                epoch=max(final_epoch, 0),
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
            )
        )
    finally:
        try:
            if source.close is not None:
                source.close()
        finally:
            close_batches = getattr(device_batches, "close", None)
            if callable(close_batches):
                close_batches()
            logger.finish()
