from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src import ranger22
from src.data import Batch, iter_device_batches
from src.model import ModelConfig, NNUEModel, QuantizationConfig
from src.train.config import BaseTrainingConfig
from src.train.distributed import DistributedRuntime, init_training_runtime
from src.train.log import TrainingLogger

LoaderMetricsFn = Callable[[], dict[str, float | int]]


@dataclass(slots=True)
class TrainBatchSource:
    batches: Iterable[Batch]
    metrics: LoaderMetricsFn
    close: Callable[[], None]


TrainBatchSourceFactory = Callable[[DistributedRuntime, int], TrainBatchSource]


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
    runtime: DistributedRuntime,
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
        aux_loss_alpha=args.aux_loss_alpha,
        z_loss_alpha=args.z_loss_alpha,
        router_load_floor=args.router_load_floor,
        router_load_cap=args.router_load_cap,
        router_teacher_alpha=args.router_teacher_alpha,
        router_teacher_anneal_epochs=args.router_teacher_anneal_epochs,
    )
    model = NNUEModel(args.features, model_cfg, QuantizationConfig()).to(device)

    # Separate param groups: router gets lower LR
    router_module = getattr(model.layer_stacks, "router", None)
    router_params = (
        list(router_module.parameters()) if router_module is not None else []
    )
    router_param_ids = {id(p) for p in router_params}
    main_params = [p for p in model.parameters() if id(p) not in router_param_ids]
    param_groups: list[dict[str, Any]] = [{"params": main_params}]
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
            emit_log=runtime.is_main_process,
        )

    setattr(torch._dynamo.config, "cache_size_limit", 32)
    train_model: torch.nn.Module = model
    if runtime.is_distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
        )
    compiled_model = torch.compile(train_model, backend=args.compile_backend)
    return model, compiled_model, optimizer, scheduler, start_epoch, global_step


def restore_training_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    emit_log: bool = True,
) -> tuple[int, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    if emit_log:
        print(
            f"resumed checkpoint {path} at epoch={start_epoch:03d} global_step={global_step}",
            flush=True,
        )
    return start_epoch, global_step


def run_training(
    args: BaseTrainingConfig,
    source_factory: TrainBatchSourceFactory,
    *,
    allow_distributed: bool,
) -> None:
    ensure_datasets_exist(args.datasets)
    os.makedirs(args.default_root_dir, exist_ok=True)

    runtime = init_training_runtime(args, allow_distributed=allow_distributed)
    source: TrainBatchSource | None = None
    logger: TrainingLogger | None = None
    device_batches = None

    try:
        set_seed(args.seed)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training path")
        local_batch_size = args.batch_size
        if runtime.is_distributed:
            if args.batch_size % runtime.world_size != 0:
                raise ValueError(
                    "For DDP, `batch_size` is the global batch size and must be divisible by WORLD_SIZE. "
                    f"Got batch_size={args.batch_size}, world_size={runtime.world_size}."
                )
            local_batch_size = args.batch_size // runtime.world_size
        device_index = (
            runtime.local_rank
            if runtime.is_distributed
            else torch.cuda.current_device()
        )
        device = torch.device("cuda", device_index)
        if runtime.is_main_process:
            print(
                "starting training: world_size={} local_batch_size={} global_batch_size={} device={}".format(
                    runtime.world_size,
                    local_batch_size,
                    args.batch_size,
                    device,
                ),
                flush=True,
            )

        source = source_factory(runtime, local_batch_size)
        model, compiled_model, optimizer, scheduler, start_epoch, global_step = (
            build_training_state(args, device, runtime)
        )

        logger = TrainingLogger(args, runtime, local_batch_size)

        num_batches = num_batches_for_size(
            args.epoch_size,
            args.batch_size,
        )
        final_epoch = start_epoch - 1
        device_batches = iter_device_batches(
            source.batches,
            device,
            queue_size_limit=args.data_loader_queue_size,
        )

        for epoch in range(start_epoch, args.max_epochs):
            compiled_model.train()
            model.set_epoch(epoch)
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
            elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
            runtime.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor.item())
            runtime.all_reduce(epoch_loss_sum)
            reduced_batches_tensor = torch.tensor(
                processed_batches,
                device=device,
                dtype=torch.int64,
            )
            runtime.all_reduce(reduced_batches_tensor)
            reduced_batches = int(reduced_batches_tensor.item())
            lr = optimizer.param_groups[0]["lr"]
            epoch_loss = float(epoch_loss_sum.item()) / reduced_batches
            final_epoch = epoch
            logger.finish_epoch(
                epoch=epoch,
                epoch_loss=epoch_loss,
                lr=lr,
                elapsed=elapsed,
                processed_batches=processed_batches,
                reduced_batches=reduced_batches,
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
                runtime.barrier()

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
        runtime.barrier()
    finally:
        try:
            if source is not None and source.close is not None:
                source.close()
        finally:
            close_batches = getattr(device_batches, "close", None)
            if callable(close_batches):
                close_batches()
            if logger is not None:
                logger.finish()
            runtime.destroy()
