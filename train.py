import os
import random
import time
from dataclasses import asdict

import torch
import tyro
from torch.utils.data import DataLoader

import wandb
from config import TrainingConfig
from src.data import DataloaderSkipConfig, make_sparse_batch_dataset
from src.model import ModelConfig, NNUEModel, QuantizationConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_batches_for_size(total_positions: int, batch_size: int) -> int:
    if total_positions <= 0:
        return 0
    return max(1, (total_positions + batch_size - 1) // batch_size)


def make_train_loader(args: TrainingConfig) -> DataLoader:
    skip_cfg = DataloaderSkipConfig(
        filtered=args.filtered,
        wld_filtered=args.wld_filtered,
        random_fen_skipping=args.random_fen_skipping,
        early_fen_skipping=args.early_fen_skipping,
        simple_eval_skipping=args.simple_eval_skipping,
    )
    stream = make_sparse_batch_dataset(
        feature_set=args.features,
        filenames=list(args.datasets),
        batch_size=args.batch_size,
        cyclic=True,
        loader_threads=args.loader_threads,
        config=skip_cfg,
        chunk_entries=args.chunk_entries,
        encode_threads=args.encode_threads,
    )
    prefetch_factor = (
        None if args.data_loader_workers == 0 else args.data_loader_queue_size
    )
    return DataLoader(
        stream,
        batch_size=None,
        batch_sampler=None,
        num_workers=args.data_loader_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.data_loader_workers > 0,
        prefetch_factor=prefetch_factor,
    )


def compute_loss(
    scorenet: torch.Tensor,
    outcome: torch.Tensor,
    score: torch.Tensor,
    args: TrainingConfig,
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


def main() -> None:
    args = tyro.cli(TrainingConfig)

    for dataset in args.datasets:
        if not os.path.exists(dataset):
            raise FileNotFoundError(dataset)

    os.makedirs(args.default_root_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this training path")

    model_cfg = ModelConfig(
        L1=args.l1, L2=args.l2, L3=args.l3, layer_stacks=args.layer_stacks
    )
    model = NNUEModel(args.features, model_cfg, QuantizationConfig()).to(device)
    setattr(torch._dynamo.config, "cache_size_limit", 32)
    compiled_model = torch.compile(model, backend=args.compile_backend)

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

    train_loader = make_train_loader(args)

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=asdict(args),
    )

    global_step = 0
    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_start = time.time()
        num_batches = num_batches_for_size(args.epoch_size, args.batch_size)

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
            ) = batch

            us = us.to(device, non_blocking=True)
            them = them.to(device, non_blocking=True)
            white_indices = white_indices.to(
                device, non_blocking=True, dtype=torch.int32
            )
            white_values = white_values.to(device, non_blocking=True)
            black_indices = black_indices.to(
                device, non_blocking=True, dtype=torch.int32
            )
            black_values = black_values.to(device, non_blocking=True)
            outcome = outcome.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)
            psqt_indices = psqt_indices.to(device, non_blocking=True, dtype=torch.int64)
            layer_stack_indices = layer_stack_indices.to(
                device,
                non_blocking=True,
                dtype=torch.int64,
            )

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

            if (
                batch_idx % max(1, num_batches // 4) == 0
                or batch_idx == num_batches - 1
            ):
                print(
                    f"epoch={epoch:03d} step={batch_idx + 1}/{num_batches} "
                    f"loss={loss_value:.6f}",
                    flush=True,
                )

        scheduler.step()
        epoch_loss = epoch_loss_sum / max(1, num_batches)
        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} done loss={epoch_loss:.6f} lr={lr:.8g} time={elapsed:.1f}s it/s={num_batches / elapsed:.1f}",
            flush=True,
        )
        wandb.log(
            {
                "train/epoch": epoch,
                "train/loss_epoch": epoch_loss,
                "train/lr": lr,
                "train/epoch_time_sec": elapsed,
                "train/it_per_sec": num_batches / elapsed,
            },
            step=global_step,
        )

    run.finish()


if __name__ == "__main__":
    main()
