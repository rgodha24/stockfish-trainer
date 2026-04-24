from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F
import tyro
import wandb
from torch import nn
from tyro.conf import Positional

from src.data import (
    DataloaderSkipConfig,
    iter_device_batches,
    make_sparse_batch_dataset,
)
from src.model.modules import LayerStacks
from src.scripts.oracle_headroom import (
    _all_expert_scores,
    _load_layerstacks_checkpoint,
    _loss_matrix,
    _resolve_device,
    _shared_forward,
)


@dataclass(kw_only=True)
class RouterTrainingConfig:
    checkpoint: Positional[str]
    datasets: Positional[tuple[str, ...]] = ()
    val_datasets: tuple[str, ...] = ()

    batch_size: int = 8192
    train_batches_per_epoch: int = 1024
    val_batches: int = 128
    max_epochs: int = 32
    device: str = "auto"
    loader_threads: int = -1
    shuffle_buffer_entries: int = 65536
    pin_memory: bool = True

    hidden_dim: int = 128
    dropout: float = 0.0
    lr: float = 3.0e-5
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    rank_temperature: float = 1.0e-3
    rank_margin_clip: float = 4.0
    weight_floor: float = 0.05
    weight_clip: float = 5.0

    filtered: bool | None = None
    wld_filtered: bool | None = None
    random_fen_skipping: int | None = None
    early_fen_skipping: int | None = None
    simple_eval_skipping: int | None = None
    param_index: int | None = None
    pc_y1: float | None = None
    pc_y2: float | None = None
    pc_y3: float | None = None

    output_dir: str = "logs/router"
    resume_from_checkpoint: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if self.train_batches_per_epoch <= 0:
            raise ValueError("`train_batches_per_epoch` must be positive.")
        if self.val_batches <= 0:
            raise ValueError("`val_batches` must be positive.")
        if self.max_epochs <= 0:
            raise ValueError("`max_epochs` must be positive.")
        if self.loader_threads == 0:
            raise ValueError("`loader_threads` must be positive or -1 for auto.")
        if self.shuffle_buffer_entries < 0:
            raise ValueError("`shuffle_buffer_entries` must be non-negative.")
        if self.hidden_dim < 0:
            raise ValueError("`hidden_dim` must be non-negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("`dropout` must be in [0, 1).")
        if self.lr <= 0.0:
            raise ValueError("`lr` must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("`weight_decay` must be non-negative.")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("`grad_clip_norm` must be positive.")
        if self.rank_temperature <= 0.0:
            raise ValueError("`rank_temperature` must be positive.")
        if self.rank_margin_clip <= 0.0:
            raise ValueError("`rank_margin_clip` must be positive.")
        if self.weight_floor < 0.0:
            raise ValueError("`weight_floor` must be non-negative.")
        if self.weight_clip <= 0.0:
            raise ValueError("`weight_clip` must be positive.")


class RouterHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim == 0:
            self.net = nn.Linear(input_dim, num_experts)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_experts),
            )
            first = self.net[0]
            last = self.net[-1]
            assert isinstance(first, nn.Linear)
            assert isinstance(last, nn.Linear)
            nn.init.kaiming_uniform_(first.weight, a=5**0.5)
            nn.init.uniform_(first.bias, -0.02, 0.02)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pick_override(
    override: bool | int | float | None,
    ckpt_cfg: dict,
    key: str,
    default: bool | int | float,
) -> bool | int | float:
    if override is not None:
        return override
    return ckpt_cfg.get(key, default)


def _build_skip_config(
    args: RouterTrainingConfig, ckpt_cfg: dict
) -> DataloaderSkipConfig:
    return DataloaderSkipConfig(
        filtered=bool(_pick_override(args.filtered, ckpt_cfg, "filtered", True)),
        wld_filtered=bool(
            _pick_override(args.wld_filtered, ckpt_cfg, "wld_filtered", True)
        ),
        random_fen_skipping=int(
            _pick_override(args.random_fen_skipping, ckpt_cfg, "random_fen_skipping", 0)
        ),
        early_fen_skipping=int(
            _pick_override(args.early_fen_skipping, ckpt_cfg, "early_fen_skipping", -1)
        ),
        simple_eval_skipping=int(
            _pick_override(
                args.simple_eval_skipping, ckpt_cfg, "simple_eval_skipping", -1
            )
        ),
        param_index=int(_pick_override(args.param_index, ckpt_cfg, "param_index", 0)),
        pc_y1=float(_pick_override(args.pc_y1, ckpt_cfg, "pc_y1", 1.0)),
        pc_y2=float(_pick_override(args.pc_y2, ckpt_cfg, "pc_y2", 2.0)),
        pc_y3=float(_pick_override(args.pc_y3, ckpt_cfg, "pc_y3", 1.0)),
    )


def _resolve_binpacks(paths: tuple[str, ...]) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    skipped: list[str] = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".binpack"):
            resolved.append(path)
        else:
            skipped.append(path)
    return sorted(set(resolved)), skipped


def _resolve_train_files(
    args: RouterTrainingConfig, ckpt_cfg: dict
) -> tuple[list[str], list[str]]:
    requested = args.datasets if args.datasets else tuple(ckpt_cfg.get("datasets", ()))
    files, skipped = _resolve_binpacks(requested)
    if not files:
        raise ValueError("No readable .binpack files were found for training.")
    return files, skipped


def _resolve_val_files(args: RouterTrainingConfig, train_files: list[str]) -> list[str]:
    if not args.val_datasets:
        return train_files
    files, _ = _resolve_binpacks(args.val_datasets)
    if not files:
        raise ValueError("No readable .binpack files were found in `val_datasets`.")
    return files


def _make_loader(
    args: RouterTrainingConfig,
    ckpt_cfg: dict,
    files: list[str],
):
    return make_sparse_batch_dataset(
        feature_set=ckpt_cfg.get("features", "Full_Threats+HalfKAv2_hm^"),
        filenames=files,
        batch_size=args.batch_size,
        cyclic=True,
        loader_threads=args.loader_threads,
        config=_build_skip_config(args, ckpt_cfg),
        shuffle_buffer_entries=args.shuffle_buffer_entries,
        pin_memory=args.pin_memory,
    )


def _routing_stats(
    loss_matrix: torch.Tensor,
    bucket_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    oracle_loss, oracle_indices = loss_matrix.min(dim=1)
    bucket_loss = loss_matrix.gather(1, bucket_indices.unsqueeze(1)).squeeze(1)
    gain = (bucket_loss - oracle_loss).clamp_min(0.0)
    return bucket_loss, oracle_loss, oracle_indices, gain


def _ranking_loss(
    logits: torch.Tensor,
    loss_matrix: torch.Tensor,
    oracle_indices: torch.Tensor,
    oracle_loss: torch.Tensor,
    *,
    temperature: float,
    margin_clip: float,
) -> torch.Tensor:
    best_logits = logits.gather(1, oracle_indices.unsqueeze(1))
    target_margin = ((loss_matrix - oracle_loss.unsqueeze(1)) / temperature).clamp(
        min=0.0,
        max=margin_clip,
    )
    violations = F.softplus(target_margin - (best_logits - logits))
    mask = torch.ones_like(violations)
    mask.scatter_(1, oracle_indices.unsqueeze(1), 0.0)
    return (violations * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _example_weights(gain: torch.Tensor, *, floor: float, clip: float) -> torch.Tensor:
    gain = gain.clamp_min(0.0)
    mean_gain = float(gain.mean().item())
    if mean_gain <= 1.0e-12:
        return torch.ones_like(gain)
    return (gain / mean_gain + floor).clamp(max=clip)


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_checkpoint(
    path: str,
    *,
    epoch: int,
    global_step: int,
    router: RouterHead,
    optimizer: torch.optim.Optimizer,
    args: RouterTrainingConfig,
    best_val_top1_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "router_state_dict": router.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": asdict(args),
            "base_checkpoint": args.checkpoint,
            "best_val_top1_loss": best_val_top1_loss,
        },
        path,
    )


def _restore_checkpoint(
    path: str,
    *,
    router: RouterHead,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, float]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    router.load_state_dict(raw["router_state_dict"])
    optimizer.load_state_dict(raw["optimizer_state_dict"])
    start_epoch = int(raw.get("epoch", -1)) + 1
    global_step = int(raw.get("global_step", 0))
    best_val_top1_loss = float(raw.get("best_val_top1_loss", float("inf")))
    print(
        f"resumed router checkpoint {path} at epoch={start_epoch:03d} global_step={global_step}",
        flush=True,
    )
    return start_epoch, global_step, best_val_top1_loss


def _run_epoch(
    *,
    router: RouterHead,
    frozen_model,
    ckpt_cfg: dict,
    checkpoint_epoch: int,
    loader,
    batches_per_epoch: int,
    optimizer: torch.optim.Optimizer | None,
    args: RouterTrainingConfig,
    device: torch.device,
    prefix: str,
    global_step: int,
) -> tuple[dict[str, Any], int]:
    training = optimizer is not None
    router.train(training)
    num_experts = frozen_model.layer_stacks.count

    total_objective = 0.0
    total_top1_loss = 0.0
    total_bucket_loss = 0.0
    total_oracle_loss = 0.0
    oracle_agreement = 0
    bucket_agreement = 0
    route_hist = torch.zeros(num_experts, dtype=torch.int64)
    positions = 0

    device_batches = iter_device_batches(loader, device, queue_size_limit=16)
    for batch_idx, batch in enumerate(device_batches):
        if batch_idx >= batches_per_epoch:
            break

        with torch.no_grad():
            l0, psqt, outcome, score, bucket_indices = _shared_forward(
                frozen_model, batch
            )
            expert_scores = _all_expert_scores(frozen_model, l0, psqt)
            loss_matrix, _ = _loss_matrix(
                expert_scores,
                outcome,
                score,
                ckpt_cfg,
                checkpoint_epoch,
            )
            bucket_loss, oracle_loss, oracle_indices, gain = _routing_stats(
                loss_matrix,
                bucket_indices,
            )
            weights = _example_weights(
                gain,
                floor=args.weight_floor,
                clip=args.weight_clip,
            )

        if training:
            logits = router(l0)
            per_example_loss = _ranking_loss(
                logits,
                loss_matrix.detach(),
                oracle_indices,
                oracle_loss.detach(),
                temperature=args.rank_temperature,
                margin_clip=args.rank_margin_clip,
            )
            objective = (weights * per_example_loss).sum() / weights.sum().clamp_min(
                1.0e-12
            )

            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), args.grad_clip_norm)
            optimizer.step()
            global_step += 1
        else:
            with torch.no_grad():
                logits = router(l0)
                per_example_loss = _ranking_loss(
                    logits,
                    loss_matrix,
                    oracle_indices,
                    oracle_loss,
                    temperature=args.rank_temperature,
                    margin_clip=args.rank_margin_clip,
                )
                objective = (
                    weights * per_example_loss
                ).sum() / weights.sum().clamp_min(1.0e-12)

        top1_indices = logits.argmax(dim=1)
        top1_loss = loss_matrix.gather(1, top1_indices.unsqueeze(1)).squeeze(1)
        batch_size = int(top1_indices.numel())

        total_objective += float(objective.detach().item()) * batch_size
        total_top1_loss += float(top1_loss.detach().mean().item()) * batch_size
        total_bucket_loss += float(bucket_loss.detach().mean().item()) * batch_size
        total_oracle_loss += float(oracle_loss.detach().mean().item()) * batch_size
        oracle_agreement += int((top1_indices == oracle_indices).sum().item())
        bucket_agreement += int((top1_indices == bucket_indices).sum().item())
        route_hist.add_(
            torch.bincount(top1_indices.detach().cpu(), minlength=num_experts)
        )
        positions += batch_size

        if (
            batch_idx % max(1, batches_per_epoch // 4) == 0
            or batch_idx + 1 == batches_per_epoch
        ):
            print(
                f"{prefix} step={batch_idx + 1}/{batches_per_epoch} objective={objective.detach().item():.6f}",
                flush=True,
            )

    if positions == 0:
        raise RuntimeError(f"{prefix} loader produced no batches")

    bucket_mean = total_bucket_loss / positions
    oracle_mean = total_oracle_loss / positions
    top1_mean = total_top1_loss / positions
    headroom = max(bucket_mean - oracle_mean, 1.0e-12)
    return {
        f"{prefix}/objective": total_objective / positions,
        f"{prefix}/top1_loss": top1_mean,
        f"{prefix}/bucket_loss": bucket_mean,
        f"{prefix}/oracle_loss": oracle_mean,
        f"{prefix}/oracle_agreement": oracle_agreement / positions,
        f"{prefix}/bucket_agreement": bucket_agreement / positions,
        f"{prefix}/regret_reduction": (bucket_mean - top1_mean) / headroom,
        f"{prefix}/route_hist": route_hist.tolist(),
        f"{prefix}/positions": positions,
    }, global_step


def _print_epoch_summary(epoch: int, prefix: str, metrics: dict[str, Any]) -> None:
    print(
        (
            f"epoch={epoch:03d} {prefix} objective={metrics[f'{prefix}/objective']:.6f} "
            f"top1={metrics[f'{prefix}/top1_loss']:.6f} "
            f"bucket={metrics[f'{prefix}/bucket_loss']:.6f} "
            f"oracle={metrics[f'{prefix}/oracle_loss']:.6f} "
            f"regret_reduction={metrics[f'{prefix}/regret_reduction'] * 100.0:.2f}% "
            f"oracle_agree={metrics[f'{prefix}/oracle_agreement'] * 100.0:.2f}%"
        ),
        flush=True,
    )
    print(
        f"epoch={epoch:03d} {prefix} route_hist={metrics[f'{prefix}/route_hist']}",
        flush=True,
    )


def _default_run_name(args: RouterTrainingConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    return args.wandb_run_name or f"router-{stem}-{stamp}"


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    args = tyro.cli(RouterTrainingConfig)
    device = _resolve_device(args.device)
    frozen_model, ckpt_cfg, checkpoint_epoch = _load_layerstacks_checkpoint(
        args.checkpoint
    )
    layer_stacks = frozen_model.layer_stacks
    if not isinstance(layer_stacks, LayerStacks):
        raise ValueError("Router training expects a layerstacks checkpoint.")
    frozen_model = frozen_model.to(device)
    frozen_model.requires_grad_(False)

    train_files, skipped = _resolve_train_files(args, ckpt_cfg)
    val_files = _resolve_val_files(args, train_files)
    for path in skipped:
        print(f"skipped={path}", flush=True)

    run_name = _default_run_name(args)
    run_dir = os.path.join(args.output_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    _save_json(os.path.join(run_dir, "config.json"), asdict(args))

    router = RouterHead(
        input_dim=frozen_model.L1,
        num_experts=int(layer_stacks.count),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    best_val_top1_loss = float("inf")
    if args.resume_from_checkpoint is not None:
        start_epoch, global_step, best_val_top1_loss = _restore_checkpoint(
            args.resume_from_checkpoint,
            router=router,
            optimizer=optimizer,
        )

    run = None
    if args.wandb_project is not None:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            config=asdict(args),
        )

    train_batches = iter(_make_loader(args, ckpt_cfg, train_files))
    val_batches = iter(_make_loader(args, ckpt_cfg, val_files))
    best_path = os.path.join(checkpoint_dir, "best.pt")
    last_path = os.path.join(checkpoint_dir, "last.pt")
    final_metrics: dict[str, Any] = {}

    try:
        for epoch in range(start_epoch, args.max_epochs):
            epoch_start = time.time()
            train_metrics, global_step = _run_epoch(
                router=router,
                frozen_model=frozen_model,
                ckpt_cfg=ckpt_cfg,
                checkpoint_epoch=checkpoint_epoch,
                loader=train_batches,
                batches_per_epoch=args.train_batches_per_epoch,
                optimizer=optimizer,
                args=args,
                device=device,
                prefix="train",
                global_step=global_step,
            )
            val_metrics, _ = _run_epoch(
                router=router,
                frozen_model=frozen_model,
                ckpt_cfg=ckpt_cfg,
                checkpoint_epoch=checkpoint_epoch,
                loader=val_batches,
                batches_per_epoch=args.val_batches,
                optimizer=None,
                args=args,
                device=device,
                prefix="val",
                global_step=global_step,
            )

            final_metrics = {
                **train_metrics,
                **val_metrics,
                "train/epoch": epoch,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch_time_sec": max(time.time() - epoch_start, 1.0e-9),
                "train/global_step": global_step,
            }

            _print_epoch_summary(epoch, "train", train_metrics)
            _print_epoch_summary(epoch, "val", val_metrics)
            if run is not None:
                wandb.log(final_metrics, step=global_step)

            _save_checkpoint(
                last_path,
                epoch=epoch,
                global_step=global_step,
                router=router,
                optimizer=optimizer,
                args=args,
                best_val_top1_loss=best_val_top1_loss,
            )
            val_top1_loss = float(val_metrics["val/top1_loss"])
            if val_top1_loss < best_val_top1_loss:
                best_val_top1_loss = val_top1_loss
                _save_checkpoint(
                    best_path,
                    epoch=epoch,
                    global_step=global_step,
                    router=router,
                    optimizer=optimizer,
                    args=args,
                    best_val_top1_loss=best_val_top1_loss,
                )
                print(f"saved best checkpoint {best_path}", flush=True)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary = {
            "run_dir": run_dir,
            "best_checkpoint": best_path if os.path.exists(best_path) else None,
            "last_checkpoint": last_path if os.path.exists(last_path) else None,
            "best_val_top1_loss": best_val_top1_loss,
            **final_metrics,
        }
        summary_path = os.path.join(run_dir, "summary.json")
        _save_json(summary_path, summary)
        print(f"finished training; wrote summary {summary_path}", flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
