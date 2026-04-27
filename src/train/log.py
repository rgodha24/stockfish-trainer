from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any

import torch
import wandb

from src.train.distributed import DistributedRuntime
from src.train.config import BaseTrainingConfig


def _loader_metric_delta(
    start: dict[str, float | int] | None,
    end: dict[str, float | int] | None,
    key: str,
) -> float | int | None:
    if start is None or end is None or key not in start or key not in end:
        return None
    return end[key] - start[key]


def _build_loader_metrics(
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


def _print_loader_summary(
    metrics: dict[str, Any],
    loader_end: dict[str, float | int] | None,
) -> None:
    if loader_end is None:
        return
    encoded_pos_per_sec = metrics.get("loader/encoded_positions_per_sec")
    wait_fraction = metrics.get("loader/wait_fraction")
    print(
        "loader encoded_pos/s={} wait={:.1f}% pending_batches={} inflight={}".format(
            f"{encoded_pos_per_sec:.0f}" if encoded_pos_per_sec is not None else "n/a",
            float(wait_fraction or 0.0) * 100.0,
            int(loader_end.get("pending_batches", 0)),
            int(loader_end.get("inflight_calls", 0)),
        ),
        flush=True,
    )


@dataclass(slots=True)
class RoutingMetricsAccumulator:
    router_loss_sum: torch.Tensor
    aux_loss_sum: torch.Tensor
    load_floor_loss_sum: torch.Tensor
    load_cap_loss_sum: torch.Tensor
    z_loss_sum: torch.Tensor
    probe_loss_sum: torch.Tensor
    probe_weight_sum: torch.Tensor
    entropy_sum: torch.Tensor
    top1_prob_sum: torch.Tensor
    target_prob_sum: torch.Tensor
    bucket_agreement_sum: torch.Tensor
    teacher_ce_sum: torch.Tensor
    teacher_alpha_sum: torch.Tensor
    fraction_routed_sum: torch.Tensor
    avg_gate_prob_sum: torch.Tensor

    @classmethod
    def maybe_create(
        cls, log_dict: dict[str, torch.Tensor]
    ) -> RoutingMetricsAccumulator | None:
        fraction_routed = log_dict.get("routing/fraction_routed")
        avg_gate_prob = log_dict.get("routing/avg_gate_prob")
        if fraction_routed is None or avg_gate_prob is None:
            return None
        device = fraction_routed.device
        return cls(
            router_loss_sum=torch.zeros((), device=device, dtype=torch.float32),
            aux_loss_sum=torch.zeros((), device=device, dtype=torch.float32),
            load_floor_loss_sum=torch.zeros((), device=device, dtype=torch.float32),
            load_cap_loss_sum=torch.zeros((), device=device, dtype=torch.float32),
            z_loss_sum=torch.zeros((), device=device, dtype=torch.float32),
            probe_loss_sum=torch.zeros((), device=device, dtype=torch.float32),
            probe_weight_sum=torch.zeros((), device=device, dtype=torch.float32),
            entropy_sum=torch.zeros((), device=device, dtype=torch.float32),
            top1_prob_sum=torch.zeros((), device=device, dtype=torch.float32),
            target_prob_sum=torch.zeros((), device=device, dtype=torch.float32),
            bucket_agreement_sum=torch.zeros((), device=device, dtype=torch.float32),
            teacher_ce_sum=torch.zeros((), device=device, dtype=torch.float32),
            teacher_alpha_sum=torch.zeros((), device=device, dtype=torch.float32),
            fraction_routed_sum=torch.zeros_like(
                fraction_routed.detach(), dtype=torch.float32
            ),
            avg_gate_prob_sum=torch.zeros_like(
                avg_gate_prob.detach(), dtype=torch.float32
            ),
        )

    def update(self, log_dict: dict[str, torch.Tensor]) -> None:
        router_loss = log_dict.get("routing/router_loss")
        aux_loss = log_dict.get("routing/aux_loss")
        load_floor_loss = log_dict.get("routing/load_floor_loss")
        load_cap_loss = log_dict.get("routing/load_cap_loss")
        z_loss = log_dict.get("routing/z_loss")
        probe_loss = log_dict.get("routing/probe_loss")
        probe_weight = log_dict.get("routing/probe_weight")
        fraction_routed = log_dict.get("routing/fraction_routed")
        avg_gate_prob = log_dict.get("routing/avg_gate_prob")
        target_prob = log_dict.get("routing/target_prob")
        bucket_agreement = log_dict.get("routing/bucket_agreement")
        teacher_ce = log_dict.get("routing/teacher_ce")
        teacher_alpha = log_dict.get("routing/teacher_alpha")
        entropy = log_dict.get("routing/entropy")
        top1_prob = log_dict.get("routing/top1_prob")
        if (
            router_loss is None
            or aux_loss is None
            or load_floor_loss is None
            or load_cap_loss is None
            or z_loss is None
            or probe_loss is None
            or probe_weight is None
            or fraction_routed is None
            or avg_gate_prob is None
            or target_prob is None
            or bucket_agreement is None
            or teacher_ce is None
            or teacher_alpha is None
            or entropy is None
            or top1_prob is None
        ):
            return

        self.router_loss_sum.add_(router_loss.detach().to(dtype=torch.float32))
        self.aux_loss_sum.add_(aux_loss.detach().to(dtype=torch.float32))
        self.load_floor_loss_sum.add_(load_floor_loss.detach().to(dtype=torch.float32))
        self.load_cap_loss_sum.add_(load_cap_loss.detach().to(dtype=torch.float32))
        self.z_loss_sum.add_(z_loss.detach().to(dtype=torch.float32))
        self.probe_loss_sum.add_(probe_loss.detach().to(dtype=torch.float32))
        self.probe_weight_sum.add_(probe_weight.detach().to(dtype=torch.float32))
        self.entropy_sum.add_(entropy.detach().to(dtype=torch.float32))
        self.top1_prob_sum.add_(top1_prob.detach().to(dtype=torch.float32))
        self.target_prob_sum.add_(target_prob.detach().to(dtype=torch.float32))
        self.bucket_agreement_sum.add_(
            bucket_agreement.detach().to(dtype=torch.float32)
        )
        self.teacher_ce_sum.add_(teacher_ce.detach().to(dtype=torch.float32))
        self.teacher_alpha_sum.add_(teacher_alpha.detach().to(dtype=torch.float32))
        self.fraction_routed_sum.add_(fraction_routed.detach().to(dtype=torch.float32))
        self.avg_gate_prob_sum.add_(avg_gate_prob.detach().to(dtype=torch.float32))

    def synchronize(self, runtime: DistributedRuntime) -> None:
        if not runtime.is_distributed:
            return
        # Batch all routing metrics into a single all-reduce to minimize
        # NCCL kernel launches (each launch has ~10-20us latency).
        scalars = [
            self.router_loss_sum,
            self.aux_loss_sum,
            self.load_floor_loss_sum,
            self.load_cap_loss_sum,
            self.z_loss_sum,
            self.probe_loss_sum,
            self.probe_weight_sum,
            self.entropy_sum,
            self.top1_prob_sum,
            self.target_prob_sum,
            self.bucket_agreement_sum,
            self.teacher_ce_sum,
            self.teacher_alpha_sum,
        ]
        flat = torch.cat(
            [s.unsqueeze(0) for s in scalars]
            + [self.fraction_routed_sum.flatten(), self.avg_gate_prob_sum.flatten()]
        )
        runtime.all_reduce(flat)
        # Scatter reduced values back into the individual accumulators.
        num_scalars = len(scalars)
        for i, s in enumerate(scalars):
            s.copy_(flat[i])
        offset = num_scalars
        fr_numel = self.fraction_routed_sum.numel()
        self.fraction_routed_sum.copy_(
            flat[offset : offset + fr_numel].view_as(self.fraction_routed_sum)
        )
        offset += fr_numel
        ag_numel = self.avg_gate_prob_sum.numel()
        self.avg_gate_prob_sum.copy_(
            flat[offset : offset + ag_numel].view_as(self.avg_gate_prob_sum)
        )

    def finalize(self, processed_batches: int) -> tuple[dict[str, Any], str]:
        denom = max(processed_batches, 1)
        avg_fraction_routed = (self.fraction_routed_sum / denom).detach().cpu()
        avg_gate_prob = (self.avg_gate_prob_sum / denom).detach().cpu()
        experts_used = int((avg_fraction_routed > 1e-4).sum().item())
        dead_experts = int((avg_fraction_routed <= 1e-4).sum().item())
        uniform = 1.0 / max(avg_fraction_routed.numel(), 1)
        load_std = float(avg_fraction_routed.std(unbiased=False).item())
        load_cv = load_std / uniform if uniform > 0.0 else 0.0

        metrics: dict[str, Any] = {
            "routing/router_loss_epoch": float(
                (self.router_loss_sum / denom).detach().cpu().item()
            ),
            "routing/aux_loss_epoch": float(
                (self.aux_loss_sum / denom).detach().cpu().item()
            ),
            "routing/load_floor_loss_epoch": float(
                (self.load_floor_loss_sum / denom).detach().cpu().item()
            ),
            "routing/load_cap_loss_epoch": float(
                (self.load_cap_loss_sum / denom).detach().cpu().item()
            ),
            "routing/z_loss_epoch": float(
                (self.z_loss_sum / denom).detach().cpu().item()
            ),
            "routing/probe_loss_epoch": float(
                (self.probe_loss_sum / denom).detach().cpu().item()
            ),
            "routing/probe_weight_epoch": float(
                (self.probe_weight_sum / denom).detach().cpu().item()
            ),
            "routing/entropy_epoch": float(
                (self.entropy_sum / denom).detach().cpu().item()
            ),
            "routing/top1_prob_epoch": float(
                (self.top1_prob_sum / denom).detach().cpu().item()
            ),
            "routing/target_prob_epoch": float(
                (self.target_prob_sum / denom).detach().cpu().item()
            ),
            "routing/bucket_agreement_epoch": float(
                (self.bucket_agreement_sum / denom).detach().cpu().item()
            ),
            "routing/teacher_ce_epoch": float(
                (self.teacher_ce_sum / denom).detach().cpu().item()
            ),
            "routing/teacher_alpha_epoch": float(
                (self.teacher_alpha_sum / denom).detach().cpu().item()
            ),
            "routing/load_max_epoch": float(avg_fraction_routed.max().item()),
            "routing/load_min_epoch": float(avg_fraction_routed.min().item()),
            "routing/load_std_epoch": load_std,
            "routing/load_cv_epoch": load_cv,
            "routing/experts_used_epoch": experts_used,
            "routing/dead_experts_epoch": dead_experts,
            "routing/load_hist_epoch": wandb.Histogram(avg_fraction_routed.tolist()),
            "routing/prob_hist_epoch": wandb.Histogram(avg_gate_prob.tolist()),
        }

        table = wandb.Table(columns=["expert", "load", "prob"])
        for index, (load, prob) in enumerate(zip(avg_fraction_routed, avg_gate_prob)):
            load_value = float(load.item())
            prob_value = float(prob.item())
            metrics[f"routing/load_epoch/expert_{index:02d}"] = load_value
            metrics[f"routing/prob_epoch/expert_{index:02d}"] = prob_value
            table.add_data(index, load_value, prob_value)
        metrics["routing/expert_load_bar"] = wandb.plot.bar(
            table,
            "expert",
            "load",
            title="Routing load by expert",
        )

        summary = (
            "routing aux={aux:.6f} floor={floor:.6f} cap={cap:.6f} "
            "probe={probe:.6f}@{probe_weight:.3f} "
            "z={z:.6f} entropy={entropy:.3f} top1={top1:.3f} "
            "agree={agree:.3f} target_prob={target_prob:.3f} teacher_ce={teacher_ce:.3f} "
            "teacher_alpha={teacher_alpha:.3f} "
            "load[min,max]=[{load_min:.4f},{load_max:.4f}] used={used}/{total}"
        ).format(
            aux=metrics["routing/aux_loss_epoch"],
            floor=metrics["routing/load_floor_loss_epoch"],
            cap=metrics["routing/load_cap_loss_epoch"],
            probe=metrics["routing/probe_loss_epoch"],
            probe_weight=metrics["routing/probe_weight_epoch"],
            z=metrics["routing/z_loss_epoch"],
            entropy=metrics["routing/entropy_epoch"],
            top1=metrics["routing/top1_prob_epoch"],
            agree=metrics["routing/bucket_agreement_epoch"],
            target_prob=metrics["routing/target_prob_epoch"],
            teacher_ce=metrics["routing/teacher_ce_epoch"],
            teacher_alpha=metrics["routing/teacher_alpha_epoch"],
            load_min=metrics["routing/load_min_epoch"],
            load_max=metrics["routing/load_max_epoch"],
            used=experts_used,
            total=avg_fraction_routed.numel(),
        )
        return metrics, summary


class TrainingLogger:
    def __init__(
        self,
        args: BaseTrainingConfig,
        runtime: DistributedRuntime,
        local_batch_size: int,
    ):
        self.args = args
        self.runtime = runtime
        self.local_batch_size = local_batch_size
        self.run = None
        self.checkpoint_dir: str | None = None
        if runtime.is_main_process:
            self.run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    **asdict(args),
                    "world_size": runtime.world_size,
                },
            )
            self.checkpoint_dir = os.path.join(self.run.dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._loader_start: dict[str, float | int] | None = None
        self._routing_acc: RoutingMetricsAccumulator | None = None

    def start_epoch(self, loader_start: dict[str, float | int] | None) -> None:
        self._loader_start = loader_start
        self._routing_acc = None

    def on_batch(
        self,
        log_dict: dict[str, torch.Tensor],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self._routing_acc is None:
            self._routing_acc = RoutingMetricsAccumulator.maybe_create(log_dict)
        if self._routing_acc is not None:
            self._routing_acc.update(log_dict)
        return log_dict.get("routing/router_loss", reference_tensor.new_zeros(()))

    def log_step(
        self,
        *,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        loss: torch.Tensor,
    ) -> None:
        if not self.runtime.is_main_process:
            return
        if batch_idx % max(1, num_batches // 4) != 0 and batch_idx != num_batches - 1:
            return
        print(
            f"epoch={epoch:03d} step={batch_idx + 1}/{num_batches} loss={loss.detach().item():.6f}",
            flush=True,
        )

    def finish_epoch(
        self,
        *,
        epoch: int,
        epoch_loss: float,
        lr: float,
        elapsed: float,
        processed_batches: int,
        reduced_batches: int,
        global_step: int,
        loader_end: dict[str, float | int] | None,
    ) -> None:
        if self._routing_acc is not None:
            self._routing_acc.synchronize(self.runtime)
        if not self.runtime.is_main_process:
            return

        global_batch_size = self.args.batch_size
        metrics: dict[str, Any] = {
            "train/epoch": epoch,
            "train/loss_epoch": epoch_loss,
            "train/lr": lr,
            "train/epoch_time_sec": elapsed,
            "train/it_per_sec": processed_batches / elapsed,
            "train/positions_per_sec": (processed_batches * global_batch_size)
            / elapsed,
            "train/world_size": self.runtime.world_size,
            "train/local_batch_size": self.local_batch_size,
            "train/global_batch_size": global_batch_size,
        }
        metrics.update(_build_loader_metrics(self._loader_start, loader_end, elapsed))

        routing_summary = None
        if self._routing_acc is not None:
            routing_metrics, routing_summary = self._routing_acc.finalize(
                reduced_batches
            )
            metrics.update(routing_metrics)

        print(
            f"epoch={epoch:03d} done loss={epoch_loss:.6f} lr={lr:.8g} time={elapsed:.1f}s it/s={processed_batches / elapsed:.1f} pos/s={(processed_batches * global_batch_size) / elapsed:.0f}",
            flush=True,
        )
        if routing_summary is not None:
            print(routing_summary, flush=True)
        _print_loader_summary(metrics, loader_end)
        wandb.log(metrics, step=global_step)

    def save_periodic_checkpoint(
        self,
        *,
        epoch: int,
        save_fn,
    ) -> None:
        if not self.runtime.is_main_process or self.checkpoint_dir is None:
            return
        epoch_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        last_path = os.path.join(self.checkpoint_dir, "last.pt")
        save_fn(epoch_path)
        shutil.copyfile(epoch_path, last_path)
        print(f"saved checkpoints {epoch_path} -> {last_path}", flush=True)

    def save_final_checkpoint(self, save_fn) -> None:
        if not self.runtime.is_main_process or self.checkpoint_dir is None:
            return
        final_path = os.path.join(self.checkpoint_dir, "final.pt")
        save_fn(final_path)
        print(f"saved final checkpoint {final_path}", flush=True)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
