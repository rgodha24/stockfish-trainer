from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import cast

import torch
import tyro
from tyro.conf import Positional

from src.data import (
    DataloaderSkipConfig,
    iter_device_batches,
    make_sparse_batch_dataset,
)
from src.model import ModelConfig, NNUEModel, QuantizationConfig
from src.model.modules import LayerStacks


@dataclass(kw_only=True)
class OracleHeadroomConfig:
    checkpoint: Positional[str]
    datasets: Positional[tuple[str, ...]] = ()
    batch_size: int = 16384
    max_batches: int = 64
    device: str = "auto"
    loader_threads: int = -1
    shuffle_buffer_entries: int = 65536
    pin_memory: bool = True
    filtered: bool | None = None
    wld_filtered: bool | None = None
    random_fen_skipping: int | None = None
    early_fen_skipping: int | None = None
    simple_eval_skipping: int | None = None
    param_index: int | None = None
    pc_y1: float | None = None
    pc_y2: float | None = None
    pc_y3: float | None = None
    output_json: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if self.max_batches <= 0:
            raise ValueError("`max_batches` must be positive.")
        if self.loader_threads == 0:
            raise ValueError("`loader_threads` must be positive or -1 for auto.")
        if self.shuffle_buffer_entries < 0:
            raise ValueError("`shuffle_buffer_entries` must be non-negative.")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _pick_override(
    override: bool | int | float | None,
    ckpt_cfg: dict,
    key: str,
    default: bool | int | float,
):
    if override is not None:
        return override
    return ckpt_cfg.get(key, default)


def _build_skip_config(
    args: OracleHeadroomConfig, ckpt_cfg: dict
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


def _load_layerstacks_checkpoint(path: str) -> tuple[NNUEModel, dict, int]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "model_state_dict" not in raw:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    ckpt_cfg = dict(raw.get("training_config", {}))
    raw_stacks = ckpt_cfg.get("stacks")
    if raw_stacks is None:
        legacy_stacks = ckpt_cfg.get("layer_stacks")
        if isinstance(legacy_stacks, bool):
            stack_mode = "layer" if legacy_stacks else "none"
        else:
            stack_mode = legacy_stacks or "layer"
    elif isinstance(raw_stacks, bool):
        stack_mode = "layer" if raw_stacks else "none"
    else:
        stack_mode = raw_stacks

    model_cfg = ModelConfig(
        L1=int(ckpt_cfg.get("l1", 1024)),
        L2=int(ckpt_cfg.get("l2", 31)),
        L3=int(ckpt_cfg.get("l3", 32)),
        stacks=stack_mode,
        num_experts=int(ckpt_cfg.get("num_experts", 8)),
        aux_loss_alpha=float(ckpt_cfg.get("aux_loss_alpha", 0.0)),
        z_loss_alpha=float(ckpt_cfg.get("z_loss_alpha", 0.0)),
        router_teacher_alpha=float(ckpt_cfg.get("router_teacher_alpha", 0.0)),
        router_teacher_anneal_epochs=int(
            ckpt_cfg.get("router_teacher_anneal_epochs", 0)
        ),
    )
    model = NNUEModel(
        ckpt_cfg.get("features", "Full_Threats+HalfKAv2_hm^"),
        model_cfg,
        QuantizationConfig(),
    )
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    if not isinstance(model.layer_stacks, LayerStacks):
        raise ValueError(
            "Oracle headroom script currently expects a layerstacks checkpoint."
        )
    return model, ckpt_cfg, int(raw.get("epoch", -1))


def _resolve_datasets(
    args: OracleHeadroomConfig, ckpt_cfg: dict
) -> tuple[list[str], list[str]]:
    requested = args.datasets if args.datasets else tuple(ckpt_cfg.get("datasets", ()))
    resolved: list[str] = []
    skipped: list[str] = []
    for path in requested:
        if os.path.isfile(path) and path.endswith(".binpack"):
            resolved.append(path)
        else:
            skipped.append(path)
    resolved = sorted(set(resolved))
    if not resolved:
        raise ValueError("No readable .binpack files were found for evaluation.")
    return resolved, skipped


def _layer_params(layer) -> tuple[torch.Tensor, torch.Tensor]:
    weight = layer.linear.weight
    bias = layer.linear.bias
    factorized = getattr(layer, "factorized_linear", None)

    if factorized is not None:
        if weight.shape[0] == layer.out_features * layer.count:
            weight = weight + factorized.weight.repeat(layer.count, 1)
            bias = bias + factorized.bias.repeat(layer.count)
        elif weight.shape[0] == layer.out_features:
            weight = weight + factorized.weight
            bias = bias + factorized.bias

    if weight.shape[0] == layer.out_features * layer.count:
        weight = weight.view(layer.count, layer.out_features, layer.in_features)
        bias = bias.view(layer.count, layer.out_features)
    elif weight.shape[0] == layer.out_features:
        weight = weight.unsqueeze(0).expand(layer.count, -1, -1).contiguous()
        bias = bias.unsqueeze(0).expand(layer.count, -1).contiguous()
    else:
        raise ValueError("Unexpected layer shape while expanding experts.")

    return weight.contiguous(), bias.contiguous()


def _shared_forward(
    model: NNUEModel, batch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    wp, bp = model.input(white_indices, white_values, black_indices, black_values)
    w, wpsqt = torch.split(wp, model.L1, dim=1)
    b, bpsqt = torch.split(bp, model.L1, dim=1)

    l0 = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    l0 = torch.clamp(l0, 0.0, 1.0)

    pairwise_chunk_size = model.L1 // 2
    l0 = l0.reshape(l0.shape[0], 2, 2, pairwise_chunk_size)
    l0 = (l0[:, :, 0, :] * l0[:, :, 1, :]).reshape(l0.shape[0], -1) * (127 / 128)

    psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
    wpsqt = wpsqt.gather(1, psqt_indices_unsq)
    bpsqt = bpsqt.gather(1, psqt_indices_unsq)
    psqt = (wpsqt - bpsqt) * (us - 0.5)
    return (
        l0,
        psqt.reshape(-1, 1),
        outcome.reshape(-1),
        score.reshape(-1),
        layer_stack_indices.reshape(-1),
    )


def _all_expert_scores(
    model: NNUEModel, l0: torch.Tensor, psqt: torch.Tensor
) -> torch.Tensor:
    stacks = cast(LayerStacks, model.layer_stacks)
    l1_w, l1_b = _layer_params(stacks.l1)
    l1c = torch.einsum("bi,eoi->beo", l0, l1_w) + l1_b

    l1x, l1_out = l1c.split(model.L2, dim=2)
    l1x = torch.clamp(
        torch.cat([l1x.square() * (255 / 256), l1x], dim=2),
        0.0,
        1.0,
    )

    l2_w, l2_b = _layer_params(stacks.l2)
    l2c = torch.einsum("bei,eoi->beo", l1x, l2_w) + l2_b
    l2x = torch.clamp(l2c, 0.0, 1.0)

    out_w, out_b = _layer_params(stacks.output)
    l3c = torch.einsum("bei,eoi->beo", l2x, out_w) + out_b
    raw_scores = l3c.squeeze(-1) + l1_out.squeeze(-1) + psqt
    return raw_scores * model.quantization.nnue2score


def _loss_matrix(
    scorenet: torch.Tensor,
    outcome: torch.Tensor,
    score: torch.Tensor,
    ckpt_cfg: dict,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lambda_default = float(ckpt_cfg.get("lambda_", 1.0))
    start_lambda = float(ckpt_cfg.get("start_lambda", lambda_default) or lambda_default)
    end_lambda = float(ckpt_cfg.get("end_lambda", lambda_default) or lambda_default)
    max_epochs = max(int(ckpt_cfg.get("max_epochs", 1)), 1)

    in_offset = float(ckpt_cfg.get("in_offset", 270.0))
    out_offset = float(ckpt_cfg.get("out_offset", 270.0))
    in_scaling = float(ckpt_cfg.get("in_scaling", 340.0))
    out_scaling = float(ckpt_cfg.get("out_scaling", 380.0))
    pow_exp = float(ckpt_cfg.get("pow_exp", 2.5))
    qp_asymmetry = float(ckpt_cfg.get("qp_asymmetry", 0.0))
    w1 = float(ckpt_cfg.get("w1", 0.0))
    w2 = float(ckpt_cfg.get("w2", 0.5))

    q = (scorenet - in_offset) / in_scaling
    qm = (-scorenet - in_offset) / in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

    score = score.reshape(-1)
    outcome = outcome.reshape(-1)
    s = (score - out_offset) / out_scaling
    sm = (-score - out_offset) / out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

    actual_lambda = start_lambda + (end_lambda - start_lambda) * (epoch / max_epochs)
    pt = pf * actual_lambda + outcome * (1.0 - actual_lambda)

    loss = torch.pow(torch.abs(pt.unsqueeze(1) - qf), pow_exp)
    if qp_asymmetry != 0.0:
        loss = loss * (((qf > pt.unsqueeze(1)).to(loss.dtype) * qp_asymmetry) + 1.0)

    weights = 1.0 + (2.0**w1 - 1.0) * torch.pow(
        (pf - 0.5).square() * pf * (1.0 - pf),
        w2,
    )
    return loss, weights


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    quantiles = torch.quantile(
        values, torch.tensor([0.5, 0.9, 0.99], dtype=values.dtype)
    )
    return {
        "mean": float(values.mean().item()),
        "median": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "max": float(values.max().item()),
    }


def main() -> None:
    args = tyro.cli(OracleHeadroomConfig)
    device = _resolve_device(args.device)
    model, ckpt_cfg, checkpoint_epoch = _load_layerstacks_checkpoint(args.checkpoint)
    model = model.to(device)
    layer_stacks = cast(LayerStacks, model.layer_stacks)

    datasets, skipped_inputs = _resolve_datasets(args, ckpt_cfg)
    skip_cfg = _build_skip_config(args, ckpt_cfg)
    loader = make_sparse_batch_dataset(
        feature_set=ckpt_cfg.get("features", model.feature_name),
        filenames=datasets,
        batch_size=args.batch_size,
        cyclic=False,
        loader_threads=args.loader_threads,
        config=skip_cfg,
        shuffle_buffer_entries=args.shuffle_buffer_entries,
        pin_memory=args.pin_memory,
    )

    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"device={device}", flush=True)
    print(f"checkpoint_epoch={checkpoint_epoch}", flush=True)
    print(f"datasets={len(datasets)}", flush=True)
    for path in datasets:
        print(f"  data={path}", flush=True)
    if skipped_inputs:
        print(f"skipped_inputs={len(skipped_inputs)}", flush=True)
        for path in skipped_inputs:
            print(f"  skipped={path}", flush=True)

    num_experts = int(layer_stacks.count)
    forced_loss_num = torch.zeros(num_experts, dtype=torch.float64)
    bucket_hist = torch.zeros(num_experts, dtype=torch.int64)
    oracle_hist = torch.zeros(num_experts, dtype=torch.int64)
    margin_chunks: list[torch.Tensor] = []
    improvement_chunks: list[torch.Tensor] = []

    processed_batches = 0
    processed_positions = 0
    weight_sum = 0.0
    bucket_loss_num = 0.0
    oracle_loss_num = 0.0
    agreement_count = 0
    exhausted = False

    with torch.inference_mode():
        for batch in iter_device_batches(loader, device, queue_size_limit=16):
            l0, psqt, outcome, score, layer_stack_indices = _shared_forward(
                model, batch
            )
            expert_scores = _all_expert_scores(model, l0, psqt)
            loss_matrix, weights = _loss_matrix(
                expert_scores,
                outcome,
                score,
                ckpt_cfg,
                checkpoint_epoch,
            )

            bucket_loss = loss_matrix.gather(
                1, layer_stack_indices.unsqueeze(1)
            ).squeeze(1)
            oracle_loss, oracle_indices = loss_matrix.min(dim=1)
            second_best = torch.topk(loss_matrix, k=2, dim=1, largest=False).values[
                :, 1
            ]
            margin = second_best - oracle_loss
            improvement = bucket_loss - oracle_loss

            weights64 = weights.to(dtype=torch.float64)
            bucket_loss_num += float(
                (bucket_loss.to(dtype=torch.float64) * weights64).sum().item()
            )
            oracle_loss_num += float(
                (oracle_loss.to(dtype=torch.float64) * weights64).sum().item()
            )
            forced_loss_num.add_(
                (loss_matrix.to(dtype=torch.float64) * weights64.unsqueeze(1))
                .sum(dim=0)
                .cpu()
            )
            weight_sum += float(weights64.sum().item())

            bucket_hist.add_(
                torch.bincount(layer_stack_indices.cpu(), minlength=num_experts)
            )
            oracle_hist.add_(
                torch.bincount(oracle_indices.cpu(), minlength=num_experts)
            )
            agreement_count += int((oracle_indices == layer_stack_indices).sum().item())
            margin_chunks.append(margin.cpu())
            improvement_chunks.append(improvement.cpu())

            processed_batches += 1
            processed_positions += int(layer_stack_indices.numel())
            if processed_batches % 8 == 0 or processed_batches == args.max_batches:
                print(
                    f"processed_batches={processed_batches} positions={processed_positions}",
                    flush=True,
                )
            if processed_batches >= args.max_batches:
                break
        else:
            exhausted = True

    if processed_batches == 0:
        raise RuntimeError("The evaluation loader produced no batches.")

    margin_values = torch.cat(margin_chunks) if margin_chunks else torch.empty(0)
    improvement_values = (
        torch.cat(improvement_chunks) if improvement_chunks else torch.empty(0)
    )

    forced_losses = (forced_loss_num / max(weight_sum, 1e-12)).tolist()
    bucket_loss = bucket_loss_num / max(weight_sum, 1e-12)
    oracle_loss = oracle_loss_num / max(weight_sum, 1e-12)
    absolute_gain = bucket_loss - oracle_loss
    relative_gain = absolute_gain / bucket_loss if bucket_loss != 0.0 else 0.0

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "features": ckpt_cfg.get("features", model.feature_name),
        "datasets": datasets,
        "skipped_inputs": skipped_inputs,
        "processed_batches": processed_batches,
        "processed_positions": processed_positions,
        "loader_exhausted": exhausted,
        "piece_count_loss": bucket_loss,
        "oracle_loss": oracle_loss,
        "absolute_gain": absolute_gain,
        "relative_gain": relative_gain,
        "oracle_piece_count_agreement": agreement_count / processed_positions,
        "positions_with_oracle_gain": float(
            (improvement_values > 1e-12).float().mean().item()
        ),
        "mean_oracle_gain_per_position": float(improvement_values.mean().item()),
        "margin_stats": _tensor_stats(margin_values),
        "improvement_stats": _tensor_stats(improvement_values),
        "piece_count_hist": bucket_hist.tolist(),
        "oracle_hist": oracle_hist.tolist(),
        "forced_expert_losses": forced_losses,
    }

    print(f"piece_count_loss={result['piece_count_loss']:.8f}", flush=True)
    print(f"oracle_loss={result['oracle_loss']:.8f}", flush=True)
    print(
        f"absolute_gain={result['absolute_gain']:.8f} relative_gain={result['relative_gain'] * 100.0:.2f}%",
        flush=True,
    )
    print(
        f"oracle_piece_count_agreement={result['oracle_piece_count_agreement'] * 100.0:.2f}%",
        flush=True,
    )
    print(
        f"positions_with_oracle_gain={result['positions_with_oracle_gain'] * 100.0:.2f}%",
        flush=True,
    )
    print(f"piece_count_hist={result['piece_count_hist']}", flush=True)
    print(f"oracle_hist={result['oracle_hist']}", flush=True)
    print(
        "forced_expert_losses="
        + ", ".join(f"e{i}={loss:.8f}" for i, loss in enumerate(forced_losses)),
        flush=True,
    )
    print(
        "margin_stats="
        + ", ".join(
            f"{key}={value:.8f}" for key, value in result["margin_stats"].items()
        ),
        flush=True,
    )
    print(
        "improvement_stats="
        + ", ".join(
            f"{key}={value:.8f}" for key, value in result["improvement_stats"].items()
        ),
        flush=True,
    )

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote_json={args.output_json}", flush=True)


if __name__ == "__main__":
    main()
