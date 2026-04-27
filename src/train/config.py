from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import tyro
from tyro.conf import Positional

from src.data import DataloaderSkipConfig
from src.model.modules.config import Stacks


@dataclass(kw_only=True)
class BaseTrainingConfig:
    datasets: Positional[tuple[str, ...]] = ()
    max_epochs: int = 800
    batch_size: int = 16384
    features: str = "Full_Threats+HalfKAv2_hm^"

    l1: int = 1024
    l2: int = 31
    l3: int = 32
    stacks: Stacks = "layer"
    num_experts: int = 8
    aux_loss_alpha: float = 0.001
    z_loss_alpha: float = 0.0
    router_load_floor: float = 0.0
    router_load_cap: float = 1.0
    router_lr_multiplier: float = 1.0
    router_teacher_alpha: float = 0.0
    router_teacher_anneal_epochs: int = 0

    lr: float = 8.75e-4
    gamma: float = 0.992

    in_offset: float = 270.0
    out_offset: float = 270.0
    in_scaling: float = 340.0
    out_scaling: float = 380.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0
    w1: float = 0.0
    w2: float = 0.5
    lambda_: Annotated[float, tyro.conf.arg(name="lambda")] = 1.0
    start_lambda: float | None = None
    end_lambda: float | None = None

    random_fen_skipping: int = 0
    early_fen_skipping: int = -1
    simple_eval_skipping: int = -1
    filtered: bool = True
    wld_filtered: bool = True
    param_index: int = 0
    pc_y1: float = 1.0
    pc_y2: float = 2.0
    pc_y3: float = 1.0

    epoch_size: int = 100_000_000
    loader_threads: int = -1
    shuffle_buffer_entries: int = 65536
    pin_memory: bool = True
    data_loader_queue_size: int = 16

    seed: int = 42
    default_root_dir: str = "logs"
    checkpoint_every_epochs: int = 10
    wandb_project: str = "stockfish-trainer"
    wandb_run_name: str | None = None
    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    compile_backend: Literal["inductor", "cudagraphs"] = "inductor"
    resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("Argument `datasets` is required.")
        if self.max_epochs <= 0 or self.epoch_size <= 0 or self.batch_size <= 0:
            raise ValueError(
                "Arguments `max_epochs`, `epoch_size` and `batch_size` must be positive."
            )
        if self.shuffle_buffer_entries < 0:
            raise ValueError("`shuffle_buffer_entries` must be non-negative.")
        if self.data_loader_queue_size <= 0:
            raise ValueError("`data_loader_queue_size` must be positive.")
        if self.checkpoint_every_epochs < 0:
            raise ValueError("`checkpoint_every_epochs` must be non-negative.")
        if self.l1 % 2 != 0:
            raise ValueError("`l1` must be even.")
        if self.num_experts <= 0:
            raise ValueError("`num_experts` must be positive.")
        if self.aux_loss_alpha < 0.0:
            raise ValueError("`aux_loss_alpha` must be non-negative.")
        if self.z_loss_alpha < 0.0:
            raise ValueError("`z_loss_alpha` must be non-negative.")
        if not 0.0 <= self.router_load_floor <= 1.0:
            raise ValueError("`router_load_floor` must be in [0, 1].")
        if not 0.0 <= self.router_load_cap <= 1.0:
            raise ValueError("`router_load_cap` must be in [0, 1].")
        if self.router_load_floor > self.router_load_cap:
            raise ValueError("`router_load_floor` must be <= `router_load_cap`.")
        if self.router_teacher_alpha < 0.0:
            raise ValueError("`router_teacher_alpha` must be non-negative.")
        if self.router_teacher_anneal_epochs < 0:
            raise ValueError("`router_teacher_anneal_epochs` must be non-negative.")

    def loader_skip_config(self) -> DataloaderSkipConfig:
        return DataloaderSkipConfig(
            filtered=self.filtered,
            wld_filtered=self.wld_filtered,
            random_fen_skipping=self.random_fen_skipping,
            early_fen_skipping=self.early_fen_skipping,
            simple_eval_skipping=self.simple_eval_skipping,
            param_index=self.param_index,
            pc_y1=self.pc_y1,
            pc_y2=self.pc_y2,
            pc_y3=self.pc_y3,
        )


@dataclass(kw_only=True)
class SingleNodeTrainingConfig(BaseTrainingConfig):
    pass


@dataclass(kw_only=True)
class MultiNodeTrainingConfig(BaseTrainingConfig):
    batch_size: int = 65536
    loader_threads: int = 16
    encode_threads: int = 6
    random_fen_skipping: int = 0
    early_fen_skipping: int = 12
    ray_address: str | None = None
    ray_namespace: str = "stockfish-trainer"
    ray_log_to_driver: bool = True

    feeder_count: int = 10
    feeder_cpus: float = 16.0
    decode_threads: int = -1
    inflight_per_feeder: int = 4
    report_interval_sec: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.encode_threads <= 0:
            raise ValueError("`encode_threads` must be positive.")
        if self.feeder_count <= 0:
            raise ValueError("`feeder_count` must be positive.")
        if self.feeder_cpus <= 0:
            raise ValueError("`feeder_cpus` must be positive.")
        if self.decode_threads == 0:
            raise ValueError("`decode_threads` must be positive or -1 for auto.")
        if self.inflight_per_feeder <= 0:
            raise ValueError("`inflight_per_feeder` must be positive.")
        if self.report_interval_sec <= 0:
            raise ValueError("`report_interval_sec` must be positive.")

    def distributed_loader_config(self):
        from src.distributed import DistributedLoaderConfig

        return DistributedLoaderConfig(
            datasets=self.datasets,
            feature_set=self.features,
            batch_size=self.batch_size,
            feeder_count=self.feeder_count,
            cyclic=True,
            loader_threads=self.loader_threads,
            decode_threads=self.decode_threads,
            encode_threads=self.encode_threads,
            shuffle_buffer_entries=self.shuffle_buffer_entries,
            filtered=self.filtered,
            wld_filtered=self.wld_filtered,
            random_fen_skipping=self.random_fen_skipping,
            early_fen_skipping=self.early_fen_skipping,
            simple_eval_skipping=self.simple_eval_skipping,
            param_index=self.param_index,
            pc_y1=self.pc_y1,
            pc_y2=self.pc_y2,
            pc_y3=self.pc_y3,
            seed=self.seed,
            ray_address=self.ray_address,
            ray_namespace=self.ray_namespace,
            log_to_driver=self.ray_log_to_driver,
            feeder_cpus=self.feeder_cpus,
            inflight_per_feeder=self.inflight_per_feeder,
            report_interval_sec=self.report_interval_sec,
            pin_memory=self.pin_memory,
        )


TrainingConfig = SingleNodeTrainingConfig
