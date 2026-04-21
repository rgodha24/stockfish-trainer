from __future__ import annotations

from dataclasses import dataclass

from tyro.conf import Positional

from src.data import DataloaderSkipConfig


@dataclass(kw_only=True)
class DistributedLoaderConfig:
    datasets: Positional[tuple[str, ...]] = ()
    feature_set: str = "Full_Threats+HalfKAv2_hm^"
    batch_size: int = 65536
    feeder_count: int = 10
    cyclic: bool = True

    loader_threads: int = 16
    decode_threads: int = -1
    encode_threads: int = 4
    shuffle_buffer_entries: int = 65536

    filtered: bool = True
    wld_filtered: bool = True
    random_fen_skipping: int = 6
    early_fen_skipping: int = 12
    simple_eval_skipping: int = -1
    param_index: int = 0
    pc_y1: float = 1.0
    pc_y2: float = 2.0
    pc_y3: float = 1.0

    seed: int | None = None
    ray_address: str | None = None
    ray_namespace: str = "stockfish-trainer"
    log_to_driver: bool = True
    feeder_cpus: float = 16.0

    inflight_per_feeder: int = 4
    report_interval_sec: float = 5.0
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("Argument `datasets` is required.")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if self.feeder_count <= 0:
            raise ValueError("`feeder_count` must be positive.")
        if self.loader_threads == 0:
            raise ValueError("`loader_threads` must be positive or -1 for auto.")
        if self.decode_threads == 0:
            raise ValueError("`decode_threads` must be positive or -1 for auto.")
        if self.encode_threads <= 0:
            raise ValueError("`encode_threads` must be positive.")
        if self.inflight_per_feeder <= 0:
            raise ValueError("`inflight_per_feeder` must be positive.")
        if self.feeder_cpus <= 0:
            raise ValueError("`feeder_cpus` must be positive.")
        if self.report_interval_sec <= 0:
            raise ValueError("`report_interval_sec` must be positive.")
        if self.shuffle_buffer_entries < 0:
            raise ValueError("`shuffle_buffer_entries` must be non-negative.")

    def skip_config(self) -> DataloaderSkipConfig:
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
