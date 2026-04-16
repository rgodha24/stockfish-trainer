from __future__ import annotations

import time
from dataclasses import dataclass

import tyro

from src.distributed.config import DistributedLoaderConfig
from src.distributed.pipeline import RayBatchStream


@dataclass(kw_only=True)
class SmokeConfig(DistributedLoaderConfig):
    target_batches: int = 2000
    max_seconds: float = 0.0
    pin_memory: bool = False
    report_interval_sec: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.target_batches < 0:
            raise ValueError("`target_batches` must be non-negative.")
        if self.max_seconds < 0:
            raise ValueError("`max_seconds` must be non-negative.")
        if self.target_batches == 0 and self.max_seconds == 0:
            raise ValueError(
                "Set at least one stop condition: `target_batches > 0` or `max_seconds > 0`."
            )


def main() -> None:
    cfg = tyro.cli(SmokeConfig)
    loader = RayBatchStream(cfg)
    start_time = time.perf_counter()

    try:
        batches = 0
        while True:
            if cfg.target_batches > 0 and batches >= cfg.target_batches:
                break
            if (
                cfg.max_seconds > 0
                and (time.perf_counter() - start_time) >= cfg.max_seconds
            ):
                break

            try:
                batch = next(loader)
            except StopIteration:
                break
            _ = batch[0].shape[0]
            batches += 1
    finally:
        loader.close(emit_summary=True)


if __name__ == "__main__":
    main()
