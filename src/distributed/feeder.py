from __future__ import annotations

from typing import Any

import ray
import rust

from src.data import DataloaderSkipConfig


@ray.remote(max_restarts=0)
class BatchFeeder:
    def __init__(
        self,
        *,
        filenames: list[str],
        feature_set: str,
        batch_size: int,
        encode_threads: int,
        total_threads: int,
        decode_threads: int | None,
        shuffle_buffer_entries: int,
        seed: int | None,
        cyclic: bool,
        skip_cfg: DataloaderSkipConfig,
        rank: int,
        world_size: int,
    ) -> None:
        self.stream = rust.BatchStream(
            feature_set,
            list(filenames),
            int(batch_size),
            total_threads=total_threads,
            decode_threads=decode_threads,
            encode_threads=encode_threads,
            slab_count=None,
            shuffle_buffer_entries=shuffle_buffer_entries,
            seed=seed,
            cyclic=cyclic,
            filtered=skip_cfg.filtered,
            random_fen_skipping=int(skip_cfg.random_fen_skipping),
            wld_filtered=skip_cfg.wld_filtered,
            early_fen_skipping=int(skip_cfg.early_fen_skipping),
            simple_eval_skipping=int(skip_cfg.simple_eval_skipping),
            param_index=int(skip_cfg.param_index),
            pc_y1=float(skip_cfg.pc_y1),
            pc_y2=float(skip_cfg.pc_y2),
            pc_y3=float(skip_cfg.pc_y3),
            rank=rank,
            world_size=world_size,
        )
        self.done = False
        self.returned_batches = 0
        self.returned_entries = 0

    def next_batch(self) -> tuple[dict | None, int]:
        if self.done:
            return None, 0

        encoded = self.stream.next_batch()
        if encoded is None:
            self.done = True
            return None, 0

        entries = int(encoded["size"])
        self.returned_batches += 1
        self.returned_entries += entries
        return encoded, entries

    def stats(self) -> dict[str, Any]:
        return {
            "stream": self.stream.stats(),
            "returned_batches": self.returned_batches,
            "returned_entries": self.returned_entries,
            "done": self.done,
        }

    def close(self) -> None:
        self.stream.close()
        self.done = True
