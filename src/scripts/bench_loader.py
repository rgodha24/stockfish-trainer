from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, Iterator

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tyro
from tyro.conf import Positional

from src.data import (
    Batch,
    DataloaderSkipConfig,
    SparseBatchDataset,
    make_sparse_batch_dataset,
)


@dataclass(kw_only=True)
class BenchLoaderConfig:
    datasets: Positional[tuple[str, ...]] = ()
    batch_size: int = 65536
    features: str = "Full_Threats+HalfKAv2_hm^"
    loader_threads: int = -1
    decode_threads: int = -1
    encode_threads: int = -1
    shuffle_buffer_entries: int = 65536
    pin_memory: bool = True
    filtered: bool = True
    wld_filtered: bool = True
    random_fen_skipping: int = 0
    early_fen_skipping: int = -1
    simple_eval_skipping: int = -1
    param_index: int = 0
    pc_y1: float = 1.0
    pc_y2: float = 2.0
    pc_y3: float = 1.0
    warmup_batches: int = 8
    measure_batches: int = 64

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("Argument `datasets` is required.")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if self.loader_threads == 0:
            raise ValueError("`loader_threads` must be positive or -1 for auto.")
        if self.shuffle_buffer_entries < 0:
            raise ValueError("`shuffle_buffer_entries` must be non-negative.")
        if self.warmup_batches < 0:
            raise ValueError("`warmup_batches` must be non-negative.")
        if self.measure_batches <= 0:
            raise ValueError("`measure_batches` must be positive.")

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


def resolve_binpack_paths(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    resolved = []
    ignored = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".binpack"):
            resolved.append(path)
        elif os.path.isdir(path):
            ignored.append(path)
        elif path.endswith(".chunks"):
            ignored.append(path)
        else:
            ignored.append(path)

    if not resolved:
        raise ValueError("No .binpack files found in the provided dataset arguments.")

    return sorted(set(resolved)), ignored


def make_loader(cfg: BenchLoaderConfig, files: list[str]) -> SparseBatchDataset:
    dataset = make_sparse_batch_dataset(
        feature_set=cfg.features,
        filenames=files,
        batch_size=cfg.batch_size,
        cyclic=True,
        loader_threads=cfg.loader_threads,
        config=cfg.loader_skip_config(),
        shuffle_buffer_entries=cfg.shuffle_buffer_entries,
        pin_memory=cfg.pin_memory,
    )
    from src.data.loader import _auto_thread_counts, _default_slab_count

    if cfg.decode_threads > 0 or cfg.encode_threads > 0:
        total = dataset.total_threads
        auto_dec, auto_enc = _auto_thread_counts(total)
        enc = cfg.encode_threads if cfg.encode_threads > 0 else auto_enc
        dec = cfg.decode_threads if cfg.decode_threads > 0 else (total - enc)
        dataset.decode_threads = dec
        dataset.encode_threads = enc
        dataset.slab_count = _default_slab_count(enc)
    return dataset


def consume_batches(iterator: Iterator[Batch], batch_count: int) -> tuple[int, int]:
    batches = 0
    positions = 0
    while batches < batch_count:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        positions += int(batch[0].shape[0])
        batches += 1
    return batches, positions


def main() -> None:
    cfg = tyro.cli(BenchLoaderConfig)
    files, ignored = resolve_binpack_paths(cfg.datasets)
    loader = make_loader(cfg, files)
    iterator = iter(loader)

    warmup_done, warmup_positions = consume_batches(iterator, cfg.warmup_batches)
    start = time.perf_counter()
    measured_batches, measured_positions = consume_batches(
        iterator, cfg.measure_batches
    )
    elapsed = max(time.perf_counter() - start, 1e-9)

    print(
        "cpu_only=True files={} total_threads={} decode_threads={} encode_threads={} batch_size={} features={}".format(
            len(files),
            loader.total_threads,
            loader.decode_threads,
            loader.encode_threads,
            cfg.batch_size,
            cfg.features,
        ),
        flush=True,
    )
    if ignored:
        print(f"ignored_inputs={len(ignored)}", flush=True)
    print(
        "warmup_batches={}/{} warmup_positions={}".format(
            warmup_done,
            cfg.warmup_batches,
            warmup_positions,
        ),
        flush=True,
    )
    print(
        "measured_batches={}/{} positions={} elapsed_sec={:.3f}".format(
            measured_batches,
            cfg.measure_batches,
            measured_positions,
            elapsed,
        ),
        flush=True,
    )
    print(
        "pos/s={:.0f} batch_ms={:.2f}".format(
            measured_positions / elapsed,
            elapsed * 1000.0 / max(measured_batches, 1),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
