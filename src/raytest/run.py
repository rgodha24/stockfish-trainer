from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import ray
import rust
import torch
import tyro
from tyro.conf import Positional

from src.data import DataloaderSkipConfig

PACKED_ENTRY_BYTES = 32


@dataclass(kw_only=True)
class RayTestConfig:
    datasets: Positional[tuple[str, ...]] = ()
    feature_set: str = "Full_Threats+HalfKAv2_hm^"
    batch_size: int = 16384
    chunk_entries: int = 8192
    feeder_count: int = 4
    target_batches: int = 200
    max_seconds: float = 0.0
    cyclic: bool = True

    loader_threads: int = -1
    decode_threads: int = -1
    encode_threads: int = 1
    shuffle_buffer_entries: int = 16384

    filtered: bool = True
    wld_filtered: bool = True
    random_fen_skipping: int = 0
    early_fen_skipping: int = -1
    simple_eval_skipping: int = -1

    seed: int | None = None
    ray_address: str | None = None
    ray_namespace: str = "stockfish-raytest"
    feeder_cpus: float = 1.0

    bundle_chunks: int = 1
    inflight_per_feeder: int = 1
    report_interval_sec: float = 5.0
    materialize_tensors: bool = False

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("Argument `datasets` is required.")
        if self.batch_size <= 0 or self.chunk_entries <= 0:
            raise ValueError("`batch_size` and `chunk_entries` must be positive.")
        if self.batch_size % self.chunk_entries != 0:
            raise ValueError("`batch_size` must be divisible by `chunk_entries`.")
        if self.feeder_count <= 0:
            raise ValueError("`feeder_count` must be positive.")
        if self.target_batches < 0:
            raise ValueError("`target_batches` must be non-negative.")
        if self.max_seconds < 0:
            raise ValueError("`max_seconds` must be non-negative.")
        if self.target_batches == 0 and self.max_seconds == 0:
            raise ValueError(
                "Set at least one stop condition: `target_batches > 0` or `max_seconds > 0`."
            )
        if self.loader_threads == 0:
            raise ValueError("`loader_threads` must be positive or -1 for auto.")
        if self.decode_threads == 0:
            raise ValueError("`decode_threads` must be positive or -1 for auto.")
        if self.encode_threads < 0:
            raise ValueError("`encode_threads` must be non-negative.")
        if self.bundle_chunks <= 0:
            raise ValueError("`bundle_chunks` must be positive.")
        if self.inflight_per_feeder <= 0:
            raise ValueError("`inflight_per_feeder` must be positive.")
        if self.feeder_cpus <= 0:
            raise ValueError("`feeder_cpus` must be positive.")
        if self.report_interval_sec <= 0:
            raise ValueError("`report_interval_sec` must be positive.")


@dataclass
class RuntimeCounters:
    received_chunks: int = 0
    received_entries: int = 0
    received_bytes: int = 0
    dropped_partial_chunks: int = 0
    encoded_batches: int = 0
    encoded_entries: int = 0
    wait_sec: float = 0.0
    get_sec: float = 0.0
    encode_sec: float = 0.0


@ray.remote(max_restarts=0)
class PackedFeeder:
    def __init__(
        self,
        *,
        filenames: list[str],
        total_threads: int,
        decode_threads: int | None,
        chunk_entries: int,
        shuffle_buffer_entries: int,
        seed: int | None,
        cyclic: bool,
        skip_cfg: DataloaderSkipConfig,
        rank: int,
        world_size: int,
    ) -> None:
        self.chunk_entries = int(chunk_entries)
        self.expected_chunk_bytes = self.chunk_entries * PACKED_ENTRY_BYTES
        self.stream = rust.PackedEntryStream(
            list(filenames),
            total_threads=total_threads,
            decode_threads=decode_threads,
            chunk_entries=self.chunk_entries,
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
        self.returned_chunks = 0
        self.returned_entries = 0
        self.dropped_partial_chunks = 0

    def next_bundle(self, bundle_chunks: int) -> tuple[list[bytes], int, int]:
        if self.done:
            return [], 0, 0

        chunks: list[bytes] = []
        entries = 0
        dropped = 0

        while len(chunks) < bundle_chunks:
            chunk = self.stream.next_chunk()
            if chunk is None:
                self.done = True
                break
            if len(chunk) != self.expected_chunk_bytes:
                dropped += 1
                continue
            chunks.append(chunk)
            entries += self.chunk_entries

        self.returned_chunks += len(chunks)
        self.returned_entries += entries
        self.dropped_partial_chunks += dropped
        return chunks, entries, dropped

    def stats(self) -> dict[str, Any]:
        return {
            "stream": self.stream.stats(),
            "returned_chunks": self.returned_chunks,
            "returned_entries": self.returned_entries,
            "dropped_partial_chunks": self.dropped_partial_chunks,
            "done": self.done,
        }

    def close(self) -> None:
        self.stream.close()
        self.done = True


def _resolve_total_threads(loader_threads: int) -> int:
    if loader_threads > 0:
        return loader_threads
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 8
    return max(1, cpu_count - 1)


def _materialize_batch(batch: dict[str, Any]) -> None:
    us = torch.from_numpy(batch["is_white"])
    them = 1.0 - us
    white_indices = torch.from_numpy(batch["white"])
    black_indices = torch.from_numpy(batch["black"])
    outcome = torch.from_numpy(batch["outcome"])
    score = torch.from_numpy(batch["score"])
    psqt_indices = torch.from_numpy(batch["psqt_indices"])
    layer_stack_indices = torch.from_numpy(batch["layer_stack_indices"])
    _ = (
        us,
        them,
        white_indices,
        black_indices,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    )


def _print_progress(
    counters: RuntimeCounters,
    *,
    start_time: float,
    pending_chunks: int,
    inflight_calls: int,
) -> None:
    elapsed = max(1e-9, time.perf_counter() - start_time)
    entries_per_sec = counters.encoded_entries / elapsed
    batches_per_sec = counters.encoded_batches / elapsed
    gib_per_sec = counters.received_bytes / elapsed / (1024**3)
    wait_pct = (counters.wait_sec / elapsed) * 100.0
    encode_pct = (counters.encode_sec / elapsed) * 100.0
    print(
        "elapsed={:.1f}s batches={} entries={} batch/s={:.2f} entry/s={:.0f} "
        "recv_gib/s={:.2f} wait={:.1f}% encode={:.1f}% pending_chunks={} inflight={}".format(
            elapsed,
            counters.encoded_batches,
            counters.encoded_entries,
            batches_per_sec,
            entries_per_sec,
            gib_per_sec,
            wait_pct,
            encode_pct,
            pending_chunks,
            inflight_calls,
        ),
        flush=True,
    )


def main() -> None:
    cfg = tyro.cli(RayTestConfig)

    for dataset in cfg.datasets:
        if not os.path.exists(dataset):
            raise FileNotFoundError(dataset)

    feature_set = cfg.feature_set.replace("^", "")
    chunks_per_batch = cfg.batch_size // cfg.chunk_entries
    total_threads = _resolve_total_threads(cfg.loader_threads)
    decode_threads = cfg.decode_threads if cfg.decode_threads > 0 else None
    skip_cfg = DataloaderSkipConfig(
        filtered=cfg.filtered,
        wld_filtered=cfg.wld_filtered,
        random_fen_skipping=cfg.random_fen_skipping,
        early_fen_skipping=cfg.early_fen_skipping,
        simple_eval_skipping=cfg.simple_eval_skipping,
    )

    print(
        "starting raytest: feeders={} batch_size={} chunk_entries={} chunks_per_batch={} "
        "bundle_chunks={} inflight_per_feeder={} encode_threads={}".format(
            cfg.feeder_count,
            cfg.batch_size,
            cfg.chunk_entries,
            chunks_per_batch,
            cfg.bundle_chunks,
            cfg.inflight_per_feeder,
            cfg.encode_threads,
        ),
        flush=True,
    )

    ray_init_kwargs: dict[str, Any] = {
        "namespace": cfg.ray_namespace,
        "log_to_driver": True,
    }
    if cfg.ray_address:
        ray_init_kwargs["address"] = cfg.ray_address
    ray.init(**ray_init_kwargs)

    actors: list[Any] = []
    inflight: dict[Any, int] = {}
    pending_chunks: deque[bytes] = deque()
    counters = RuntimeCounters()

    try:
        for rank in range(cfg.feeder_count):
            actor = PackedFeeder.options(num_cpus=cfg.feeder_cpus).remote(
                filenames=list(cfg.datasets),
                total_threads=total_threads,
                decode_threads=decode_threads,
                chunk_entries=cfg.chunk_entries,
                shuffle_buffer_entries=cfg.shuffle_buffer_entries,
                seed=None if cfg.seed is None else cfg.seed + rank,
                cyclic=cfg.cyclic,
                skip_cfg=skip_cfg,
                rank=rank,
                world_size=cfg.feeder_count,
            )
            actors.append(actor)

        for feeder_idx, actor in enumerate(actors):
            for _ in range(cfg.inflight_per_feeder):
                ref = actor.next_bundle.remote(cfg.bundle_chunks)
                inflight[ref] = feeder_idx

        start_time = time.perf_counter()
        next_report_time = start_time + cfg.report_interval_sec

        while inflight:
            now = time.perf_counter()
            if (
                cfg.target_batches > 0
                and counters.encoded_batches >= cfg.target_batches
            ):
                break
            if cfg.max_seconds > 0 and (now - start_time) >= cfg.max_seconds:
                break

            wait_start = time.perf_counter()
            ready, _ = ray.wait(list(inflight.keys()), num_returns=1, timeout=1.0)
            counters.wait_sec += time.perf_counter() - wait_start

            if not ready:
                if time.perf_counter() >= next_report_time:
                    _print_progress(
                        counters,
                        start_time=start_time,
                        pending_chunks=len(pending_chunks),
                        inflight_calls=len(inflight),
                    )
                    next_report_time += cfg.report_interval_sec
                continue

            ref = ready[0]
            feeder_idx = inflight.pop(ref)

            get_start = time.perf_counter()
            chunks, entries, dropped = ray.get(ref)
            counters.get_sec += time.perf_counter() - get_start
            counters.dropped_partial_chunks += dropped

            if chunks:
                counters.received_chunks += len(chunks)
                counters.received_entries += entries
                counters.received_bytes += entries * PACKED_ENTRY_BYTES
                pending_chunks.extend(chunks)

                next_ref = actors[feeder_idx].next_bundle.remote(cfg.bundle_chunks)
                inflight[next_ref] = feeder_idx

            while len(pending_chunks) >= chunks_per_batch:
                chunk_group = [
                    pending_chunks.popleft() for _ in range(chunks_per_batch)
                ]

                encode_start = time.perf_counter()
                batch = rust.encode_packed_chunks(
                    feature_set,
                    chunk_group,
                    cfg.batch_size,
                    cfg.encode_threads,
                )
                counters.encode_sec += time.perf_counter() - encode_start

                counters.encoded_batches += 1
                counters.encoded_entries += cfg.batch_size

                if cfg.materialize_tensors:
                    _materialize_batch(batch)

                if (
                    cfg.target_batches > 0
                    and counters.encoded_batches >= cfg.target_batches
                ):
                    break

            if time.perf_counter() >= next_report_time:
                _print_progress(
                    counters,
                    start_time=start_time,
                    pending_chunks=len(pending_chunks),
                    inflight_calls=len(inflight),
                )
                next_report_time += cfg.report_interval_sec

        elapsed = max(1e-9, time.perf_counter() - start_time)
        print("raytest complete", flush=True)
        print(
            "batches={} entries={} elapsed={:.1f}s batch/s={:.2f} entry/s={:.0f} recv_gib/s={:.2f}".format(
                counters.encoded_batches,
                counters.encoded_entries,
                elapsed,
                counters.encoded_batches / elapsed,
                counters.encoded_entries / elapsed,
                counters.received_bytes / elapsed / (1024**3),
            ),
            flush=True,
        )
        print(
            "wait_sec={:.2f} get_sec={:.2f} encode_sec={:.2f} dropped_partial_chunks={}".format(
                counters.wait_sec,
                counters.get_sec,
                counters.encode_sec,
                counters.dropped_partial_chunks,
            ),
            flush=True,
        )

        feeder_stats = ray.get([actor.stats.remote() for actor in actors])
        for idx, stats in enumerate(feeder_stats):
            stream_stats = stats["stream"]
            print(
                "feeder={} returned_chunks={} returned_entries={} dropped_partial_chunks={} "
                "decoded_entries={} skipped_entries={} produced_chunks={} chunk_queue_len={}".format(
                    idx,
                    stats["returned_chunks"],
                    stats["returned_entries"],
                    stats["dropped_partial_chunks"],
                    stream_stats["decoded_entries"],
                    stream_stats["skipped_entries"],
                    stream_stats["produced_chunks"],
                    stream_stats["chunk_queue_len"],
                ),
                flush=True,
            )
    finally:
        if actors:
            ray.get([actor.close.remote() for actor in actors])
        ray.shutdown()


if __name__ == "__main__":
    main()
