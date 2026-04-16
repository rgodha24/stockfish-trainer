from __future__ import annotations

import time
from collections import deque
from typing import Any

import ray

from src.data import SparseBatchTensorizer, resolve_total_threads
from src.distributed.config import DistributedLoaderConfig
from src.distributed.feeder import PackedFeeder
from src.distributed.metrics import (
    RuntimeCounters,
    format_breakdown,
    format_feeder_stats,
    format_progress,
    format_summary,
)


class RayBatchStream:
    def __init__(self, cfg: DistributedLoaderConfig):
        self.cfg = cfg
        self.feature_set = cfg.feature_set.replace("^", "")
        self.total_threads = resolve_total_threads(cfg.loader_threads)
        self.decode_threads = (
            cfg.decode_threads if cfg.decode_threads > 0
            else max(1, self.total_threads - cfg.encode_threads)
        )
        self.tensorizer = SparseBatchTensorizer(pin_memory=cfg.pin_memory)
        self.counters = RuntimeCounters()

        self.actors: list[Any] = []
        self.inflight: dict[Any, int] = {}
        self.pending_batches: deque[dict] = deque()
        self.start_time = 0.0
        self.next_report_time = 0.0
        self.closed = False

        print(
            "starting distributed loader: feeders={} batch_size={} chunk_entries={} inflight_per_feeder={} decode_threads={} encode_threads={}".format(
                cfg.feeder_count,
                cfg.batch_size,
                cfg.chunk_entries,
                cfg.inflight_per_feeder,
                self.decode_threads,
                cfg.encode_threads,
            ),
            flush=True,
        )

        ray_init_kwargs: dict[str, Any] = {
            "namespace": cfg.ray_namespace,
            "log_to_driver": cfg.log_to_driver,
        }
        if cfg.ray_address:
            ray_init_kwargs["address"] = cfg.ray_address
        ray.init(**ray_init_kwargs)

        skip_cfg = cfg.skip_config()
        for rank in range(cfg.feeder_count):
            actor = PackedFeeder.options(num_cpus=cfg.feeder_cpus).remote(
                filenames=list(cfg.datasets),
                feature_set=self.feature_set,
                batch_size=cfg.batch_size,
                encode_threads=cfg.encode_threads,
                total_threads=self.total_threads,
                decode_threads=self.decode_threads,
                chunk_entries=cfg.chunk_entries,
                shuffle_buffer_entries=cfg.shuffle_buffer_entries,
                seed=None if cfg.seed is None else cfg.seed + rank,
                cyclic=cfg.cyclic,
                skip_cfg=skip_cfg,
                rank=rank,
                world_size=cfg.feeder_count,
            )
            self.actors.append(actor)

        for feeder_idx, actor in enumerate(self.actors):
            for _ in range(cfg.inflight_per_feeder):
                ref = actor.next_batch.remote()
                self.inflight[ref] = feeder_idx

        self.start_time = time.perf_counter()
        self.next_report_time = self.start_time + cfg.report_interval_sec

    def __iter__(self) -> RayBatchStream:
        return self

    def _report_progress(self, *, force: bool = False) -> None:
        if not force and time.perf_counter() < self.next_report_time:
            return
        print(format_progress(self.snapshot()), flush=True)
        self.next_report_time = time.perf_counter() + self.cfg.report_interval_sec

    def _wait_for_batch(self) -> None:
        wait_start = time.perf_counter()
        ready, _ = ray.wait(list(self.inflight.keys()), num_returns=1, timeout=1.0)
        self.counters.wait_sec += time.perf_counter() - wait_start

        if not ready:
            self._report_progress()
            return

        ref = ready[0]
        feeder_idx = self.inflight.pop(ref)

        get_start = time.perf_counter()
        encoded, entries = ray.get(ref)
        self.counters.get_sec += time.perf_counter() - get_start

        if encoded is not None:
            self.counters.received_batches += 1
            self.counters.received_entries += entries
            self.counters.received_bytes += sum(
                v.nbytes for v in encoded.values() if hasattr(v, "nbytes")
            )
            self.pending_batches.append(encoded)

            next_ref = self.actors[feeder_idx].next_batch.remote()
            self.inflight[next_ref] = feeder_idx

    def __next__(self):
        while not self.pending_batches:
            if not self.inflight:
                raise StopIteration
            self._wait_for_batch()

        batch = self.pending_batches.popleft()
        self.counters.encoded_batches += 1
        self.counters.encoded_entries += batch["size"]
        self._report_progress()
        return self.tensorizer.to_tuple(batch)

    def snapshot(self) -> dict[str, float | int]:
        return self.counters.snapshot(
            start_time=self.start_time,
            pending_batches=len(self.pending_batches),
            inflight_calls=len(self.inflight),
        )

    def feeder_stats(self) -> list[dict[str, Any]]:
        if not self.actors:
            return []
        return ray.get([actor.stats.remote() for actor in self.actors])

    def close(self, *, emit_summary: bool = False) -> None:
        if self.closed:
            return
        self.closed = True

        if emit_summary and self.start_time > 0.0:
            try:
                snapshot = self.snapshot()
                print("distributed loader complete", flush=True)
                print(format_summary(snapshot), flush=True)
                print(format_breakdown(snapshot), flush=True)
                for idx, stats in enumerate(self.feeder_stats()):
                    print(format_feeder_stats(idx, stats), flush=True)
            except Exception:
                pass

        try:
            if self.actors:
                try:
                    ray.get([actor.close.remote() for actor in self.actors])
                except Exception:
                    pass
        finally:
            self.actors = []
            self.inflight.clear()
            self.pending_batches.clear()
            ray.shutdown()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
