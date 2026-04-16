from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeCounters:
    received_batches: int = 0
    received_entries: int = 0
    received_bytes: int = 0
    encoded_batches: int = 0
    encoded_entries: int = 0
    wait_sec: float = 0.0
    get_sec: float = 0.0

    def snapshot(
        self,
        *,
        start_time: float,
        pending_batches: int,
        inflight_calls: int,
    ) -> dict[str, float | int]:
        elapsed = max(1e-9, time.perf_counter() - start_time)
        return {
            "elapsed_sec": elapsed,
            "received_batches": self.received_batches,
            "received_entries": self.received_entries,
            "received_bytes": self.received_bytes,
            "encoded_batches": self.encoded_batches,
            "encoded_entries": self.encoded_entries,
            "batch_per_sec": self.encoded_batches / elapsed,
            "entry_per_sec": self.encoded_entries / elapsed,
            "recv_gib_per_sec": self.received_bytes / elapsed / (1024**3),
            "wait_sec": self.wait_sec,
            "get_sec": self.get_sec,
            "wait_fraction": self.wait_sec / elapsed,
            "get_fraction": self.get_sec / elapsed,
            "pending_batches": pending_batches,
            "inflight_calls": inflight_calls,
        }


def format_progress(snapshot: dict[str, float | int]) -> str:
    return (
        "elapsed={elapsed_sec:.1f}s batches={encoded_batches} entries={encoded_entries} "
        "batch/s={batch_per_sec:.2f} entry/s={entry_per_sec:.0f} recv_gib/s={recv_gib_per_sec:.2f} "
        "wait={wait_pct:.1f}% pending_batches={pending_batches} inflight={inflight_calls}"
    ).format(
        elapsed_sec=float(snapshot["elapsed_sec"]),
        encoded_batches=int(snapshot["encoded_batches"]),
        encoded_entries=int(snapshot["encoded_entries"]),
        batch_per_sec=float(snapshot["batch_per_sec"]),
        entry_per_sec=float(snapshot["entry_per_sec"]),
        recv_gib_per_sec=float(snapshot["recv_gib_per_sec"]),
        wait_pct=float(snapshot["wait_fraction"]) * 100.0,
        pending_batches=int(snapshot["pending_batches"]),
        inflight_calls=int(snapshot["inflight_calls"]),
    )


def format_summary(snapshot: dict[str, float | int]) -> str:
    return (
        "batches={encoded_batches} entries={encoded_entries} elapsed={elapsed_sec:.1f}s "
        "batch/s={batch_per_sec:.2f} entry/s={entry_per_sec:.0f} recv_gib/s={recv_gib_per_sec:.2f}"
    ).format(
        encoded_batches=int(snapshot["encoded_batches"]),
        encoded_entries=int(snapshot["encoded_entries"]),
        elapsed_sec=float(snapshot["elapsed_sec"]),
        batch_per_sec=float(snapshot["batch_per_sec"]),
        entry_per_sec=float(snapshot["entry_per_sec"]),
        recv_gib_per_sec=float(snapshot["recv_gib_per_sec"]),
    )


def format_breakdown(snapshot: dict[str, float | int]) -> str:
    return "wait_sec={wait_sec:.2f} get_sec={get_sec:.2f}".format(
        wait_sec=float(snapshot["wait_sec"]),
        get_sec=float(snapshot["get_sec"]),
    )


def format_feeder_stats(index: int, stats: dict[str, Any]) -> str:
    stream_stats = stats["stream"]
    return (
        "feeder={index} returned_batches={returned_batches} returned_entries={returned_entries} "
        "decoded_entries={decoded_entries} encoded_entries={encoded_entries} skipped_entries={skipped_entries} "
        "produced_batches={produced_batches} ready_queue_len={ready_queue_len} free_queue_len={free_queue_len}"
    ).format(
        index=index,
        returned_batches=int(stats["returned_batches"]),
        returned_entries=int(stats["returned_entries"]),
        decoded_entries=int(stream_stats["decoded_entries"]),
        encoded_entries=int(stream_stats["encoded_entries"]),
        skipped_entries=int(stream_stats["skipped_entries"]),
        produced_batches=int(stream_stats["produced_batches"]),
        ready_queue_len=int(stream_stats["ready_queue_len"]),
        free_queue_len=int(stream_stats["free_queue_len"]),
    )
