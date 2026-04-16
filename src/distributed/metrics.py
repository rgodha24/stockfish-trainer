from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


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

    def snapshot(
        self,
        *,
        start_time: float,
        pending_chunks: int,
        inflight_calls: int,
    ) -> dict[str, float | int]:
        elapsed = max(1e-9, time.perf_counter() - start_time)
        return {
            "elapsed_sec": elapsed,
            "received_chunks": self.received_chunks,
            "received_entries": self.received_entries,
            "received_bytes": self.received_bytes,
            "dropped_partial_chunks": self.dropped_partial_chunks,
            "encoded_batches": self.encoded_batches,
            "encoded_entries": self.encoded_entries,
            "batch_per_sec": self.encoded_batches / elapsed,
            "entry_per_sec": self.encoded_entries / elapsed,
            "recv_gib_per_sec": self.received_bytes / elapsed / (1024**3),
            "wait_sec": self.wait_sec,
            "get_sec": self.get_sec,
            "encode_sec": self.encode_sec,
            "wait_fraction": self.wait_sec / elapsed,
            "get_fraction": self.get_sec / elapsed,
            "encode_fraction": self.encode_sec / elapsed,
            "pending_chunks": pending_chunks,
            "inflight_calls": inflight_calls,
        }


def format_progress(snapshot: dict[str, float | int]) -> str:
    return (
        "elapsed={elapsed_sec:.1f}s batches={encoded_batches} entries={encoded_entries} "
        "batch/s={batch_per_sec:.2f} entry/s={entry_per_sec:.0f} recv_gib/s={recv_gib_per_sec:.2f} "
        "wait={wait_pct:.1f}% encode={encode_pct:.1f}% pending_batches={pending_chunks} inflight={inflight_calls}"
    ).format(
        elapsed_sec=float(snapshot["elapsed_sec"]),
        encoded_batches=int(snapshot["encoded_batches"]),
        encoded_entries=int(snapshot["encoded_entries"]),
        batch_per_sec=float(snapshot["batch_per_sec"]),
        entry_per_sec=float(snapshot["entry_per_sec"]),
        recv_gib_per_sec=float(snapshot["recv_gib_per_sec"]),
        wait_pct=float(snapshot["wait_fraction"]) * 100.0,
        encode_pct=float(snapshot["encode_fraction"]) * 100.0,
        pending_chunks=int(snapshot["pending_chunks"]),
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
    return (
        "wait_sec={wait_sec:.2f} get_sec={get_sec:.2f} encode_sec={encode_sec:.2f} dropped_partial_chunks={dropped_partial_chunks}"
    ).format(
        wait_sec=float(snapshot["wait_sec"]),
        get_sec=float(snapshot["get_sec"]),
        encode_sec=float(snapshot["encode_sec"]),
        dropped_partial_chunks=int(snapshot["dropped_partial_chunks"]),
    )


def format_feeder_stats(index: int, stats: dict[str, Any]) -> str:
    stream_stats = stats["stream"]
    return (
        "feeder={index} returned_chunks={returned_chunks} returned_entries={returned_entries} "
        "dropped_partial_chunks={dropped_partial_chunks} decoded_entries={decoded_entries} "
        "skipped_entries={skipped_entries} produced_chunks={produced_chunks} chunk_queue_len={chunk_queue_len}"
    ).format(
        index=index,
        returned_chunks=int(stats["returned_chunks"]),
        returned_entries=int(stats["returned_entries"]),
        dropped_partial_chunks=int(stats["dropped_partial_chunks"]),
        decoded_entries=int(stream_stats["decoded_entries"]),
        skipped_entries=int(stream_stats["skipped_entries"]),
        produced_chunks=int(stream_stats["produced_chunks"]),
        chunk_queue_len=int(stream_stats["chunk_queue_len"]),
    )
