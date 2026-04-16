from __future__ import annotations

from typing import Any

import ray
import rust

from src.data import DataloaderSkipConfig, PACKED_ENTRY_BYTES


@ray.remote(max_restarts=0)
class PackedFeeder:
    def __init__(
        self,
        *,
        filenames: list[str],
        feature_set: str,
        encode_threads: int,
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
        self.feature_set = feature_set
        self.encode_threads = encode_threads
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

    def next_bundle(self, bundle_chunks: int) -> tuple[dict | None, int, int]:
        if self.done:
            return None, 0, 0

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

        if len(chunks) != bundle_chunks:
            # partial end-of-stream bundle — drop it
            return None, 0, dropped

        batch_size = bundle_chunks * self.chunk_entries
        encoded = rust.encode_packed_chunks(
            self.feature_set, chunks, batch_size, self.encode_threads
        )
        return encoded, entries, dropped

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
