import os
from dataclasses import dataclass

import torch

import rust


PACKED_ENTRY_BYTES = 32


@dataclass
class DataloaderSkipConfig:
    filtered: bool = True
    wld_filtered: bool = True
    random_fen_skipping: int = 0
    early_fen_skipping: int = -1
    simple_eval_skipping: int = -1
    param_index: int = 0
    pc_y1: float = 1.0
    pc_y2: float = 2.0
    pc_y3: float = 1.0


def _infer_world(rank: int | None, world_size: int | None) -> tuple[int, int]:
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def resolve_total_threads(loader_threads: int) -> int:
    if loader_threads > 0:
        return loader_threads
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 8
    return max(1, cpu_count - 1)


def default_thread_split(
    total_threads: int,
    config: DataloaderSkipConfig,
) -> tuple[int, int]:
    skip_heavy = (
        config.filtered
        or config.wld_filtered
        or config.random_fen_skipping > 0
        or config.early_fen_skipping >= 0
        or config.simple_eval_skipping > 0
    )
    if total_threads <= 1:
        return 1, 1

    if skip_heavy:
        decode_threads = max(1, (total_threads * 3) // 4)
        encode_threads = max(1, total_threads - decode_threads)
        return decode_threads, encode_threads

    decode_threads = max(1, min(8, total_threads // 4))
    encode_threads = max(1, total_threads - decode_threads)
    return decode_threads, encode_threads


def resolve_thread_split(
    total_threads: int,
    config: DataloaderSkipConfig,
    encode_threads: int,
) -> tuple[int, int]:
    decode_threads, default_encode_threads = default_thread_split(total_threads, config)
    if encode_threads > 0:
        chosen_encode_threads = max(1, encode_threads)
        decode_threads = max(1, total_threads - chosen_encode_threads)
    else:
        chosen_encode_threads = default_encode_threads
    return decode_threads, chosen_encode_threads


class SparseBatchTensorizer:
    def __init__(self, *, pin_memory: bool = False):
        self.pin_memory = pin_memory
        self._ones_buffer: torch.Tensor | None = None

    def _maybe_pin(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.pin_memory or not torch.cuda.is_available():
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    def _values_buffer(self, rows: int, cols: int) -> torch.Tensor:
        if (
            self._ones_buffer is None
            or self._ones_buffer.shape[1] != cols
            or self._ones_buffer.shape[0] < rows
        ):
            ones = torch.ones((rows, cols), dtype=torch.float32)
            try:
                ones = ones.pin_memory()
            except RuntimeError:
                pass
            self._ones_buffer = ones
        return self._ones_buffer[:rows]

    def to_tuple(self, batch: dict[str, object]) -> tuple[torch.Tensor, ...]:
        us = self._maybe_pin(torch.from_numpy(batch["is_white"]))
        them = self._maybe_pin(1.0 - us)
        white_indices = self._maybe_pin(torch.from_numpy(batch["white"]))
        black_indices = self._maybe_pin(torch.from_numpy(batch["black"]))
        outcome = self._maybe_pin(torch.from_numpy(batch["outcome"]))
        score = self._maybe_pin(torch.from_numpy(batch["score"]))
        psqt_indices = self._maybe_pin(torch.from_numpy(batch["psqt_indices"]))
        layer_stack_indices = self._maybe_pin(
            torch.from_numpy(batch["layer_stack_indices"])
        )

        rows, cols = white_indices.shape
        values = self._values_buffer(rows, cols)

        return (
            us,
            them,
            white_indices,
            values,
            black_indices,
            values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        )


class PackedChunkDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        filenames: list[str],
        chunk_entries: int,
        shuffle_buffer_entries: int,
        cyclic: bool,
        total_threads: int,
        decode_threads: int,
        config: DataloaderSkipConfig,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        super().__init__()
        self.filenames = list(filenames)
        self.chunk_entries = int(chunk_entries)
        self.shuffle_buffer_entries = int(shuffle_buffer_entries)
        self.cyclic = bool(cyclic)
        self.total_threads = int(total_threads)
        self.decode_threads = int(decode_threads)
        self.config = config
        self.rank, self.world_size = _infer_world(rank, world_size)

    def __iter__(self):
        stream = rust.PackedEntryStream(
            list(self.filenames),
            total_threads=self.total_threads,
            decode_threads=self.decode_threads,
            chunk_entries=self.chunk_entries,
            shuffle_buffer_entries=self.shuffle_buffer_entries,
            seed=None,
            cyclic=self.cyclic,
            filtered=self.config.filtered,
            random_fen_skipping=int(self.config.random_fen_skipping),
            wld_filtered=self.config.wld_filtered,
            early_fen_skipping=int(self.config.early_fen_skipping),
            simple_eval_skipping=int(self.config.simple_eval_skipping or 0),
            param_index=int(self.config.param_index),
            pc_y1=float(self.config.pc_y1),
            pc_y2=float(self.config.pc_y2),
            pc_y3=float(self.config.pc_y3),
            rank=self.rank,
            world_size=self.world_size,
        )
        try:
            while True:
                chunk = stream.next_chunk()
                if chunk is None:
                    break
                yield chunk
        finally:
            stream.close()


class EncodedBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        feature_set: str,
        chunk_dataset: PackedChunkDataset,
        batch_size: int,
        chunk_entries: int,
        encode_threads: int,
    ):
        super().__init__()
        self.feature_set = feature_set.replace("^", "")
        self.chunk_dataset = chunk_dataset
        self.batch_size = int(batch_size)
        self.chunk_entries = int(chunk_entries)
        self.encode_threads = int(encode_threads)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chunk_entries <= 0:
            raise ValueError("chunk_entries must be positive")
        if self.encode_threads <= 0:
            raise ValueError("encode_threads must be positive")
        if self.batch_size % self.chunk_entries != 0:
            raise ValueError("batch_size must be divisible by chunk_entries")
        self.chunks_per_batch = self.batch_size // self.chunk_entries
        self.tensorizer = SparseBatchTensorizer()

    def __iter__(self):
        chunk_iter = iter(self.chunk_dataset)
        while True:
            chunks: list[bytes] = []
            for _ in range(self.chunks_per_batch):
                try:
                    chunks.append(next(chunk_iter))
                except StopIteration:
                    return

            batch = rust.encode_packed_chunks(
                self.feature_set,
                chunks,
                self.batch_size,
                self.encode_threads,
            )
            yield self.tensorizer.to_tuple(batch)


def make_sparse_batch_dataset(
    feature_set: str,
    filenames: list[str],
    batch_size: int,
    cyclic: bool = True,
    loader_threads: int = -1,
    config: DataloaderSkipConfig = DataloaderSkipConfig(),
    chunk_entries: int = 8192,
    shuffle_buffer_entries: int = 16384,
    rank: int | None = None,
    world_size: int | None = None,
    encode_threads: int = 0,
) -> EncodedBatchDataset:
    total_threads = resolve_total_threads(loader_threads)
    decode_threads, chosen_encode_threads = resolve_thread_split(
        total_threads, config, encode_threads
    )

    chunk_dataset = PackedChunkDataset(
        filenames=filenames,
        chunk_entries=chunk_entries,
        shuffle_buffer_entries=shuffle_buffer_entries,
        cyclic=cyclic,
        total_threads=total_threads,
        decode_threads=decode_threads,
        config=config,
        rank=rank,
        world_size=world_size,
    )
    return EncodedBatchDataset(
        feature_set=feature_set,
        chunk_dataset=chunk_dataset,
        batch_size=batch_size,
        chunk_entries=chunk_entries,
        encode_threads=chosen_encode_threads,
    )
