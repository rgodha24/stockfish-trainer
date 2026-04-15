import os
from dataclasses import dataclass

import torch

import rust


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


class PackedChunkDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        filenames: list[str],
        chunk_entries: int,
        cyclic: bool,
        loader_threads: int,
        config: DataloaderSkipConfig,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        super().__init__()
        self.filenames = list(filenames)
        self.chunk_entries = int(chunk_entries)
        self.cyclic = bool(cyclic)
        self.loader_threads = int(loader_threads)
        self.config = config
        self.rank, self.world_size = _infer_world(rank, world_size)

    def _total_threads(self) -> int:
        if self.loader_threads > 0:
            return self.loader_threads
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpu_count = os.cpu_count() or 8
        return max(1, cpu_count - 1)

    def __iter__(self):
        stream = rust.PackedEntryStream(
            list(self.filenames),
            total_threads=self._total_threads(),
            decode_threads=None,
            chunk_entries=self.chunk_entries,
            shuffle_buffer_entries=16384,
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
        max_batches: int | None = None,
    ):
        super().__init__()
        self.feature_set = feature_set.replace("^", "")
        self.chunk_dataset = chunk_dataset
        self.batch_size = int(batch_size)
        self.chunk_entries = int(chunk_entries)
        self.max_batches = max_batches
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chunk_entries <= 0:
            raise ValueError("chunk_entries must be positive")
        if self.batch_size % self.chunk_entries != 0:
            raise ValueError("batch_size must be divisible by chunk_entries")
        self.chunks_per_batch = self.batch_size // self.chunk_entries
        self._ones_buffer: torch.Tensor | None = None

    def __iter__(self):
        chunk_iter = iter(self.chunk_dataset)
        yielded = 0
        while True:
            if self.max_batches is not None and yielded >= self.max_batches:
                return
            chunks: list[bytes] = []
            for _ in range(self.chunks_per_batch):
                try:
                    chunks.append(next(chunk_iter))
                except StopIteration:
                    return

            batch = rust.encode_packed_chunks(self.feature_set, chunks, self.batch_size)
            yielded += 1
            yield self._batch_to_tuple(batch)

    def __len__(self):
        if self.max_batches is None:
            raise TypeError("dataset length is undefined without max_batches")
        return self.max_batches

    def _batch_to_tuple(self, batch):
        us = torch.from_numpy(batch["is_white"])
        them = 1.0 - us
        white_indices = torch.from_numpy(batch["white"])
        black_indices = torch.from_numpy(batch["black"])
        outcome = torch.from_numpy(batch["outcome"])
        score = torch.from_numpy(batch["score"])
        psqt_indices = torch.from_numpy(batch["psqt_indices"])
        layer_stack_indices = torch.from_numpy(batch["layer_stack_indices"])

        rows, cols = white_indices.shape
        if (
            self._ones_buffer is None
            or self._ones_buffer.shape[1] != cols
            or self._ones_buffer.shape[0] < rows
        ):
            self._ones_buffer = torch.ones((rows, cols), dtype=torch.float32)
        values = self._ones_buffer[:rows]

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


def make_sparse_batch_dataset(
    feature_set: str,
    filenames: list[str],
    batch_size: int,
    cyclic: bool = True,
    loader_threads: int = -1,
    config: DataloaderSkipConfig = DataloaderSkipConfig(),
    chunk_entries: int = 8192,
    rank: int | None = None,
    world_size: int | None = None,
    max_batches: int | None = None,
) -> EncodedBatchDataset:
    chunk_dataset = PackedChunkDataset(
        filenames=filenames,
        chunk_entries=chunk_entries,
        cyclic=cyclic,
        loader_threads=loader_threads,
        config=config,
        rank=rank,
        world_size=world_size,
    )
    return EncodedBatchDataset(
        feature_set=feature_set,
        chunk_dataset=chunk_dataset,
        batch_size=batch_size,
        chunk_entries=chunk_entries,
        max_batches=max_batches,
    )
