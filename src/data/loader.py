import os
from dataclasses import dataclass
from typing import TypeAlias

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


Batch: TypeAlias = tuple[torch.Tensor, ...]


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

    def to_tuple(self, batch: dict[str, object]) -> Batch:
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


class RustSparseBatchProvider:
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size: int,
        cyclic: bool,
        total_threads: int,
        decode_threads: int | None,
        encode_threads: int | None,
        shuffle_buffer_entries: int,
        config: DataloaderSkipConfig,
        pin_memory: bool,
        rank: int,
        world_size: int,
    ):
        self._stream = rust.BatchStream(
            feature_set.replace("^", ""),
            list(filenames),
            batch_size,
            total_threads=total_threads,
            decode_threads=decode_threads,
            encode_threads=encode_threads,
            slab_count=None,
            shuffle_buffer_entries=shuffle_buffer_entries,
            seed=None,
            cyclic=cyclic,
            filtered=config.filtered,
            random_fen_skipping=int(config.random_fen_skipping),
            wld_filtered=config.wld_filtered,
            early_fen_skipping=int(config.early_fen_skipping),
            simple_eval_skipping=int(config.simple_eval_skipping or 0),
            param_index=int(config.param_index),
            pc_y1=float(config.pc_y1),
            pc_y2=float(config.pc_y2),
            pc_y3=float(config.pc_y3),
            rank=rank,
            world_size=world_size,
        )
        self._tensorizer = SparseBatchTensorizer(pin_memory=pin_memory)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._stream.next_batch()
        if batch is None:
            raise StopIteration
        return self._tensorizer.to_tuple(batch)

    def __del__(self):
        if hasattr(self, "_stream"):
            self._stream.close()


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size: int,
        cyclic: bool = True,
        loader_threads: int = -1,
        config: DataloaderSkipConfig | None = None,
        shuffle_buffer_entries: int = 16384,
        pin_memory: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        encode_threads: int = 0,
    ):
        super().__init__()
        self.feature_set = feature_set
        self.filenames = filenames
        self.batch_size = int(batch_size)
        self.cyclic = cyclic
        self.total_threads = resolve_total_threads(loader_threads)
        self.shuffle_buffer_entries = int(shuffle_buffer_entries)
        self.config = config if config is not None else DataloaderSkipConfig()
        self.pin_memory = pin_memory
        self.rank, self.world_size = _infer_world(rank, world_size)

        if encode_threads > 0:
            self.encode_threads = max(1, int(encode_threads))
            self.decode_threads = max(1, self.total_threads - self.encode_threads)
        else:
            self.encode_threads = None
            self.decode_threads = None

    def __iter__(self):
        return RustSparseBatchProvider(
            self.feature_set,
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            total_threads=self.total_threads,
            decode_threads=self.decode_threads,
            encode_threads=self.encode_threads,
            shuffle_buffer_entries=self.shuffle_buffer_entries,
            config=self.config,
            pin_memory=self.pin_memory,
            rank=self.rank,
            world_size=self.world_size,
        )


def make_sparse_batch_dataset(
    feature_set: str,
    filenames: list[str],
    batch_size: int,
    cyclic: bool = True,
    loader_threads: int = -1,
    config: DataloaderSkipConfig | None = None,
    shuffle_buffer_entries: int = 16384,
    pin_memory: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
    encode_threads: int = 0,
) -> SparseBatchDataset:
    resolved_config = config if config is not None else DataloaderSkipConfig()
    return SparseBatchDataset(
        feature_set=feature_set,
        filenames=filenames,
        batch_size=batch_size,
        cyclic=cyclic,
        loader_threads=loader_threads,
        config=resolved_config,
        shuffle_buffer_entries=shuffle_buffer_entries,
        pin_memory=pin_memory,
        rank=rank,
        world_size=world_size,
        encode_threads=encode_threads,
    )
