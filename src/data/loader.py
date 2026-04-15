import os
import queue
import threading
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


def _recursive_pin(obj):
    if not torch.cuda.is_available():
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory()
    if isinstance(obj, dict):
        return {k: _recursive_pin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_recursive_pin(v) for v in obj)
    return obj


class RustSparseBatchProvider:
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size: int,
        cyclic: bool,
        loader_threads: int,
        config: DataloaderSkipConfig,
    ):
        if loader_threads > 0:
            total_threads = loader_threads
        else:
            try:
                cpu_count = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                cpu_count = os.cpu_count() or 8
            total_threads = max(1, cpu_count - 1)

        skip_heavy = (
            config.filtered
            or config.wld_filtered
            or config.random_fen_skipping > 0
            or config.early_fen_skipping >= 0
            or (
                config.simple_eval_skipping is not None
                and config.simple_eval_skipping > 0
            )
        )

        if skip_heavy:
            decode_threads = max(1, (total_threads * 9) // 10)
        else:
            decode_threads = max(1, total_threads // 4)
        encode_threads = max(1, total_threads - decode_threads)
        slab_count = max(8, encode_threads + max(4, encode_threads // 2))

        self._stream = rust.BatchStream(
            feature_set.replace("^", ""),
            list(filenames),
            batch_size,
            total_threads=total_threads,
            decode_threads=decode_threads,
            encode_threads=encode_threads,
            slab_count=slab_count,
            shuffle_buffer_entries=16384,
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
            rank=0,
            world_size=1,
        )

        self._ones_buffer: torch.Tensor | None = None

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._stream.next_batch()
        if batch is None:
            raise StopIteration

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
            buf = torch.ones((rows, cols), dtype=torch.float32)
            try:
                buf = buf.pin_memory()
            except RuntimeError:
                pass
            self._ones_buffer = buf
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
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
    ):
        super().__init__()
        self.feature_set = feature_set
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.loader_threads = loader_threads
        self.config = config

    def __iter__(self):
        return RustSparseBatchProvider(
            self.feature_set,
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            loader_threads=self.loader_threads,
            config=self.config,
        )


class FixedNumBatchesDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, num_batches, pin_memory: bool = False, queue_size_limit=16
    ):
        super().__init__()
        self.dataset = dataset
        self.iter = None
        self.num_batches = num_batches
        self.pin_memory = pin_memory

        self._prefetch_queue = queue.Queue(maxsize=queue_size_limit)
        self._prefetch_thread = None
        self._stop_prefetching = threading.Event()
        self._prefetch_started = False
        self._lock = threading.Lock()

    def _safe_put(self, item):
        while not self._stop_prefetching.is_set():
            try:
                self._prefetch_queue.put(item, timeout=1.0)
                break
            except queue.Full:
                continue

    def _prefetch_worker(self):
        try:
            while not self._stop_prefetching.is_set():
                try:
                    item = next(self.iter)
                    if self.pin_memory:
                        item = _recursive_pin(item)
                    self._safe_put(item)
                except StopIteration:
                    self._safe_put(None)
                    break
        except Exception as e:
            self._safe_put(e)

    def _start_prefetching(self):
        with self._lock:
            if not self._prefetch_started:
                self.iter = iter(self.dataset)
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker, daemon=True
                )
                self._prefetch_thread.start()
                self._prefetch_started = True

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        _ = idx
        self._start_prefetching()
        try:
            item = self._prefetch_queue.get(timeout=300.0)
            if item is None:
                raise StopIteration("End of dataset reached")
            if isinstance(item, Exception):
                raise item
            return item
        except queue.Empty as e:
            raise RuntimeError("Prefetch timeout - no data available") from e

    def __del__(self):
        if hasattr(self, "_stop_prefetching"):
            self._stop_prefetching.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
