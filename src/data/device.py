from __future__ import annotations

import collections
import queue
import threading
from typing import Iterable, Iterator

import torch

from .loader import Batch


class _PrefetchedBatchIterator(Iterator[Batch]):
    def __init__(self, batches: Iterable[Batch], *, maxsize: int):
        if maxsize <= 0:
            raise ValueError("`maxsize` must be positive.")
        self._iter = iter(batches)
        self._queue: queue.Queue[Batch | BaseException | None] = queue.Queue(
            maxsize=maxsize
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _safe_put(self, item: Batch | BaseException | None) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                self._safe_put(next(self._iter))
        except StopIteration:
            if not self._stop.is_set():
                self._safe_put(None)
        except BaseException as exc:
            if not self._stop.is_set():
                self._safe_put(exc)

    def __iter__(self) -> _PrefetchedBatchIterator:
        return self

    def __next__(self) -> Batch:
        try:
            item = self._queue.get(timeout=300.0)
        except queue.Empty as exc:
            raise RuntimeError("Prefetch timeout - no data available") from exc
        if item is None:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    (
        us,
        white_indices,
        black_indices,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = batch

    return (
        us.to(device, non_blocking=True),
        white_indices.to(device, non_blocking=True, dtype=torch.int32),
        black_indices.to(device, non_blocking=True, dtype=torch.int32),
        outcome.to(device, non_blocking=True),
        score.to(device, non_blocking=True),
        psqt_indices.to(device, non_blocking=True, dtype=torch.int64),
        layer_stack_indices.to(device, non_blocking=True, dtype=torch.int64),
    )


def iter_device_batches(
    cpu_batches: Iterable[Batch],
    device: torch.device,
    *,
    queue_size_limit: int = 16,
) -> Iterator[Batch]:
    prefetched_batches = _PrefetchedBatchIterator(cpu_batches, maxsize=queue_size_limit)
    try:
        if device.type != "cuda":
            for batch in prefetched_batches:
                yield _move_batch_to_device(batch, device)
            return

        retained_cpu = collections.deque[Batch](maxlen=3)
        xfer_stream = torch.cuda.Stream(device=device)
        compute_stream = torch.cuda.current_stream(device=device)

        cpu_iter = iter(prefetched_batches)
        try:
            first_cpu_batch = next(cpu_iter)
        except StopIteration:
            return

        with torch.cuda.stream(xfer_stream):
            current_gpu_batch = _move_batch_to_device(first_cpu_batch, device)
            current_ready = torch.cuda.Event()
            current_ready.record(xfer_stream)
        retained_cpu.append(first_cpu_batch)

        for next_cpu_batch in cpu_iter:
            with torch.cuda.stream(xfer_stream):
                next_gpu_batch = _move_batch_to_device(next_cpu_batch, device)
                next_ready = torch.cuda.Event()
                next_ready.record(xfer_stream)
            retained_cpu.append(next_cpu_batch)

            compute_stream.wait_event(current_ready)
            for tensor in current_gpu_batch:
                if tensor.is_cuda:
                    tensor.record_stream(compute_stream)
            yield current_gpu_batch

            current_gpu_batch = next_gpu_batch
            current_ready = next_ready

        compute_stream.wait_event(current_ready)
        for tensor in current_gpu_batch:
            if tensor.is_cuda:
                tensor.record_stream(compute_stream)
        yield current_gpu_batch
    finally:
        prefetched_batches.close()
