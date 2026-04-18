from __future__ import annotations

import collections
from typing import Iterable, Iterator

import torch

from .loader import Batch


def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = batch

    return (
        us.to(device, non_blocking=True),
        them.to(device, non_blocking=True),
        white_indices.to(device, non_blocking=True, dtype=torch.int32),
        white_values.to(device, non_blocking=True),
        black_indices.to(device, non_blocking=True, dtype=torch.int32),
        black_values.to(device, non_blocking=True),
        outcome.to(device, non_blocking=True),
        score.to(device, non_blocking=True),
        psqt_indices.to(device, non_blocking=True, dtype=torch.int64),
        layer_stack_indices.to(device, non_blocking=True, dtype=torch.int64),
    )


def iter_device_batches(
    cpu_batches: Iterable[Batch],
    device: torch.device,
) -> Iterator[Batch]:
    if device.type != "cuda":
        for batch in cpu_batches:
            yield _move_batch_to_device(batch, device)
        return

    retained_cpu = collections.deque[Batch](maxlen=3)
    xfer_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    cpu_iter = iter(cpu_batches)
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
