from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from src.train.config import BaseTrainingConfig


@dataclass(slots=True)
class DistributedRuntime:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: Any = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        if self.is_distributed:
            dist.all_reduce(tensor, op=op)
        return tensor

    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()

    def destroy(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def init_training_runtime(
    args: BaseTrainingConfig,
    *,
    allow_distributed: bool,
) -> DistributedRuntime:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        return DistributedRuntime(rank=0, local_rank=0, world_size=1)

    if not allow_distributed:
        raise RuntimeError(
            "torchrun environment detected for an entrypoint that does not support multi-process training"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is out of range for {torch.cuda.device_count()} visible CUDA devices"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.distributed_backend, init_method="env://")
    return DistributedRuntime(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=args.distributed_backend,
    )
