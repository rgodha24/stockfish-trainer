from __future__ import annotations

import torch
import tyro

from src.data import make_sparse_batch_dataset
from src.train.common import TrainBatchSource, run_training
from src.train.config import SingleNodeTrainingConfig
from src.train.distributed import DistributedRuntime


def make_train_source(
    args: SingleNodeTrainingConfig,
    runtime: DistributedRuntime,
    local_batch_size: int,
) -> TrainBatchSource:
    batches = make_sparse_batch_dataset(
        feature_set=args.features,
        filenames=list(args.datasets),
        batch_size=local_batch_size,
        cyclic=True,
        loader_threads=args.loader_threads,
        config=args.loader_skip_config(),
        shuffle_buffer_entries=args.shuffle_buffer_entries,
        pin_memory=args.pin_memory,
        rank=runtime.rank,
        world_size=runtime.world_size,
    )
    return TrainBatchSource(batches=batches, metrics=lambda: dict(), close=lambda: None)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    args = tyro.cli(SingleNodeTrainingConfig)
    run_training(
        args,
        lambda runtime, local_batch_size: make_train_source(
            args,
            runtime,
            local_batch_size,
        ),
        allow_distributed=True,
    )


if __name__ == "__main__":
    main()
