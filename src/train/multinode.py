from __future__ import annotations

import torch
import tyro

from src.distributed import RayBatchStream
from src.train.common import TrainBatchSource, run_training
from src.train.config import MultiNodeTrainingConfig
from src.train.distributed import DistributedRuntime


def make_train_source(
    args: MultiNodeTrainingConfig,
    _runtime: DistributedRuntime,
    _local_batch_size: int,
) -> TrainBatchSource:
    stream = RayBatchStream(args.distributed_loader_config())
    return TrainBatchSource(
        batches=stream,
        metrics=stream.snapshot,
        close=lambda: stream.close(emit_summary=True),
    )


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    args = tyro.cli(MultiNodeTrainingConfig)
    run_training(
        args,
        lambda runtime, local_batch_size: make_train_source(
            args,
            runtime,
            local_batch_size,
        ),
        allow_distributed=False,
    )


if __name__ == "__main__":
    main()
