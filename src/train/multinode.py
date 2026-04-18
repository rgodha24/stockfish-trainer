from __future__ import annotations

import tyro

from src.distributed import RayBatchStream
from src.train.common import TrainBatchSource, run_training
from src.train.config import MultiNodeTrainingConfig


def make_train_source(args: MultiNodeTrainingConfig) -> TrainBatchSource:
    stream = RayBatchStream(args.distributed_loader_config())
    return TrainBatchSource(
        batches=stream,
        metrics=stream.snapshot,
        close=lambda: stream.close(emit_summary=True),
    )


def main() -> None:
    args = tyro.cli(MultiNodeTrainingConfig)
    run_training(args, make_train_source(args))


if __name__ == "__main__":
    main()
