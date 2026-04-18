from __future__ import annotations

import tyro

from src.data import make_sparse_batch_dataset
from src.train.common import TrainBatchSource, run_training
from src.train.config import SingleNodeTrainingConfig


def make_train_source(args: SingleNodeTrainingConfig) -> TrainBatchSource:
    batches = make_sparse_batch_dataset(
        feature_set=args.features,
        filenames=list(args.datasets),
        batch_size=args.batch_size,
        cyclic=True,
        loader_threads=args.loader_threads,
        config=args.loader_skip_config(),
        shuffle_buffer_entries=args.shuffle_buffer_entries,
        pin_memory=args.pin_memory,
    )
    return TrainBatchSource(batches=batches, metrics=lambda: dict(), close=lambda: None)


def main() -> None:
    args = tyro.cli(SingleNodeTrainingConfig)
    run_training(args, make_train_source(args))


if __name__ == "__main__":
    main()
