from __future__ import annotations

import tyro
from torch.utils.data import DataLoader

from src.data import make_sparse_batch_dataset
from src.train.common import run_training
from src.train.config import SingleNodeTrainingConfig


def make_train_loader(args: SingleNodeTrainingConfig) -> DataLoader:
    stream = make_sparse_batch_dataset(
        feature_set=args.features,
        filenames=list(args.datasets),
        batch_size=args.batch_size,
        cyclic=True,
        loader_threads=args.loader_threads,
        config=args.loader_skip_config(),
        shuffle_buffer_entries=args.shuffle_buffer_entries,
        pin_memory=args.pin_memory,
        encode_threads=args.encode_threads,
    )
    return DataLoader(
        stream,
        batch_size=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
    )


def main() -> None:
    args = tyro.cli(SingleNodeTrainingConfig)
    run_training(args, make_train_loader(args))


if __name__ == "__main__":
    main()
