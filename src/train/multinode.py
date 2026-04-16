from __future__ import annotations

import tyro

from src.distributed import RayBatchStream
from src.train.common import run_training
from src.train.config import MultiNodeTrainingConfig


def make_train_loader(args: MultiNodeTrainingConfig) -> RayBatchStream:
    return RayBatchStream(args.distributed_loader_config())


def main() -> None:
    args = tyro.cli(MultiNodeTrainingConfig)
    train_loader = make_train_loader(args)
    run_training(
        args,
        train_loader,
        loader_metrics_fn=train_loader.snapshot,
        loader_close_fn=lambda: train_loader.close(emit_summary=True),
    )


if __name__ == "__main__":
    main()
