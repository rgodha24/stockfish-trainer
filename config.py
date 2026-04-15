from dataclasses import dataclass
from typing import Annotated

import tyro
from tyro.conf import Positional


@dataclass(kw_only=True)
class TrainingConfig:
    datasets: Positional[tuple[str, ...]] = ()
    max_epochs: int = 800
    batch_size: int = 16384
    features: str = "Full_Threats+HalfKAv2_hm^"

    l1: int = 1024
    l2: int = 31
    l3: int = 32
    layer_stacks: bool = True

    lr: float = 8.75e-4
    gamma: float = 0.992

    in_offset: float = 270.0
    out_offset: float = 270.0
    in_scaling: float = 340.0
    out_scaling: float = 380.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0
    w1: float = 0.0
    w2: float = 0.5
    lambda_: Annotated[float, tyro.conf.arg(name="lambda")] = 1.0
    start_lambda: float | None = None
    end_lambda: float | None = None

    random_fen_skipping: int = 0
    early_fen_skipping: int = -1
    simple_eval_skipping: int = -1
    filtered: bool = True
    wld_filtered: bool = True

    epoch_size: int = 100_000_000
    loader_threads: int = -1
    pin_memory: bool = True
    data_loader_queue_size: int = 16

    seed: int = 42
    default_root_dir: str = "logs"
    wandb_project: str = "stockfish-trainer"
    wandb_run_name: str | None = None

    def __post_init__(self):
        if not self.datasets:
            raise ValueError("Argument `datasets` is required.")
        if self.max_epochs <= 0 or self.epoch_size <= 0 or self.batch_size <= 0:
            raise ValueError(
                "Arguments `max_epochs`, `epoch_size` and `batch_size` must be positive."
            )
