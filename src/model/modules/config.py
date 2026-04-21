from dataclasses import dataclass
from typing import Annotated, Literal

import tyro

type Stacks = Literal["none"] | Literal["layer"] | Literal["moe"]


# 3 layer fully connected network
@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 31
    """Size of second hidden layer."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Size of third hidden layer."""
    stacks: Stacks = "layer"
    """what types of stacks to use"""
    num_experts: int = 8
    """number of experts used when stacks='moe'"""
    aux_loss_alpha: float = 0.001
    """Switch-style load-balancing coefficient for MoE routing"""
    z_loss_alpha: float = 0.0
    """router z-loss coefficient for MoE routing"""
    gumbel_tau_start: float = 2.0
    """Gumbel-Softmax starting temperature (high = soft/exploratory)"""
    gumbel_tau_end: float = 0.3
    """Gumbel-Softmax ending temperature (low = sharp/committed)"""
    gumbel_anneal_fraction: float = 0.3
    """Fraction of total training over which to anneal tau"""

    def __post_init__(self) -> None:
        if self.L1 <= 0 or self.L2 <= 0 or self.L3 <= 0:
            raise ValueError("`l1`, `l2`, and `l3` must be positive.")
        if self.L1 % 2 != 0:
            raise ValueError("`l1` must be even.")
        if self.num_experts <= 0:
            raise ValueError("`num_experts` must be positive.")
        if self.aux_loss_alpha < 0.0:
            raise ValueError("`aux_loss_alpha` must be non-negative.")
        if self.z_loss_alpha < 0.0:
            raise ValueError("`z_loss_alpha` must be non-negative.")
