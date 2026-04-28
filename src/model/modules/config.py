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
    """Coefficient for anti-collapse per-expert load floor/cap regularization."""
    z_loss_alpha: float = 0.0
    """router z-loss coefficient for MoE routing"""
    router_load_floor: float = 0.0
    """Minimum desired routed fraction per expert (0 disables the floor term)."""
    router_load_cap: float = 1.0
    """Maximum desired routed fraction per expert (1 disables the cap term)."""
    router_teacher_alpha: float = 0.0
    """Weight of explicit CE loss toward the piece-count bucket during MoE warm-start."""
    router_teacher_anneal_epochs: int = 0
    """Linearly decay router_teacher_alpha to zero over this many epochs (0 keeps it constant)."""
    probe_loss_alpha: float = 0.0
    """Coefficient for the per-expert router probe loss."""
    probe_loss_tau: float = 0.05
    """Temperature for soft targets derived from per-expert score error."""
    probe_loss_teacher_threshold: float = 0.2
    """Probe loss ramps in once teacher alpha drops below this threshold."""
    probe_loss_ramp_power: float = 1.0
    """Exponent applied to probe ramp progress for a gentler handoff from teacher."""
    probe_loss_ramp_start_epoch: int = 0
    """Epoch where the external probe ramp cap starts increasing from zero."""
    probe_loss_ramp_end_epoch: int = 0
    """Epoch where the external probe ramp cap reaches one (<= start disables the cap)."""

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
        if not 0.0 <= self.router_load_floor <= 1.0:
            raise ValueError("`router_load_floor` must be in [0, 1].")
        if not 0.0 <= self.router_load_cap <= 1.0:
            raise ValueError("`router_load_cap` must be in [0, 1].")
        if self.router_load_floor > self.router_load_cap:
            raise ValueError("`router_load_floor` must be <= `router_load_cap`.")
        if self.router_teacher_alpha < 0.0:
            raise ValueError("`router_teacher_alpha` must be non-negative.")
        if self.router_teacher_anneal_epochs < 0:
            raise ValueError("`router_teacher_anneal_epochs` must be non-negative.")
        if self.probe_loss_alpha < 0.0:
            raise ValueError("`probe_loss_alpha` must be non-negative.")
        if self.probe_loss_tau <= 0.0:
            raise ValueError("`probe_loss_tau` must be positive.")
        if self.probe_loss_teacher_threshold < 0.0:
            raise ValueError("`probe_loss_teacher_threshold` must be non-negative.")
        if self.probe_loss_ramp_power <= 0.0:
            raise ValueError("`probe_loss_ramp_power` must be positive.")
        if self.probe_loss_ramp_start_epoch < 0:
            raise ValueError("`probe_loss_ramp_start_epoch` must be non-negative.")
        if self.probe_loss_ramp_end_epoch < 0:
            raise ValueError("`probe_loss_ramp_end_epoch` must be non-negative.")
