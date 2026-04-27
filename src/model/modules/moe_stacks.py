import math
from typing import Generator

import torch
import torch.nn.functional as F
from torch import nn

from .config import LayerStacksConfig
from .stacked_linear import FactorizedStackedLinear, StackedLinear


class MoELayerStacks(nn.Module):
    # Router reads first 32 post-pairwise features from each perspective.
    ROUTER_FEATURES_PER_PERSPECTIVE: int = 32
    ROUTER_INPUT_DIM: int = ROUTER_FEATURES_PER_PERSPECTIVE * 2  # 64

    def __init__(self, num_experts: int, config: LayerStacksConfig):
        super().__init__()

        self.num_experts = num_experts
        self.L2 = config.L2
        self.L3 = config.L3
        self.aux_loss_alpha = config.aux_loss_alpha
        self.z_loss_alpha = config.z_loss_alpha
        self.router_load_floor = config.router_load_floor
        self.router_load_cap = config.router_load_cap
        self.router_teacher_alpha = config.router_teacher_alpha
        self.router_teacher_anneal_epochs = config.router_teacher_anneal_epochs
        self.register_buffer("_teacher_alpha", torch.tensor(0.0))

        self.router = nn.Linear(self.ROUTER_INPUT_DIM, num_experts)
        self._reset_router()

        self.l1 = FactorizedStackedLinear(config.L1, self.L2 + 1, num_experts)
        self.l2 = StackedLinear(self.L2 * 2, self.L3, num_experts)
        self.output = StackedLinear(self.L3, 1, num_experts)
        self._diversify_expert_inits()

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def set_current_epoch(self, epoch: int) -> None:
        """Set current epoch for teacher CE scheduling."""
        teacher_alpha = self.router_teacher_alpha
        if self.router_teacher_anneal_epochs > 0:
            progress = min(epoch / self.router_teacher_anneal_epochs, 1.0)
            teacher_alpha *= 1.0 - progress
        self.get_buffer("_teacher_alpha").fill_(teacher_alpha)

    @torch.no_grad()
    def _reset_router(self) -> None:
        nn.init.normal_(self.router.weight, mean=0.0, std=0.1)
        self.router.bias.zero_()

    @torch.no_grad()
    def _diversify_expert_inits(self) -> None:
        """Re-initialize each expert with independent random weights to break symmetry."""
        for layer in (self.l1, self.l2, self.output):
            for i in range(self.num_experts):
                begin = i * layer.out_features
                end = (i + 1) * layer.out_features
                nn.init.kaiming_uniform_(
                    layer.linear.weight[begin:end, :], a=math.sqrt(5)
                )
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    layer.linear.weight[begin:end, :]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(layer.linear.bias[begin:end], -bound, bound)

    def _expert_params(self, layer: StackedLinear) -> tuple[torch.Tensor, torch.Tensor]:
        weight = layer.linear.weight
        bias = layer.linear.bias
        if isinstance(layer, FactorizedStackedLinear):
            weight = weight + layer.factorized_linear.weight.repeat(layer.count, 1)
            bias = bias + layer.factorized_linear.bias.repeat(layer.count)
        weight = weight.view(layer.count, layer.out_features, layer.in_features)
        bias = bias.view(layer.count, layer.out_features)
        return weight.contiguous(), bias.contiguous()

    def _all_experts_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute all experts in parallel. Returns (B, num_experts, 1)."""
        # l1: (B, in) -> (B, num_experts, L2+1)
        l1_w, l1_b = self._expert_params(self.l1)
        l1c_ = torch.einsum("bi,eoi->beo", x, l1_w) + l1_b

        l1x_, l1x_out = l1c_.split(self.L2, dim=2)
        l1x_ = torch.clamp(
            torch.cat([l1x_.square() * (255 / 256), l1x_], dim=2),
            0.0,
            1.0,
        )

        # l2: (B, num_experts, L2*2) -> (B, num_experts, L3)
        l2_w, l2_b = self._expert_params(self.l2)
        l2c_ = torch.einsum("bei,eoi->beo", l1x_, l2_w) + l2_b
        l2x_ = torch.clamp(l2c_, 0.0, 1.0)

        # output: (B, num_experts, L3) -> (B, num_experts, 1)
        out_w, out_b = self._expert_params(self.output)
        l3c_ = torch.einsum("bei,eoi->beo", l2x_, out_w) + out_b
        return l3c_ + l1x_out

    def forward(
        self,
        x: torch.Tensor,
        _ls_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        half = x.shape[1] // 2
        router_input = torch.cat(
            [
                x[:, : self.ROUTER_FEATURES_PER_PERSPECTIVE],
                x[:, half : half + self.ROUTER_FEATURES_PER_PERSPECTIVE],
            ],
            dim=1,
        )
        gate_logits = self.router(router_input)

        assert _ls_indices is not None
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_indices = gate_logits.argmax(dim=-1)

        # Teacher metrics (how well does the router match piece-count buckets?)
        target_prob = gate_probs.gather(1, _ls_indices.unsqueeze(1)).mean()
        bucket_agreement = (expert_indices == _ls_indices).to(gate_probs.dtype).mean()
        teacher_ce = F.cross_entropy(gate_logits, _ls_indices)

        fraction_routed = torch.bincount(expert_indices, minlength=self.num_experts).to(
            gate_probs.dtype
        ) / gate_probs.new_tensor(float(expert_indices.numel()))
        avg_gate_prob = gate_probs.mean(dim=0)
        load_estimate = avg_gate_prob + (fraction_routed - avg_gate_prob).detach()
        load_floor_loss = gate_probs.new_zeros(())
        if self.router_load_floor > 0.0:
            load_floor_loss = (
                F.relu(load_estimate.new_tensor(self.router_load_floor) - load_estimate)
                .square()
                .mean()
            )
        load_cap_loss = gate_probs.new_zeros(())
        if self.router_load_cap < 1.0:
            load_cap_loss = (
                F.relu(load_estimate - load_estimate.new_tensor(self.router_load_cap))
                .square()
                .mean()
            )
        aux_loss = load_floor_loss + load_cap_loss
        z_loss = torch.logsumexp(gate_logits, dim=-1).square().mean()
        entropy = -(gate_probs * gate_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        normalized_entropy = entropy / gate_probs.new_tensor(
            math.log(max(self.num_experts, 2))
        )
        top1_prob = gate_probs.max(dim=-1).values.mean()
        teacher_alpha = self.get_buffer("_teacher_alpha").to(dtype=gate_probs.dtype)
        router_loss = x.new_zeros(())
        if self.training:
            router_loss = x.new_tensor(self.aux_loss_alpha) * aux_loss
            router_loss = router_loss + x.new_tensor(self.z_loss_alpha) * z_loss
            router_loss = router_loss + teacher_alpha * teacher_ce

        # Dense all-experts path: efficient batched GEMMs with standard
        # autograd, no GPU->CPU syncs in the backward.
        l3x_ = (
            self._all_experts_forward(x)
            .gather(1, expert_indices.view(-1, 1, 1).expand(-1, 1, 1))
            .squeeze(1)
        )
        if self.training:
            # Straight-through estimator: forward value stays unscaled,
            # but router gets gradients through gate_prob.
            gate_prob = gate_probs.gather(1, expert_indices.unsqueeze(1))
            l3x_ = l3x_ * (1.0 + gate_prob - gate_prob.detach())
        return l3x_, {
            "routing/router_loss": router_loss,
            "routing/aux_loss": aux_loss,
            "routing/load_floor_loss": load_floor_loss,
            "routing/load_cap_loss": load_cap_loss,
            "routing/z_loss": z_loss,
            "routing/fraction_routed": fraction_routed,
            "routing/avg_gate_prob": avg_gate_prob,
            "routing/target_prob": target_prob,
            "routing/bucket_agreement": bucket_agreement,
            "routing/teacher_ce": teacher_ce,
            "routing/teacher_alpha": teacher_alpha,
            "routing/entropy": normalized_entropy,
            "routing/top1_prob": top1_prob,
        }

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        for i in range(self.num_experts):
            yield self.l1.at_index(i), self.l2.at_index(i), self.output.at_index(i)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        if hasattr(self.l1, "coalesce_weights"):
            self.l1.coalesce_weights()
