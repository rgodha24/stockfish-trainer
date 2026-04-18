import math
from typing import Generator

import torch
import torch.nn.functional as F
from torch import nn

from .config import LayerStacksConfig
from .stacked_linear import FactorizedStackedLinear, StackedLinear


class MoELayerStacks(nn.Module):
    def __init__(self, num_experts: int, config: LayerStacksConfig):
        super().__init__()

        self.num_experts = num_experts
        self.L2 = config.L2
        self.L3 = config.L3
        self.router_input_dim = config.router_features
        self.expert_input_dim = config.eval_features_per_perspective
        self.aux_loss_alpha = config.aux_loss_alpha
        self.z_loss_alpha = config.z_loss_alpha

        self.router = nn.Linear(self.router_input_dim, num_experts)
        self._reset_router()

        self.l1 = FactorizedStackedLinear(
            self.expert_input_dim, self.L2 + 1, num_experts
        )
        self.l2 = StackedLinear(self.L2 * 2, self.L3, num_experts)
        self.output = StackedLinear(self.L3, 1, num_experts)

        with torch.no_grad():
            self.output.linear.bias.zero_()

    @torch.no_grad()
    def _reset_router(self) -> None:
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        self.router.bias.zero_()

    def _factorized_stacked_forward_all(
        self, layer: FactorizedStackedLinear, x: torch.Tensor
    ) -> torch.Tensor:
        merged_weight = layer.linear.weight + layer.factorized_linear.weight.repeat(
            self.num_experts, 1
        )
        merged_bias = layer.linear.bias + layer.factorized_linear.bias.repeat(
            self.num_experts
        )
        expert_weight = merged_weight.reshape(
            self.num_experts, layer.out_features, layer.in_features
        )
        expert_bias = merged_bias.reshape(self.num_experts, layer.out_features)
        return torch.einsum("bi,eoi->beo", x, expert_weight) + expert_bias.unsqueeze(0)

    def _stacked_forward_all(
        self, layer: StackedLinear, x: torch.Tensor
    ) -> torch.Tensor:
        expert_weight = layer.linear.weight.reshape(
            self.num_experts, layer.out_features, layer.in_features
        )
        expert_bias = layer.linear.bias.reshape(self.num_experts, layer.out_features)
        if x.dim() == 2:
            return torch.einsum(
                "bi,eoi->beo", x, expert_weight
            ) + expert_bias.unsqueeze(0)
        return torch.einsum("bei,eoi->beo", x, expert_weight) + expert_bias.unsqueeze(0)

    def _combine_expert_outputs(
        self, gate: torch.Tensor, stacked_output: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("be,beo->bo", gate, stacked_output)

    def forward(
        self, expert_input: torch.Tensor, router_input: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate_logits = self.router(router_input.float())
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_indices = gate_logits.argmax(dim=-1)
        hard_gate = F.one_hot(expert_indices, self.num_experts).to(gate_probs.dtype)
        gate = hard_gate + gate_probs - gate_probs.detach()

        fraction_routed = hard_gate.mean(dim=0)
        avg_gate_prob = gate_probs.mean(dim=0)
        aux_loss = self.num_experts * (fraction_routed * avg_gate_prob).sum()
        z_loss = torch.logsumexp(gate_logits, dim=-1).square().mean()
        entropy = -(gate_probs * gate_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        normalized_entropy = entropy / gate_probs.new_tensor(
            math.log(max(self.num_experts, 2))
        )
        top1_prob = gate_probs.max(dim=-1).values.mean()
        router_loss = expert_input.new_zeros(())
        if self.training:
            router_loss = expert_input.new_tensor(self.aux_loss_alpha) * aux_loss
            router_loss = (
                router_loss + expert_input.new_tensor(self.z_loss_alpha) * z_loss
            )

        if self.training:
            l1c_all = self._factorized_stacked_forward_all(self.l1, expert_input)
            l1x_all, l1x_out_all = torch.split(l1c_all, [self.L2, 1], dim=2)
            l1x_all = torch.clamp(
                torch.cat([torch.pow(l1x_all, 2.0) * (255 / 256), l1x_all], dim=2),
                0.0,
                1.0,
            )
            l2c_all = self._stacked_forward_all(self.l2, l1x_all)
            l2x_all = torch.clamp(l2c_all, 0.0, 1.0)
            l3c_all = self._stacked_forward_all(self.output, l2x_all)
            l3x_all = l3c_all + l1x_out_all
            l3x_ = self._combine_expert_outputs(gate, l3x_all)
        else:
            l1c_ = self.l1(expert_input, expert_indices)
            l1x_, l1x_out = l1c_.split(self.L2, dim=1)
            l1x_ = torch.clamp(
                torch.cat([torch.pow(l1x_, 2.0) * (255 / 256), l1x_], dim=1),
                0.0,
                1.0,
            )
            l2c_ = self.l2(l1x_, expert_indices)
            l2x_ = torch.clamp(l2c_, 0.0, 1.0)
            l3c_ = self.output(l2x_, expert_indices)
            l3x_ = l3c_ + l1x_out

        return l3x_, {
            "routing/router_loss": router_loss,
            "routing/aux_loss": aux_loss,
            "routing/z_loss": z_loss,
            "routing/fraction_routed": fraction_routed,
            "routing/avg_gate_prob": avg_gate_prob,
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
        self.l1.coalesce_weights()
