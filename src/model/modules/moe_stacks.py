import math
from typing import Generator

import torch
import torch.nn.functional as F
from torch import nn

from .config import LayerStacksConfig
from .sparse_expert_linear import sparse_expert_linear
from .stacked_linear import StackedLinear


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
        self.gumbel_tau_start = config.gumbel_tau_start
        self.gumbel_tau_end = config.gumbel_tau_end
        self.gumbel_anneal_fraction = config.gumbel_anneal_fraction
        self._training_progress: float = 0.0

        self.router = nn.Linear(self.router_input_dim, num_experts)
        self._reset_router()

        self.l1 = StackedLinear(
            self.expert_input_dim, self.L2 + 1, num_experts
        )
        self.l2 = StackedLinear(self.L2 * 2, self.L3, num_experts)
        self.output = StackedLinear(self.L3, 1, num_experts)

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def set_training_progress(self, progress: float) -> None:
        """Set training progress (0.0 to 1.0) for tau annealing."""
        self._training_progress = progress

    @property
    def current_tau(self) -> float:
        """Compute current Gumbel-Softmax temperature from training progress."""
        if self._training_progress >= self.gumbel_anneal_fraction:
            return self.gumbel_tau_end
        t = self._training_progress / self.gumbel_anneal_fraction
        return self.gumbel_tau_start + (self.gumbel_tau_end - self.gumbel_tau_start) * t

    @torch.no_grad()
    def _reset_router(self) -> None:
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        self.router.bias.zero_()

    def _expert_params(
        self, layer: StackedLinear
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = layer.linear.weight.view(
            layer.count, layer.out_features, layer.in_features
        )
        bias = layer.linear.bias.view(layer.count, layer.out_features)
        return weight.contiguous(), bias.contiguous()

    def _all_experts_forward(
        self,
        expert_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute all experts in parallel. Returns (B, num_experts, 1)."""
        # l1: (B, in) -> (B, num_experts, L2+1)
        l1_w, l1_b = self._expert_params(self.l1)
        l1c_ = torch.einsum("bi,eoi->beo", expert_input, l1_w) + l1_b

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

    def _sparse_forward(
        self,
        expert_input: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        l1_weight, l1_bias = self._expert_params(self.l1)
        l1c_ = sparse_expert_linear(expert_input, expert_indices, l1_weight, l1_bias)
        l1x_, l1x_out = l1c_.split(self.L2, dim=1)
        l1x_ = torch.clamp(
            torch.cat([l1x_.square() * (255 / 256), l1x_], dim=1),
            0.0,
            1.0,
        )

        l2_weight, l2_bias = self._expert_params(self.l2)
        l2c_ = sparse_expert_linear(l1x_, expert_indices, l2_weight, l2_bias)
        l2x_ = torch.clamp(l2c_, 0.0, 1.0)

        out_weight, out_bias = self._expert_params(self.output)
        l3c_ = sparse_expert_linear(l2x_, expert_indices, out_weight, out_bias)
        return l3c_ + l1x_out

    def forward(
        self, expert_input: torch.Tensor, router_input: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate_logits = self.router(router_input.float())
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_indices = gate_logits.argmax(dim=-1)
        fraction_routed = torch.bincount(expert_indices, minlength=self.num_experts).to(
            gate_probs.dtype
        ) / gate_probs.new_tensor(float(expert_indices.numel()))
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
            # All-experts forward + Gumbel-Softmax soft mixing for gradient flow
            all_outputs = self._all_experts_forward(expert_input)  # (B, E, 1)
            tau = self.current_tau
            routing_weights = F.gumbel_softmax(
                gate_logits, tau=tau, hard=True
            )
            l3x_ = (all_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)  # (B, 1)
        else:
            # Inference: hard argmax, single expert (sparse kernel on CUDA)
            if expert_input.is_cuda and expert_input.dtype == torch.float32:
                l3x_ = self._sparse_forward(expert_input, expert_indices)
            else:
                all_outputs = self._all_experts_forward(expert_input)
                l3x_ = all_outputs.gather(
                    1, expert_indices.view(-1, 1, 1).expand(-1, 1, all_outputs.shape[2])
                ).squeeze(1)

        return l3x_, {
            "routing/router_loss": router_loss,
            "routing/aux_loss": aux_loss,
            "routing/z_loss": z_loss,
            "routing/fraction_routed": fraction_routed,
            "routing/avg_gate_prob": avg_gate_prob,
            "routing/entropy": normalized_entropy,
            "routing/top1_prob": top1_prob,
            "routing/tau": expert_input.new_tensor(tau if self.training else 0.0),
        }

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        for i in range(self.num_experts):
            yield self.l1.at_index(i), self.l2.at_index(i), self.output.at_index(i)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        pass  # No factorized weights to coalesce
