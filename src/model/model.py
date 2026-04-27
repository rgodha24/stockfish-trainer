import torch
from torch import nn

from .config import ModelConfig
from .modules import LayerStacks, MoELayerStacks, get_feature_cls
from .quantize import QuantizationConfig, QuantizationManager


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        feature_cls = get_feature_cls(feature_name)
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        self.input = feature_cls(self.L1 + self.num_psqt_buckets)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.stacks = config.stacks
        if self.stacks == "moe":
            self.layer_stacks = MoELayerStacks(config.num_experts, config)
        else:
            self.layer_stacks = LayerStacks(self.num_ls_buckets, config)

        self.quantization = QuantizationManager(quantize_config)
        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights(num_psqt_buckets, self.quantization.nnue2score)

    @torch.no_grad()
    def clip_weights(self):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data_fp32 = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                        ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                        expanded_virtual_layer = virtual_params.repeat(xs, ys)
                        if min_weight is not None:
                            min_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                        if max_weight is not None:
                            max_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                    clamped = p_data_fp32
                    if min_weight is not None:
                        if isinstance(min_weight, torch.Tensor):
                            clamped = torch.maximum(clamped, min_weight)
                        else:
                            clamped = clamped.clamp_min(min_weight)
                    if max_weight is not None:
                        if isinstance(max_weight, torch.Tensor):
                            clamped = torch.minimum(clamped, max_weight)
                        else:
                            clamped = clamped.clamp_max(max_weight)
                    p_data_fp32.copy_(clamped)

    def clip_input_weights(self):
        self.input.clip_weights(self.quantization)

    def forward(
        self,
        us: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        psqt_indices: torch.Tensor,
        layer_stack_indices: torch.Tensor,
    ):
        wp, bp = self.input(white_indices, black_indices)
        w, wpsqt = torch.split(wp, self.L1, dim=1)
        b, bpsqt = torch.split(bp, self.L1, dim=1)

        # Clamp and pairwise-product each perspective independently, then
        # select ordering (us-first) with torch.where instead of the old
        # float multiply-add. Avoids two (N, 2*L1) cat intermediates.
        chunk = self.L1 // 2
        us_bool = us.bool()
        w_c = torch.clamp(w, 0.0, 1.0).reshape(-1, 2, chunk)
        b_c = torch.clamp(b, 0.0, 1.0).reshape(-1, 2, chunk)
        # We multiply by 127/128 because in the quantized network 1.0 is represented by 127
        # and it's more efficient to divide by 128 instead.
        w_pw = w_c[:, 0, :] * w_c[:, 1, :]
        b_pw = b_c[:, 0, :] * b_c[:, 1, :]
        l0_ = torch.cat([
            torch.where(us_bool, w_pw, b_pw),
            torch.where(us_bool, b_pw, w_pw),
        ], dim=1) * (127 / 128)

        psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
        wpsqt = wpsqt.gather(1, psqt_indices_unsq)
        bpsqt = bpsqt.gather(1, psqt_indices_unsq)
        # The PSQT values are averaged over perspectives. "Their" perspective
        # has a negative influence (us-0.5 is 0.5 for white and -0.5 for black,
        # which does both the averaging and sign flip for black to move)
        psqt = (wpsqt - bpsqt) * (us - 0.5)

        stacks_out, log_dict = self.layer_stacks(l0_, layer_stack_indices)
        return stacks_out + psqt, log_dict

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch number to layer stacks for curriculum scheduling."""
        if self.stacks == "moe":
            self.layer_stacks.set_current_epoch(epoch)
