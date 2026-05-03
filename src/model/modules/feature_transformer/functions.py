import torch
from torch import autograd

from .kernel import (
    make_sparse_input_linear_forward_kernel,
    make_sparse_input_linear_backward_kernel,
)


class SparseLinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, feature_indices, weight, bias):
        ctx.save_for_backward(feature_indices, weight, bias)

        assert len(feature_indices.shape) == 2
        assert feature_indices.dtype == torch.int32
        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32
        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32
        assert feature_indices.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device
        assert feature_indices.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(
            batch_size,
            output_size,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        kernel = make_sparse_input_linear_forward_kernel(
            max_active_features, output_size
        )
        kernel(feature_indices, weight, bias, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]

        grad_output = grad_output.contiguous()

        feature_indices, weight, _bias = ctx.saved_tensors

        device = feature_indices.device
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(
            weight.shape[0], weight.shape[1], dtype=torch.float32, device=device
        )
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        kernel = make_sparse_input_linear_backward_kernel(
            max_active_features, output_size
        )
        kernel(feature_indices, weight_grad, bias_grad, grad_output)

        return None, weight_grad, bias_grad
