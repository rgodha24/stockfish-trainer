from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import autograd
from torch.utils.cpp_extension import load

_ext = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@torch.compiler.disable(recursive=False)
def _load_ext():
    global _ext
    if _ext is not None:
        return _ext

    root = _repo_root()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    _ext = load(
        name="stockfish_trainer_cuembed_ext",
        sources=[str(Path(__file__).with_name("cuembed_wrapper.cu"))],
        extra_include_paths=[str(root / "third_party" / "cuembed")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        with_cuda=True,
        verbose=bool(int(os.environ.get("STOCKFISH_TRAINER_CUEMBED_VERBOSE", "0"))),
    )
    return _ext


class CuEmbedSparseLinearFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        flat_indices_0,
        offsets_0,
        flat_indices_1,
        offsets_1,
        weight,
        bias,
    ):
        assert len(flat_indices_0.shape) == 1
        assert flat_indices_0.dtype == torch.int32
        assert len(offsets_0.shape) == 1
        assert offsets_0.dtype == torch.int32
        assert len(flat_indices_1.shape) == 1
        assert flat_indices_1.dtype == torch.int32
        assert len(offsets_1.shape) == 1
        assert offsets_1.dtype == torch.int32
        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32
        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32
        assert flat_indices_0.is_cuda
        assert offsets_0.is_cuda
        assert flat_indices_1.is_cuda
        assert offsets_1.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda
        assert weight.device == flat_indices_0.device
        assert offsets_0.device == flat_indices_0.device
        assert flat_indices_1.device == flat_indices_0.device
        assert offsets_1.device == flat_indices_0.device
        assert bias.device == flat_indices_0.device
        assert flat_indices_0.is_contiguous()
        assert offsets_0.is_contiguous()
        assert flat_indices_1.is_contiguous()
        assert offsets_1.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        ext = _load_ext()
        output_0 = ext.cuembed_forward(flat_indices_0, offsets_0, weight)
        output_1 = ext.cuembed_forward(flat_indices_1, offsets_1, weight)
        ctx.save_for_backward(flat_indices_0, offsets_0, flat_indices_1, offsets_1)
        ctx.num_features = weight.shape[0]
        return output_0 + bias, output_1 + bias

    @staticmethod
    def backward(ctx, grad_output_0, grad_output_1):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]
        assert not ctx.needs_input_grad[3]
        flat_indices_0, offsets_0, flat_indices_1, offsets_1 = ctx.saved_tensors
        grad_output_0 = grad_output_0.contiguous()
        grad_output_1 = grad_output_1.contiguous()
        ext = _load_ext()
        weight_grad = ext.cuembed_backward(
            flat_indices_0,
            offsets_0,
            grad_output_0,
            ctx.num_features,
        )
        weight_grad.add_(
            ext.cuembed_backward(
                flat_indices_1,
                offsets_1,
                grad_output_1,
                ctx.num_features,
            )
        )
        bias_grad = grad_output_0.sum(dim=0) + grad_output_1.sum(dim=0)
        return None, None, None, None, weight_grad, bias_grad
