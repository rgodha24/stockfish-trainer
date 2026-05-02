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
    def forward(ctx, flat_indices, offsets, weight, bias):
        assert len(flat_indices.shape) == 1
        assert flat_indices.dtype == torch.int32
        assert len(offsets.shape) == 1
        assert offsets.dtype == torch.int32
        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32
        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32
        assert flat_indices.is_cuda
        assert offsets.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda
        assert weight.device == flat_indices.device
        assert offsets.device == flat_indices.device
        assert bias.device == flat_indices.device
        assert flat_indices.is_contiguous()
        assert offsets.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        output = _load_ext().cuembed_forward(flat_indices, offsets, weight)
        ctx.save_for_backward(flat_indices, offsets)
        ctx.num_features = weight.shape[0]
        return output + bias

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]
        flat_indices, offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        weight_grad = _load_ext().cuembed_backward(
            flat_indices,
            offsets,
            grad_output,
            ctx.num_features,
        )
        bias_grad = grad_output.sum(dim=0)
        return None, None, weight_grad, bias_grad
