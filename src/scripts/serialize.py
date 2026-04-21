"""Export a trained model checkpoint to .nnue format for Stockfish.

Supports standard LayerStacks models, shared/no-stack models, and MoE
(Mixture of Experts) models.  All models include a router section between
the feature transformer and the per-bucket FC layers (C++ always reads it).
MoE models write quantized router weights; non-MoE models write zeros.

Binary layout (matches rgodha24/Stockfish @ fd1b945f):
    [header]  VERSION(4) | hash(4) | desc_len(4) | desc
    [FT]      ft_hash(4) | bias_i16 | per-feature weights | psqt_weights
    [router]  bias_i32[8] | weight_i8[8 * 64]
    [buckets] for each bucket: fc_hash(4) | l1 | l2 | output

Usage:
    python -m src.scripts.serialize <source.pt> <target.nnue> [options]
    python -m src.scripts.serialize <source.pt> <target.pt>   # re-save as model
"""

import hashlib
import os
import struct
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import torch
import tyro
from numba import njit
from torch import nn
from tyro.conf import OmitArgPrefixes, Positional

from src.model import ModelConfig, NNUEModel, QuantizationConfig
from src.model.modules import MoELayerStacks

# Format version (matches rgodha24/Stockfish fork — single version for all models).
VERSION = 0x7AF32F20

# Router<64, 8>::get_hash_value(0) = 0xABCD1234 + 8 = 0xABCD123C.
# Always included in the header hash (C++ Network::hash XORs it in).
ROUTER_HASH = 0xABCD123C

DEFAULT_DESCRIPTION = (
    "Network trained with the stockfish-trainer "
    "(https://github.com/rgodha24/stockfish-trainer)."
)

LEB128_MAGIC = b"COMPRESSED_LEB128"


def _encode_leb_128_array_python(arr: npt.NDArray) -> bytes:
    out = bytearray()
    for raw in arr.reshape(-1):
        value = int(raw)
        while True:
            byte = value & 0x7F
            value >>= 7
            if (value == 0 and (byte & 0x40) == 0) or (
                value == -1 and (byte & 0x40) != 0
            ):
                out.append(byte)
                break
            out.append(byte | 0x80)
    return bytes(out)


@njit
def _encode_leb_128_array_numba(arr: npt.NDArray) -> list[int]:
    out = []
    for raw in arr:
        value = int(raw)
        while True:
            byte = value & 0x7F
            value >>= 7
            if (value == 0 and (byte & 0x40) == 0) or (
                value == -1 and (byte & 0x40) != 0
            ):
                out.append(byte)
                break
            out.append(byte | 0x80)
    return out


def _encode_leb_128_array(arr: npt.NDArray) -> bytes:
    flat = np.ascontiguousarray(arr.reshape(-1))
    return bytes(_encode_leb_128_array_numba(flat))


def _fc_hash(model: NNUEModel) -> int:
    """Compute the Stockfish FC-layers hash for the model.

    The InputSlice dimension is ``L1 * 2`` (both perspectives, post-pairwise).
    This is the same for MoE and non-MoE models since the experts now see the
    full post-pairwise features.
    """
    input_slice_dim = model.L1 * 2

    prev_hash = 0xEC42E90D
    prev_hash ^= input_slice_dim

    layers = [
        model.layer_stacks.l1,
        model.layer_stacks.l2,
        model.layer_stacks.output,
    ]
    for layer in layers:
        layer_hash = 0xCC03DAE4
        layer_hash += layer.out_features
        layer_hash ^= prev_hash >> 1
        layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
        if layer.out_features != 1:
            # ClippedReLU hash
            layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
        prev_hash = layer_hash

    return prev_hash


class NNUEWriter:
    """Serialise an NNUEModel to the Stockfish .nnue binary format.

    All multi-byte integers are little-endian.

    Binary layout (matches rgodha24/Stockfish @ fd1b945f)
    -----------------------------------------------------
    [header]  VERSION(4) | hash(4) | desc_len(4) | desc
    [FT]      ft_hash(4) | bias_i16 | per-feature weights | psqt_weights
    [router]  bias_i32[8] | weight_i8[8 * 64]   (always present)
    [buckets] for each bucket: fc_hash(4) | l1 | l2 | output
    """

    def __init__(
        self,
        model: NNUEModel,
        description: str | None = None,
        ft_compression: Literal["none", "leb128"] = "leb128",
        pad_l1: int | None = None,
    ):
        if description is None:
            description = DEFAULT_DESCRIPTION

        self.buf = bytearray()

        # Effective L1 after optional padding (for SIMD tiling tests)
        self.effective_l1 = pad_l1 if pad_l1 is not None else model.L1
        self.pad_amount = self.effective_l1 - model.L1
        if self.pad_amount < 0:
            raise ValueError(f"pad_l1={pad_l1} is smaller than model L1={model.L1}")

        fc_hash = _fc_hash(model)

        # --- Header ---
        # C++ Network::hash = FT_hash ^ Arch_hash ^ Router_hash
        self.int32(VERSION)
        self.int32(fc_hash ^ model.feature_hash ^ (self.effective_l1 * 2) ^ ROUTER_HASH)
        desc_bytes = description.encode("utf-8")
        self.int32(len(desc_bytes))
        self.buf.extend(desc_bytes)

        # --- Feature transformer ---
        self.int32(model.feature_hash ^ (self.effective_l1 * 2))
        self._write_feature_transformer(model, ft_compression)

        # --- Router (always present; zeros for non-MoE) ---
        self._write_router(model)

        # --- Per-bucket / per-expert FC layers ---
        for l1, l2, output in model.layer_stacks.get_coalesced_layer_stacks():
            self.int32(fc_hash)
            self._write_fc_layer(model, l1, is_output=False)
            self._write_fc_layer(model, l2, is_output=False)
            self._write_fc_layer(model, output, is_output=True)

    # -- FT ---------------------------------------------------------------

    def _write_feature_transformer(
        self, model: NNUEModel, ft_compression: Literal["none", "leb128"]
    ) -> None:
        layer = model.input

        bias = layer.bias.data[: model.L1]
        export_weight = layer.get_export_weights()
        weight = export_weight[:, : model.L1]
        psqt_weight = export_weight[:, model.L1 :]

        bias, weight, psqt_weight = model.quantization.quantize_feature_transformer(
            bias, weight, psqt_weight
        )

        # Pad FT to effective_l1 (zero-fill extra neurons at the end)
        if self.pad_amount > 0:
            bias = torch.nn.functional.pad(bias, (0, self.pad_amount))
            weight = torch.nn.functional.pad(weight, (0, self.pad_amount))

        self._write_tensor(bias.flatten().numpy(), ft_compression)
        offset = 0
        for f in layer.features:
            n = int(getattr(f, "NUM_REAL_FEATURES"))
            segment = weight[offset : offset + n]
            if f.EXPORT_WEIGHT_DTYPE == torch.int8:
                self._write_tensor(segment.to(torch.int8).flatten().numpy())
            else:
                self._write_tensor(segment.flatten().numpy(), ft_compression)
            offset += n
        self._write_tensor(psqt_weight.flatten().numpy(), ft_compression)

    # -- Router -----------------------------------------------------------

    def _write_router(self, model: NNUEModel) -> None:
        """Write Router<64, 8> data: int32 biases[8] then int8 weights[8*64].

        C++ router.read_parameters() reads biases then weights with no
        framing (no hash prefix, no magic marker, no metadata).
        For non-MoE models we write zeros so the C++ reader stays happy.
        """
        num_experts = 8
        router_input_dim = 64  # Router<64, 8>

        stacks = model.layer_stacks
        if isinstance(stacks, MoELayerStacks):
            bias, weight = model.quantization.quantize_fc_layer(
                stacks.router.bias.data,
                stacks.router.weight.data,
                output_layer=False,
            )
        else:
            # Non-MoE: write zeros (C++ still reads the router section)
            bias = torch.zeros(num_experts, dtype=torch.int32)
            weight = torch.zeros(num_experts, router_input_dim, dtype=torch.int8)

        self.buf.extend(bias.flatten().numpy().tobytes())
        self.buf.extend(weight.flatten().numpy().tobytes())

    # -- FC layers --------------------------------------------------------

    def _write_fc_layer(
        self, model: NNUEModel, layer: nn.Linear, *, is_output: bool
    ) -> None:
        bias, weight = model.quantization.quantize_fc_layer(
            layer.bias.data, layer.weight.data, is_output
        )

        # FC inputs are padded to multiples of 32 by spec.
        num_in = weight.shape[1]
        if num_in % 32 != 0:
            padded = ((num_in + 31) // 32) * 32
            new_w = torch.zeros(weight.shape[0], padded, dtype=torch.int8)
            new_w[:, :num_in] = weight
            weight = new_w

        self.buf.extend(bias.flatten().numpy().tobytes())
        self.buf.extend(weight.flatten().numpy().tobytes())

    # -- Primitives -------------------------------------------------------

    def _write_leb_128_array(self, arr: npt.NDArray) -> None:
        payload = _encode_leb_128_array(arr)
        self.int32(len(payload))
        self.buf.extend(payload)

    def _write_tensor(
        self, arr: npt.NDArray, compression: Literal["none", "leb128"] = "none"
    ) -> None:
        if compression == "none":
            self.buf.extend(arr.tobytes())
        elif compression == "leb128":
            self.buf.extend(LEB128_MAGIC)
            self._write_leb_128_array(arr)
        else:
            raise ValueError(f"Invalid compression mode: {compression}")

    def int32(self, v: int) -> None:
        self.buf.extend(struct.pack("<I", v & 0xFFFFFFFF))


# ── Checkpoint loading ─────────────────────────────────────────────────────────


def _model_from_checkpoint(path: str) -> NNUEModel:
    """Reconstruct an NNUEModel from a training checkpoint .pt file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    cfg_dict: dict = ckpt.get("training_config", {})
    model_cfg = ModelConfig(
        L1=cfg_dict.get("l1", 1024),
        L2=cfg_dict.get("l2", 31),
        L3=cfg_dict.get("l3", 32),
        stacks=cfg_dict.get("stacks") or "layer",
        num_experts=cfg_dict.get("num_experts", 8),
        aux_loss_alpha=cfg_dict.get("aux_loss_alpha", 0.0),
        z_loss_alpha=cfg_dict.get("z_loss_alpha", 1e-3),
    )
    features = cfg_dict.get("features", "Full_Threats+HalfKAv2_hm^")

    model = NNUEModel(features, model_cfg, QuantizationConfig())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_model(path: str) -> NNUEModel:
    """Load a model from a .pt file (either a checkpoint dict or a saved model object)."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return _model_from_checkpoint(path)
    if isinstance(raw, NNUEModel):
        raw.eval()
        return raw
    raise ValueError(
        f"Unrecognised .pt format in {path!r}. "
        "Expected a training checkpoint dict or a saved NNUEModel."
    )


def _describe_model_for_export(model: NNUEModel) -> str:
    if isinstance(model.layer_stacks, MoELayerStacks):
        stacks = model.layer_stacks
        return (
            f"  MoE model: {stacks.num_experts} experts, "
            f"router_input_dim={stacks.ROUTER_INPUT_DIM}, L1={model.L1}"
        )

    count = model.layer_stacks.count
    if getattr(model, "stacks", "layer") == "none":
        return f"  Shared/no-stack model: {count} identical buckets, L1={model.L1}"

    return f"  LayerStacks model: {count} buckets, L1={model.L1}"


# ── CLI ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SerializeConfig:
    out_sha: bool = False
    """Ignore target filename and write as nn-<sha256[:12]>.nnue.
    If target is a directory the file is placed there; otherwise it goes to
    dirname(target) or CWD."""

    description: Optional[str] = None
    """Description string embedded in the .nnue header."""

    ft_compression: Literal["none", "leb128"] = "leb128"
    """Compression method for FT arrays in .nnue output."""

    pad_l1: Optional[int] = None
    """Pad feature transformer from model's L1 to this value (zero-fill).
    Useful for testing SIMD tiling effects without retraining."""


@dataclass(frozen=True)
class CliConfig:
    source: Positional[str]
    """Source file (.pt checkpoint)."""

    target: Positional[str]
    """Target file (.pt or .nnue) or directory (with --out-sha)."""

    serialize_config: OmitArgPrefixes[SerializeConfig] = field(
        default_factory=SerializeConfig
    )


def main() -> None:
    args = tyro.cli(CliConfig)
    cfg = args.serialize_config

    print(f"Converting {args.source!r} → {args.target!r}")

    model = _load_model(args.source)
    print(_describe_model_for_export(model))

    target_is_nnue = cfg.out_sha or args.target.endswith(".nnue")

    if args.target.endswith(".ckpt"):
        raise ValueError("Cannot write .ckpt files; use .pt or .nnue as target.")

    if args.target.endswith(".pt"):
        torch.save(model, args.target)
        print(f"Wrote {args.target}")
        return

    if target_is_nnue:
        writer = NNUEWriter(
            model,
            description=cfg.description,
            ft_compression=cfg.ft_compression,
            pad_l1=cfg.pad_l1,
        )
        buf = bytes(writer.buf)

        if cfg.out_sha:
            sha = hashlib.sha256(buf).hexdigest()
            if os.path.isdir(args.target):
                out_dir = os.path.abspath(args.target)
            else:
                out_dir = os.path.abspath(os.path.dirname(args.target) or os.getcwd())
            os.makedirs(out_dir, exist_ok=True)
            final = os.path.join(out_dir, f"nn-{sha[:12]}.nnue")
        else:
            final = args.target
            os.makedirs(os.path.dirname(os.path.abspath(final)) or ".", exist_ok=True)

        with open(final, "wb") as f:
            f.write(buf)
        print(f"Wrote {final}  ({len(buf):,} bytes)")
        return

    raise ValueError(f"Unknown target format for {args.target!r}. Use .pt or .nnue.")


if __name__ == "__main__":
    main()
