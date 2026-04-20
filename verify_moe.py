"""Compare Python MoE model vs serialized .nnue file to find mismatches."""

import struct
import sys

import numpy as np
import torch

from src.model import ModelConfig, NNUEModel, QuantizationConfig
from src.model.modules import MoELayerStacks


def load_checkpoint(path: str) -> NNUEModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict: dict = ckpt.get("training_config", {})
    model_cfg = ModelConfig(
        L1=cfg_dict.get("l1", 1024),
        L2=cfg_dict.get("l2", 31),
        L3=cfg_dict.get("l3", 32),
        stacks=cfg_dict.get("stacks") or "layer",
        num_experts=cfg_dict.get("num_experts", 8),
        router_features=cfg_dict.get("router_features", 32),
        aux_loss_alpha=cfg_dict.get("aux_loss_alpha", 1e-3),
        z_loss_alpha=cfg_dict.get("z_loss_alpha", 0.0),
    )
    features = cfg_dict.get("features", "Full_Threats+HalfKAv2_hm^")
    model = NNUEModel(features, model_cfg, QuantizationConfig())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def read_nnue_router(path: str, num_experts: int = 8, router_dim: int = 32):
    """Read router weights from a .nnue MoE file."""
    with open(path, "rb") as f:
        data = f.read()

    # Find router magic: 0x00B50000 followed by 0x0000000B
    magic_lo = struct.pack("<I", 0x00B50000)
    magic_hi = struct.pack("<I", 0x0000000B)

    idx = data.find(magic_lo + magic_hi)
    if idx < 0:
        print("ERROR: Router magic not found in .nnue file!")
        return None, None, None

    pos = idx + 8  # skip magic
    ne = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    rd = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    ef = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    print(f"  Router metadata: num_experts={ne}, router_dim={rd}, eval_features={ef}")

    # Read weights [num_experts, router_dim]
    w_size = ne * rd
    weights = np.frombuffer(data, dtype=np.float32, count=w_size, offset=pos).reshape(ne, rd)
    pos += w_size * 4

    # Read biases [num_experts]
    biases = np.frombuffer(data, dtype=np.float32, count=ne, offset=pos)
    pos += ne * 4

    return weights, biases, ef


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/external/models/moe-fr/final.pt"
    nnue_path = sys.argv[2] if len(sys.argv) > 2 else "/mnt/external/models/moe-fr/final.nnue"

    print(f"Loading checkpoint: {ckpt_path}")
    model = load_checkpoint(ckpt_path)
    stacks = model.layer_stacks
    assert isinstance(stacks, MoELayerStacks), "Not a MoE model!"

    print(f"  num_experts={stacks.num_experts}")
    print(f"  router_input_dim={stacks.router_input_dim}")
    print(f"  expert_input_dim={stacks.expert_input_dim}")
    print(f"  L2={stacks.L2}, L3={stacks.L3}")
    print(f"  eval_features_per_perspective={model.eval_features_per_perspective}")

    # Python router weights
    py_router_w = stacks.router.weight.data.cpu().float().numpy()  # [num_experts, router_dim]
    py_router_b = stacks.router.bias.data.cpu().float().numpy()  # [num_experts]

    print(f"\nPython router weight shape: {py_router_w.shape}, range: [{py_router_w.min():.6f}, {py_router_w.max():.6f}]")
    print(f"Python router bias: {py_router_b}")

    # Read .nnue file
    print(f"\nReading .nnue: {nnue_path}")
    nnue_w, nnue_b, ef = read_nnue_router(nnue_path, stacks.num_experts, stacks.router_input_dim)

    if nnue_w is None:
        return

    print(f"NNUE router weight shape: {nnue_w.shape}, range: [{nnue_w.min():.6f}, {nnue_w.max():.6f}]")
    print(f"NNUE router bias: {nnue_b}")

    # Compare
    w_match = np.allclose(py_router_w, nnue_w, atol=1e-6)
    b_match = np.allclose(py_router_b, nnue_b, atol=1e-6)
    print(f"\nRouter weights match: {w_match}")
    print(f"Router biases match: {b_match}")

    if not w_match:
        diff = np.abs(py_router_w - nnue_w)
        print(f"  Max weight diff: {diff.max():.8f}")
        print(f"  Mean weight diff: {diff.mean():.8f}")

    # Compare FC layers per expert
    print("\n--- FC Layer Comparison ---")
    quant = model.quantization
    for i, (l1, l2, out) in enumerate(stacks.get_coalesced_layer_stacks()):
        print(f"\nExpert {i}:")
        # Check l1 (hidden, not output)
        l1_bias_q, l1_weight_q = quant.quantize_fc_layer(l1.bias.data, l1.weight.data, False)
        print(f"  l1: in={l1.in_features} out={l1.out_features}, weight range [{l1.weight.data.min():.4f}, {l1.weight.data.max():.4f}], "
              f"quantized weight range [{l1_weight_q.min()}, {l1_weight_q.max()}]")
        l2_bias_q, l2_weight_q = quant.quantize_fc_layer(l2.bias.data, l2.weight.data, False)
        print(f"  l2: in={l2.in_features} out={l2.out_features}, weight range [{l2.weight.data.min():.4f}, {l2.weight.data.max():.4f}], "
              f"quantized weight range [{l2_weight_q.min()}, {l2_weight_q.max()}]")
        out_bias_q, out_weight_q = quant.quantize_fc_layer(out.bias.data, out.weight.data, True)
        print(f"  out: in={out.in_features} out={out.out_features}, weight range [{out.weight.data.min():.4f}, {out.weight.data.max():.4f}], "
              f"quantized weight range [{out_weight_q.min()}, {out_weight_q.max()}]")

    # Test: given a random router input, which expert is selected?
    print("\n--- Router Test ---")
    torch.manual_seed(42)
    test_input = torch.rand(1, stacks.router_input_dim)
    logits = stacks.router(test_input.float())
    py_expert = logits.argmax(dim=-1).item()
    print(f"Random router input -> Python selects expert {py_expert}")
    print(f"  logits: {logits.detach().numpy().flatten()}")

    # Simulate C++ router
    cpp_logits = nnue_w @ test_input.numpy().flatten() + nnue_b
    cpp_expert = np.argmax(cpp_logits)
    print(f"  C++ would select expert {cpp_expert}")
    print(f"  C++ logits: {cpp_logits}")

    # Check if the model's expert weights are all similar (collapsed)
    print("\n--- Expert Diversity Check ---")
    experts = list(stacks.get_coalesced_layer_stacks())
    for layer_name in ['l1', 'l2', 'out']:
        layer_idx = {'l1': 0, 'l2': 1, 'out': 2}[layer_name]
        weights = [e[layer_idx].weight.data for e in experts]
        # Pairwise cosine similarity
        sims = []
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                flat_i = weights[i].flatten()
                flat_j = weights[j].flatten()
                sim = torch.nn.functional.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0))
                sims.append(sim.item())
        print(f"  {layer_name}: mean pairwise cosine similarity = {np.mean(sims):.4f} (1.0 = identical experts)")


if __name__ == "__main__":
    main()
