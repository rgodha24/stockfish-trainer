"""Benchmark GPU forward+backward pass in isolation."""
from __future__ import annotations
import glob, time
import torch
from src.data import make_sparse_batch_dataset
from src.data.device import iter_device_batches
from src.model.model import NNUEModel
from src.model.config import ModelConfig
from src.model.quantize import QuantizationConfig

FEATURE_SET = "Full_Threats+HalfKAv2_hm^"
BATCH_SIZE = 16384
WARMUP = 30
MEASURE = 200
DEVICE = torch.device("cuda")

files = sorted(glob.glob("/tmp/nnue-data/*.binpack"))
if not files:
    raise SystemExit("No binpack files found in /tmp/nnue-data/")

dataset = make_sparse_batch_dataset(
    feature_set=FEATURE_SET,
    filenames=files,
    batch_size=BATCH_SIZE,
    cyclic=True,
    pin_memory=True,
)

cfg = ModelConfig(L1=1024, L2=31, L3=32, stacks="layer")
model = NNUEModel(FEATURE_SET, cfg, QuantizationConfig()).to(DEVICE)
model.train()

batches_gpu = []
print("Prefetching batches...")
for i, b in enumerate(iter_device_batches(dataset, DEVICE)):
    batches_gpu.append(b)
    if len(batches_gpu) >= WARMUP + MEASURE:
        break
print(f"Got {len(batches_gpu)} batches")

def run_one(batch):
    (us, wi, bi, outcome, score, psqt_idx, ls_idx) = batch
    out, _ = model(us, wi, bi, psqt_idx, ls_idx)
    loss = out.sum()
    loss.backward()
    model.zero_grad(set_to_none=True)

# warmup (also triggers kernel autotune)
print(f"Warming up ({WARMUP} batches)...")
for b in batches_gpu[:WARMUP]:
    run_one(b)
torch.cuda.synchronize()

# measure with CUDA events
print(f"Measuring ({MEASURE} batches)...")
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)
start_ev.record()
for b in batches_gpu[WARMUP:WARMUP + MEASURE]:
    run_one(b)
end_ev.record()
torch.cuda.synchronize()

elapsed_ms = start_ev.elapsed_time(end_ev)
per_batch_ms = elapsed_ms / MEASURE
pos_per_s = BATCH_SIZE * MEASURE / (elapsed_ms / 1000)
print(f"total_ms={elapsed_ms:.1f} per_batch_ms={per_batch_ms:.3f} pos/s={pos_per_s:.0f}")
