# Distributed Training

`src/distributed/` contains the shared Ray-backed loader pipeline used by:

- `python -m src.distributed.smoke` for multinode throughput smoke tests
- `python -m src.train.multinode` for real training on the same data path

The design keeps decode on feeder nodes, transfers packed 32-byte entries over Ray,
and performs batch encoding on the training node so smoke-test throughput matches the
real head-node bottleneck we care about.

## Core pieces

- `config.py`: shared loader/runtime config
- `feeder.py`: Ray actor owning a sharded Rust `PackedEntryStream`
- `pipeline.py`: trainer-side pull/encode/tensorize loop
- `metrics.py`: progress + final summary formatting
- `smoke.py`: no-GPU throughput validation entrypoint

## Local smoke usage

```bash
python -m src.distributed.smoke \
  --feeder-count 4 \
  --batch-size 65536 \
  --chunk-entries 8192 \
  --bundle-chunks 1 \
  --inflight-per-feeder 1 \
  --target-batches 200 \
  /path/to/data1.binpack /path/to/data2.binpack
```

## Slurm launchers

- `src/scripts/slurm_multinode_smoke.sbatch`: CPU-only smoke test
- `src/scripts/slurm_multinode_h100.sbatch`: GPU training launcher

Both scripts start a Ray head on the trainer node, attach worker nodes as feeder-only
Ray workers, wait for the expected worker CPU capacity, and then launch either the
smoke test or the real trainer.
