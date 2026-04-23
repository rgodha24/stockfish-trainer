# Distributed Training

`src/distributed/` contains the shared Ray-backed loader pipeline used by:

- `python -m src.distributed.smoke` for multinode throughput smoke tests
- `python -m src.train.multinode` for real training on the same data path

For single-node multi-GPU training on one host, use `torchrun -m src.train.singlenode` instead. The Ray-backed `src.train.multinode` path still assumes one trainer process.

The design keeps full Rust decode+encode on feeder nodes, transfers already-encoded
batches over Ray, and reuses the same direct `BatchStream` path as single-node
training.

## Core pieces

- `config.py`: shared loader/runtime config
- `feeder.py`: Ray actor owning a sharded Rust `BatchStream`
- `pipeline.py`: trainer-side pull/tensorize loop
- `metrics.py`: progress + final summary formatting
- `smoke.py`: no-GPU throughput validation entrypoint

## Local smoke usage

```bash
python -m src.distributed.smoke \
  --feeder-count 4 \
  --batch-size 65536 \
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
