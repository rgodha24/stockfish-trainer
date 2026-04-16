# Ray Loader Test Harness

`src/raytest/` is a no-training benchmark harness for the packed-entry pipeline.

It uses the same Rust loader APIs as training (`PackedEntryStream` and `encode_packed_chunks`), but skips model forward/backward so you can isolate loader + transport behavior.

## What it measures

- Feeder decode throughput (entries/chunks returned)
- Trainer-side pull + encode throughput (batches/sec, entries/sec)
- Time split across `ray.wait`, `ray.get`, and Rust encode calls

## Core shape

- 1 trainer process (driver)
- N Ray feeder actors
- Each feeder owns one sharded Rust `PackedEntryStream` (`rank=i`, `world_size=N`)
- Trainer pulls bundles with `ray.wait` and encodes full batches with `rust.encode_packed_chunks`

## Local usage

```bash
python -m src.raytest.run \
  --feeder-count 4 \
  --batch-size 16384 \
  --chunk-entries 8192 \
  --bundle-chunks 1 \
  --inflight-per-feeder 1 \
  --target-batches 200 \
  /path/to/data1.binpack /path/to/data2.binpack
```

## Useful knobs

- `--chunk-entries`: packed entries per chunk emitted by Rust
- `--bundle-chunks`: chunks returned per feeder RPC
- `--inflight-per-feeder`: in-flight pull requests per feeder
- `--feeder-cpus`: Ray CPU reservation per feeder actor
- `--encode-threads`: threads used by `encode_packed_chunks`
- `--materialize-tensors`: include numpy->torch conversion in the test path

## Slurm usage

Use `src/raytest/slurm_raytest.sbatch` and pass datasets as script args:

```bash
sbatch src/raytest/slurm_raytest.sbatch /path/to/data1.binpack /path/to/data2.binpack
```

Default hetjob shape in the script:

- Group 0: 1 head node, 4 CPUs
- Group 1: 2 worker nodes, 16 CPUs each

The script starts Ray with `uv run ray ...` on all nodes.

Note: the launcher resolves hostnames to IPv4 addresses and uses IPs for `--address` / `--node-ip-address` to avoid GCS hostname resolution issues.

Tune via environment variables before `sbatch`:

- `FEEDER_COUNT`
- `FEEDER_CPUS`
- `BATCH_SIZE`
- `CHUNK_ENTRIES`
- `BUNDLE_CHUNKS`
- `INFLIGHT_PER_FEEDER`
- `TARGET_BATCHES`
- `MAX_SECONDS`
- `LOADER_THREADS`, `DECODE_THREADS`, `ENCODE_THREADS`
