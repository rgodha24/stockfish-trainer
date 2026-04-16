#!/bin/bash
#SBATCH --job-name=raytest
#SBATCH --account=coc
#SBATCH --partition=ice-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=raytest-%j.out

#SBATCH hetjob
#SBATCH --account=coc
#SBATCH --partition=ice-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: sbatch $0 <dataset1.binpack> [dataset2.binpack ...]"
  exit 1
fi

DATASETS=("$@")
readarray -t HEAD_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0")
readarray -t WORKER_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1")
HEAD_NODE=${HEAD_NODES[0]}

RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_ADDRESS="${HEAD_NODE}:${RAY_PORT}"

FEEDER_COUNT=${FEEDER_COUNT:-2}
FEEDER_CPUS=${FEEDER_CPUS:-16}
BATCH_SIZE=${BATCH_SIZE:-16384}
CHUNK_ENTRIES=${CHUNK_ENTRIES:-8192}
BUNDLE_CHUNKS=${BUNDLE_CHUNKS:-1}
INFLIGHT_PER_FEEDER=${INFLIGHT_PER_FEEDER:-1}
TARGET_BATCHES=${TARGET_BATCHES:-400}
MAX_SECONDS=${MAX_SECONDS:-0}

LOADER_THREADS=${LOADER_THREADS:--1}
DECODE_THREADS=${DECODE_THREADS:--1}
ENCODE_THREADS=${ENCODE_THREADS:-1}
SHUFFLE_BUFFER_ENTRIES=${SHUFFLE_BUFFER_ENTRIES:-16384}

cleanup() {
  srun --het-group=0 --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    bash -lc "uv run ray stop --force >/dev/null 2>&1 || true" || true
  for NODE in "${WORKER_NODES[@]}"; do
    srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="${NODE}" \
      bash -lc "uv run ray stop --force >/dev/null 2>&1 || true" || true
  done
}

trap cleanup EXIT

echo "head nodes: ${HEAD_NODES[*]}"
echo "worker nodes: ${WORKER_NODES[*]}"
echo "ray head node: ${HEAD_NODE}"
echo "ray address: ${RAY_ADDRESS}"
echo "datasets: ${DATASETS[*]}"

srun --het-group=0 --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
  bash -lc "uv run ray stop --force >/dev/null 2>&1 || true; uv run ray start --head --node-ip-address=${HEAD_NODE} --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --num-cpus=0 --disable-usage-stats" &

sleep 4

for NODE in "${WORKER_NODES[@]}"; do
  srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="${NODE}" \
    bash -lc "uv run ray stop --force >/dev/null 2>&1 || true; uv run ray start --address=${RAY_ADDRESS} --num-cpus=16 --disable-usage-stats" &
done
wait

uv run python -m src.raytest.run \
  --ray-address "${RAY_ADDRESS}" \
  --feeder-count "${FEEDER_COUNT}" \
  --feeder-cpus "${FEEDER_CPUS}" \
  --batch-size "${BATCH_SIZE}" \
  --chunk-entries "${CHUNK_ENTRIES}" \
  --bundle-chunks "${BUNDLE_CHUNKS}" \
  --inflight-per-feeder "${INFLIGHT_PER_FEEDER}" \
  --target-batches "${TARGET_BATCHES}" \
  --max-seconds "${MAX_SECONDS}" \
  --loader-threads "${LOADER_THREADS}" \
  --decode-threads "${DECODE_THREADS}" \
  --encode-threads "${ENCODE_THREADS}" \
  --shuffle-buffer-entries "${SHUFFLE_BUFFER_ENTRIES}" \
  "${DATASETS[@]}"
