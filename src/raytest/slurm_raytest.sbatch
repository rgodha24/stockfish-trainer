#!/bin/bash
#SBATCH --job-name=raytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=raytest-%j.out

#SBATCH hetjob
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

resolve_ipv4() {
  local host="$1"
  while read -r ip _; do
    if [[ "${ip}" == *.* ]]; then
      printf '%s\n' "${ip}"
      return 0
    fi
  done < <(getent ahostsv4 "${host}" 2>/dev/null || true)
  return 1
}

is_port_free() {
  local port="$1"
  python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

pick_free_port() {
  local preferred="$1"
  local low="$2"
  local high="$3"
  python - "$preferred" "$low" "$high" <<'PY'
import random
import socket
import sys

preferred = int(sys.argv[1])
low = int(sys.argv[2])
high = int(sys.argv[3])


def is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


if is_free(preferred):
    print(preferred)
    raise SystemExit(0)

for _ in range(256):
    port = random.randint(low, high)
    if is_free(port):
        print(port)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

if ! HEAD_IP=$(resolve_ipv4 "${HEAD_NODE}"); then
  echo "failed to resolve IPv4 address for head node ${HEAD_NODE}"
  exit 1
fi

if [[ -n "${RAY_PORT:-}" ]]; then
  if ! is_port_free "${RAY_PORT}"; then
    echo "requested RAY_PORT=${RAY_PORT} is already in use on ${HEAD_NODE}"
    exit 1
  fi
else
  RAY_PORT=$(pick_free_port "$((20000 + (SLURM_JOB_ID % 20000)))" 20000 50000)
fi

if [[ -n "${RAY_DASHBOARD_PORT:-}" ]]; then
  if ! is_port_free "${RAY_DASHBOARD_PORT}"; then
    echo "requested RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT} is already in use on ${HEAD_NODE}"
    exit 1
  fi
else
  RAY_DASHBOARD_PORT=$(pick_free_port "$((50001 + (SLURM_JOB_ID % 10000)))" 50001 62000)
fi

RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

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

HEAD_PID=""
WORKER_SRUN_PIDS=()

cleanup() {
  uv run --no-sync ray stop --force >/dev/null 2>&1 || true
  for NODE in "${WORKER_NODES[@]}"; do
    srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="${NODE}" \
      bash -lc "uv run --no-sync ray stop --force >/dev/null 2>&1 || true" || true
  done

  if [[ -n "${HEAD_PID}" ]]; then
    kill "${HEAD_PID}" >/dev/null 2>&1 || true
  fi
  for PID in "${WORKER_SRUN_PIDS[@]}"; do
    kill "${PID}" >/dev/null 2>&1 || true
  done
}

trap cleanup EXIT

echo "head nodes: ${HEAD_NODES[*]}"
echo "worker nodes: ${WORKER_NODES[*]}"
echo "ray head node: ${HEAD_NODE}"
echo "ray head ip: ${HEAD_IP}"
echo "ray gcs port: ${RAY_PORT}"
echo "ray dashboard port: ${RAY_DASHBOARD_PORT}"
echo "ray address: ${RAY_ADDRESS}"
echo "datasets: ${DATASETS[*]}"

uv run --no-sync ray stop --force >/dev/null 2>&1 || true
uv run --no-sync ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" --dashboard-port="${RAY_DASHBOARD_PORT}" --num-cpus=0 --disable-usage-stats --block &
HEAD_PID=$!

sleep 4

if ! kill -0 "${HEAD_PID}" >/dev/null 2>&1; then
  echo "ray head failed to stay alive after startup"
  wait "${HEAD_PID}" || true
  exit 1
fi

for NODE in "${WORKER_NODES[@]}"; do
  if ! NODE_IP=$(resolve_ipv4 "${NODE}"); then
    echo "failed to resolve IPv4 address for worker node ${NODE}"
    exit 1
  fi
  echo "starting worker ${NODE} (${NODE_IP})"
  srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="${NODE}" \
    bash -lc "uv run --no-sync ray stop --force >/dev/null 2>&1 || true; uv run --no-sync ray start --address=${RAY_ADDRESS} --node-ip-address=${NODE_IP} --num-cpus=16 --disable-usage-stats --block" &
  WORKER_SRUN_PIDS+=("$!")
done

READY=0
for _ in $(seq 1 30); do
  if uv run --no-sync python - <<PY
import ray

ray.init(address="${RAY_ADDRESS}", log_to_driver=False)
cpu = float(ray.cluster_resources().get("CPU", 0.0))
ray.shutdown()
raise SystemExit(0 if cpu >= (${FEEDER_COUNT} * ${FEEDER_CPUS}) else 2)
PY
  then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
  echo "ray cluster did not become ready with enough worker CPU"
  uv run --no-sync ray status --address "${RAY_ADDRESS}" || true
  exit 1
fi

uv run --no-sync ray status --address "${RAY_ADDRESS}" || true

uv run --no-sync python -m src.raytest.run \
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
