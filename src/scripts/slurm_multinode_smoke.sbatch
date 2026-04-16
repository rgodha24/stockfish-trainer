#!/bin/bash
#SBATCH --job-name=dist-smoke
#SBATCH --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:45:00
#SBATCH --output=dist-smoke-%j.out
#SBATCH hetjob
#SBATCH --nodes=10 --ntasks=10 --ntasks-per-node=1 --cpus-per-task=16

set -euo pipefail
[[ "$#" -lt 1 ]] && { echo "usage: sbatch $0 <dataset.binpack> ..."; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
RAY=".venv/bin/ray"
PY=".venv/bin/python"
WORKER_CPUS=${WORKER_CPUS:-16}

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
WORKERS=($(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1"))
IB_IFACE="${IB_IFACE:-ib0}"
# Ask each node for its own ib0 IP via srun; falls back to DNS if ib0 is missing.
node_ip() { srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="$1" \
              ip -4 -o addr show dev "$IB_IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }

HEAD_IP=$(ip -4 -o addr show dev "$IB_IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
[[ -z "$HEAD_IP" ]] && HEAD_IP=$(getent ahostsv4 "$HEAD" | awk '/STREAM/{print $1; exit}')
echo "head $HEAD: $HEAD_IP"
RAY_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
ADDR="${HEAD_IP}:${RAY_PORT}"
EXPECTED_CPUS=$(( ${#WORKERS[@]} * WORKER_CPUS ))

trap '"$RAY" stop --force &>/dev/null || true; for n in "${WORKERS[@]}"; do srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="$n" "$RAY" stop --force &>/dev/null & done; wait || true' EXIT

"$RAY" stop --force &>/dev/null || true
"$RAY" start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-cpus=0 --disable-usage-stats &
sleep 4

for NODE in "${WORKERS[@]}"; do
  NODE_IP=$(node_ip "$NODE")
  [[ -z "$NODE_IP" ]] && NODE_IP=$(getent ahostsv4 "$NODE" | awk '/STREAM/{print $1; exit}')
  echo "worker $NODE: $NODE_IP"
  srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="$NODE" \
    bash -lc '"$0" start --block --address="$1" --node-ip-address="$2" --num-cpus="$3" --disable-usage-stats || true' \
    "$RAY" "$ADDR" "$NODE_IP" "$WORKER_CPUS" &
done

until "$PY" -c "import ray; ray.init(address='$ADDR', log_to_driver=False); assert ray.cluster_resources().get('CPU', 0) >= $EXPECTED_CPUS; ray.shutdown()" >/dev/null 2>&1; do sleep 2; done

"$PY" -m src.distributed.smoke --ray-address "$ADDR" "$@"
