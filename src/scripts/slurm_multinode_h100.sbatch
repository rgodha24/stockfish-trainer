#!/bin/bash
#SBATCH --job-name=dist-train
#SBATCH --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=8 --time=04:00:00
#SBATCH --output=dist-train-%j.out
#SBATCH hetjob
#SBATCH --nodes=10 --ntasks=10 --ntasks-per-node=1 --cpus-per-task=16

set -euo pipefail
[[ "$#" -lt 1 ]] && { echo "usage: sbatch $0 <dataset.binpack> ... [-- trainer overrides]"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
RAY=".venv/bin/ray"
PY=".venv/bin/python"
WORKER_CPUS=${WORKER_CPUS:-16}
IB_IFACE="${IB_IFACE:-ib0}"

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
WORKERS=($(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1"))

# Resolve InfiniBand IPs -- same logic that the smoke test uses.
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

# threats.yaml step-1 hyperparameters
"$PY" -m src.train.multinode \
  --ray-address "$ADDR" \
  --max-epochs 350 \
  --lr 4.375e-4 \
  --gamma 0.995 \
  --start-lambda 1.0 \
  --end-lambda 0.75 \
  --random-fen-skipping 10 \
  --early-fen-skipping 12 \
  --pc-y1 0.6893201149773951 \
  --pc-y2 2.9285769485515805 \
  --pc-y3 1.4386005301749225 \
  --w1 3.3553547771220007 \
  --w2 0.7006821612968052 \
  --default-root-dir "$HOME/scratch/stockfish-training" \
  --wandb-run-name "dist-train-$SLURM_JOB_ID" \
  "$@"
