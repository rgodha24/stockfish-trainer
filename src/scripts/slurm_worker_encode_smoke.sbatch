#!/bin/bash
#SBATCH --job-name=worker-encode-smoke
#SBATCH --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 --time=00:30:00
#SBATCH --output=worker-encode-smoke-%j.out
#SBATCH hetjob
#SBATCH --nodes=10 --ntasks=10 --ntasks-per-node=1 --cpus-per-task=16

set -euo pipefail
[[ "$#" -lt 1 ]] && { echo "usage: sbatch $0 <dataset.binpack> ..."; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
RAY=".venv/bin/ray"
PY=".venv/bin/python"
WORKER_CPUS=16

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
WORKERS=($(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1"))

# Prefer the IB (ib0) interface for Ray traffic — falls back to hostname DNS.
# Override IB_IFACE if your cluster uses a different name (ib1, hsn0, etc).
IB_IFACE="${IB_IFACE:-ib0}"
local_ib_ip() {
  local ip
  ip=$(ip addr show "$IB_IFACE" 2>/dev/null | awk '/inet /{print $2; exit}' | cut -d/ -f1)
  [[ -n "$ip" ]] && echo "$ip" && return
  hostname -I | awk '{print $1}'  # last-resort fallback
}

HEAD_IP=$(local_ib_ip)
RAY_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
ADDR="${HEAD_IP}:${RAY_PORT}"
echo "Ray head: $HEAD_IP:$RAY_PORT (IB_IFACE=$IB_IFACE)"
EXPECTED_CPUS=$(( ${#WORKERS[@]} * WORKER_CPUS ))

trap '"$RAY" stop --force &>/dev/null || true; for n in "${WORKERS[@]}"; do srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="$n" "$RAY" stop --force &>/dev/null & done; wait || true' EXIT

"$RAY" stop --force &>/dev/null || true
"$RAY" start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-cpus=0 --disable-usage-stats &
sleep 4

for NODE in "${WORKERS[@]}"; do
  srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="$NODE" \
    bash -lc 'IB_IFACE="$4"; ip=$(ip addr show "$IB_IFACE" 2>/dev/null | awk "/inet /{print \$2; exit}" | cut -d/ -f1); [[ -z "$ip" ]] && ip=$(hostname -I | awk "{print \$1}"); echo "worker $(hostname): $ip"; "$0" start --block --address="$1" --node-ip-address="$ip" --num-cpus="$3" --disable-usage-stats || true' \
    "$RAY" "$ADDR" "$WORKER_CPUS" "$IB_IFACE" &
done

until "$PY" -c "import ray; ray.init(address='$ADDR', log_to_driver=False); assert ray.cluster_resources().get('CPU', 0) >= $EXPECTED_CPUS; ray.shutdown()" >/dev/null 2>&1; do sleep 2; done

"$PY" -m src.distributed.smoke \
  --ray-address "$ADDR" \
  --feeder-count 10 \
  --feeder-cpus 16 \
  --loader-threads 16 \
  --encode-threads 4 \
  --batch-size 65536 \
  --chunk-entries 8192 \
  --bundle-chunks 8 \
  --inflight-per-feeder 4 \
  --target-batches 500 \
  --max-seconds 300 \
  "$@"
