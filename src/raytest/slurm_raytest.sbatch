#!/bin/bash
#SBATCH --job-name=raytest
#SBATCH --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=4 --time=00:45:00
#SBATCH --output=raytest-%j.out
#SBATCH hetjob
#SBATCH --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=16

set -euo pipefail
[[ "$#" -lt 1 ]] && { echo "usage: sbatch $0 <dataset.binpack> ..."; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
RAY=".venv/bin/ray"
PY=".venv/bin/python"

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
WORKERS=($(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1"))
ip4() { getent ahostsv4 "$1" | awk '/STREAM/{print $1; exit}'; }

HEAD_IP=$(ip4 "$HEAD")
RAY_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
ADDR="${HEAD_IP}:${RAY_PORT}"

trap '"$RAY" stop --force &>/dev/null; for n in "${WORKERS[@]}"; do srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="$n" "$RAY" stop --force &>/dev/null & done; wait' EXIT

"$RAY" stop --force &>/dev/null || true
"$RAY" start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-cpus=0 --disable-usage-stats &
sleep 4

for NODE in "${WORKERS[@]}"; do
  srun --het-group=1 --nodes=1 --ntasks=1 --nodelist="$NODE" \
    "$RAY" start --block --address="$ADDR" --node-ip-address="$(ip4 "$NODE")" --num-cpus=16 --disable-usage-stats &
done

until "$PY" -c "import ray; ray.init('$ADDR', log_to_driver=False); assert ray.cluster_resources().get('CPU',0) >= 32; ray.shutdown()"; do sleep 2; done

"$PY" -m src.raytest.run --ray-address "$ADDR" "$@"
