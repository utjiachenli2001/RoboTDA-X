#!/usr/bin/env bash
# Chain the retrain campaigns so the GPU never idles between them.
set -u
cd /mnt/sdb/ljc/RoboTDA-X
export CUDA_VISIBLE_DEVICES=0
count () { ls if_repair/runs/campaigns/"$1" 2>/dev/null | wc -l; }
wait_for () { while [ "$(count "$1")" -lt "$2" ]; do sleep 60; done; }

wait_for A 240
echo "[chain] A complete ($(count A)) -> C at $(date -u)"
for w in 0 1 2; do
  .venv/bin/python -m if_repair.retrain --campaign C --worker $w --nworkers 3 \
    > if_repair/runs/logs/C_w$w.log 2>&1 &
done
wait
echo "[chain] C complete ($(count C)) -> B at $(date -u)"
for w in 0 1 2; do
  .venv/bin/python -m if_repair.retrain --campaign B --worker $w --nworkers 3 \
    > if_repair/runs/logs/B_w$w.log 2>&1 &
done
wait
echo "[chain] B complete ($(count B)) at $(date -u)"
touch if_repair/runs/logs/chain.done
