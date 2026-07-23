#!/usr/bin/env bash
# Regenerate the 5 diffusion ensemble members from their recorded seed + demo list.
# Same recipe as regen_ckpt.sh; output stays under if_repair/runs/regen_dp/.
set -e
cd /mnt/sdb/ljc/RoboTDA-X
export CUDA_VISIBLE_DEVICES=0
for S in 621 622 623 624 625; do
  OUT="if_repair/runs/regen_dp/dpens_s$S"
  [ -f "$OUT/final.pt" ] && { echo "skip dpens_s$S"; continue; }
  mkdir -p "$OUT"
  .venv/bin/python phase3/src/train_diffusion.py \
      --run_dir "$OUT" --demos "phase3/runs/P10ens/dpens_s$S/demos.json" \
      --seed "$S" --device cuda
done
echo "[regen_dp] done"
