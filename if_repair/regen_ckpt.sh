#!/usr/bin/env bash
# Regenerate a stage_E member's checkpoints from its recorded seed + demos.
# Repo files are never touched: output goes to if_repair/runs/regen/<member>/.
set -e
cd /mnt/sdb/ljc/RoboTDA-X
. .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0     # bootstrap.py would pin 4-7, which do not exist here
M="$1"
SEED="${M#ens_s}"
OUT="if_repair/runs/regen/$M"
mkdir -p "$OUT"
/usr/bin/time -f "%e" -o "$OUT/wall.txt" \
  python src/train.py --run_dir "$OUT" --demos "runs/stage_E/$M/demos.json" \
                      --seed "$SEED" --device cuda
echo "$M wall_s=$(cat $OUT/wall.txt)"
ls "$OUT"
