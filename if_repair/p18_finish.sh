#!/usr/bin/env bash
# Chain the remaining campaign-U GPU work: wait for the reserve-pair re-runs, then compute the
# GradDot scoring features for both arms at every pool, one cell per GPU.
#
# H1f at pool 50 is by definition identical to H1 at pool 50 (same surrogate, same members), so it
# is aliased rather than recomputed -- recomputing it would burn 5 trainings to reproduce a file
# byte for byte.
set -u
cd "$(dirname "$0")/.." || exit 1

echo "[finish] waiting for reserve-pair re-runs..."
while pgrep -f 'p18_gate.py --run-reruns' > /dev/null; do sleep 60; done
echo "[finish] re-runs done: $(ls if_repair/runs/campaigns/U/*_i440[3-6]_o440[3-6].npz 2>/dev/null | wc -l) files"

echo "[finish] computing GradDot features (2 arms x 4 pools, one cell per GPU)"
g=0
for arm in H1 H1f; do
  for pool in 50 100 200 370; do
    if [ "$arm" = "H1f" ] && [ "$pool" = "50" ]; then continue; fi   # identical to H1/50
    CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python if_repair/p18_gram.py \
      --pool "$pool" --arm "$arm" > "logs/gram_${arm}_${pool}.log" 2>&1 &
    g=$(( (g + 1) % 8 ))
  done
done
wait
echo "[finish] gram logs:"
grep -h "wrote\|Traceback" logs/gram_*.log | tail -10
cp if_repair/results/p18_graddot_H1_pool50.npz if_repair/results/p18_graddot_H1f_pool50.npz
echo "[finish] aliased H1f/pool50 -> H1/pool50"
ls -1 if_repair/results/p18_graddot_*.npz | wc -l
echo "[finish] DONE"
