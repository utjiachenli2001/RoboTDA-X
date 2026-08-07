#!/usr/bin/env bash
# Measure campaign-U throughput on an 8-GPU box.
#
# The H200 figure (85 retrains/hour, 3 workers, one card) does not transfer: different card,
# and eight of them. The budget and the ETA both rest on this number, so it is measured rather
# than scaled by a guess.
#
# One worker process per (GPU, slot). CUDA_VISIBLE_DEVICES is set PER WORKER, which is also what
# keeps src/bootstrap.py's ALLOWED_GPUS=(4,5,6,7) out of the way -- it only pins when the variable
# is unset, so an explicit per-worker assignment is respected and GPUs 0-3 stay usable.
set -u
cd "$(dirname "$0")/.." || exit 1
GPUS=${GPUS:-8}
SLOTS=${SLOTS:-3}        # workers per GPU
REPEATS=${REPEATS:-2}    # retrains per worker
SIZE=${SIZE:-25}         # campaign U trains on 25 demos at every pool

echo "[scale8] gpus=$GPUS slots=$SLOTS repeats=$REPEATS size=$SIZE"
rm -f /tmp/s8_*.log
start=$(date +%s)
for g in $(seq 0 $((GPUS - 1))); do
  for s in $(seq 1 "$SLOTS"); do
    CUDA_VISIBLE_DEVICES=$g .venv/bin/python if_repair/p18_costmodel.py \
      --sizes "$SIZE" --repeats "$REPEATS" > "/tmp/s8_g${g}_s${s}.log" 2>&1 &
  done
done
wait
end=$(date +%s)
wall=$((end - start))
total=$((GPUS * SLOTS * REPEATS))
fails=$(grep -lciE "traceback|out of memory|refusing to run" /tmp/s8_*.log 2>/dev/null | wc -l)
echo "[scale8] $total retrains in ${wall}s -> $(echo "scale=1; $total*3600/$wall" | bc) retrains/hour  (logs with errors: $fails)"
echo "[scale8] per-retrain wall: $(echo "scale=1; $wall*$GPUS*$SLOTS/$total" | bc)s"
grep -h "^\[u0\] n=" /tmp/s8_g0_s1.log | head -3
