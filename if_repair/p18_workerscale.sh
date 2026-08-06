#!/usr/bin/env bash
# U0b -- measure retrains/hour as a function of WORKER COUNT on one H200.
#
# The ladder's 68-hour estimate came from "~88 retrains/hour with 3 workers", measured while
# this card was shared. It is now idle, and a single worker leaves the GPU at ~40% utilisation,
# so the real ceiling is unknown. This is the number the campaign budget actually depends on --
# more than the size question, which the size sweep shows is flat.
#
# Method: K concurrent workers each do R retrains at a fixed training-set size; measure total
# wall time; throughput = K*R / wall. Fixed size means worker count is the only moving part.
set -u
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=0

SIZE=${SIZE:-258}          # a mid-ladder training set; cost is flat in size anyway
REPEATS=${REPEATS:-2}      # retrains per worker
WORKERS=${WORKERS:-"1 2 3 4 6"}

echo "[u0b] size=$SIZE repeats=$REPEATS worker counts: $WORKERS"
for K in $WORKERS; do
  rm -f /tmp/u0b_w*.log
  start=$(date +%s)
  for w in $(seq 1 "$K"); do
    .venv/bin/python if_repair/p18_costmodel.py --sizes "$SIZE" --repeats "$REPEATS" \
      > "/tmp/u0b_w${w}.log" 2>&1 &
  done
  wait
  end=$(date +%s)
  wall=$((end - start))
  # every worker pays one warmup retrain (200 steps) that is not part of the measured load
  total=$((K * REPEATS))
  if [ "$wall" -gt 0 ]; then
    rate=$(echo "scale=1; $total * 3600 / $wall" | bc)
  else
    rate="inf"
  fi
  fails=$(grep -lciE "traceback|out of memory" /tmp/u0b_w*.log 2>/dev/null | wc -l)
  echo "[u0b] workers=$K retrains=$total wall=${wall}s  ->  ${rate} retrains/hour  (worker logs with errors: $fails)"
done
echo "[u0b] done"
