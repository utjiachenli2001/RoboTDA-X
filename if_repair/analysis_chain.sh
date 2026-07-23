#!/usr/bin/env bash
# Run each downstream analysis the moment its campaign lands, so no result waits on a poll.
set -u
cd /mnt/sdb/ljc/RoboTDA-X
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4      # leave cores for the trainers
PY=.venv/bin/python
count () { ls if_repair/runs/campaigns/"$1" 2>/dev/null | wc -l; }
wait_for () { while [ "$(count "$1")" -lt "$2" ]; do sleep 60; done; }

wait_for A 240
echo "[analysis] A complete -> B2 campaign arm $(date -u)"
$PY -m if_repair.b2_functionals --obs campaign --campaign A \
    > if_repair/runs/logs/b2_campaign.log 2>&1

wait_for C 72
echo "[analysis] C complete -> B5 $(date -u)"
$PY -m if_repair.b5_variance > if_repair/runs/logs/b5.log 2>&1

wait_for B 144
echo "[analysis] B complete -> confirm3 $(date -u)"
$PY -m if_repair.confirm3 > if_repair/runs/logs/confirm3.log 2>&1
echo "[analysis] ALL DONE $(date -u)"
touch if_repair/runs/logs/analysis.done
