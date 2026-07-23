#!/bin/bash
# Phase-5 training chain. Only P15 trains (48 diffusion retrains). Resumable: every job is
# marker-gated, so re-running skips completed jobs. P16 and P17 are zero-retrain analysis stages
# run separately (p16_analyze.py, p17_graddot_diffusion.py).
set -u
PY=/home/ljc/miniconda/envs/robotda_x/bin/python
cd "$(dirname "$0")"

echo "=== PHASE 5 TRAINING CHAIN starting $(date -Is) ==="
echo "--- P15: diffusion ground truth S=10 (48 retrains, C1-only rollouts + all-9 L2) ---"
$PY orch5.py --jobs ../results/p15_jobs.json --stage P15 2>&1 || echo "P15 orchestrator rc=$?"
echo "=== PHASE 5 TRAINING CHAIN done $(date -Is) ==="
