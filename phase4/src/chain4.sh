#!/bin/bash
# Phase-4 training chain. Resumable: every stage is marker-gated, so re-running skips completed
# jobs. Order is the PREREGISTERED cut order in reverse (the first-to-be-cut stage runs LAST):
#   P12 (96 BC retrains)  -> P11 depends on it (S=10 ground truth)
#   P13 (48 diffusion)    -> the diffusion settlement
#   P14 (64 BC)           -> first on the cut list, so it runs last
set -u
PY=/home/ljc/miniconda/envs/robotda_x/bin/python
cd "$(dirname "$0")"

echo "=== PHASE 4 TRAINING CHAIN starting $(date -Is) ==="

echo "--- P12: BC ground truth S=10 (96 retrains) ---"
$PY orch4.py --jobs ../results/p12_jobs.json --stage P12 2>&1 || echo "P12 orchestrator rc=$?"

echo "--- P13: diffusion ground truth S=8 (48 retrains) ---"
$PY orch4.py --jobs ../results/p13_jobs.json --stage P13 2>&1 || echo "P13 orchestrator rc=$?"

echo "--- P14: fresh heterogeneity corpus (64 retrains) ---"
$PY orch4.py --jobs ../results/p14_jobs.json --stage P14 2>&1 || echo "P14 orchestrator rc=$?"

echo "=== PHASE 4 TRAINING CHAIN done $(date -Is) ==="
