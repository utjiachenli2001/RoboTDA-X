# HANDOFF — GPU / retrain follow-ups (all out of scope for this pass)

This pass was pure re-scoring of cached Gram matrices against cached outcomes: no training,
no GPU, no repo file edited. Everything below needs resources this pass did not have.
Ordered by what would actually change a conclusion.

## 1. Confirm the Hodges–Lehmann effect with fresh masks (highest value)

**What was seen.** Within the champion's unit-L2 normalization, Hodges–Lehmann aggregation
beat the mean on **all four** hold-out targets (Δratio +0.032…+0.067) and on C1
(0.5130 → 0.5678). Uniform sign on 4/4 unseen targets is the only consistent effect this
project found. It did **not** survive Bonferroni-8 (best cell C4, p=0.0076 vs α=0.003125).

**Why it cannot be settled with what is cached.** The 24 demo-grain masks are fixed and
already used. Re-testing the same masks cannot raise the evidence — only a fresh mask draw
can. At n=24 and Bonferroni-8 the critical ρ is ≈ 0.60, which no estimator here approaches;
either more masks or a smaller preregistered family is required.

**What to run.** Draw a fresh set of demo-grain masks with the same generator
(`results/demo_mask_manifest.json` was produced at seed/K recorded in its header), retrain
S=10 seeds per mask on the targets where HL led (C4 first, then C2/C9), and test the single
preregistered hypothesis "HL > mean within unit-L2". A family of 1 needs p < 0.025, which
C4's 0.0076 already clears — so this is a genuinely decidable experiment, not a fishing trip.

## 2. Common-random-seed ground truth

Every outcome here is a seed *mean* (or median) over independently seeded retrains, so the
mask-to-mask contrast carries the full between-seed variance. Re-running the mask sweep with
**common random seeds** across masks (same init and data order, only the mask differing)
removes that variance from the comparison and should raise the ceiling materially. The
ceilings are currently 0.75–0.96 (BC) and 0.76–0.90 (diffusion); every ratio in `RESULTS.md`
and `FINDINGS.md` is divided by one of those. This is the cheapest way to make all existing
numbers sharper without inventing a new estimator.

## 3. Task 4 — per-layer / subspace Φ (blocked, see BLOCKERS.md #3)

Not attempted: the repo contains **no checkpoints** (`find . -name '*.pt' | wc -l` → 0).
`runs/stage_E/ens_s201..210/` hold only `demos.json`, `outcomes.json`, `train_meta.json` and
markers. The cached `G` has already collapsed layers, so per-layer Φ cannot be recovered
from it — it must be recomputed from weights.

**Why this is now the most interesting blocked experiment.** The k-sweep found k\* ≈ 1:
the 135×135 Gram carries about one eigendirection distinguishable from random demo pairing.
That is a statement about the *dimensionality of Φ relative to the number of demos*, and the
direct remedy is to shrink Φ rather than to regularize the solve. Restricting Φ to the action
head or last block is exactly that experiment, and the k-sweep predicts it should move the
needle where truncation and damping did not.

**What to run.** With `runs/stage_E/ens_s*/final.pt` present:
`src/attribution.py:build_targets` + `demo_gradient`, restricted per layer group; rebuild
`G`, `K` per group; then reuse `if_repair/eval.py:evaluate` unchanged — the tier plumbing,
LDS, ceilings and CLI guards all work on any `(G, K)` of the same shape. Layer-ablation
table: which layer groups carry LDS signal.

## 4. Port to robomimic / Can

Everything here is LIBERO-goal, 135 demos, 9 clusters, one policy class plus a 5-member
diffusion arm. Whether "the identity is the best preconditioner" is a property of influence
functions at this demo count or a property of *this* dataset is not answerable from inside
this repo. A second task suite with a different demo count is the cleanest external check —
and the k\*≈1 diagnostic is cheap to compute there before committing to a full LDS sweep.

## 5. Re-run the diffusion arm with more members

The diffusion Gram has E=5. At that depth Hodges–Lehmann and the mean induce the same
ranking over the 24 masks (`FINDINGS.md`), so the diffusion comparison carries only one
effective estimator and cannot separate the two frozen candidates. E=10 would make the
cross-policy-class check informative rather than merely directional.

---

## Reproducing this pass

```bash
ln -s /path/to/RoboTDA-X /mnt/sdb/ljc/RoboTDA-X   # bootstrap.py hardcodes this ROOT
python -m venv .venv && . .venv/bin/activate
pip install numpy==2.2.6 scipy==1.18.0 pandas==2.3.3 pyarrow==24.0.0 \
            matplotlib==3.11.0 scikit-learn==1.9.0 pytest==9.1.1 PyYAML==6.0.3
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu  # import-only

python -m pytest if_repair/tests -q
python -m if_repair.run anchors
python -m if_repair.run dev     --config if_repair/configs/dev.yaml
python -m if_repair.aggregate            # Task 1 + tail diagnostic
python -m if_repair.spectral             # Task 2 k-sweep, spectrum null, damping
python -m if_repair.shrinkage            # Task 3
python -m if_repair.figs                 # figs/k_sweep.png, figs/spectrum.png
# hold-out: already computed once; re-running overwrites results/holdout_table.csv
python -m if_repair.run holdout   --config if_repair/configs/frozen.yaml
python -m if_repair.run diffusion --config if_repair/configs/frozen.yaml
```

Runtime is seconds to ~1 minute per step on CPU (135×135 × ≤20 members × 9 targets); the
k-sweep is the slowest at ~21 s. No GPU is used anywhere.
