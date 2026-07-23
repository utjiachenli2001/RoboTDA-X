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

---

# Pass 2 follow-ups — B2/B3/B4 are now UNBLOCKED and cheap

Pass 1 listed Task 4 as blocked for want of checkpoints. That is resolved: checkpoints are
**regenerable** in 94 s/member via `if_repair/regen_ckpt.sh <member>`, which also writes the
5 intermediate `ckpt_0..4.pt` that TracIn needs. Only B1 was run (session limit, not budget:
**0.15 of 12 GPU-h used**). The remaining Phase-B items now cost well under an hour each.

**Read first:** regenerated weights are NOT the originals (BLOCKERS #6). Recompute every
baseline on whichever ensemble you use; never mix regenerated numbers with the E=20 cached
Gram rows.

## B2 — target-functional redesign (highest upside, ~2-3 h)

`if_repair/gradients.py:build_gram` already isolates the test-side term (`AT.build_targets`
→ `TG`). Swap the uniform held-out-L2 target for (a) ensemble-action-variance weighting, (b)
failure-divergence-state weighting, (c) rollout-visited states. **Gate each new functional
through its own split-half ceiling ≥0.4 BEFORE scoring any estimator against it** — a
misaligned proxy scored −0.48 in the sibling study, and B1 already shows this corpus will
happily produce large negative LDS (block_01/C7 = −0.142).

## B3 — KFAC / EK-FAC (~2-3 h)

The one genuinely different `H^-1`. Given B1's mechanism result, the specific prediction is
sharp: KFAC's *structure* is what should help, because it estimates curvature in a factored
space far smaller than 19.2M, i.e. it raises the effective `k*` the way layer restriction
did. Test it per-block as well as globally — if KFAC beats GradDot only where `k*` is already
> 1, that confirms the p/N account rather than the "better solve" account.

## B4 — TracIn density & grain (~1-2 h)

`ckpt_0..4.pt` per regenerated member are already on disk. TracIn is the diffusion winner
(0.479 vs GradDot 0.414), so this is the direct probe of the policy-class flip, and the
cheapest remaining item.

## B5 — common-random-seed estimand (~4-5 h, unchanged from pass 1)

Still the highest-value retrain: init is ~72 % of outcome variance, and every ratio in
RESULTS.md is divided by a ceiling inflated by it.

## The two results most worth defending

1. **Datamodel on C2** — the only hold-out pass in either pass (ratio 0.639, p = 0.00115).
   Confirm on fresh masks; note it needs outcome data, so it is a different estimator class.
2. **`k*` rises when Φ is restricted** (1 → 6-9). This is the mechanism claim. Strengthen it
   by sweeping Φ width directly (random parameter subsets of increasing size) and showing
   `k*` and the best-k LDS move together — that turns a 5-group observation into a curve.

---

# Pass 3 follow-ups — read FINDINGS.md §1 first

## What changed about the project's story

Pass 2 ended "generality NO, unification NO, **mechanism YES**". Pass 3 **retracts the
mechanism claim** (B6: k\* is a property of which subspace, not of p/N — two groups of identical
dimension differ 6 vs 1, and random subspaces never leave k\* ≈ 1), and adds one thing pass 2
could not have known: **the demo-grain LDS is mask-draw-dependent at the scale of every effect
this project chases** (BLOCKERS #17). Four of five preregistered hypotheses failed on a fresh
mask draw — and so did the baseline.

The one confirmed result is the **datamodel on C2**: 0.639 on the archived masks, **0.729 on the
fresh masks, p = 0.0002**. Two independent draws.

## Do NOT re-run these

- **B5 (common random seeds).** Retired, not deferred. Stage G is already common-init (read
  `src/train.py`: the model is seeded before it is built), the batch-order main effect is
  0.1–0.5% of variance, and the binding noise is the mask×init interaction, which no seeding
  protocol removes. The three designs differ by ≤0.05 in reliability. See FINDINGS §6.
- **Either confirmatory family.** Both are consumed (BLOCKERS #18).
- **Six concurrent trainers.** Slower than three (BLOCKERS #16).

## Open threads, ordered by what would change a conclusion

1. **More masks is now the only thing that moves the needle.** BLOCKERS #17 shows the mask draw
   is worth ~0.14 in C1 ratio, larger than any estimator effect reported. Everything else is
   being measured through that noise. A K = 48 draw at 6 seeds costs ~7 solo-h and would roughly
   halve the sampling error on every comparison; nothing else in the backlog buys as much. This
   is the one place where spending GPU on *more data* is not a violation of the prime directive
   but its precondition — the directive was about not buying stability with seeds, and this buys
   adjudicability with masks.
2. **Why `embed` and `block_00` and not the others?** B6 establishes that k\* tracks the subspace
   rather than its size and leaves the obvious question open. The cheap probe is *structured*
   subspaces of matched size — one attention head at a time, input-side vs output-side
   projections within a block — to see whether "input-side" or "early" is the predictor. All of
   it reuses the cached Φ; zero retrains.
3. **The target functional is the largest untried lever** (+0.161 on C5 dev, vs +0.130 for the
   whole curvature sweep) and per-frame held-out losses are now on disk, so a **new weighting
   costs zero GPU**: `functionals.weights()` plus `campaign_outcomes()` give it an outcome and a
   ceiling immediately. The obvious untried ones are behavioural rather than model-derived, since
   the two model-derived ones (`ens_var`, `fail_div`) did not transfer across targets.
4. **The intermediate-checkpoint effect.** Step 6400/8000 attributes better than the final
   weights on C1 (0.598 vs 0.506) while adjacent checkpoints swing by 0.42. Either there is a
   real "attribute from the middle of training" effect or the LDS is far noisier than n = 24
   suggests — and #1 says the second. A denser checkpoint grid on one member separates them for
   ~0.2 solo-h.
5. **Rollout-based functionals** remain blocked on LIBERO/robosuite (BLOCKERS #15). Success rate
   is still the only behaviourally meaningful outcome the study never predicts.

## What is cheap now that was not

- **Per-frame held-out losses for 456 retrains** (`runs/campaigns/{A,B,C}/*.npz`). Any frame
  weighting `w` yields an outcome `-Σ w_f l2_f / Σ w_f` and its own split-half ceiling for zero
  GPU. **Machine-local and gitignored (~40 MB); ~11 solo-h to rebuild.** The derived outcome
  table is committed at `results/campaign_outcomes.parquet` (153 KB) — enough for everything
  except defining a *new* weighting.
- **A split init/order trainer** (`retrain.train_one`) the repo's single `--seed` cannot express,
  with bit-for-bit legacy equivalence pinned by a test.
- **A regenerated diffusion ensemble with checkpoints** (`runs/regen_dp/`), which p17 never had.
  Machine-local; 0.46 solo-h to rebuild via `if_repair/regen_dp.sh`.
- **A ceiling recipe** (`functionals.split_half_ceiling`) that reproduces all nine archived p12
  ceilings to <1e−12, so any new outcome can be gated the same way the archived ones were.

## Invariants for the next session

- Run from `/mnt/sdb/ljc/RoboTDA-X`; `export CUDA_VISIBLE_DEVICES=0` before anything GPU.
- Three concurrent trainers.
- Never mix regenerated Gram numbers with the E=20 cached-Gram rows. **Outcomes are the
  exception** and the reason is measured (BLOCKERS #12): mask-level outcomes regenerate with
  Spearman 0.61–0.93.
- Any estimator fitted on mask outcomes goes through the leave-one-mask-out path.
- Report LDS, ceiling, ratio, p, pass. Never bare ρ. And after pass 3, never a single mask draw.
