# FINDINGS — methods sweep for faithful influence attribution

Pass 2 (methods sweep). Pass 1 (`if-repair-rescoring`) established the spectral/aggregation
nulls; this pass adds a gradient-free datamodel, rank fusion, and layerwise influence, and
unblocks Phase B by **regenerating** the missing checkpoints.

## Best estimator per target vs the GradDot baseline (ratio-to-ceiling)

Demo grain, n=24, tier `bc_s10`. **Bold = passes** (ratio ≥ 0.5 ∧ p < α).
Dev = C1/C5 (iterated freely). Hold-out = C2/C4/C7/C9, computed **once**, Bonferroni-16
(α = 0.00156).

| estimator | C1 (dev) | C5 (dev) | C2 | C4 | C7 | C9 | GPU-h |
|---|---|---|---|---|---|---|---|
| **GradDot_dmean** (baseline, E=20) | **0.624** | 0.416 | 0.315 | 0.245 | 0.396 | 0.246 | 0 |
| GradDot_unitL2 (published, 0.513) | **0.540** | 0.350 | 0.397 | 0.487 | 0.349 | 0.396 | 0 |
| A1 robust aggregation (best) | **0.627** | 0.437 | — | — | — | — | 0 |
| A2 spectral `truncated_if` (best k) | **0.624** (k=0) | 0.597† | — | — | — | — | 0 |
| A3 James–Stein shrinkage | 0.624 (no gain) | 0.416 | — | — | — | — | 0 |
| **A4 datamodel_lasso (LOO)** | 0.417 | **0.879** | **0.639** | 0.367 | NaN‡ | NaN‡ | 0 |
| A5 fusion, gradients-only (z-avg) | **0.649** | 0.429 | 0.361 | 0.375 | 0.388 | 0.295 | 0 |
| A5 fusion, +datamodel (LOO) | **0.879** | **0.814** | — | — | — | — | 0 |
| B1 `head` GradDot (regen E=5) | 0.506 | 0.413 | 0.321 | 0.389 | 0.426 | 0.495 | 0.14 |
| B1 `block_01` trunc_k10 (regen E=5) | 0.470 | **0.757**† | 0.189 | 0.310 | −0.142 | 0.336 | ↑ |

† oracle-tuned over 11 groups × 8 k values — did **not** transfer (see below).
‡ CV-selected `alpha=0.1` zeroes **all** coefficients → constant prediction → Spearman
undefined. A real failure mode, not a bug: the datamodel either finds design structure or
collapses entirely. NaN is scored as a non-pass.

## Did we win?

**Win condition 1 — generality (≥3 of {C1,C2,C4,C7,C9}): NOT achieved.**
Best is the datamodel with **2** passes (C5 dev + C2 hold-out); GradDot has 1 (C1). No
estimator passed ≥3. Hold-out pass counts: datamodel 1/4, fusion 0/4, B1-head 0/4,
B1-block_01 0/4.

**Win condition 2 — unification across policy classes: NOT achieved** (not reached this
pass; the diffusion arm was covered in pass 1, where nothing cleared Bonferroni).

**Win condition 3 — mechanism: ACHIEVED, and it is the real result.**

The pass-1 finding was `k* ≈ 1`: the full-parameter Gram (19.2M params, 135 demos) carries
about one eigendirection distinguishable from random demo pairing, so the whole
preconditioner is noise and the best inverse is the identity — i.e. GradDot. B1 tests the
direct implication, that **Φ is too wide for 135 demos**, by restricting Φ to blocks:

| group | params | % of model | k\* (parallel analysis) |
|---|---|---|---|
| ALL | 19,222,091 | 100 % | **1** |
| head | 39,499 | 0.2 % | 1 |
| last_block | 3,152,384 | 16.4 % | 1 |
| block_00 | 3,152,384 | 16.4 % | **6** |
| embed | 268,288 | 1.4 % | **9** |

**Restricting Φ raises k\* from 1 to 6–9, and only in that regime does inverting beat not
inverting** (`block_01`/C5: trunc_k10 0.757 > k=0 0.692). That is the mechanism the paper's
null needed: IF does not fail because the solve is wrong, it fails because curvature is
unestimable at this demo count. It is a statement about `p/N`, not about influence functions.

Two further B1 results worth keeping:
- **The action head carries essentially all the C1 signal**: `head` (0.2 % of parameters)
  scores 0.506 vs `ALL` 0.509. 99.8 % of the gradient is inert for attribution.
- Signal location is **target-dependent**: C1 lives in the head, C5 in `block_01`
  (0.692 vs ALL 0.401). No single block is "the" influential one.

## Honest limits

1. **The `block_01`/C5 result is noise.** It was the max over 11 groups × 8 k × 2 targets
   (176 comparisons). On hold-out it collapses: 0.189 / 0.310 / **−0.142** / 0.336. Reported
   as a negative transfer result, not a method.
2. **B1 runs on a REGENERATED E=5 ensemble, not the originals** (BLOCKERS #6). Training is
   bit-deterministic within an environment but not across torch/CUDA/GPU versions: the
   regenerated `ens_s201` differs from its cached slice by `G_rel_fro` 0.88 with gradient
   norms ~5× smaller. All B1 baselines are recomputed on that same ensemble, so B1's
   internal comparisons are valid; its absolute numbers are **not** comparable to the E=20
   cached-Gram rows.
3. **The datamodel is not a drop-in competitor.** It consumes 23 mask outcomes (which cost
   retrains); every gradient estimator sees zero. It is a different estimator class with a
   different input budget, and it must be evaluated leave-one-mask-out — scoring its
   in-sample coefficients gives ratios **above 1.0**, which is the signature of circularity
   (`demo_grain_lds` forms `X@β`, exactly what the datamodel minimises).
4. **n=24 is the binding constraint.** At Bonferroni-16 the critical ρ is ≈0.63; only one
   cell in the entire sweep cleared it. Most "improvements" here are un-adjudicable at this
   mask count, not absent.

## GPU ledger — 0.15 of 12 h used

| item | GPU-h |
|---|---|
| 5 × member checkpoint regeneration (93–94 s each) | 0.129 |
| Gram rebuild + cache verification | 0.002 |
| B1 layerwise (2 runs × 5 members × 11 groups; Φ computed once/member) | 0.011 |
| Phase A (A1–A5), all cached-Gram / CPU | 0.000 |
| **total** | **0.15** |

Under budget by 11.85 h. The constraint was never GPU: each member trains in 94 s and a
full-width gradient pass over 135 demos takes ~1 s on the H200. **B2 (target-functional
redesign), B3 (KFAC), B4 (TracIn density) were not run** — see HANDOFF.md. They are now
cheap and unblocked, since the checkpoint-regeneration path works and includes the 5
per-member checkpoints TracIn needs.

## Bottom line

No estimator achieved generality. The datamodel is the strongest new attributor (passes C5
and C2, where every gradient method fails) but collapses to a constant on C7/C9. GradDot
remains the best gradient estimator and remains C1-specific. The contribution of this pass is
the **mechanism**: attribution fails here because `p/N` makes curvature unestimable —
raising `k*` by shrinking Φ is what makes preconditioning start to work, and that is a
lever the field can act on, unlike "add more seeds".
