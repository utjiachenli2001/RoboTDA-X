# MAP — which (cache, outcome, ceiling) triple reproduces which paper number

All LDS is **demo grain**, n=24 masks, via `phase3/src/p6_lambda_sweep.py:demo_grain_lds`.
Ceilings and manifests are read verbatim from the archived artifacts. Nothing is recomputed.

## Tiers (defined once in `if_repair/data.py:TIERS`)

| tier | Gram cache | outcome table | seeds | seed agg | functional | ceiling |
|---|---|---|---|---|---|---|
| `bc_s10` | p6 + p11 (**E=20**) | `phase4/results/p12_outcomes_S10.parquet` | 401–410 | mean | `neg_plain_loss` | `p12_ceilings.json → targets[t].ceiling_10seed_SB` |
| `dev_s6` | p6 (**E=10**) | `phase2/results/stage_G6_outcomes.parquet` | 6-seed | mean | `neg_plain_loss` | `phase2/results/p1_demo_grain.json → all_targets[t].neg_plain_loss.ceiling_6seed_SB` |
| `diff_s10` | p17 (**E=5**) | `phase5/results/p15_outcomes_S10.parquet` | 601–610 | **median** | `neg_plain_loss` | `p15_verdict.json → all_targets_DESCRIPTIVE[t].ceiling_median_10seed_SB` |

`diff_s10` uses the **median** seed aggregator because phase 5 established the seed-*mean*
outcome is broken for the diffusion arm (`p15_verdict.seed_mean_brokenness_series`: the
mean-aggregated C1 ceiling collapses to 0.201 while the median-aggregated one is 0.830).
The preregistered aggregator there is the median; matching it is required for the ceiling and
the prediction to measure the same quantity.

## Anchors reproduced (Task 0, `python -m if_repair.run anchors`)

| anchor | estimator | tier / target | reproduced | archived target | diff |
|---|---|---|---|---|---|
| A1 | **GradDot_unitL2** | `bc_s10` C1, E=20 | **0.5130434782608695** | 0.5130434782608695 (`p16_analyze.py:C1_GRADDOT_ARCHIVED`) | **0.0e+00** |
| A1 | ratio / p | `bc_s10` C1 | ratio 0.540, p 0.0052 | ~0.54, ~0.005 | ✓ |
| A2 | GradDot_dmean (`scores_graddot`) | `bc_s10` C1, E=20 | 0.5930 (ratio 0.624) | — | different estimator, see BLOCKERS #1 |
| A3 | CV-IF tuned C1 → eval C5 | `bc_s10`, E=20 | 0.3939 (ratio 0.420, FAILS) | ~0.40 | 0.0096 |
| A3b | CV-IF tuned C1 → eval C5 | `dev_s6`, E=10 | **0.40347826086956523** | 0.40347826086956523 (`p6_lambda_sweep.json`) | **0.0e+00** |
| A4 | λ→∞ collapse | `bc_s10`, E=20 | IF and TRAK → GradDot_dmean, **max gap 0.0e+00** at ridge_rel ≥ 1e5 | gap → 0 | ✓ |
| A5 | GradDot_dmean | `dev_s6` C1, E=10 | 0.5043 | ~0.504 (`p6_lambda_extend.py` docstring) | 0.0004 |

Additionally `tests/test_anchors.py::test_both_graddot_variants_match_archived_p16` reproduces
**both** GradDot variants against all 7 non-focal rows of `phase5/results/p16_lds_table.csv`
to `<1e-12` — i.e. the whole loader + LDS path is verified against archived ground truth, not
just the two focal numbers.

## The two GradDot estimators (do not conflate — BLOCKERS #1)

| name | definition | C1 @ E=20 `bc_s10` | where |
|---|---|---|---|
| `GradDot_unitL2` | `K_m[:,j] / ‖K_m[:,j]‖₂`, then mean over m | **0.5130** (the paper's headline) | p16 PRIMARY `GradDot_E20_normalized`; `p5lib.normalized_ensemble_scores(normalize=True)` |
| `GradDot_dmean` | `K_m / mean(diag G_m)`, then mean over m | **0.5930** | p16 SECONDARY `GradDot_E20_dmean`; `p6_lambda_extend.scores_graddot(normalize_per_member=True)` |

`GradDot_dmean` is the mathematically true λ→∞ limit of the ensemble-mean IF/TRAK (verified: A4).
`GradDot_unitL2` is the preregistered champion whose value the paper reports. **Both are carried
as baselines throughout**; any claimed improvement must beat `GradDot_dmean` (0.5930), the
stronger of the two on C1.
