# BLOCKERS — stated assumptions that turned out to be wrong

## 1. (RESOLVED, but the brief is wrong) `scores_graddot` is **not** the 0.513 champion

The brief says the paper's headline `GradDot = 0.513` at E=20 is produced by
`phase3/src/p6_lambda_extend.py:scores_graddot(normalize_per_member=True)`, and warns that the
per-member `1/d_m` normalization is the subtlety to preserve.

Measured, on the exact triple the brief specifies (E=20 = p6+p11, `p12_outcomes_S10`
`neg_plain_loss`, ceiling 0.9506):

| estimator | C1 LDS | ratio |
|---|---|---|
| `scores_graddot(normalize_per_member=True)` — i.e. `K_m / mean(diag G_m)` | **0.5930** | 0.624 |
| per-member **unit-L2** normalization of `K_m`, then mean | **0.5130434782608695** | 0.540 |

The archived constant in `phase5/src/p16_analyze.py` is
`C1_GRADDOT_ARCHIVED = 0.5130434782608695`, described there as
*"P11 archived C1 GradDot_E20_normalized"*, and `p5lib.normalized_ensemble_scores` implements
that normalization as **unit L2** (`M / ‖M‖₂` per member column), not `1/d_m`.

So the repo contains **two** GradDot variants and the brief conflates them. Both are reproduced
here bit-for-bit against `p16_lds_table.csv` (`<1e-12`, all 7 non-focal targets).

**Impact / how it is handled.** The brief's `0.513` anchor is real and is reproduced exactly —
by `GradDot_unitL2`, implemented in `if_repair/anchors.py:graddot_unit_l2`. The brief's
*attribution of it to `scores_graddot`* is wrong. Both baselines are carried through every table,
and **the bar for "beats the champion" is the stronger of the two (`GradDot_dmean`, 0.5930 on C1)**
— not the 0.513 the brief names. This matters: several variants below beat 0.513 while losing to
0.5930, and calling those an improvement would be an artifact of picking the weaker baseline.

Note the brief's own secondary numbers ("unnormalized 0.397, normalized 0.504") *do* match
`scores_graddot` — but at the **E=10 `dev_s6`** triple (reproduced: 0.3965 / 0.5043), not E=20.

## 2. (RESOLVED, brief imprecise) "Cross-validated exact IF, C1" is tuned-on-C1, evaluated on C5

The brief asks for "cross-validated exact IF, C1, E=20 → ρ ≈ 0.40". A CV estimator *evaluated*
on C1 (tuned on C5) gives 0.3409 at E=10 / 0.5096 at E=20 — neither is 0.40. The repo's archived
`p6_lambda_sweep.json → MANDATORY_CROSS_VALIDATION` has exactly one 0.40:
`tune_on_C1__evaluate_on_C5 = 0.40347826086956523`. So "C1" names the **tuning** target.
Reproduced bit-for-bit at `dev_s6`/E=10 (diff 0.0e+00) and as 0.3939 at `bc_s10`/E=20 — both
within the required 0.03 of 0.40, and both FAIL the half-ceiling + p bar as the brief states.

## 3. (BLOCKING — Task 4 not attempted) No checkpoints in the repo

Task 4 requires per-layer Φ recomputed from `runs/stage_E/ens_s*/final.pt` via
`src/attribution.py:demo_gradient` / `build_targets`.

```
$ find . \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) | wc -l
0
```

`runs/stage_E/ens_s201..s210/` exist but contain only `demos.json`, `outcomes.json`,
`train_meta.json`, `train_log.jsonl` and markers — **no weights**. The published repo carries
code, configs, reports and small results only (see its initial-commit message); the 19.2M-param
checkpoints were never committed. Task 4 (subspace / per-layer, layer-ablation) is therefore
**not attempted**, exactly as the brief instructs. It is the only task that needed a GPU, so
nothing else is affected. Carried to `HANDOFF.md` as a GPU follow-up.

## 4. (RESOLVED, minor) Diffusion cache is E=5, and its tier needs the MEDIAN aggregator

`phase5/results/p17_diffusion_gram_cache.npz` has `G (5,135,135)`, `K (5,135,9)` — **5** members
(`dpens_s621..s625`), not the 10 of the BC arms. The brief does not state an E for it; noted so
that "E=20 = p6 ∪ p11" is not mistakenly applied here.

Separately, the diffusion tier must aggregate seeds by **median**, not mean: `p15_verdict.json →
seed_mean_brokenness_series` shows the mean-aggregated C1 ceiling collapsing to 0.201 vs 0.830
for the median. Using the mean would score predictions against a ceiling computed a different
way. Encoded in `data.py:TIERS["diff_s10"]["agg"] = "median"`.

## 5. (RESOLVED, environment) `src/bootstrap.py` hardcodes an absolute ROOT

`ROOT = "/mnt/sdb/ljc/RoboTDA-X"` — the original host's path — and every repo path constant
(`RESULTS`, `RUNS`, `P2_RESULTS`, `GK_CACHE`, …) derives from it. On any other machine the repo
modules are unimportable. Since the brief forbids editing repo files, this is handled **outside**
the repo by making the clone visible at the expected path:

```
sudo mkdir -p /mnt/sdb/ljc && sudo chown $USER /mnt/sdb/ljc
ln -s ~/code/RoboTDA-X /mnt/sdb/ljc/RoboTDA-X
```

`bootstrap.py` also pins `CUDA_VISIBLE_DEVICES` to one of GPUs 4–7 (a rule from the original
host). This box has one GPU (index 0), so the pin selects a nonexistent device — harmless here,
since Tasks 0–3 and 5 are pure NumPy and torch is only imported, never used for compute.
`if_repair` code itself derives its own ROOT from `__file__` and does not depend on the symlink.

---

# Pass 2 (methods sweep) — additional blockers

## 6. (RESOLVED with a caveat) Checkpoints are regenerable, but NOT reproducible

Pass 1 recorded (#3) that the repo ships no weights, blocking all of Phase B. That was only
half right. The training **data** is present (700 `.npz` under `data/proc`; `dataset.Bank`
builds fine), every member's seed + demo list + cfg is in `runs/stage_E/ens_s*/`, and a
member trains in **94 s** on the H200. So the checkpoints can be regenerated, which is what
unblocked B1.

They are **not the original weights**, and this was tested rather than assumed. Rebuilding
`(G,K)` for a regenerated `ens_s201` and comparing to its cached slice
(`if_repair/results/regen_verify_ens_s201.json`):

| quantity | value |
|---|---|
| `G` relative Frobenius difference | 0.879 |
| `K` relative Frobenius difference | 0.843 |
| per-target Spearman of `K` columns | 0.02 … 0.85 |
| median ratio of `diag(G)` regen/cached | **0.190** |

Gradient norms are ~5× smaller and the rank correlations are partial. The repo's determinism
gate (`p8b_determinism*.json`) established determinism of *rollouts on the original box*; it
does not imply bit-reproducibility across a different torch/CUDA/GPU stack, and this is a
direct measurement that it does not hold here (final loss −18.84 regen vs −18.70 archived).

**Consequence, applied throughout:** every B1 baseline is recomputed on the same regenerated
E=5 ensemble. B1's internal comparisons (group vs group, k vs k) are valid; B1's absolute
numbers are **not** comparable to the E=20 cached-Gram rows, and are labelled `regenE5`.

## 7. (RESOLVED) `demo_grain_lds` is circular for any estimator fit on mask outcomes

The demo-grain LDS forms its mask prediction as `sum_{d in mask} score_d`, which for a linear
datamodel is exactly `X @ beta` — the quantity the datamodel minimised. Scoring in-sample
coefficients therefore measures fit, not faithfulness, and duly returns ρ up to 0.986 with
**ratio-to-ceiling above 1.0** — impossible for a real estimator, since the ceiling is the
reliability of the outcome itself.

This bit twice: first in A4, then again in A5 when datamodel coefficients were fused into a
rank ensemble (it inflated C5 from an honest 0.43 to a fake 0.97). Both are now evaluated
**leave-one-mask-out**: each mask is predicted by a fit that excluded it. Gradient estimators
are unaffected — their scores never touch outcomes — and use the direct path.

Any future estimator that consumes outcome data must go through the LOO path in
`if_repair/confirm.py`, not `eval.evaluate`.

## 8. (NOT A BUG — real failure mode) The datamodel collapses to a constant on C7/C9

`A4_datamodel_lasso_LOO` returns NaN on hold-out targets C7 and C9. Cause: the
cross-validated `alpha` is selected at 0.1, which zeroes **every** coefficient, so the model
predicts the mean for all 24 masks and Spearman is undefined.

| target | selected alpha | non-zero coefs |
|---|---|---|
| C2 | 0.01 | 6 |
| C4 | 0.001 | 22 |
| C7 | 0.1 | **0** |
| C9 | 0.1 | **0** |

This is honest behaviour — CV decided the design explains nothing for those targets — but it
means the datamodel does not degrade gracefully: it either finds structure (C5, C2) or
returns nothing at all. NaN is scored as a non-pass, never dropped.

## 9. (NOT ATTEMPTED, not blocked) B2 / B3 / B4 were not run

Only B1 was executed from Phase B. This was a wall-clock/session limit, not a resource or
data limit: 0.15 of 12 GPU-h were used, and the checkpoint-regeneration path that unblocked
B1 unblocks the rest equally (including the 5 per-member checkpoints `ckpt_0..4.pt` that B4's
TracIn density needs). Carried to HANDOFF.md with the exact entry points.
