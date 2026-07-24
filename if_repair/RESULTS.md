# RESULTS — dev tables (C1, C5 only)

Everything below is **demo-grain LDS, n=24 masks**, tier `bc_s10` (E=20 Gram = p6∪p11,
`p12_outcomes_S10` `neg_plain_loss` 10-seed mean, `ceiling_10seed_SB`) unless stated.
Every cell is reported as **LDS / ceiling / ratio / p**; ratio = LDS ÷ ceiling; the bar is
ratio ≥ 0.5 **and** p < α (α = 0.025 on dev, Bonferroni-corrected on hold-out).
No hold-out target appears anywhere in this file.

Ceilings: C1 = 0.950559, C5 = 0.938088. Bars: 0.475279, 0.469044.

## Baselines (Task 0)

| estimator | C1 LDS | C1 ratio | C1 p | C5 LDS | C5 ratio | C5 p |
|---|---|---|---|---|---|---|
| **GradDot_unitL2** (published champion) | 0.5130 | 0.540 | 0.0052 | 0.3287 | 0.350 | 0.0584 |
| **GradDot_dmean** (`scores_graddot`, true λ→∞ limit) | **0.5930** | **0.624** | 0.0011 | 0.3904 | 0.416 | 0.0296 |
| exact IF (λ→0) | −0.1626 | −0.171 | 0.776 | 0.0148 | 0.016 | 0.473 |
| CV-IF (tuned C1, frozen, held-out) | — | — | — | 0.3939 | 0.420 | 0.0284 |

The champion to beat is **GradDot_dmean = 0.5930**, not the 0.513 the brief names
(BLOCKERS #1). Reporting against 0.513 would make three variants below look like wins.

## Task 1 — member aggregation (`results/exp_01_bc_s10.csv`)

**C1 LDS**, aggregator across columns:

| config | mean | median | trimmed 0.1 | trimmed 0.2 | Hodges–Lehmann |
|---|---|---|---|---|---|
| GradDot / dmean | **0.5930** | 0.3852 | 0.5287 | 0.5000 | 0.5400 |
| GradDot / unitL2 | 0.5130 | 0.5470 | 0.5209 | 0.5426 | **0.5678** |
| IF / none @ ridge 1e+1 | **0.5965** | 0.3939 | 0.5304 | 0.5139 | 0.5409 |
| IF / none @ ridge 1e−2 | 0.3217 | 0.4800 | 0.4730 | 0.4922 | 0.4635 |
| IF / unitL2 @ ridge 1e+1 | 0.5130 | 0.5496 | 0.5209 | 0.5409 | 0.5678 |
| TRAK / none @ ridge 1e−2 | −0.1913 | −0.3157 | −0.2296 | −0.2791 | −0.2817 |
| TRAK / unitL2 @ ridge 1e+1 | 0.4861 | 0.5113 | 0.4878 | 0.4991 | 0.5061 |

**C5 LDS**:

| config | mean | median | trimmed 0.1 | trimmed 0.2 | Hodges–Lehmann |
|---|---|---|---|---|---|
| GradDot / dmean | **0.3904** | 0.1713 | 0.3243 | 0.2843 | 0.3557 |
| GradDot / unitL2 | 0.3287 | 0.1157 | 0.2835 | 0.2861 | 0.3287 |
| IF / none @ 1e+1 | 0.3939 | 0.1704 | 0.3235 | 0.2913 | 0.3496 |
| IF / none @ 1e−2 | **0.4104** | 0.3139 | 0.4130 | 0.3339 | 0.3313 |
| IF / unitL2 @ 1e+1 | 0.3287 | 0.0913 | 0.2835 | 0.2878 | 0.3261 |
| TRAK / none @ 1e−2 | 0.2217 | 0.1809 | 0.3000 | 0.2217 | 0.3130 |
| TRAK / unitL2 @ 1e+1 | 0.3548 | 0.3096 | 0.3687 | 0.3643 | 0.3652 |

**GATE — does any robust aggregator raise C1 LDS vs `mean`? NO.**
Best over all configs × aggregators = 0.5965 (IF/none@1e+1, **mean**); the best
mean-aggregated value is the same 0.5965. Robust aggregation never produces a new maximum.

It does help *within* the champion's own normalization: GradDot/unitL2 goes
0.5130 → 0.5678 under Hodges–Lehmann (+0.055 on C1). But that only recovers part of what
the dmean convention already gets for free with a plain mean (0.5930).

### Member-tail diagnostic (`results/exp_01_tail_stats.csv`)

Excess kurtosis of each demo's score across members (0 = Gaussian):

| config | target | median kurt | p90 kurt | frac demos kurt>1 | (max−min)/\|mean\| |
|---|---|---|---|---|---|
| GradDot / unitL2 | C1 | **−0.764** | 2.175 | 0.141 | 9.17 |
| GradDot / unitL2 | C5 | **−0.961** | 2.710 | 0.170 | 11.54 |
| GradDot / dmean | C1 | 3.276 | 9.478 | 0.859 | 18.34 |
| GradDot / dmean | C5 | 2.104 | 6.080 | 0.726 | 12.19 |
| IF / none | C1 | 3.340 | 9.619 | 0.852 | 18.79 |
| TRAK / none | C1 | 4.454 | 11.531 | 0.904 | 17.42 |

This is the explanation for the gate. The unnormalized families *are* heavy-tailed across
members (73–90 % of demos with excess kurtosis > 1), which is exactly the condition under
which a robust aggregator should win. But unit-L2 normalization **removes the heavy tails at
the source** (kurtosis goes negative, i.e. lighter than Gaussian), and once it has, the mean
is already the right summary. The tails were a per-member *scale* artifact, not outlier
members — so normalize, don't robustify.

## Task 2 — spectral truncated inverse (`results/k_sweep_C1.json`, `k_sweep_C5.json`, `figs/k_sweep.png`)

`truncated_if(k)` inverts the top-k eigendirections of each `G_m` and acts as the identity on
the rest. Endpoints verified numerically (`tests/test_spectral.py`):
`k=0` ≡ GradDot (≤1e-6), `k=N` ≡ exact IF (≤1e-3).

**C1 (dmean convention)** — LDS by k:

| k | 0 | 1 | 2 | 5 | 10 | 20 | 40 | 80 | 120 | 132 | 133 | 134 | 135 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LDS | **+0.593** | +0.537 | +0.250 | +0.052 | −0.093 | −0.165 | −0.141 | −0.327 | −0.255 | −0.104 | +0.023 | +0.162 | −0.238 |

**C5 (dmean convention)**:

| k | 0 | 1 | 2 | 5 | 20 | 100 | 130 | 131 | 132 | **133** | 134 | 135 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LDS | +0.390 | +0.410 | +0.357 | +0.328 | +0.320 | +0.143 | +0.167 | +0.241 | +0.398 | **+0.560** | −0.059 | +0.110 |

**Shape of the curve — the deliverable.** On C1 the maximum is at **k = 0** and the curve
falls away immediately: *any* amount of inverse preconditioning hurts, not merely the noisy
tail of it. This is a stronger statement than the brief's hypothesis ("interior peak ⇒ the
noise part of the preconditioner hurts") — there is no interior peak on C1 at all.

C5 does show an interior maximum, at k=133 (0.390 → **0.560**, ratio 0.597, p=0.0022, would
pass). **It is not real.** Its neighbours are k=132 → 0.398 and k=134 → −0.059: a
single-k spike of width 1 in a 136-point sweep, i.e. the sweep's own multiple-comparison
noise. Cross-validation confirms it:

| tune on | selected k | evaluate on | LDS | ratio | p | pass |
|---|---|---|---|---|---|---|
| C1 | k=0 | C5 | 0.3904 | 0.416 | 0.0296 | ✗ |
| C5 | k=133 | C1 | **0.0235** | 0.025 | 0.457 | ✗ |

Tuning k on C5 and freezing it destroys C1 (0.593 → 0.024). No choice of k transfers.

**Spectrum / noise floor** (`figs/spectrum.png`, `results/spectrum_null.json`): parallel
analysis against a 200-permutation null (off-diagonal permutation of each `G_m`) gives
**k\* = 1** (median over members; range 1–3). Only the leading eigendirection is
distinguishable from random demo pairing — consistent with the k-sweep, where everything
past k≈1 is noise being amplified by 1/λ.

**Damped variant** (`damped_if`, γ by the repo's CV protocol):

| mode | tune → eval | γ | LDS | ratio | p | pass |
|---|---|---|---|---|---|---|
| diag | C1 → C5 | 1e−1 | −0.0061 | −0.006 | 0.511 | ✗ |
| diag | C5 → C1 | 1e+3 | 0.0817 | 0.086 | 0.352 | ✗ |
| muI | C1 → C5 | 1e+2 | **0.4878** | **0.520** | 0.0078 | **✓** |
| muI | C5 → C1 | 1e+1 | 0.1670 | 0.176 | 0.218 | ✗ |

`muI` damping cross-validates onto C5 above the half-ceiling bar — the only dev result that
does so on C5 — but the reverse direction collapses (0.167 on C1), so it is unstable, not a
champion. Reported, not promoted.

## Task 3 — variance-aware shrinkage (`results/exp_02_bc_s10.csv`)

James–Stein on each demo's ensemble mean, shrunk by cross-member variance.

| config | variant | C1 LDS | C1 ratio | top-20 Jaccard |
|---|---|---|---|---|
| GradDot/unitL2 | baseline mean | 0.5130 | 0.540 | 0.399 |
| GradDot/unitL2 | JS → cluster mean | 0.5009 | 0.527 | 0.344 |
| GradDot/unitL2 | JS → grand mean | 0.5130 | 0.540 | 0.406 |
| GradDot/dmean | baseline mean | **0.5930** | 0.624 | 0.196 |
| GradDot/dmean | JS → cluster mean | **−0.0844** | −0.089 | 0.225 |
| GradDot/dmean | JS → grand mean | NaN (constant) | — | 0.263 |
| IF/none@1e1 | baseline mean | 0.5965 | 0.628 | 0.209 |
| IF/none@1e1 | JS → cluster mean | −0.0705 | −0.074 | 0.308 |

**GATE — does shrinkage help? NO**, on either target (best shrunk = best baseline).

Shrinking toward *cluster means* is actively destructive for the unnormalized families
(0.593 → −0.084): almost all of the usable signal is *between-cluster* contrast, and pulling
demos toward their cluster centre deletes precisely that. Shrinking toward zero is a no-op
(the JS factor saturates at 1). Stability and faithfulness move in opposite directions here —
JS raises top-20 Jaccard for dmean (0.196 → 0.225–0.271) while collapsing LDS — which is why
stability was reported alongside, not instead of, faithfulness.

## Summary of dev gates

| task | question | answer |
|---|---|---|
| 1 | any robust aggregator raises C1 LDS vs mean? | **No** |
| 2 | interior peak in the k-sweep that survives CV? | **No** |
| 3 | variance-aware shrinkage helps? | **No** |

No variant from Tasks 1–3 beats `GradDot_dmean` (0.5930) on C1. See `FINDINGS.md` for the
frozen hold-out run.

---

# Pass 2 — methods sweep (A4, A5, B1) and GPU ledger

Pass-1 tables above are unchanged. New estimators below; same conventions
(demo grain, n=24, tier `bc_s10`, reported as LDS / ceiling / ratio / p).

## A4 — design-based datamodel (`results/a4_datamodel.csv`)

Headline `lds` is **out-of-fold** (each mask predicted by a fit excluding it). The in-sample
column is retained only to expose the circularity (BLOCKERS #7).

| model | target | alpha | LDS (OOF) | ratio | p | pass | LDS in-sample (INVALID) | non-zero coefs |
|---|---|---|---|---|---|---|---|---|
| ridge | C1 | 100 | 0.043 | 0.045 | 0.422 | ✗ | 0.793 | 135 |
| lasso | C1 | 0.001 | 0.397 | 0.417 | 0.028 | ✗ | 0.986 | 19 |
| elasticnet | C1 | 0.01 | 0.300 | 0.316 | 0.077 | ✗ | 0.891 | 14 |
| ridge | C5 | 10 | 0.784 | 0.836 | 2.9e−06 | **✓** | 0.978 | 135 |
| **lasso** | **C5** | 0.001 | **0.824** | **0.879** | 3.7e−07 | **✓** | 0.985 | 19 |
| elasticnet | C5 | 0.01 | 0.795 | 0.847 | 1.7e−06 | **✓** | 0.970 | 8 |

The in-sample column reaching ratio > 1.0 is the tell. Note the datamodel wins C5 by a wide
margin (0.879 vs GradDot's 0.416) and loses C1 (0.417 vs 0.624) — exactly opposite to every
gradient estimator.

## A5 — rank fusion (`results/a5_fusion.csv`), target-blind rules

Ratio-to-ceiling. Recipes containing the datamodel are evaluated leave-one-mask-out.

| recipe | fusion | C1 | C5 |
|---|---|---|---|
| gradients_only | z-score avg | **0.649** | 0.429 |
| gradients_only | borda | 0.626 | 0.400 |
| GradDot+datamodel | z-score avg | 0.879 | **0.814** |
| GradDot+HL+datamodel | z-score avg | 0.851 | 0.793 |

`gradients_only` z-avg = **0.6165 LDS (ratio 0.649)** on C1 is a genuine, circularity-free
improvement over GradDot_dmean (0.5930 / 0.624) — the best C1 number in either pass.

## B1 — layerwise influence (`results/b1_layerwise.csv`), regenerated E=5 ensemble

Ratio-to-ceiling. **Not comparable to the E=20 rows** (BLOCKERS #6).

| group | params | C1 GradDot | C1 best-k | C5 GradDot | C5 best-k |
|---|---|---|---|---|---|
| ALL | 19.2M | 0.509 | 0.509 (k=0) | 0.401 | 0.572 (k=20) |
| head | 39.5K (0.2 %) | **0.506** | 0.506 (k=0) | 0.413 | 0.454 (k=2) |
| embed | 268K | 0.277 | 0.277 (k=0) | 0.485 | 0.485 (k=0) |
| block_00 | 3.15M | 0.042 | 0.042 | 0.559 | 0.559 |
| **block_01** | 3.15M | −0.072 | −0.072 | **0.692** | **0.757 (k=10)** |
| block_05 / last | 3.15M | 0.470 | 0.470 | 0.399 | 0.489 (k=2) |

`k*` by group (parallel analysis, 100 perms): ALL 1, head 1, last_block 1, **block_00 6,
embed 9**. Restricting Φ makes curvature estimable, and only there does truncated inversion
beat GradDot.

## Task 9 — confirmatory hold-out (`results/holdout_phase2.csv`), computed once

C2/C4/C7/C9, Bonferroni over the 16 tested cells (α = 0.00156). Full table in FINDINGS.md.
One pass: **datamodel_lasso on C2, ratio 0.639, p = 0.00115**. Everything else fails.
`block_01`/k=10 does not transfer (C7 = −0.142), confirming it as sweep noise.

## GPU ledger (budget 12 h, hard stop)

| # | item | detail | GPU-h | cumulative |
|---|---|---|---|---|
| 1 | rate measurement | `ens_s201` retrain, 94 s wall | 0.026 | 0.026 |
| 2 | Gram rebuild + cache verify | full-width Φ, 135 demos, ~1 s | 0.002 | 0.028 |
| 3 | member regeneration ×4 | `ens_s202..205`, 93 s each | 0.103 | 0.131 |
| 4 | B1 layerwise | 5 members × 11 groups, Φ once/member, 19.5 s | 0.005 | 0.136 |
| 5 | B1 re-run inside `confirm.py` | same, for hold-out | 0.006 | **0.142** |

**Total 0.15 GPU-h of 12.** Phase A (A1–A5) used zero. No GPU was spent on larger ensembles
or more seeds, per the prime directive.

---

# Pass 3 — Phase B completed, plus a retraction

Pass 2 ran only B1 and left B2/B3/B4/B5 unrun. This pass runs all four, adds the diffusion arm
(B7) and the width control (B6), retrains the ground truth so that redesigned functionals have
outcomes of their own, and tests five preregistered hypotheses on a fresh mask draw.

Ensemble for every gradient estimator below: the **regenerated E=5** (BLOCKERS #6). Every
baseline is recomputed on it. Where an outcome is used it is named explicitly, because this pass
has two of them — the archived `p12_outcomes_S10` and the regenerated campaign A table.

Full grids are in `if_repair/results/*.csv`; the slices below are the ones that carry the
conclusions.

## B6 — Φ width vs Φ role (`b6_width.csv`, `b6_kstar_compare.csv`)

`python -m if_repair.b6_width --kstar_compare`

| Φ restricted to | params | k\* (median of 5 members) | per-member |
|---|---|---|---|
| ALL | 19,222,091 | 1 | 1,1,1,1,2 |
| head | 39,499 | 1 | 1,1,1,1,2 |
| embed | 268,288 | **9** | 9,10,2,7,10 |
| block_00 | 3,152,384 | **6** | 4,5,11,6,6 |
| last_block | **3,152,384** | **1** | 1,1,2,1,2 |
| random ×3 | 1,000 → 3,000,000 | 1–3 | never above 3 |
| random ×3 | 10,000,000 | 1 | 1,1,1,1,2 |

`block_00` and `last_block` are the same size and differ 6 vs 1; random subsets never leave
k\* ≈ 1 at any width. **k\* is a property of which subspace, not its dimension** (BLOCKERS #10).

The width curve itself (mean over 3 draws, C1/C5):

| width | frac of model | k\* | ratio at k=0 | best-k ratio | gain |
|---|---|---|---|---|---|
| 1,000 | 0.00005 | 1.0 | 0.164 / −0.195 | 0.284 / 0.401 | 0.114 / 0.559 |
| 10,000 | 0.0005 | 2.0 | 0.121 / 0.323 | 0.220 / 0.454 | 0.094 / 0.123 |
| 100,000 | 0.005 | 1.3 | 0.457 / 0.098 | 0.505 / 0.544 | 0.046 / 0.419 |
| 1,000,000 | 0.052 | 1.0 | 0.532 / 0.396 | 0.532 / 0.493 | 0.000 / 0.090 |
| 19,222,091 | 1.000 | 1.0 | 0.509 / 0.401 | 0.509 / 0.572 | 0.000 / 0.160 |

Spearman(width, k\*) = −0.365; Spearman(k\*, gain from inverting) = +0.455. Both weak, and the
second is driven by the narrowest cells, where the unpreconditioned score is itself broken
(C5 at width 1,000 has ratio −0.195 at k=0, so "inverting helps" means it rescues a broken
score).

## B3 — KFAC / EK-FAC (`b3_kfac.csv`)

`python -m if_repair.b3_kfac`. Φ **and** the preconditioner are restricted together
(BLOCKERS #13); `method=none` is B1's GradDot on the same restricted Φ.

C1 / C5 ratio-to-ceiling, archived outcomes:

| method, λ_rel | ALL | head | last_block | block_00 | embed |
|---|---|---|---|---|---|
| none (GradDot) | 0.509 / 0.401 | 0.506 / 0.413 | 0.470 / 0.399 | 0.042 / 0.559 | 0.277 / 0.485 |
| kfac 1e−4 | 0.353 / 0.407 | 0.504 / 0.401 | 0.467 / 0.402 | 0.037 / 0.437 | **0.320 / 0.615** |
| kfac 1e−2 | 0.353 / 0.407 | 0.501 / 0.401 | 0.467 / 0.402 | 0.037 / 0.437 | 0.294 / 0.613 |
| kfac 1e+0 | 0.353 / 0.407 | 0.506 / 0.401 | 0.467 / 0.402 | 0.037 / 0.437 | 0.243 / 0.498 |
| kfac 1e+2 | 0.353 / 0.407 | 0.506 / 0.401 | 0.467 / 0.402 | 0.037 / 0.437 | 0.031 / 0.319 |
| kfac 1e+4 | 0.353 / 0.407 | 0.506 / 0.401 | 0.467 / 0.402 | 0.037 / 0.437 | 0.017 / 0.298 |
| ekfac 1e−4 | 0.418 / 0.427 | 0.506 / 0.401 | 0.447 / 0.430 | 0.182 / 0.514 | **0.017** / 0.298 |

Fraction of each group's parameters that KFAC actually covers (nn.Linear only; the fused
attention projection, LayerNorms, biases and positional embeddings stay identity): ALL 0.752,
head 0.972, embed 0.977, block_00 0.748, last_block 0.748.

`embed` is the only group where preconditioning helps, on both dev targets, with a monotone
dose-response in λ. EK-FAC — same eigenvectors, eigenvalues re-estimated from the 135 demo
gradients instead of ~92k frames — takes `embed`/C1 from 0.320 to 0.017 (BLOCKERS #11).

## B4 — TracIn (`b4_tracin.csv`, `b4_single_ckpt.csv`)

`python -m if_repair.b4_tracin` and `--decompose`. Checkpoints at steps 1600–8000; η from the
frozen cosine schedule (9.46e−5 down to 1.00e−5).

Top C1 cells (archived outcomes, α = 0.025):

| group | density | LR-weighted | estimator | ratio | p |
|---|---|---|---|---|---|
| head | last5 | yes | TracIn | **0.602** | 0.0017 |
| ALL | last3 | no | TracIn_trunc_k1 | 0.594 | 0.0020 |
| ALL | last5 | yes | TracIn | 0.594 | 0.0020 |
| head | last2 | no | TracIn | 0.578 | 0.0027 |

Every top C5 cell is `last1` (block_00 0.559, embed 0.485) — trajectory integration does not help
C5.

Identity check: `last1` with no LR weighting reproduces GradDot_dmean on the same ensemble
(ALL 0.509, head 0.506), so the sweep contains the baseline as an endpoint.

Per-checkpoint decomposition (Φ = head):

| cell | C1 | C5 |
|---|---|---|
| ckpt_0 (step 1600) | 0.215 | 0.134 |
| ckpt_1 (3200) | 0.550 | 0.104 |
| ckpt_2 (4800) | 0.436 | 0.131 |
| ckpt_3 (6400) | **0.598** | 0.217 |
| ckpt_4 (8000, = GradDot) | 0.506 | **0.413** |
| last5 LR-weighted | 0.602 | 0.127 |
| last5 unweighted | 0.553 | 0.192 |

The 5-checkpoint sum equals the best single checkpoint to within 0.004, and adjacent checkpoints
differ by up to 0.42 — four times the gain over GradDot. The C1 result is an intermediate-
checkpoint effect sitting inside the n=24 noise, not trajectory integration.

## B2 — target functionals (`b2_ceilings_*.csv`, `b2_scores_*.csv`)

`python -m if_repair.b2_functionals --obs archived` and `--obs campaign`.

Protocol: weighting → outcome → **its own split-half ceiling** → only then score. The ceiling
recipe reproduces all nine archived `p12` ceilings to <1e−12 (`tests/test_pass3.py`).

Ceilings (campaign A outcomes, 24 masks × 10 seeds): every functional clears the 0.4 gate.

| target | plain | transport | interaction | ens_var | ens_var_q75 | fail_div | fail_div_q75 |
|---|---|---|---|---|---|---|---|
| C1 | 0.941 | 0.929 | 0.962 | 0.914 | 0.914 | 0.950 | 0.934 |
| C5 | 0.920 | 0.912 | 0.924 | 0.929 | 0.898 | 0.929 | 0.917 |

GradDot_dmean ratio-to-ceiling, **campaign A outcomes** (fully self-consistent — regenerated Φ,
regenerated outcomes):

| functional | C1 | C5 |
|---|---|---|
| interaction | **0.526** (p=0.0058, pass) | 0.491 |
| plain | 0.475 | 0.374 |
| fail_div_q75 | 0.031 | 0.502 |
| fail_div | −0.078 | **0.535** (p=0.0067, pass) |
| ens_var_q75 | 0.274 | 0.430 |
| ens_var | −0.054 | 0.368 |
| transport | −0.233 | −0.490 |

Same table against the **archived** outcomes, for the three functionals the archive contains:
plain 0.509 / 0.401, interaction 0.461 / 0.474, transport −0.239 / −0.502.

`transport` is negative on both targets under both independent ground truths, with ceilings of
0.89–0.96 throughout.

NOT ATTEMPTED: rollout-visited-state weighting (BLOCKERS #15).

## B5 — init vs order (`b5_variance.csv`, `b5_design_reliability.csv`)

See FINDINGS. The premise was wrong (Stage G is already common-init), the order main effect is
0.1–0.5%, the binding noise is mask×init at 9.8–34.9%, and the three seeding designs differ by
≤0.05 without a consistent ordering. Retired.

## B7 — diffusion arm (`b7_diffusion.csv`, `b7_regen_verify.csv`)

`bash if_repair/regen_dp.sh` then `python -m if_repair.b7_diffusion`. Tier `diff_s10`
(median aggregator, BLOCKERS #4). Groups are role analogues of the BC ones, not name matches.

Regeneration check against the archived p17 cache — far tighter than the BC arm:

| member | G_rel_fro | K_rel_fro | diag ratio | mean K Spearman |
|---|---|---|---|---|
| dpens_s621 | 0.131 | 0.081 | 1.115 | 0.997 |
| dpens_s622 | 0.062 | 0.189 | 0.928 | 0.994 |
| dpens_s623 | 0.328 | 0.667 | 0.824 | 0.964 |
| dpens_s624 | 0.053 | 0.114 | 0.956 | 0.996 |
| dpens_s625 | 0.270 | 0.184 | 0.732 | 0.998 |
| *BC ens_s201, for contrast* | *0.879* | *0.843* | *0.190* | *0.02–0.85* |

C1 ratio-to-ceiling, plain TracIn:

| density | LR | ALL | act_out | den_blocks_05 | embed | obs_blocks_00 |
|---|---|---|---|---|---|---|
| last1 | — | 0.516 | 0.489 | 0.530 | −0.346 | −0.026 |
| last5 | no | 0.582 | 0.613 | 0.580 | −0.132 | −0.061 |
| last5 | yes | **0.637** | **0.622** | 0.586 | −0.131 | −0.001 |
| evenly3 | no | 0.613 | **0.653** | 0.566 | −0.162 | 0.017 |

Best C5 cell is 0.458 (obs_blocks_00, last5, LR, trunc_k1, p=0.031) — fails.

Win-condition-2 transfer: TracIn/last5/LR-weighted passes C1 in both classes and beats its
GradDot control in both (BC head 0.602 vs 0.506, BC ALL 0.594 vs 0.509, diffusion act_out
0.622 vs 0.489, diffusion ALL 0.637 vs 0.516).

## Fresh-mask confirmatory family (`confirm3_fresh_masks.csv`)

`python -m if_repair.confirm3`. 24 fresh masks (Stage-G generator, seed 4711; disjoint from the
archived 24, pinned by `tests/test_pass3.py`), campaign B outcomes at 6 seeds, Bonferroni
α = 0.025/5 = 0.005. `PREREG` frozen and committed at `aca92e6` with campaign B at zero runs.

Fresh ceilings: plain C1 0.925, C2 0.912, C5 0.936; interaction C1 0.947, C2 0.941, C5 0.900 —
all far above the 0.4 gate, so every hypothesis was adjudicable.

| hypothesis | weighting | LDS | ceiling | ratio | bar | p | PASS |
|---|---|---|---|---|---|---|---|
| H1_datamodel_C2 | plain | 0.6643 | 0.9119 | **0.729** | 0.456 | 0.0002 | **YES** |
| H2_tracin_head_C1 | plain | 0.3791 | 0.9248 | 0.410 | 0.462 | 0.0338 | no |
| H3_kfac_embed_C5 | plain | 0.4670 | 0.9357 | 0.499 | 0.468 | 0.0107 | no |
| H4_interaction_functional_C5 | interaction | −0.0296 | 0.9002 | −0.033 | 0.450 | 0.5545 | no |
| H5_graddot_head_C1 | plain | 0.3113 | 0.9248 | 0.337 | 0.462 | 0.0693 | no |

Reference rows (not in the family, uncorrected α = 0.025):

| estimator | C1 | C2 | C5 |
|---|---|---|---|
| GradDot_dmean, ALL | 0.337 (p 0.069) | 0.493 (p 0.014) | −0.106 (p 0.678) |
| TracIn head, last1 (= GradDot head) | 0.337 | 0.491 | −0.106 |

The reference rows are the point: the **baseline** also fails on the fresh draw. GradDot_dmean on
C1 goes 0.509 (archived masks, archived outcomes) → 0.475 (archived masks, campaign A outcomes)
→ 0.337 (fresh masks). On C5 it goes 0.401 → 0.374 → −0.106. The mask draw is worth ~0.14 on C1
and a sign flip on C5, which is larger than every estimator effect this project has reported.

## GPU ledger (`gpu_ledger_pass3.csv`)

`python -m if_repair.gpu_ledger`. Three readings, because concurrency makes the single word
"GPU-hours" ambiguous by a factor of three.

| stage | jobs | job_h | solo_h | occupancy_h |
|---|---|---|---|---|
| campaign A | 240 | 10.793 | 5.800 | 3.436 |
| campaign B | 144 | 5.276 | 3.480 | 1.690 |
| campaign C | 72 | 2.598 | 1.740 | 0.831 |
| diffusion regeneration | 5 | 0.656 | 0.460 | overlapped |
| B7 diffusion TracIn cache | 25 | 1.013 | 1.014 | overlapped |
| B4 TracIn cache | 25 | 0.064 | 0.064 | overlapped |
| B3 KFAC | 5 | 0.006 | 0.038 | overlapped |
| B6 width cache | 5 | 0.016 | 0.013 | overlapped |
| B2 Gram | 5 | 0.010 | 0.038 | overlapped |
| **total** | | **20.46** | **12.65** | **6.00** |

`job_h` double-counts contention (3 workers stretch an 87 s retrain to 137–160 s). `solo_h` is
the work actually done and is the figure comparable to the pass-1/2 ledger, which ran serially.
Project total **12.80 solo-h against a 12 h budget** — 0.8 h over, spent on the diffusion arm.

## Artifacts

- `results/campaign_outcomes.parquet` — 28,728 rows: every (campaign, weighting, target, mask,
  seed_init, seed_order) outcome for all three campaigns. 153 KB, committed.
- `results/campaign_ceilings.csv` — the split-half ceiling of each (campaign, weighting, target).
- `runs/campaigns/{A,B,C}/*.npz` — the per-frame held-out losses. **Machine-local and gitignored**
  (~40 MB); rebuilding them costs ~11 solo-h. Keep them if the box survives.
- `runs/regen_dp/` — the regenerated diffusion ensemble with checkpoints. Machine-local.

---

# Pass 3, continued — B8 (mask-draw sampling) and the confirmatory re-run at matched depth

After the first write-up I found two errors of my own (both in BLOCKERS #17/#18): campaign B was
run at 6 seeds and compared against depth-10 dev numbers, and the 0.509->0.337 C1 drop was
mis-attributed to the mask draw when most of it was seed depth. Campaign B was extended to 10
seeds (240 runs total) and two analyses added.

## B8 -- sampling distribution over mask draws (`b8_maskdraw.py`, `b8_maskdraw_*_d{6,10}.csv`)

`python -m if_repair.b8_maskdraw --targets C1,C5,C2 --n_boot 2000 --n_seeds 10`. Pools the 24
G-series + 24 H-series masks (exchangeable; `tests/test_pass3.py` pins same-generator, same-size,
same-universe, disjoint-id) and bootstraps 2000 random 24-mask subsets. Matched 10-seed depth.

Pooled-48 point estimates (ceiling ~0.94):

| target | GradDot_ALL | GradDot_head | TracIn_head_last5 | KFAC_embed | datamodel_LOO |
|---|---|---|---|---|---|
| C1 | 0.398 | 0.394 | 0.473 | 0.124 | 0.416 |
| C5 | 0.137 | 0.138 | 0.020 | **0.631** | **0.684** |
| C2 | 0.434 | 0.437 | 0.466 | 0.211 | 0.614 |

Bootstrap level sd (why absolute numbers are unresolvable): 0.13-0.17 on every target for
GradDot, against a 0.5 pass bar. The ceiling itself has sd 0.014-0.023.

Paired vs GradDot_ALL on the SAME subsets (the draw cancels):

| target | estimator | mean Δ | paired sd | p05 | p95 | beats GradDot |
|---|---|---|---|---|---|---|
| C5 | KFAC_embed | +0.483 | 0.153 | +0.240 | +0.733 | **100.0%** |
| C5 | datamodel_LOO | +0.557 | 0.202 | +0.222 | +0.885 | 99.7% |
| C5 | TracIn_head_last5 | −0.117 | 0.094 | −0.268 | +0.035 | 10.2% |
| C1 | TracIn_head_last5 | +0.076 | 0.104 | −0.098 | +0.249 | 77.1% |
| C1 | KFAC_embed | −0.274 | 0.222 | −0.650 | +0.083 | 10.4% |
| C2 | datamodel_LOO | +0.121 | 0.230 | −0.258 | +0.469 | 70.9% |
| any | GradDot_head | ~0 | **0.005** | | | — |

Depth-6 (`_d6.csv`) agrees: C5 KFAC +0.389 sd 0.157 99.6%, datamodel +0.496 sd 0.216 98.9%,
head-vs-ALL sd 0.004. The paired sd is far below the level sd everywhere, which is the whole
point: at n=24 you cannot say what an estimator's ratio IS, but you can say whether it beats the
baseline on the same masks.

## Confirmatory family at matched 10-seed depth (`confirm3_fresh_masks_d10.csv`)

`python -m if_repair.confirm3` on campaign B extended to 10 seeds. Fresh ceilings all 0.94-0.97.

| hypothesis | weighting | LDS | ceiling | ratio | p | Bonf α | PASS |
|---|---|---|---|---|---|---|---|
| H1 datamodel on C2 | plain | 0.5887 | 0.940 | 0.626 | 0.0012 | 0.005 | **YES** |
| H3 KFAC embed on C5 | plain | 0.5409 | 0.960 | 0.563 | 0.0032 | 0.005 | **YES** |
| H2 TracIn head on C1 | plain | 0.3252 | 0.958 | 0.340 | 0.060 | 0.005 | no |
| H4 interaction on C5 | interaction | −0.0522 | 0.956 | −0.055 | 0.60 | 0.005 | no |
| H5 GradDot head on C1 | plain | 0.2713 | 0.958 | 0.283 | 0.100 | 0.005 | no |

2 of 5, vs 1 of 5 at depth 6. H3 crossed the bar (0.499 -> 0.563) purely from matching the seed
depth to the dev protocol. Reference GradDot_ALL at depth 10: C1 0.283, C2 0.448, C5 −0.189 --
the baseline fails its own absolute bar on the fresh draw, which is BLOCKERS #17.

---

# Passes 4-6 -- new estimator classes, the leverage family, and six-draw confirmation

All numbers demo-grain LDS, n=24 masks. Bar = GradDot_dmean recomputed on the same ensemble
(BLOCKERS #1/#23). "Pooled paired Delta_rho" = point Spearman(estimator) - Spearman(GradDot) over
the pooled 72 G/H/I masks; per-draw signs show out-of-sample robustness. Bootstrap p one-sided.

## Pass 4 -- four new estimator classes

### W4 -- RelatIF (b14_rescoring.py), cached E=20 Gram, vs GradDot_dmean
| estimator | C1 | C5 |
|---|---|---|
| GradDot_dmean (bar) | 0.593 | 0.390 |
| relatif_lin/unitL2 (K/G_dd) | 0.424 | **0.611** |
RelatIF paired vs GradDot_dmean, C5: G +0.226, H +0.384, I +0.135, **pooled +0.210 (p=0.016)**.
C1 is a wash. (relatif_sqrt/unitL2 C5 +0.100; relatif_lin/unitL2 is the winner.)

### W3 -- TRAK exact dual (b13_trak.py), regen E=5 head-Phi, per-target lambda
| Phi | lambda_rel | C2 ratio | note |
|---|---|---|---|
| regen E=5 head | 0.1 | 0.68 | inverting the head Gram |
TRAK-head/C2 paired vs GradDot_head: **pooled +0.222 (p=0.016)**. At lambda->inf TRAK == GradDot
(A4 identity), so per-target lambda is essential; a C5-only selection misses this.

### W2 -- exact frozen-trunk head surrogate-LOO (b12_headloo.py), regen E=5
Freeze trunk, ridge head W=(PhiT Phi + lam I)^-1 PhiT A on 512-D features, exact LOO by downdating
the 512x512 normal equations, score_d = Delta held-out L2. lambda_rel=0.001.
Paired vs GradDot_head: C5 pooled **+0.222 (p=0.044)** (G+0.18,H+0.36,I+0.22); C1 +0.17 ns, C2 +0.01
ns -- so exact LOO beats GradDot on C5 (real leave-one-out structure) but not C1/C2 (linearization
exonerated there).

### W1/W5/W6 negatives
- W1 unlearning-LOO: 1-member pilot, ascent ~= GradDot (C1 -0.04, C5 +0.06), finetune-forget dead. Killed.
- W5 mid-training GradDot: 20-ckpt grid, adjacent-ckpt LDS swing up to 1.03 -> ckpt noise. Closed.
- W6 hybrid datamodel: win-cond-2 fails (no hybrid@12 reaches plain datamodel@24 on C2 gate 0.324 /
  C5 gate 0.560). Secondary: the gradient prior regularizes the datamodel at low K (kernel-ridge@12
  beats plain-lasso@12 by +0.43 on C5, +0.22 on C2); GradDot-alone dominates the datamodel to K~16-20.

## Pass 5 -- the unified leverage family

### P1 (p1_leverage_family.py) -- family = diag(G)^-beta (G+lam I)^-1 K, unit-L2 then mean
Per-target best pooled paired Delta_rho vs GradDot_dmean, sign-consistent (beats GradDot on ALL of
G/H/I):
| target | best config | pooled Delta | sign-consistent |
|---|---|---|---|
| C8 | regenE5-head/dmean/lam0.3/beta1 | +0.387 | yes |
| C7 | cachedE20-full/dmean/lam1/beta1 | +0.316 | yes |
| C2 | regenE5-head/dmean/lam0.3/beta1 | +0.276 | yes |
| C5 | cachedE20-full/dmean/lam-inf/beta0.5 | +0.243 | yes |
| C4 | (weak) | +0.146 | yes |
| C1,C3,C6,C9 | - | - | no |
Best SINGLE config vs GradDot_dmean: regenE5-head/dmean/lam0.3/beta1 -> C2,C8 (2 targets). No config
reaches 3 vs the strict bar; the C2,C3,C8 "generality" held only vs the unit-L2 baseline (#23).

### P4 (p4_why.py) -- predictors of leverage responsiveness (n=9 targets, descriptive)
Spearman(predictor, best leverage Delta): datamodel_lds **+0.66**, contam_cached +0.45,
contam_head -0.18, ceiling -0.10, outcome_var -0.05. Leverage correction tracks the datamodel but
also wins where the datamodel collapses (C7,C8). diag(G) dispersion CV ~1.2.

### P2 (p2_ensemble.py) -- complementarity + ensemble
Rank corr over 135 demos: C5 RelatIF vs surrogate-LOO **+0.20** (complementary); C2 TRAK vs leverage
+0.81 (same). Ensemble (z-avg) pooled vs GradDot_dmean: **C5 +0.298 (p=0.0018)** (> either
component), C2 +0.289 (p=0.0010).

### P6.1 (p6_multiview.py) -- multi-view fixed estimator, vs MAX(GradDot cached, GradDot head)
| MV variant | qualifying targets (sign-consistent + pooled Delta>=+0.10 vs max-bar) |
|---|---|
| MV-A (z-avg of 3 views) | C7 only (1) |
| MV-B (rank-avg) | none |
| MV-C (z-avg of 2 leverage views) | none |
Averaging across target-specific views dilutes; a fixed single estimator does not generalize.

## Campaign J confirmation (pass 4, seed 20260723, PREREG_J, scored once)
| hypothesis | lds | GradDot | ceiling | ratio | abs pass | paired Delta | paired p |
|---|---|---|---|---|---|---|---|
| J1 RelatIF/C5 | 0.516 | 0.175 | 0.968 | **0.533** | **YES** (p=0.005) | +0.341 | 0.073 |
| J3 surrogate-LOO/C5 | 0.464 | 0.124 | 0.968 | 0.479 | no (0.479<0.5) | +0.340 | 0.128 |
| J2 TRAK-head/C2 | -0.301 | -0.337 | 0.902 | -0.334 | no | +0.036 | 0.47 |
RelatIF/C5 clears the absolute half-ceiling bar out of sample (first gradient estimator to do so);
both C5 estimators beat GradDot by +0.34, but the paired p misses 0.05 at n=24. C2 unresolvable on
the J draw (GradDot itself -0.34).

## Campaign K confirmation (pass 5, seed 20260724, PREREG_K, scored once)
| hypothesis | lds | GradDot | ceiling | ratio | abs pass | paired Delta | paired p |
|---|---|---|---|---|---|---|---|
| K1 leverage-head/C2 | -0.166 | -0.162 | 0.938 | -0.177 | no | -0.004 | 0.50 |
| K2 leverage-head/C8 | 0.420 | 0.471 | 0.927 | 0.453 | no | -0.051 | 0.61 |
| K3 leverage-cachedE20/C7 | 0.416 | 0.130 | 0.970 | 0.429 | no | +0.286 | 0.195 |
0/3 confirm. C2 unresolvable (both negative); C8 dev win did not replicate (GradDot strong on K);
C7 keeps the right direction (+0.29 over GradDot) but misses both bars at n=24. The pass-5 family
dev wins were largely mask-draw overfitting; only C5 (campaign J) survived a fresh draw.

## Campaign L confirmation (pass 6 C5 capstone, seed 20260725, PREREG_L, scored once)
| hypothesis | lds | GradDot | ceiling | ratio | paired Delta | paired p |
|---|---|---|---|---|---|---|
| L1 C5 ensemble | 0.217 | 0.143 | 0.958 | 0.227 | +0.075 | 0.36 |
| L2 RelatIF/C5 | 0.217 | 0.143 | 0.958 | 0.227 | +0.075 | 0.38 |
| L3 surrogate-LOO/C5 | 0.127 | 0.212 | 0.958 | 0.133 | -0.085 | 0.63 |
0/3. Same estimator as campaign J (ratio 0.533, paired +0.34) now scores ratio 0.227, paired +0.075
-- J was a favourable draw. Across J+L the C5 effect is directional but small and not robust to the
draw; the surrogate even loses to GradDot on L.

## Verdict across all six draws (G/H/I/J/K/L)
Gradient leverage/self-influence corrections beat GradDot in DIRECTION out of sample (RelatIF>=GradDot
on C5 for J and L; leverage>=GradDot on C7 for K) -- a first for the project. But no gradient
estimator robustly clears the absolute half-ceiling bar across fresh draws: the one absolute-bar pass
(RelatIF/C5, J) did not replicate on L. The datamodel remains the only estimator with a large
replicable OOS advantage. Binding limitation: n=24. Pass 7 should spend GPU on a higher-power draw.

---

# Pass 7 -- full tables

All contrasts are paired against `GradDot_dmean` on the same masks (BLOCKERS #1/#23), with 95% CIs
from a percentile mask bootstrap stratified within draw. Primary statistic Kendall tau_b, secondary
Spearman (BLOCKERS #30). "Honest" = the config's discovery draw removed (BLOCKERS #28/#31).

## 7.1 Pooled out-of-sample contrast of the frozen pass-4/5/6 configs (W0.1, zero GPU)

`results/p7_pooled_oos.csv`. Spearman, depth 10.

| config | analysis | draws | n | lds | GradDot | ceiling | ratio | Delta rho | 95% CI | p |
|---|---|---|---|---|---|---|---|---|---|---|
| RelatIF/C5 | full information | J+K+L | 72 | 0.184 | 0.100 | 0.964 | 0.190 | +0.083 | [-0.108, +0.284] | 0.202 |
| RelatIF/C5 | **honest** | K+L | 48 | 0.006 | 0.050 | 0.962 | 0.006 | **-0.044** | [-0.264, +0.185] | 0.659 |
| surrogate-LOO/C5 | full information | J+K+L | 72 | 0.121 | 0.119 | 0.964 | 0.125 | +0.002 | [-0.330, +0.308] | 0.502 |
| surrogate-LOO/C5 | **honest** | K+L | 48 | -0.082 | 0.154 | 0.962 | -0.085 | **-0.236** | [-0.628, +0.143] | 0.886 |
| C5 ensemble | full information | K+L | 48 | -0.044 | 0.050 | 0.962 | -0.046 | -0.094 | [-0.377, +0.175] | 0.753 |
| leverage/C7 | full information | J+K+L | 72 | 0.307 | 0.209 | 0.960 | 0.320 | +0.098 | [-0.199, +0.403] | 0.255 |
| leverage/C7 | **honest** | J+L | 48 | 0.268 | 0.240 | 0.950 | 0.282 | **+0.028** | [-0.338, +0.395] | 0.734 |

Per-draw for RelatIF/C5 (`results/p7_per_draw.csv`): G +0.226, H +0.384, I +0.135 (all dev);
J +0.341 (discovery); **K -0.171**; L +0.075. Between-draw heterogeneity over the OOS draws:
I^2 = 46%, Cochran Q p = 0.16 -- the draws disagree about as much as sampling alone predicts.

Pooling validity (`results/p7_pooling_validity.csv`): the between-draw variance component is
**zero to numerical precision** on every set tested (C5 and C7, J+K+L and all six), F = 0.003-0.93,
p = 0.46-1.00. The six draws are exchangeable, so raw pooling is valid and the prespecified
within-draw-rank fallback was not triggered.

## 7.2 Allocation -- the exchange rate (W0.2, zero GPU)

`results/p7_allocation.csv`, `results/p7_iso_budget.csv`. Sampling sd of the paired contrast, C5,
masks bootstrapped with replacement.

| paired sd | depth 2 | depth 3 | depth 5 | depth 8 | depth 10 |
|---|---|---|---|---|---|
| 24 masks | 0.179 | 0.187 | 0.172 | 0.182 | 0.183 |
| 48 masks | 0.127 | 0.127 | 0.121 | 0.121 | 0.129 |
| 72 masks | 0.103 | 0.108 | 0.107 | 0.097 | 0.098 |
| 96 masks | 0.089 | 0.082 | 0.082 | 0.085 | 0.082 |
| 144 masks | 0.078 | 0.073 | 0.070 | 0.066 | 0.069 |

Iso-budget, C5 (BEST per budget in bold):

| budget | split | ceiling | mean ratio | paired sd | 95% CI width | power(0.15) |
|---|---|---|---|---|---|---|
| 240 | **120 x 2** | 0.809 | 0.405 | **0.077** | **0.301** | 0.62 |
| 240 | 48 x 5 | 0.908 | 0.373 | 0.121 | 0.475 | 0.34 |
| 240 | 24 x 10 (archived) | 0.945 | 0.377 | 0.183 | 0.719 | 0.20 |
| 720 | **144 x 5** | 0.920 | 0.372 | **0.070** | **0.273** | 0.70 |
| 720 | 72 x 10 | 0.955 | 0.372 | 0.068 | 0.267 | 0.71 |

Statistic selection (`results/p7_statistic_selection.csv`), scored on reliability and noise only:
kendall_tau_b **5.32**, pearson_raw 4.61, spearman 4.52, quartile_gap 3.76, top6_overlap 3.57.
Same winner on C5 and C7.

Baseline instability (`results/p7_baseline_instability.csv`): `GradDot_dmean` C5 LDS spans
**-0.066 to +0.489** across the six draws (sd 0.217). C5 var(GradDot) 0.047, var(RelatIF) 0.097,
cov +0.052, var(paired) 0.041. C7 cov is **negative** (-0.009) so var(paired) 0.071 exceeds
var(level) 0.011.

Contrast under every candidate statistic on the honest sets (`results/p7_contrast_by_statistic.csv`):
RelatIF/C5 on K+L is negative under all six (kendall -0.051, spearman -0.044, pearson -0.046,
top6 -0.167, quartile_gap -0.247). leverage/C7 on J+L is null under all six.

## 7.3 Campaign M -- the resolving campaign (W1)

144 virgin masks (6 disjoint sub-draws) x depth 2 = 288 retrains. `PREREG_M` frozen at zero runs
(`bdab9af`), family of one (alpha_abs = 0.05), scored once. `results/confirm_mseries.csv`.

| statistic | lds | GradDot | ceiling | ratio | p_abs | PASS_abs | Delta rho | 95% CI | p | PASS_paired |
|---|---|---|---|---|---|---|---|---|---|---|
| kendall_tau_b (primary) | 0.284 | 0.199 | 0.688* | 0.413 | <1e-4 | no | **+0.085** | [-0.004, +0.171] | **0.031** | **yes** |
| spearman (secondary) | 0.413 | 0.301 | 0.844 | 0.489 | <1e-4 | no | **+0.111** | [-0.013, +0.231] | **0.041** | **yes** |

\* the Kendall ceiling reuses the archived disjoint-half construction and Spearman-Brown step,
which is derived for correlations; it is an approximation and the Spearman row is the one
comparable to project history.

Per sub-draw, Kendall (`results/p7_forest_final_kendall_tau_b.csv`): M0 +0.058, M1 +0.116,
M2 +0.145, **M3 +0.341 [+0.121, +0.550]**, M4 -0.058, M5 -0.022. **M3 is a single 24-mask draw
that reproduces campaign J's headline number exactly** -- the clearest possible demonstration that
+0.34 is within the sampling distribution of a 24-mask draw under a true effect of ~+0.06.

Pooled over every clean out-of-sample mask (`results/confirm_mseries_grandpooled.csv`,
`results/p7_matched_depth.csv`):

| pooling | statistic | n | Delta rho | 95% CI | width | p |
|---|---|---|---|---|---|---|
| K+L (pre-M) | kendall | 48 | -0.051 | [-0.197, +0.103] | 0.300 | 0.757 |
| M alone (virgin) | kendall | 144 | +0.085 | [-0.005, +0.172] | 0.177 | 0.032 |
| **K+L+M, mixed depth** | **kendall** | **192** | **+0.060** | **[-0.017, +0.136]** | **0.153** | **0.057** |
| K+L+M, mixed depth | spearman | 192 | +0.084 | [-0.027, +0.193] | 0.220 | 0.064 |
| K+L+M, matched depth 2 | kendall | 192 | +0.055 | [-0.020, +0.131] | 0.150 | 0.074 |
| K+L+M, matched depth 2 | spearman | 192 | +0.075 | [-0.034, +0.184] | 0.218 | 0.085 |

K+L alone is negative at both depths (-0.094 at depth 2, -0.051 at depth 10), so depth mixing is
not driving the pooled sign.

## 7.4 W2 duels -- killed by the pilot

6 duels, depth 4, 48 retrains. `results/p7_duels_pilot.csv`, manifest `results/p7_duel_manifest.json`.

| quantity | value |
|---|---|
| mean within-duel abs diff / seed-noise sd | **0.444** (kill threshold 1.0) |
| mean paired sd | 0.0365 |
| mean unpaired sd | 0.0872 |
| **pairing gain** | **2.39x** |
| RelatIF correct | 1 / 6 (sign test p = 0.98) -- **UNINTERPRETABLE, see BLOCKERS #35** |

Duel availability (`p7_duel_manifest.json → availability`): RelatIF vs GradDot demo-level rank
correlation **0.547**; disagreeing within-cluster pairs at gap >= {3,4,5,6,7,8,9} =
{108, 62, 37, 23, 11, 3, 1}; demo-disjoint duels = {29, 18, 12, 11, 5, 1, 1}.

## 7.5 GPU ledger

`results/gpu_ledger_pass7.csv`. W0.1 and W0.2 -- including the result that overturned passes 4-6 --
cost **zero GPU**.
