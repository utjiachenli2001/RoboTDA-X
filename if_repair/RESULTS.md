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
