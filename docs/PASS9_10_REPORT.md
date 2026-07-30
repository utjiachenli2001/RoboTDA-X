# RoboTDA-X `if_repair` — progress report, passes 9 and 10

**Jiachen Li · 2026-07-30 · repo at `cb0dfea` (github.com/utjiachenli2001/RoboTDA-X, branch `main`)**

Corpus: 135 robot imitation-learning demonstrations in 9 clusters of 15. The measurement is LDS
(linear datamodeling score) — how well an estimator's per-demo influence scores predict the actual
outcome change when a subset of demos is removed and the model is retrained. Compute: one H200,
~38 GPU-hours across these two passes (3,332 retrains).

---

## 1. Headline

**The positive result reported in pass 8 does not survive a control that pass 8's own preregistration
promised but did not apply.** Pass 8 reported that at cluster grain (removing whole 15-demo clusters)
a plain, uncorrected gradient dot-product cleared an absolute "half-ceiling" bar at ratio **0.707** —
the first estimator to clear it in seven passes. Pass 9 found that number is substantially a
**training-set-size effect**, not attribution.

After correcting for it, the current position is:

- Gradient-based attribution on this corpus is **real but weak**. It is statistically distinguishable
  from zero at every unit size tested, and reaches only **22–45% of the achievable ceiling**.
- **No unit size from 3 to 15 demos clears the usefulness bar** once training-set size is held fixed.
- The only estimator that does clear the bar is the one that **reads outcomes directly**, which
  raises a question about what the benchmark measures.
- The grain question is now **structurally unresolvable on this corpus** for combinatorial reasons.
  Progress requires a larger corpus.

---

## 2. The correction (pass 9)

Cluster masks retain 4, 5 or 6 of 9 clusters — i.e. 60, 75 or 90 training demos. Training-set size
moves the outcome directly, and every estimator's mask prediction is a **sum** over retained demos,
so it grows with the count too. Correlating the two while pooling over sizes credits both sides for
counting demos.

The diagnostic is a permutation null on a **fixed** estimator: shuffle outcomes within size stratum
and re-correlate. Nothing is fit, so any surviving correlation cannot be leakage.

| scope | n | LDS | ceiling | ratio | 95% CI | permutation null | clears bar |
|---|---|---|---|---|---|---|---|
| pooled (as published in pass 8) | 149 | 0.475 | 0.672 | **0.707** | [0.61, 0.82] | **0.353** | yes |
| within 4-of-9 (60 demos) | 56 | 0.208 | 0.505 | 0.411 | [0.09, 0.78] | 0.001 | no |
| within 5-of-9 (75 demos) | 37 | 0.258 | 0.521 | 0.496 | [0.12, 0.94] | 0.002 | no |
| within 6-of-9 (90 demos) | 56 | 0.169 | 0.532 | 0.317 | [-0.02, 0.72] | -0.002 | no |

*Kendall τ_b. The pooled row reproduces the committed pass-8 result exactly (asserted as a regression
test), so this is the same computation, not a competing one.*

**0.353 of the pooled 0.475 survives an outcome shuffle.** Within stratum the null collapses to ~0
— the control works — and the bar clears in no stratum. Pass 8's claim that the bar is reachable is
**not refuted; it is unproven.**

---

## 3. Removing the confound by design (pass 9, campaign O — preregistered)

The cluster grain cannot fix itself: its mask space is capped at 336 subsets and a prior campaign
consumed 278, so per-stratum sample sizes are stuck. Sub-cluster grain escapes the cap
combinatorially — a 75-demo mask keeps 25 of 45 groups at k=3, and C(45,25) ≈ 3×10¹². Hundreds of
masks can therefore share **exactly one training-set size**, so the confound does not exist rather
than being adjusted for.

800 fresh masks, every one at exactly 75 retained demos, preregistered before any run and scored
once. 1,600 retrains, 18.2 h, zero failures.

| unit size k | n | LDS | ceiling | ratio | 95% CI | ratio/√r | beats null | clears bar |
|---|---|---|---|---|---|---|---|---|
| 3 demos | 400 | 0.134 | 0.376 | 0.356 | [0.180, 0.550] | 0.218 | yes | **no** |
| 5 demos | 400 | 0.141 | 0.386 | 0.365 | [0.204, 0.559] | 0.227 | yes | **no** |
| 15 demos (cluster) | 37 | 0.300 | 0.451 | 0.666 | [0.194, 2.349] | 0.447 | yes | **no** |

Three things this result is **not**:

1. **Not an absence of signal.** Every rung beats its permutation null decisively (nulls ~0, 97.5th
   percentiles 0.06–0.23, against observed LDS 0.13–0.30). Attribution is real; it is weak.
2. **Not a marginal failure.** These use the project's historical `ρ/r` convention at the noisiest
   depth, which is the most flattering the design can produce. On the attainable `ρ/√r` scale the
   rungs reach **22%, 23% and 45%**.
3. **Not a located transition.** k=3 and k=5 are statistically indistinguishable.

---

## 4. Pass 10

**(a) The k=15 census — the population is exhausted.** Conditional on the target cluster, the
relevant mask population is exactly C(8,4) = 70. Completing it (132 retrains) gives a census ratio of
**0.487 — below the bar**, confirming pass 9 on the full population rather than a 37-mask sample.

Two methodological by-products:

- **No detectable winner's curse on this hypothesis.** The discovery-draw subset reads *lower*
  (0.397 vs 0.496) and its LDS is essentially identical (0.265 vs 0.258). Earlier work measured ~4×
  inflation, but that was selection over many *configurations*; here the selection was over the
  *grain*, and the estimator itself was never selected.
- **The ratio's denominator can dominate a comparison.** Two halves of the same population at the
  same depth give ceilings of 0.521 vs 0.668. LDS stayed flat (0.258 / 0.265 / 0.280) while the ratio
  swung 0.397–0.496. At these sample sizes, normalising by the ceiling *adds* noise. This may affect
  historical cross-subset ratio comparisons in the project and is queued for audit.

**(b) The datamodel clears the bar — with a caveat.** A design-based datamodel (regress observed mask
outcomes on the inclusion vector; ignore gradients entirely), scored leave-one-mask-out with
regularisation refit inside each fold, at a fixed training-set size:

| grain | n | coefficients | LDS | GradDot LDS | ratio | ratio/√r | paired Δ | 95% CI |
|---|---|---|---|---|---|---|---|---|
| k=3 | 400 | 45 | 0.393 | 0.134 | 1.044 | **0.640** | +0.259 | [0.185, 0.329] |
| k=5 | 400 | 27 | 0.419 | 0.141 | 1.084 | **0.674** | +0.278 | [0.209, 0.345] |

*Ratios above 1 are expected, not anomalous: the ceiling is a reliability r and the attainable maximum
is ~1/√r ≈ 1.6 here. Permutation control ≈ 0, so this is not leakage.*

This reaches 64–67% of attainable against the gradient estimator's 22–23%. **But it is the only
competitor that sees outcomes at all** — gradient estimators never observe an outcome — and here it
is heavily over-determined (400 observations, 45 or 27 parameters), the opposite of the
under-determined regime where it originally earned its reputation. Whether it is *attributing* or
*fitting the outcome surface* is the open question. This read is descriptive, not preregistered.

**(c) Partition robustness (preregistered).** The sub-cluster results depend on one arbitrary
partition of each cluster into groups. A second, fully independent partition (sharing zero groups),
1,600 retrains:

| grain | first partition | second partition | Δ | agrees with prereg? |
|---|---|---|---|---|
| k=3 | 0.356 | **0.200** | −0.156 | yes, by 0.020 |
| k=5 | 0.365 | 0.320 | −0.045 | yes, comfortably |

Both preregistered hypotheses pass, but that is the least informative part. **k=5 is
partition-robust; k=3 is not** — its LDS nearly halved and it stayed inside the interval only because
that interval is wide. The movement is in the LDS, not the ceiling, so it is a genuine change in
predictive performance rather than the denominator artifact from (a). **Both rungs read lower on the
second partition**, so the earlier figures were if anything optimistic.

---

## 5. What is established, and what is not

**Established**
- Attribution signal on this corpus is real and weak (22–45% of attainable) at every unit size tested.
- The published cluster-grain success was substantially a training-set-size artifact.
- The self-influence / leverage corrections developed over three earlier passes are **not salvageable**:
  keeping their ranking on a well-behaved scale still reverses, and within stratum their ordering is
  actively anti-predictive.
- k=5 sub-cluster results are partition-robust.

**Not established**
- Whether attribution improves with unit size. The point estimates rise (0.36 → 0.37 → 0.49–0.67) but
  intervals overlap, and this is now **unresolvable on this corpus**: the k=15 population is capped at
  70 masks and has been exhausted. No purchasable design tightens it below a CI width of ~0.6.
- Whether the datamodel's advantage reflects attribution or outcome-surface fitting.
- Whether the half-ceiling bar is the right standard. Nine passes have failed it; the one apparent
  success was a confound. Either the bar is mis-specified for a 135-demo corpus, or small-unit
  attribution genuinely does not work at this scale. **The evidence in hand does not discriminate
  these** — both predict exactly what is observed — so I am not claiming a resolution.

---

## 6. Next

1. **Port to a corpus of 500+ demonstrations.** This has moved from "the biggest available move" to
   "the only move" for the grain question, and it is also the only clean discriminator for the
   bar-standard question, since it varies corpus size directly.
2. **Audit historical ratio comparisons** for the ceiling-noise effect in §4(a).
3. **Resolve the datamodel question** with a design where it is not over-determined — which this
   corpus cannot supply at any grain.

## 7. Reproducibility

All results are committed with the code that produced them. Every campaign was preregistered before
any run existed and scored exactly once, enforced mechanically (the scorer refuses to overwrite, and
refuses to write at all unless every preregistered arm has usable data). Entry points:
`if_repair/HANDOFF.md` (pass-9 and pass-10 sections), `if_repair/p9_prereg.md`,
`if_repair/p10_prereg.md`, `if_repair/BLOCKERS.md` #41–44. Test suite: 151 passing.
