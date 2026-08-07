# Campaign U preregistration — the fixed-retained selection ladder

**Status: DRAFT, not frozen.** Supersedes the campaign-T draft of the same date, which the
variance pilot killed. **Revision 3**, after two rounds of adversarial review (three reviewers, then one against the revision) and 112 diagnostic retrains.

---

## 0. What happened to campaign T, and why this document exists

Campaign T was a corpus-size ladder that removed **50% of each rung**. A variance pilot of **64
retrains** (4 pools x 8 masks x depth 2), run before freezing precisely to test this, found the
design's ground truth is unmeasurable at its top rung. Convergence-gated (§3.4), it measures
(every run's outcome and training loss is committed in `results/p18_runs_raw.json`):

| pool N | across-mask signal sd | across-seed noise sd | implied ceiling r @ depth 2 |
|---|---|---|---|
| 50 | 0.02115 | 0.01322 | 0.837 |
| 100 | 0.01229 | 0.02169 | 0.391 |
| 200 | 0.00890 | 0.00866 | 0.679 |
| 370 | **0.00000** | 0.00897 | **0.000** |

At 370 demonstrations, **which** 185 you train on moves held-out loss less than the random seed
does. The signal decays monotonically (0.021 → 0.012 → 0.009 → 0) and is indistinguishable from
zero at the top. This is not the outlier: dropping the one under-converged run cuts rung-370 noise
8× (0.0745 → 0.0090) and leaves the signal at exactly zero.

Campaign T's primary statistic was `ρ/√r`. At r = 0 that is 0/0. **Campaign T does not launch.**
Its one-shot score is unconsumed.

**A 48-retrain probe then located the cause, and it is not the pool size.** Holding the pool at 370
and varying only the RETAINED count:

| pool | retained | removed | signal sd | noise sd | r @ depth 2 |
|---|---|---|---|---|---|
| 370 | 25 | 93% | 0.04162 | 0.02693 | **0.827** |
| 370 | 50 | 86% | 0.02389 | 0.01189 | **0.890** |
| 370 | 100 | 73% | 0.01138 | 0.00832 | **0.789** |
| 370 | 185 | 50% | 0.00000 | 0.00897 | **0.000** |

**Retained count governs measurability, not pool size.** Campaign T did not die because 370 demos
is too many; it died because training on 185 puts the model past the point where which demos it
received still matters. Three independent cells pass, none near the bar.

Better still, the signal at retained-25 is **0.04162 at pool 370 against 0.02115 at pool 50 — a
1.97x rise**, against 1.35x predicted by the finite-population factor 25(N-25)/(N-1). Measurability
does not merely survive up this ladder, it improves. That is the property campaign U rests on, and
it is measured rather than assumed.

Three further faults the audit found, all of which this revision must fix rather than inherit:

**(a) `ρ/√r` does not normalize for r, and r varied along the regression axis by construction.**
The `√r` disattenuation is a *Pearson* identity; the statistic is *Kendall*. With
`τ = (2/π)·arcsin(g√r)`, the ratio drifts mechanically as r falls. Verified numerically:

| latent estimator strength g | mechanical drift over r ∈ [0.837, 0.01] |
|---|---|
| 1.0 | −0.058 per doubling |
| 0.7 | −0.013 per doubling |
| 0.5 | −0.004 per doubling |
| 0.3 | −0.001 per doubling |

The artifact is **largest for the strongest estimator**, so it preferentially manufactures the
"the datamodel's edge was a small-corpus artifact" verdict. Separately, Jensen inflation of
`1/√r̂` runs the other way at small r (+9% at r = 0.10, +38% at r = 0.05), so the two artifacts
have opposite signs and both track N. This is BLOCKERS #42 in a new coordinate: *two ratios at
different r are not comparable.*

**(b) The primary test had ~16% power against a doubling of the gradient ratio, and the decision
rule labelled the resulting non-rejection "the strong result".** A design that emits its most
quotable claim regardless of truth. This is the shape of the four claims already retracted.

**(c) Fixed `total_steps = 8000` confounds corpus size with convergence.** Epochs-per-demo vary
~20× along any ladder. A training-loss gate flags 1–3 runs in 16 at *every* rung. There was no
exclusion rule of record.

---

## 1. The question, unchanged

`WHAT_STANDS.md` §3: **is the gradient failure a property of the corpus size or of the approach?**

## 2. The design change, and why it is not merely a smaller version of T

**Campaign U holds the RETAINED count fixed at 25 demonstrations and grows the candidate POOL.**

| | campaign T (dead) | campaign U |
|---|---|---|
| what varies | pool N, removal fixed at 50% | pool N, retained fixed at 25 |
| training-set size | 25 / 50 / 100 / 185 | **25 at every pool** |
| ceiling r | 0.84 → 0.00 (dies) | 0.83 measured at the far end (§0) |
| estimand | ranking half-corpus subsets | **ranking fixed-size selections** |

Three reasons this is the right design rather than the surviving one:

1. **Every retrain sits at the operating point the pilot certified healthy.** Retained-25 *is*
   rung 50's mask regime, where r = 0.837. Measurability is guaranteed by construction, not hoped
   for. This is the §7 gate.
2. **It dissolves fault (a) rather than patching it.** The mechanical drift in `ρ/√r` is a
   function of r; a design whose r does not collapse along the regression axis gives it nothing to
   move along. r is **not** expected to be exactly constant — the finite-population factor
   25(N−25)/(N−1) predicts a benign *rise* of ~1.35× in signal sd, and §0 measures 1.97×. §3.3
   therefore tests r against an **equivalence band**, not a point null: an expected drift must not
   be able to fire a check designed to catch collapse.
3. **It is the estimand that matches what TDA is for.** `WHAT_STANDS` §1 already states that any
   per-demonstration use is selection or pruning. R5 was always a fixed-count selection test.
   "Does attribution improve with more data" in the deployable sense means "with a larger pool to
   select from" — which is exactly what this varies.

**Rejected: fixed removed-count** (always remove 25). One reviewer proposed it; a second refuted
it on arithmetic and is correct. If removing 185 of 370 moves the outcome less than seed noise,
removing 25 of 370 moves it less still. It is the fastest-saturating option of the three.

### 2.1 Design of record

| element | value |
|---|---|
| corpus | `libero_goal`; the old campaign's 25 demos excluded (#28, #31) |
| pools | N ∈ {50, 100, 200, 370}, **nested** |
| retained | **exactly 25 demos = 5 groups of 5, at every pool** |
| eval bank | 100 held-out demos, 10/task, disjoint from every pool |
| outcome | `plain_loss` — mean per-frame l2 over the eval bank. Primary and sole. |
| grain | groups of 5, each spanning 5 distinct tasks |
| partitions | 2 per pool, sharing zero groups |
| depth | **2 at every pool**, seeds (4401, 4402); reserves 4403-4406 (§3.4) |
| conditioning | every mask must cover all 10 tasks (§2.2) |

**Depth 2, and this reverses revision 2 — the reason is a forced choice, stated plainly.**
Revision 2 specified depth 4 for robustness against divergent runs. Depth 4 and constant
masks-per-coefficient (§6) each cost a 2x doubling, and the budget affords one. Constant
masks-per-coefficient wins because the confound it removes is **undetectable by any preregistered
instrument** (§6), whereas depth 4's purpose is served directly by the convergence gate and reserve
seeds (§3.4). An undetectable bias is worse than a detectable noise source.

Depth 2 is also now adequate on measured grounds rather than hoped ones: the probe puts r at
0.79-0.89 at campaign U's actual operating point, against campaign T's 0.00-0.84. A
well-determined ceiling does not need depth 4. Depth is identical at every pool, as #42 requires,
so absolute figures are depth-2 figures and are NOT comparable to `WHAT_STANDS` §1's depth-4 numbers.

### 2.2 The conditioning rule, preregistered

Five groups of 5 drawn from a pool need not cover all ten tasks. Task coverage moves the outcome
*and* moves every estimator's summed prediction — **BLOCKERS #41 exactly**, the nuisance axis that
credits both sides. It is also plausibly a large part of rung 50's measured signal.

**Rule of record: every campaign-U mask must contain at least one demonstration of each of the 10
tasks.** Enforced by rejection sampling at construction; the rejection rate is reported. At pool 50
this is checked against the complete enumeration rather than sampled.

## 3. Statistics — every change here is a fix to an audited fault

### 3.1 The ceiling is a split-half KENDALL, reported without a Pearson correction

Numerator and denominator must attenuate identically, so the ceiling is the split-half **Kendall τ**
between the outcome rankings of the two seed halves (1 vs 1 at depth 2). **No Spearman–Brown
correction is applied** — Spearman–Brown is itself a Pearson identity, and applying it to a Kendall
statistic would reintroduce fault (a) in the denominator, which round 2 caught. The consequence is
stated rather than corrected away: the reported ceiling is the *depth-1 split-half*, it understates
the depth-2 reliability, and it is compared only against other depth-2 ceilings in this campaign.

### 3.2 The primary is reported as a pair, and the ratio is secondary

Per pool, report **(τ, r)** as a bivariate primary. Test the **τ trend** and the **r trend**
separately. No quantity with an estimated denominator is regressed. `ρ/√r` is reported for
continuity with campaigns O/R and carries no alpha.

### 3.3 The r check is an EQUIVALENCE BAND, and the expected drift is stated in advance

Round 2 caught that a point null here is the same error as version 1's "flat = strong result",
mirrored: r is *expected* to drift upward (finite population, §2 rationale 2), so a powered point
null fires on benign drift and destroys the campaign, while an unpowered one is decoration.

**Band of record: r ∈ [0.65, 0.95] at every pool.** Justification, not taste: propagating that
range through the arcsine table of §0(a) moves τ by ≈0.005 per doubling at g = 0.5 — an order of
magnitude below the §3.5 margin, so drift inside the band cannot masquerade as a τ trend. Measured
values so far (0.827 / 0.890 / 0.789 at pool 370, 0.837 at pool 50) sit inside it.

**Consequence if a pool falls outside the band:** that pool's operating point was not held fixed;
it is reported and **excluded from the slope fit**, and the fit is refit on the remainder with the
exclusion stated. If two or more fall outside, the τ trend is not interpretable and nothing is
claimed.

### 3.4 Convergence gate — outcome-blind, with reserve seeds

Round 2 caught that revision 2's rule breaks under retained-25. Under campaign T every mask trained
on a 50% sample, so training-loss spread across masks was small and "far from the pool median" meant
"bad seed". Under campaign U, **which** 25 demos a mask holds is the dominant driver of its training
loss (two pool-370 masks share ~1.7 demos on average). A pool-median rule would flag masks whose
*content* is hard to fit, the reserve-seed re-run would reproduce the same training loss because it
is a property of the mask, and the eventual censoring would correlate with the outcome — biasing the
across-mask signal itself, which is the quantity being measured.

**Rule of record: the flag is computed WITHIN mask, across that mask's seeds.** A run is flagged if
its final training loss deviates from *its own mask's* median by more than 3×MAD of the pooled
within-mask deviations. This is a seed property by construction and cannot flag a hard mask.
Training loss is not the outcome, so the rule stays outcome-blind.

Disposition: a flagged run is **re-run at the next reserve seed** (4403, 4404, 4405, 4406, in
order), not dropped. A re-run is **re-tested** under the same rule. If all four reserves are
exhausted for one mask, that mask is reported and dropped whole (both seeds), and the count is
reported per pool. The within-mask median/MAD is **not** recomputed after replacement — the
threshold is fixed from the first complete pass, so the rule cannot chase its own tail. Budget
assumes a ~6% re-run rate.

### 3.5 Flatness is an equivalence claim, tested as one

The headline "gradient attribution does not improve with pool size" is the **null**. Claiming it
from a failure to reject is fault (b).

**Margin of record: Δ = 0.066 per doubling.** Round 2 caught that revision 2's stated derivation did
not produce its stated number. Derived properly: carrying the gradient arm from ~0.15 to the 0.5
usefulness bar by N = 2,000 requires +0.35 over log₂(2000/50) = 5.32 doublings from the bottom of
the ladder, i.e. **0.066**. (From the top of the ladder the same target gives 0.144; the bottom-of-
ladder figure is the conservative choice and is the one of record.) Flatness is claimed **only by TOST**: the 95% CI on the τ slope must lie entirely
inside (−Δ, +Δ). If the CI contains zero *and* extends beyond Δ, the preregistered verdict is
**"uninformative"** — not "the strong result".

### 3.6 Power, stated in advance

§6's allocation must achieve MDE ≤ Δ at 80% power, or the campaign is not worth running. The
achieved MDE is computed from the §7 gate's measured variances **before launch** and recorded here.
If MDE > Δ, the preregistered response is to reallocate or to report only the arms that are
powered — decided before data, not after.

### 3.7 Bootstrap

Round 2 caught that revision 2's unit does not exist. **There are no complementary pairs in this
design**: the complement of "retain 25 of 370" is "retain 345", which is not a campaign-U mask. That
machinery was inherited from campaign T without re-derivation, and 74 groups do not divide into
5-group rounds anyway (74 = 14×5 + 4).

**Construction of record:** masks are drawn **i.i.d. uniformly** over 5-group subsets of the pool's
groups, subject to §2.2's coverage condition, with rejection of duplicate signatures. Inclusion
balance is therefore *statistical*, not exact — the expected count per group is equal by symmetry —
and the realised spread is reported per pool rather than asserted to be zero. This is a real loss
against campaign T's exact balance and is accepted because exact balance is unobtainable when the
retained set is a small fraction of the pool.

**Resampling unit: the mask**, which i.i.d. construction makes exchangeable — the property
complementary pairing destroyed and this construction restores. No pool is enumerated (§6), so the
mask bootstrap applies uniformly at all four pools with no special case. Seed-draw uncertainty is
**not** captured by it; the CI is conditional on the seed pair and §8 says so.

Slope fits use **inverse-variance weights** across pools, with each pool's variance taken from **its
own bootstrap**, not from the §7 gate — the gate measures pool 370 only and cannot weight the rest.

### 3.8 The #41 permutation null, which costs zero GPU

A fixed estimator, outcomes shuffled within pool, run through the **entire** pipeline including the
ceiling, the ratio and the slope. `WHAT_STANDS` §4 lesson 4 names this as the detector that caught
the last instance of the nuisance-axis failure. If the null pipeline produces a slope, the pipeline
manufactures slopes.

## 4. Primary hypotheses

> **H1 (gradient, full-pool surrogate).** The slope of split-half τ on log₂N is zero, pools 50–370.
> **H1f (gradient, FIXED surrogate).** The same, with scores from a **50-demo** model at every pool.
> **H2 (datamodel).** The same, for the design-based datamodel, **fit on one partition and scored on
> the other** (out-of-partition).

**Why H1f is co-primary and not a footnote.** Gradient scores are a local linearization at the model
that produced them, so scoring 25-demo retrains from a full-pool model is an extrapolation *whose
distance grows with the pool*, while the datamodel is fit on the retained-25 masks and extrapolates
nowhere. That biases H1's slope negative as N grows — toward the quotable headline — and it is a
different mechanism from the one §6 fixes, so both arms would otherwise carry negative-slope
confounds of independent origin. Revision 2 mitigated this with one descriptive point at a single
pool, which cannot estimate a trend component. **H1f holds the surrogate distance fixed across all
four pools**, so the difference H1 − H1f isolates the extrapolation effect. It costs only scoring,
not retrains, because it reuses the same outcome table.

**Why H2 is explicitly out-of-partition.** `WHAT_STANDS` #50/#53 established that within-partition
datamodel figures are inflated and only cross-partition transfer counts; revision 2 named "the
design-based datamodel" without saying which, leaving load-bearing work unstated. Budget the ~20%
transfer penalty §5 of `WHAT_STANDS` reports.

Family size 3, α = 0.05, Bonferroni → 0.0167. Unit of the fit is the **pool** (partition-averaged,
4 points), because the two partitions re-group the same demos and are correlated.

## 5. Decision rule

| observed | reading of record |
|---|---|
| τ slope CI excludes 0, positive | pool size relieves the failure; report extrapolated crossing N with CI |
| CI inside (−Δ, +Δ) by TOST | **flat**: attribution does not improve with pool size over this range |
| CI contains 0 and extends beyond Δ | **uninformative** — underpowered, not a finding |
| r trend CI excludes 0 (§3.3) | operating point not held fixed; τ trend not interpretable |
| permutation null shows a slope (§3.8) | pipeline artifact; nothing is reported |
| τ not monotone in N (bootstrap P(correctly ordered) < 0.5) | **reported as non-monotone.** No slope is fit through it, and no monotone sub-range is selected. Restores the branch the parent plan required and revision 2 dropped; the ordering call is bootstrap-based, not read off point estimates (#47). |

## 6. Budget — pending the §7 gate

**Masks scale with the coefficient count, at exactly 10 per coefficient at every pool.** This is
the fix for the fault round 2 showed no preregistered instrument can detect: at a fixed mask count,
masks-per-coefficient would fall 24 → 2 along the ladder, starving the datamodel fit progressively
and biasing H2 toward "the datamodel's edge does not scale" — and the permutation null **cannot**
see it, because under shuffled outcomes there is no real signal to attenuate differentially. Holding
the ratio constant removes the confound by construction, which is the only available remedy.

| pool | groups | masks / partition | 
|---|---|---|
| 50 | 10 | 100 |
| 100 | 20 | 200 |
| 200 | 40 | 400 |
| 370 | 74 | 740 |

**No pool is enumerated.** Revision 2 called pool 50 "the complete space C(10,5) = 252"; round 2
showed §2.2's coverage rule removes exactly 10 of those, so the conditioned space is 242 and the
claim was false. Sampling 100 conditioned masks uniformly at pool 50 also makes the construction
identical at all four pools, which the enumeration special case broke.

**Cross-pool signature collisions are now possible** — pools nest and every mask is a 25-demo set —
so `jobs()` dedupes by signature and shares those retrains; `cross_rung_shared_signatures` flips
from "must be empty" to "shared, deduped".

**Budget: 1,440 masks/partition × 2 partitions × depth 2 = 5,760 retrains ≈ 67.8 h**, plus ~6%
re-runs ≈ **71.8 h**, plus the §7 gate and the fixed-surrogate arm. (Arithmetic shown because this
project's `WHAT_STANDS` records four prior slips of exactly this kind, and a fifth was caught in
this document before it shipped: 1,440 × 2 × 2, not 1,440 × 2.)

**This is the same budget campaign T carried, spent differently.** Depth 4 was traded away (§2.1) to
buy the constant masks-per-coefficient this table provides. **Final counts are contingent on §3.6's
power requirement**, computed before launch.

## 7. The viability gate — PASSED, and the criterion restated consistently

Round 2 caught that revision 2 stated the bar two ways that differ by 40%: under the pipeline's own
depth-2 convention `r = s²_mask/(s²_mask + s²_seed/2)`, **S/N = 0.7 implies r = 0.495**, not the
0.33 also written. **Criterion of record: r ≥ 0.495 (equivalently S/N ≥ 0.7)**, one form, derived.

Round 2 also showed by Monte Carlo that an 8-mask cell decides a threshold case by coin flip
(P(pass) ≈ 0.50 at the bar; ≈ 0.19 false-pass at a truly dead cell). That criticism is correct in
general and does not bind here, for two reasons recorded before campaign U launches:

1. **Nothing observed is near the bar.** Measured r = 0.827 / 0.890 / 0.789 against a bar of 0.495.
   Round 2's own Monte Carlo puts P(pass) at 0.79 for a *threshold-adjacent* true value of 1.17 S/N;
   the observed cells are above that.
2. **Three independent cells pass**, not one. A 19% per-cell false-pass rate gives ≈ 0.7% for three.

**The gate is therefore recorded as passed on 48 retrains** (`results/p18_probe.json`,
`results/p18_runs_raw.json`), and campaign U is cleared to build. The probe's masks were drawn
*without* §2.2's coverage conditioning; §8.6 records the residual mismatch this leaves.

**Had it failed**, the preregistered fallback stood: the saturation boundary itself becomes the
result — a ceiling map with CIs on r and on the crossing point N*, reported as a property of
subset-removal evaluation on robot imitation data rather than of any estimator.

## 8. Known limits, recorded before the run

1. **One nested corpus draw, one suite.** The claim is conditional on both. Not "any scale" —
   "any pool size reachable on standard LIBERO-scale corpora".
2. **2.9 octaves.** 50 → 370 cannot rule out a turn-on at 5,000 demonstrations.
3. **Fixed `total_steps`.** A defensible choice (fixed compute budget is the realistic regime) but
   a choice, plausibly implicated in both the under-convergence and the saturation. Stated in the
   limitations before a reviewer finds it.
4. **The extrapolation asymmetry** is now handled by the co-primary H1f arm (§4), not by a single
   descriptive point. What remains is that H1f's fixed 50-demo surrogate is itself a choice.
5. **Masks-per-coefficient is now constant by construction** (§6). The prior revision cited §3.8's
   permutation null as the check; round 2 showed the null **cannot** detect this class of fault, so
   the design removes it instead of testing for it. The null is retained for what it can detect.
6. **The gate's masks were unconditioned; the campaign's are conditioned** (§2.2). Task coverage is
   plausibly part of the measured signal, so the campaign's r may sit below the gate's 0.79–0.89.
   The §3.3 band's lower edge (0.65) is where that stops being tolerable, and the first complete
   pool is checked against it before the rest are launched.
7. **The estimand is narrower than the question, and the paper must say so unconditionally.** The
   question of record is whether the *gradient failure on a full corpus* is a size or an approach
   property. Campaign U never trains an evaluation model on more than 25 demonstrations; what grows
   is the candidate pool. A flat result supports *"gradient attribution does not improve at
   fixed-size selection as the pool grows"* — **not** *"the failure is not a corpus-size artifact"*.
   The claim is scoped to **selection**, unconditionally, not contingent on any arm being run.
8. **The other escape was changing the outcome, not the question.** The pilot proved the original
   estimand unmeasurable *in held-out l2 at fixed 8,000 steps*; Stage C's rollout success moves
   48% → 94% over this same range and does not saturate until the top. Rollout was rejected on cost
   (11–20× per retrain) and on binomial saturation at the top pool, **and that rejection is recorded
   here rather than left implicit**, because a reviewer will ask why the question moved instead of
   the metric.
9. **R3 (agreement-vs-N) and R5 (pruning) are DE-SCOPED from this preregistration**, not silently
   dropped. R3's per-demo scores are group coefficients copied to members, so it measures group-level
   agreement with a per-demo label; it is reported descriptively at group level with that caveat, and
   carries no alpha. R5 gets its own preregistration after campaign U scores, since its arms depend
   on the scores campaign U produces.

## 9. Scored once

`confirm_useries.py` writes once, refuses to overwrite, and refuses to write at all unless every
arm in §4 has usable data.
