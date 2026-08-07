# Campaign U preregistration — the fixed-retained selection ladder

**Status: FROZEN.** Supersedes the campaign-T draft of the same date, which the
variance pilot killed. **Revision 6 — FROZEN**, after two rounds of adversarial review (three reviewers, then one against the revision) and ~350 diagnostic retrains, all committed to `results/p18_runs_raw.json`. Target box: **h100-2** (8x H100, idle).

---

## 0. What happened to campaign T, and why this document exists

Campaign T was a corpus-size ladder that removed **50% of each rung**. A variance pilot of **64
retrains** (4 pools x 8 masks x depth 2), run before freezing precisely to test this, found the
design's ground truth is unmeasurable at its top rung. It measures:

**Provenance, and a correction round 5 forced.** Revisions 4–5 published a "convergence-gated"
version of this table (pool 50: r = 0.837). Round 5 showed that number is **irreproducible under any
stated or consistent rule** — it emerges only by dropping one mask, which happens to be the
leave-one-out *maximum* of the whole landscape (range 0.389–0.837), excluded in the
*over*-converged direction while runs at 3.3 and 3.1 MAD were kept. That is the fault family
`WHAT_STANDS` §4 exists to record, committed by this document. **The gated table is withdrawn.**

The table below is the **ungated** read, which recomputes exactly from
`results/p18_runs_raw.json`. The §3.4 convergence gate is a **campaign-time** rule with a
preregistered threshold; it is deliberately *not* applied to these diagnostic tables, because a
post-hoc diagnostic gate with no rule of record is what went wrong. Nothing turns on the change:
r = 0.733 clears the §7 bar of 0.495 exactly as 0.837 did, and it contradicts the pool-50 n=8 read
of 0.000 just as sharply.

| pool N | across-mask signal sd | across-seed noise sd | implied ceiling r @ depth 2 |
|---|---|---|---|
| 50 | 0.01842 | 0.01572 | 0.733 |
| 100 | 0.01229 | 0.02169 | 0.391 |
| 200 | 0.00767 | 0.00685 | 0.715 |
| 370 | **0.00000** | 0.07448 | **0.000** |

(Rung 370's raw noise is inflated ~8× by one under-converged run; dropping it leaves noise 0.00897
and the signal still **exactly zero**, so the conclusion is the outlier's independent of it. That
single-run sensitivity is *why* §3.4 exists as a preregistered campaign-time rule.)

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

**Band of record: r ∈ [0.65, 0.95] at every pool.** Round 4 caught that revision 3's justification
arithmetic was wrong by 4×: propagating the band edges at g = 0.5 moves τ by ≈0.060 end-to-end,
i.e. ≈0.021 per doubling if drifted across the whole ladder — a third of Δ_τ, not "an order of
magnitude below" it. **The band alone is therefore not the protection.** What protects the design is
the separate r-trend veto (§3.2, §5): an *undetected* r drift is bounded by that test's own
resolution, and dτ/dr ≈ 0.196 at the operating point turns it into a residual τ artifact of
≈0.005/doubling — which is where the mis-stated figure came from. Both numbers are now shown.

All four measured values (0.872 / 0.866 / 0.805 / 0.920, §7) sit inside the band with no trend.

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

Round 3 then caught that revision 3's within-mask rule is **undefined at depth 2**: with two seeds
the mask's median is their midpoint, so both runs deviate from it identically and "re-run the
flagged run" has no referent. That rule was written for depth 4 and the depth reversal (§2.1) broke
it. Restated at the level the design actually has:

**Rule of record — the unit is the SEED PAIR.** A mask is flagged if the absolute difference of its
two runs' final training losses, |Δ|, exceeds `median(|Δ|) + 3 × MAD(|Δ|)` computed **within its own
pool** (pool
scope is required: training-loss scale differs ~1.6× across pools). |Δ| is a seed property at fixed
mask content, so the rule cannot flag a mask merely for being hard to fit — which is the bias
round 3 identified, and which would have censored in a way correlated with the outcome, distorting
the across-mask signal that is the measured quantity.

**Disposition:** a flagged mask has **both seeds re-run at the next reserve pair** — (4403, 4404),
then (4405, 4406). A replacement pair is **re-tested** under the same rule. If both reserve pairs
are exhausted, the mask is reported and dropped whole. The per-pool MAD threshold is **fixed from
the first complete pass** and never recomputed, so the rule cannot chase its own tail. Budget
assumes a ~6% re-run rate at the pair level.

**Why the median term is not optional.** Round 4 caught that revision 4 wrote the threshold as
`3 × MAD(|Δ|)` with the median dropped. Under Gaussian seed noise |Δ| is **half-normal**, for which
median = 0.674 σ_Δ and MAD = 0.399 σ_Δ. Verified by simulation (4×10⁶ draws):

| threshold | value | flag rate under pure noise |
|---|---|---|
| `3 × MAD` (as mis-written) | 1.197 σ_Δ | **23.1%** |
| `median + 3 × MAD` (rule of record) | 1.871 σ_Δ | **6.1%** |

The mis-written form would have quadrupled the re-run budget *and* put a quarter of masks through
selection on small |Δ|, truncating the seed-noise distribution and inflating the ceiling r — the
denominator of everything. The rule of record reproduces its own budgeted rate, which is the check
that caught it.

### 3.5 Flatness is an equivalence claim, tested as one

The headline "gradient attribution does not improve with pool size" is the **null**. Claiming it
from a failure to reject is fault (b).

**Margin of record: Δ_τ = 0.054 per doubling, in RAW KENDALL τ.** Two corrections, in order.

Round 2 caught that revision 2's derivation did not produce its number. Derived properly: carrying
the gradient arm from ~0.15 to the 0.5 usefulness bar by N = 2,000 needs +0.35 over
log₂(2000/50) = 5.32 doublings, i.e. **0.066 per doubling** — in *fraction-of-attainable* units.

Round 3 then caught that **those are the wrong units**: §3.2 made raw τ the tested statistic, and a
fraction-of-attainable margin applied to raw τ would certify as "flat" a gradient arm improving
fast enough to cross the usefulness bar by N ≈ 500. **Unit convention, declared:** the fraction is `τ / τ_max`, where `τ_max = (2/π)·arcsin(√r)` is the
attainable Kendall τ for a perfect latent ranker at reliability r. Under that convention the
conversion `Δ_τ = Δ_fraction × τ_max` is exact. (Round 4 noted the frozen margin spans ±25% across
undeclared conventions — latent-g units give 0.043, the historical τ/√r convention 0.063. Declaring
it is what makes the number auditable.)

**Conversion factor, preregistered:** `C = (2/π)·arcsin(√r̄₂)`, r̄₂ the **mean depth-2 ceiling across
ALL FOUR pools** — as §3.5 has always defined it. Revision 4 froze C from the pool-370 cell alone
(r = 0.920), which round 4 correctly called out: that is one 8-mask cell, the conditioned-vs-
unconditioned gap it rested on is not significant at n = 8, and the freeze picked the luckier
sibling **in the anti-conservative direction** (larger C → wider band → easier "flat"). That is
`WHAT_STANDS` §4 lesson 1 — a normalising constant inheriting a small draw's luck — in a new
coordinate.

The four-pool conditioned gate (§7), **every cell at n = 30**, supplies all four:
r = **0.872 / 0.849 / 0.871 / 0.833**, so **r̄₂ = 0.85625**, **C = 0.75244**, and

> **Δ_τ = 0.066 × 0.75244 = 0.04966 per doubling.**

All four cells were deepened from n = 8 to n = 30 on round 5's recommendation — at n = 8 a single
cell's r̂ has SD ≈ 0.18 against ≈ 0.06 at n = 30, and the margin inherited that. The deepening cost
15 minutes on h100-2 and halved the margin's own sampling uncertainty.

The single-cell freeze was **7.5% anti-conservative**. C is fixed here, before launch, and is never
recomputed from campaign data. Flatness is claimed **only by TOST**: the **(1−2α) = 95.0% CI** on the τ slope (α = 0.025,
family of 2 per §4) must lie entirely inside (−Δ_τ, +Δ_τ). Round 3 caught that revision 3 wrote
"95% CI" alongside α = 0.0167, which are inconsistent; demoting H1f to descriptive (§4) makes the
family 2 and the two conventions agree at 95.0%. If the CI contains zero *and* extends beyond Δ_τ, the preregistered verdict is
**"uninformative"** — not "the strong result".

### 3.6 Power, stated in advance

Computed before launch from §6's allocation (184/368/736/1361 masks per partition, depth 2), using
the null-approximation var(τ̂) = 2(2n+5)/(9n(n−1)) and inverse-variance WLS on x = log₂(N/50):

| assumption on the two partitions | SE(slope) | MDE at 80% power, α = 0.025 | vs Δ_τ = 0.04966 |
|---|---|---|---|
| independent draws (÷√2) | 0.0104 | **0.03360** | **passes**, 47.8% headroom |
| perfectly correlated (no gain) | 0.0147 | **0.04751** | **passes**, 4.5% headroom |

**The design passes its own gate under either assumption.** The MDE factor 3.2416 = z₀.₉₇₅ + z₀.₉₀
is the exact TOST condition for 80% joint power at a true slope of zero.

Allocation history, so the number is auditable: ×1.0 gives conservative MDE 0.0648 (fails); ×1.6
gives 0.0510, which passed against the *mis-frozen* Δ_τ = 0.054 but **fails** the corrected 0.0502;
×1.84 gives 0.04751 and passes. The top-up cost **2.5 hours** on h100-2, which is why it was taken
rather than argued about.

**The 4.5% conservative headroom is thin, and is named as such** — but the bound it clears is
conservative twice over: masks are drawn i.i.d. so the pairs are i.i.d. and the null-variance
formula applies, while (a) duplicate rejection makes the draw effectively without replacement, a
large ignored finite-population correction at pool 50 (184 of a 242 space), and (b) the alternative
of interest has τ ≠ 0, where Kendall variance is strictly below its null value. And if realised SEs
do come in wider, TOST sends the result to "uninformative" (§3.5) — underpower cannot manufacture a
false "flat".

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
> **H1f (gradient, fixed surrogate) — DESCRIPTIVE, no alpha.** Scores from a 50-demo model at every
> pool.
> **H2 (datamodel).** The same, for the design-based datamodel, **fit on one partition and scored on
> the other** (out-of-partition).

**Why H1f is DESCRIPTIVE and not co-primary.** Round 3 showed it does not cleanly isolate what it
was added for: a fixed 50-demo surrogate sits *inside* the ladder, so the fraction of scored demos
it has **seen** falls 100% → 50% → 25% → 13.5% along the pools. Seen and unseen demos get
systematically different gradient scores (self- versus cross-influence — passes 4–7 of this project
are the cautionary tale), so H1 − H1f confounds extrapolation distance with a seen-fraction trend of
unknown sign. A disjoint surrogate corpus would fix it and the free pool is exhausted (475 = 100
eval + 370 ladder + 5 spare). It is retained descriptively, with the confound named. Demoting it
also returns the family to 2, which is what §3.6's power arithmetic needs.

**The asymmetry it was meant to address, restated as a limit.** Gradient scores are a local linearization at the model
that produced them, so scoring 25-demo retrains from a full-pool model is an extrapolation *whose
distance grows with the pool*, while the datamodel is fit on the retained-25 masks and extrapolates
nowhere. That biases H1's slope negative as N grows — toward the quotable headline — and it is a
different mechanism from the one §6 fixes, so both arms would otherwise carry negative-slope
confounds of independent origin. Revision 2 mitigated this with one descriptive point at a single
pool, which cannot estimate a trend component. H1f holds the surrogate distance fixed across all four pools, but does **not** cleanly isolate the
extrapolation effect (see the demotion rationale above); it costs only scoring, not retrains.

**Why H2 is explicitly out-of-partition.** `WHAT_STANDS` #50/#53 established that within-partition
datamodel figures are inflated and only cross-partition transfer counts; revision 2 named "the
design-based datamodel" without saying which, leaving load-bearing work unstated. Budget the ~20%
transfer penalty §5 of `WHAT_STANDS` reports.

Family size 2, α = 0.05, Bonferroni → 0.025. Unit of the fit is the **pool** (partition-averaged,
4 points), because the two partitions re-group the same demos and are correlated.

## 5. Decision rule

| observed | reading of record |
|---|---|
| τ slope CI excludes 0, positive | pool size relieves the failure; report extrapolated crossing N with CI |
| CI inside (−Δ, +Δ) by TOST | **flat**: attribution does not improve with pool size over this range |
| CI contains 0 and extends beyond Δ | **uninformative** — underpowered, not a finding |
| r trend CI excludes 0 (§3.3) | operating point not held fixed; τ trend not interpretable |
| permutation null shows a slope (§3.8) | pipeline artifact; nothing is reported |
| — | **Branch precedence:** permutation null → r-band (§3.3) → non-monotonicity → slope/TOST. |
| τ shows **evidence against monotonicity**: bootstrap P(some interior pool deviates from the fitted line by more than Δ_τ) ≥ 0.95 | **reported as non-monotone.** No slope is quoted, and no monotone sub-range is selected. |

## 6. Budget — pending the §7 gate

**Masks scale with the coefficient count, at exactly 18.4 per coefficient at every pool** (10 × the
×1.84 power allocation of §3.6). This addresses the fault round 2 showed no preregistered instrument
can detect: at a fixed mask count,
masks-per-coefficient would fall 24 → 2 along the ladder, starving the datamodel fit progressively
and biasing H2 toward "the datamodel's edge does not scale" — and the permutation null **cannot**
see it, because under shuffled outcomes there is no real signal to attenuate differentially. Holding
the ratio constant removes the confound by construction, which is the only available remedy.

| pool | groups G | masks / partition | H2 **fit** subsample | per-coefficient information |
|---|---|---|---|---|
| 50 | 10 | 184 | 184 | 46.0 |
| 100 | 20 | 368 | 245 | 45.9 |
| 200 | 40 | 736 | 421 | 46.0 |
| 370 | 74 | 1361 | 730 | 46.0 |

**Constant masks-per-coefficient is not sufficient, and round 3 showed why.** The quantity that
governs coefficient noise is per-coefficient *information* `n·p(1−p)` with `p = 5/G`, not `n/G`.
Under constant masks-per-coefficient it still **rises 1.87×** along the ladder (25 → 37.5 → 43.8 →
46.6), so coefficient noise *falls* with pool size, attenuating H2's τ most at pool 50 and
manufacturing a **positive** slope — the mirror image of the fault revision 3 was fixing, and
equally invisible to §3.8's null.

**Residual tilt, recorded:** for the exactly-5-ones design the precise invariant is
`n·p(G−5)/(G−1)`, so equalising `n·p(1−p)` leaves per-coefficient precision at roughly
46 → 44 → 43 → 42 across pools — a ~10% tilt in the *opposite* (slightly negative) direction,
negligible against Δ_τ. Recorded rather than corrected.

**Rule of record: H2's alpha-carrying fit uses an information-equalised subsample** of
`n = 9.2G²/(G−5)` masks (the table's fourth column), holding `n·p(1−p) = 46` at every pool, and
**scores on all** masks. Zero GPU cost — it is a subsample of runs the campaign already makes.

**No pool is enumerated.** Revision 2 called pool 50 "the complete space C(10,5) = 252"; round 2
showed §2.2's coverage rule removes exactly 10 of those, so the conditioned space is 242 and the
claim was false. Sampling 184 conditioned masks uniformly at pool 50 also makes the construction
identical at all four pools, which the enumeration special case broke.

**Cross-pool signature collisions are now possible** — pools nest and every mask is a 25-demo set —
so `jobs()` dedupes by signature and shares those retrains; `cross_rung_shared_signatures` flips
from "must be empty" to "shared, deduped".

**Budget: (184+368+736+1361) = 2,649 masks/partition × 2 partitions × depth 2 = 10,596 retrains.**
Round 4 caught that this paragraph still carried the ×1.0 figure (1,440/partition, 5,760 retrains)
while §6's own table and §3.6's power arithmetic had moved to ×1.6 — the *sixth* instance of the
arithmetic-provenance slip `WHAT_STANDS` records, and in the paragraph that congratulates itself on
catching the fifth. Both prior forms are shown so the correction is auditable.

**Wall-clock, measured on the target box.** Campaign U runs on **h100-2** (8× H100), where a
48-retrain measurement gives **564.7 retrains/hour** at 3 workers per GPU (24 concurrent), against
85/hour on one H200. Per card the H100 is *slower* (70.6/h vs 85/h; 137 s vs 124 s per retrain);
eight of them net 6.6×.

| | retrains | wall |
|---|---|---|
| 10,596 at 564.7/h | 10,596 | **18.8 h** |
| plus ~6% re-runs (§3.4) | ~11,230 | **≈19.9 h** |

The ×1.84 allocation costs ~8.6 h more than ×1.0 would, rather than the ~57 H200-hours the same
step would have cost on one card — which is why robustness to the partition assumption (§3.6) was
bought rather than argued for.

## 7. The viability gate — PASSED at all four pools, on conditioned masks

**Criterion of record: r ≥ 0.495 (equivalently S/N ≥ 0.7)**, one form, derived from the pipeline's
own depth-2 convention `r = s²_mask/(s²_mask + s²_seed/2)`. (Revision 2 stated the bar two ways that
differ by 40%; round 3 caught it.)

**Result — campaign U's actual mask distribution, task-coverage conditioned per §2.2:**

| pool | masks | signal sd | noise sd | r @ depth 2 |
|---|---|---|---|---|
| 50 | 30 | 0.02804 | 0.01516 | **0.872** |
| 100 | 30 | 0.03215 | 0.01915 | **0.849** |
| 200 | 30 | 0.03752 | 0.02038 | **0.871** |
| 370 | 30 | 0.03407 | 0.02156 | **0.833** |

Every pool clears the bar by a wide margin, every value sits inside §3.3's band, and there is no
monotone trend in r — which is the design's central premise (§2 rationale 2) confirmed rather than
assumed. The **signal rises** with pool size (0.028 → 0.035 → 0.038 → 0.049) as the
finite-population factor predicts, while noise stays flat.

**All four cells are at n = 30; pool 50 got there first, for cause, and the reason is recorded.** At n = 8 that cell
returned r = 0.000 — contradicting the campaign-T pilot's 0.837 at the same operating point. Round 2
had already shown by Monte Carlo that 8-mask cells decide threshold cases by coin flip
(P(pass) ≈ 0.50 at the bar, ≈ 0.19 false-pass at a dead cell), so the contradiction was resolved by
adding masks rather than by choosing whichever number suited. At n = 30 it is 0.872, consistent with
the pilot. **No r in this table is quoted as a difference against another**; §3.5's C uses their
mean, which is what it was always defined to use.

**Had the gate failed**, the preregistered fallback stood: the saturation boundary itself becomes
the result — a ceiling map with CIs on r and on the crossing point N*, as a property of
subset-removal evaluation rather than of any estimator.

## 8. Known limits, recorded before the run

1. **One nested corpus draw, one suite.** The claim is conditional on both. Not "any scale" —
   "any pool size reachable on standard LIBERO-scale corpora".
2. **2.9 octaves.** 50 → 370 cannot rule out a turn-on at 5,000 demonstrations.
3. **Fixed `total_steps`.** A defensible choice (fixed compute budget is the realistic regime) but
   a choice, plausibly implicated in both the under-convergence and the saturation. Stated in the
   limitations before a reviewer finds it.
4. **The extrapolation asymmetry** is named, measured descriptively by H1f (§4), and **not fully
   controlled**. H1f is descriptive, not co-primary, because its fixed 50-demo surrogate carries its
   own seen-fraction trend. H1's slope may be biased negative as the pool grows; the claim is scoped
   to selection (limit 7) partly for this reason.
5. **Masks-per-coefficient is now constant by construction** (§6). The prior revision cited §3.8's
   permutation null as the check; round 2 showed the null **cannot** detect this class of fault, so
   the design removes it instead of testing for it. The null is retained for what it can detect.
6. **The gate now runs on conditioned masks at all four pools** (§7), so the distribution it
   certifies is the one the campaign runs. The conditioned-vs-unconditioned difference at pool 370
   (0.920 vs 0.827) is **not significant at n = 8** and no claim rests on its direction.
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

---

## AMENDMENT 1 — the ceiling's scale, recorded before the score was consumed

**Found during the pre-score dry run, after the outcome table existed and before
`confirm_useries.csv` was written.** Recorded here in full because the reconciliation was chosen
knowing which reading yields a result, and a reader is entitled to weigh that.

**The defect.** §3.1 defines the reported ceiling as a **split-half Kendall τ** between the two
seeds' outcome rankings — a *depth-1 rank* quantity, and §3.1 explicitly says it "understates the
depth-2 reliability". But §3.3's band [0.65, 0.95], §7's gate values (0.872/0.849/0.871/0.833) and
§3.5's `C = (2/π)·arcsin(√r̄₂)` are all on the **variance-based depth-2 `r₂`**. Every quantity in
this document is on `r₂` except the ceiling statistic itself. The band was therefore being applied
to a number it was never calibrated against.

**What the two readings give.**

| pool | split-half Kendall (A/B) | → r₂ | §7 gate r₂ (n=30) | in band on Kendall scale | in band on r₂ scale |
|---|---|---|---|---|---|
| 50 | 0.4825 / 0.4664 | 0.8082 | 0.8720 | no | yes |
| 100 | 0.5252 / 0.5368 | 0.8510 | 0.8490 | no | yes |
| 200 | 0.5568 / 0.5698 | 0.8725 | 0.8710 | no | yes |
| 370 | 0.5682 / 0.5759 | 0.8779 | 0.8330 | no | yes |

Converted to `r₂` the campaign reproduces the gate — pools 100 and 200 to three decimals — so the
data are not in question; only which scale the veto is read on is.

**Disposition: both readings are reported, neither is suppressed.** `confirm_useries` emits
`r_in_band_kendall_scale` and `r_in_band_r2_scale` side by side. Read literally (Kendall scale) the
§3.3 veto fires at every pool and **no trend claim stands**. Read on the scale the rest of the
document uses (`r₂`) the veto does not fire and the τ trend is interpretable. The slope, its CI,
the TOST verdict and the permutation null are **identical under both**, because the band is a veto
gate and not an input to any of them.

**Honest weighting.** This is a post-hoc reconciliation of a preregistration defect, and reporting
a veto condition on two scales is weaker than having gotten the scale right in advance. It is
recorded as such rather than resolved silently — which is the failure mode `WHAT_STANDS` §4 exists
to prevent, and which this campaign has already had to withdraw one table over.

---

## AMENDMENT 2 — two scoring defects found on reading the scored output

Both are defects in `p18_score.py`, not in the campaign. Recorded with their consequences.

**(a) The two arms predict opposite-signed quantities, and `score()` regressed the raw signed τ.**
H1/H1f sum GradDot influence, where a *higher* score means a training set expected to *lower*
held-out loss — so a working estimator gives **negative** τ. H2 regresses the outcome directly on
group indicators, so a working estimator gives **positive** τ. Regressing raw τ therefore compared
a slope in "influence" units against one in "predicted outcome" units. On the **quality** scale
(oriented so higher = better) the arms read:

| arm | pool 50 | 100 | 200 | 370 | quality slope (unweighted) |
|---|---|---|---|---|---|
| H1 | +0.0568 | +0.0360 | +0.1299 | +0.0318 | +0.0028 |
| H2 | +0.1282 | +0.2034 | +0.2148 | +0.2262 | +0.0319 |

H1's headline `slope=+0.0355 → TREND` was a sign artefact: on the quality scale its slope is
**+0.0028**, and its own permutation null puts |slope| at p95 = 0.0234 — an order of magnitude
larger. **There is no H1 trend.**

**(b) The §5 non-monotonicity branch was never implemented**, though §5 gives it precedence over
slope/TOST. Computed now (max deviation of an interior pool from the fitted line, against
Δ_τ = 0.04966):

| arm | max interior deviation | branch |
|---|---|---|
| H1 | 0.0648 | **FIRES** |
| H2 | 0.0253 | does not fire |
| H1f | 0.0838 | **FIRES** |

**Under the preregistered precedence, H1 is reported as NON-MONOTONE and no slope is quoted for
it.** Pool 200 sits 0.065 off the line — H1's τ is noise around zero, not a curve.

**What stands.** H2 is monotone; its preregistered inverse-variance-weighted slope is **+0.0182,
CI [−0.0047, +0.0409]**, inside ±Δ_τ → **FLAT** by TOST. H1 carries no trend claim; what it carries
is a **level**: 5.6%–23.1% of attainable at the four pools, against the datamodel's 27%–40%.
