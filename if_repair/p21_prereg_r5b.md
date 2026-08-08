# R5b preregistration — a powered replication of the selection-vs-random comparison

**Frozen before any R5b policy is trained or rolled out.** Written 2026-08-07, after R5 scored.
R5's result stands as scored; **this does not re-score it.** R5b is a separate, better-powered
experiment, and its result will be reported alongside R5's rather than replacing it.

---

## 0. Why R5 was not enough, in two respects

**(a) It was underpowered, as its own prereg said in advance.** 100 episodes per policy gives a
binomial SE of ~4.9 points at p ≈ 0.38 — about the same size as the selection-to-selection spread
it was competing against (the 10 random replicates had SD ≈ 0.05). DM-top came in **+5.4 points**
over random on success and landed inside the band. That is equally consistent with a real effect of
that size and with none.

**(b) The test answered the wrong question, and this is the more important fault.** R5 asked
whether DM-top falls outside the 2.5–97.5 percentile band of 10 random replicates — i.e. *"is
DM-top an unusual draw from the distribution of random selections?"* But DM-top is not a draw from
that distribution; it is one fixed, deliberately-chosen selection. The question of interest is
*"does its expected success exceed the expectation over random selections?"* — a comparison of
means, not an outlier test. The band test is strictly weaker and answers something nobody asked.
R5's loss verdict is unaffected (every fixed arm was outside the band by a wide margin), but its
success verdict rests on the weaker test and is superseded here.

## 1. Design

| element | R5 | **R5b** |
|---|---|---|
| random replicates | 10 | **20** |
| seeds per arm | 5 trained, 3 rolled out | **5 trained and rolled out** |
| episodes per policy | 100 (10/task) | **400 (40/task)** |
| test | percentile band | **difference of means, bootstrapped** |

23 arms (DM-top, DM-bottom, GD-top, RANDOM-00…19) × 5 seeds = **115 policies**, 400 episodes each
= **46,000 episodes**. The 65 R5 policies are reused where the arm and seed already exist; only the
10 new random replicates need training.

Everything else is inherited from `p20_prereg_r5.md` unchanged: pool 370, budget 25, task-coverage
enforced on every arm, `libero_goal`'s 10 tasks, same trainer and eval bank.

## 2. Hypotheses

> **H-B2 (success).** DM-top's expected rollout success equals the expectation over random
> 25-demonstration selections.
> **H-A2 (loss).** The same, for held-out loss.

Family of 2, α = 0.05, Bonferroni → **0.025**. GD-top and DM-bottom remain descriptive.

**Test of record.** The estimate is `mean(DM-top over its 5 seeds) − mean(RANDOM over its 20
replicates × 5 seeds)`. Its CI comes from a **hierarchical bootstrap**: resample the 20 random
replicates with replacement, and within every arm resample its 5 seeds with replacement, so both
the selection-to-selection and seed-to-seed components are propagated. Episode-level binomial noise
is already inside each policy's measured rate. Reject if the 95% CI excludes zero.

## 3. Power, computed in advance

Measured in R5: random replicate-to-replicate SD ≈ 0.050; binomial SE at 400 episodes ≈ 0.024,
so per-arm seed-mean SE ≈ 0.024/√5 ≈ 0.011.

- SE of the random-arm grand mean: √(0.050²/20 + 0.011²) ≈ **0.015**
- SE of DM-top's mean: dominated by its 5 seeds ≈ **0.020**
- SE of the difference ≈ **0.025**

**Minimum detectable difference at 80% power, α = 0.025: 2.80 × 0.025 ≈ 7.0 points.** R5's observed
+5.4 sits just below that, so a true effect of the size R5 hinted at would be detected roughly
**60–65%** of the time. That is honest rather than comfortable, and it is stated now: **R5b can
resolve an effect of ~7 points or larger; it cannot resolve one of ~3.**

If the result is null, the preregistered reading is *"selection does not buy more than ~7 points of
success at this budget"* — a bound, not a proof of no effect.

## 4. Decision rule

| observed | reading of record |
|---|---|
| DM-top − RANDOM CI excludes 0, positive, on success | **selection pays**: the datamodel buys X points of task success at a fixed budget |
| CI contains 0 on success, excludes 0 on loss | **the scores optimise the proxy, not the task** — R5's finding, now on the correct test and at 4× the episodes. Bound the success effect by the CI's upper limit. |
| CI contains 0 on both | the scores carry no usable ordering at this budget; R5's loss result would then be contradicted and BOTH are reported as unresolved |
| DM-bottom not worse than RANDOM on loss | the loss ordering is not real either; nothing is claimed |

## 5. Limits, unchanged from R5 plus one

1–5 as in `p20_prereg_r5.md` §6 (one pool, one budget, one suite; scores fit on this same pool, so
selection is not out-of-sample; coverage enforced; `total_steps` fixed).
6. **R5b shares R5's policies where they exist.** The 3 fixed arms and RANDOM-00…09 are the same
   checkpoints at the same seeds, re-rolled at 400 episodes. So R5b is not independent of R5 — it
   is R5 with more episodes, more replicates, more seeds and a correct test. Reported as such.

## 6. Scored once

`confirm_r5b.csv` is written once and refuses to overwrite.
