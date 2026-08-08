# R5 preregistration — does attribution-based selection buy anything?

**Frozen before any R5 retrain.** Written 2026-08-07, after campaign U scored. Deferred from
`p18_prereg.md` §8.9 by design: R5's arms depend on the influence scores campaign U produced, so it
could not be preregistered earlier without inventing them.

---

## 1. The question

Campaign U measured that attribution is **poorly measurable** on this data. It never asked what that
costs anyone. R5 does:

> Given a 370-demonstration pool and a budget of 25, does selecting by influence beat selecting at
> random — on held-out loss, and on task success?

This is what TDA is *for*. `WHAT_STANDS` §1 already states that per-demonstration use means
selection or pruning; every number this project reports is upstream of that claim and none of them
tests it.

## 2. Arms

Pool 370, retain **25 demonstrations** — campaign U's exact operating point, where the ceiling is
measured healthy (r = 0.833–0.920).

| arm | selection rule | n policies |
|---|---|---|
| **DM-top** | 25 highest by datamodel influence (fit on partition A, per `p18_score`) | 1 × 5 seeds |
| **DM-bottom** | 25 lowest by datamodel influence | 1 × 5 seeds |
| **GD-top** | 25 highest by gradient (GradDot) influence, H1 arm | 1 × 5 seeds |
| **RANDOM** | 25 drawn uniformly, **10 independent replicates** | 10 × 5 seeds |

**65 retrains.** Every arm is subject to §2.2's task-coverage rule from `p18_prereg.md` — a
selection that drops a whole task would win or lose for a reason that has nothing to do with
attribution. Selections violating coverage are repaired by swapping in the highest-ranked
demonstration of each missing task, and the number of swaps is reported per arm.

**DM-bottom is not decoration.** Without it, "top beats random" is consistent with *any* non-constant
score. Top-vs-bottom bounds the total range the scores can move the outcome; random sits between
them if the scores mean anything.

## 3. Outcomes

1. **Held-out loss** on campaign U's 100-demo eval bank. Primary for the loss claim, free.
2. **Rollout success rate**, 10 `libero_goal` tasks × 10 episodes = 100 episodes per policy,
   `src/rollout.py`. **Primary for the claim that matters**, and the metric ICRA-style venues read.

Both are reported for every arm. Success is preregistered as primary **because it is the deployable
quantity and because held-out loss is the metric this project has just spent a campaign showing has
a measurability horizon** — using it alone to adjudicate a selection claim would be circular.

## 4. Hypotheses

> **H-A (loss).** DM-top's mean held-out loss equals the RANDOM arm's.
> **H-B (success).** DM-top's mean rollout success equals the RANDOM arm's.

Family of 2, α = 0.05, Bonferroni → **0.025**. GD-top and DM-bottom are **descriptive** — GD-top
because campaign U already showed the gradient scores are near-noise and a null there is expected
rather than informative, DM-bottom because it is a range-bound, not a test.

**Test.** The RANDOM arm's 10 replicates give the null distribution directly. A fixed arm is
declared different if its 5-seed mean falls outside the **2.5th–97.5th percentile** of the 10
random replicate means. This is a permutation-style test against the design's own control, with no
distributional assumption — deliberately, because success rates are binomial and losses are not.

## 5. Decision rule, fixed now

| observed | reading of record |
|---|---|
| DM-top beats RANDOM on success, outside the random band | **attribution-based selection works.** The headline: what the datamodel buys is X points of success at a fixed budget. |
| DM-top inside the random band on success | **selection does not pay** at this budget and pool, even though the datamodel measurably attributes. This is the more likely outcome given cross-partition per-demo agreement is ~0.5, and it is stated in advance so it cannot be reframed as a disappointment. |
| DM-top beats RANDOM on loss but not success | the scores optimise the measured proxy and not the task — **a result about the proxy**, and a caution for every LDS-based selection paper. |
| DM-bottom not worse than RANDOM | the scores carry no usable per-demonstration ordering at all; H-A/H-B are then uninterpretable regardless of their verdicts. |
| GD-top ≈ RANDOM | expected; consistent with campaign U. Carries no alpha. |

**A null here is a real result and is recorded as such in advance.** "Attribution is measurable but
selection does not pay" is more useful to the field than a weak positive, and this project has
already shown what happens when a design can only comfortably report one direction.

## 6. Known limits

1. **One pool, one budget, one suite.** 25 of 370 on `libero_goal`. Nothing here licenses a claim
   about other ratios or corpora.
2. **The datamodel scores are fit on this same pool**, so DM-top is selected using information
   derived from retrainings of this pool. That is the realistic deployment setting (you fit the
   datamodel to pick your training set) but it is **not** out-of-sample selection, and the
   write-up must say so.
3. **100 episodes per policy** gives a binomial SE of ~5 points at p = 0.5. Differences smaller
   than ~10 points will not be resolvable, and that is accepted in advance rather than pursued.
4. **Task coverage is enforced**, so the arms cannot differ by dropping a task. This makes the test
   harder and is deliberate.
5. `total_steps` fixed at 8000, as everywhere in this project.

## 7. Scored once

`confirm_r5.csv` is written once, refuses to overwrite, and refuses to write unless both arms of
§4 have usable data.
