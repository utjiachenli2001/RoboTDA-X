# PREREG_R -- pass 10, campaign R (second independent partition at k=3 and k=5)

**Frozen 2026-07-29, while campaign R has ZERO runs.** Base commit 257665a. Do not edit after the
first run dir under `if_repair/runs/campaigns/R/` exists.

## What motivated this campaign (and why that does not contaminate it)

Pass 9's k=3 and k=5 rungs rest on ONE committed partition of the corpus into groups
(`p9_grain.GROUP_SEED = 20260728`). Each cluster's 15 demos were shuffled once and chunked, so a
"group of 3" is one particular triple among many. Those rungs therefore carry partition-sampling
variance that the k=15 rung structurally cannot -- a cluster has no composition freedom. Pass 9
recorded that caveat, and `p9_prereg.md` named this exact check as "the cheap robustness check if the
curve comes out close". It came out close: **0.356 vs 0.365**.

The motivation is a *known limitation of a completed design*, not a pattern spotted in campaign R's
outcomes -- campaign R has none. Nothing here is selected on.

**Why this rather than a depth-4 re-read of campaign O.** The depth threat's direction is already
measured (BLOCKERS #42: the same 5of9 masks give ratio 0.666 at depth 2 and 0.496 at depth 4) and
the allocation question is half-answered (#29: the mean ratio-to-ceiling is flat across allocations).
The partition threat's direction is **unknown**. Confirming a known direction while an
unknown-direction threat sits deferred is the wrong use of 18 hours.

## Hypothesis of record

**FAMILY OF TWO**, so **alpha = 0.025 one-sided each** by Bonferroni.

> **R1.** At k=3, the campaign-R ratio point estimate falls **inside** campaign O's committed 95%
> bootstrap CI for k=3, which is **[0.180, 0.550]**.
>
> **R2.** The same at k=5, against campaign O's **[0.204, 0.559]**.

**The hypothesis is AGREEMENT, and a failure is the interesting outcome.** If a rung lands outside,
pass 9's k=3/k=5 numbers are partly an artifact of one partition draw and its curve needs amending --
which would be pass 10's main result. Prespecifying it this way means the pass cannot quietly reframe
a disagreement as noise after seeing it.

**Secondary, preregistered but not alpha-bearing:** the signed difference in ratio between partitions
at each grain, with a bootstrap CI on the difference. This is what gets reported if the point-estimate
containment rule is passed but the difference is nonetheless large.

## Statistic

**Primary Kendall tau_b, mandatory secondary Spearman**, fixed by continuity with campaign O. Not
re-selected: the comparison is against campaign O's committed intervals, which are Kendall, and a
partition comparison computed under a different statistic would not be a partition comparison.

## Design

Identical to campaign O in every respect except the partition seed. That is what makes this a
partition comparison rather than a design comparison.

| held identical to campaign O | value |
|---|---|
| masks per grain | 400 |
| retained demos (fixed) | 75 |
| conditioning | all of C5's groups retained |
| construction | complementary-pair permutation blocks |
| balance | exact, spread 0, every free group in exactly 200 masks |
| depth / seed slots | 2, {4401, 4402} |
| job order | seed-major |

| changed | pass 9 | campaign R |
|---|---|---|
| `GROUP_SEED` | 20260728 | **20260730** |
| `MASK_SEED` | 20260729 | **20260731** |

**Verified at build time: the two partitions share ZERO groups** at either grain (0/45 at k=3, 0/27
at k=5), so this is a genuinely independent partition and not a perturbation of pass 9's.

**Disjointness is over DEMO SETS, not group ids.** Two different partitions can produce the same
75-demo set, so the signature check spans campaigns A-P over demo sets. Measured: 0 collisions.

Total GPU: 800 masks x depth 2 = **1600 retrains**, ~18.2 h wall at the measured 88/hour.

## Stopping rule (prespecified, outcome-blind)

The analysis uses the largest EVEN depth for which all masks of a grain have completed seeds, read
off the run directory and never off the outcomes. Seed-major ordering makes every prefix a complete
balanced design. `confirm_rseries.py` inherits pass 9's `missing_grains` guard: it refuses to write
unless BOTH preregistered grains have a complete even depth, so a partial campaign cannot consume the
one-shot score and leave R2 permanently unanswerable.

## Scored exactly once

`confirm_rseries.py` refuses to overwrite an existing result file. `confirm_oseries.csv` is never
touched.

## Secondary and descriptive (no alpha)

- The absolute half-ceiling bar on campaign R. Campaign R is a genuinely fresh draw, so this read is
  legitimate out-of-sample -- it is simply not the point of the campaign, and pass 9 already answered
  it in the negative at both grains.
- The datamodel on campaign R's masks, for comparison with U2's campaign-O read.
- Per-grain permutation nulls, which must sit at ~0 at a fixed training-set size. If they do not, the
  design has a size channel and the primary is not interpretable.
- `rho/sqrt(r)` alongside `rho/r` at both grains (#42).

## What this campaign cannot do

It cannot resolve the grain trend. The k=15 rung's conditional population is combinatorially capped at
C(8,4) = 70 and pass 10's census exhausted it; no purchasable design tightens that rung below a CI
width of ~0.6, which still overlaps k=3 and k=5. Campaign R tests whether pass 9's two sub-cluster
rungs are partition-robust, and nothing more.
