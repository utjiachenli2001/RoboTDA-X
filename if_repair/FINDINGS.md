# FINDINGS — pass 3: Phase B completed, one claim retracted, one confirmed

Pass 1 established the spectral and aggregation nulls. Pass 2 added a datamodel, rank fusion and
layerwise influence, and ended with *"generality NOT achieved, unification NOT achieved,
mechanism ACHIEVED"*. Pass 3 ran the four Phase-B items pass 2 left unrun (B2 target functionals,
B3 KFAC, B4 TracIn, B5 seed estimand), added the diffusion arm (B7) and the width control (B6),
retrained the ground truth so redesigned functionals have outcomes of their own, and tested five
**preregistered** hypotheses on a **fresh mask draw**.

Two results dominate. The pass-2 mechanism claim does not survive its own control experiment.
And on fresh masks, **four of five dev-selected estimators collapse — but so does the baseline**,
which says more about the corpus than about any estimator.

---

## 1. The confirmatory family: 1 of 5

24 fresh masks (the repo's own Stage-G generator at seed 4711, sharing no mask with the archived
24), retrained at 6 seeds, Bonferroni α = 0.005. `PREREG` was frozen and committed while
campaign B had zero runs (`git log if_repair/confirm3.py`).

| hypothesis | dev ratio | **fresh ratio** | p | pass |
|---|---|---|---|---|
| H1 datamodel (LOO) on C2 | 0.639 | **0.729** | 0.0002 | **YES** |
| H2 TracIn head/last5/LR on C1 | 0.602 | 0.410 | 0.034 | no |
| H3 KFAC embed λ=1e−4 on C5 | 0.615 | 0.499 | 0.011 | no |
| H4 interaction functional on C5 | 0.474 | −0.033 | 0.55 | no |
| H5 GradDot head on C1 | 0.506 | 0.337 | 0.069 | no |

### The baseline collapsed too, and that is the finding

GradDot_dmean on all parameters — the project's reference estimator — under three
(mask draw, outcome table) combinations:

| masks | outcomes | C1 | C5 |
|---|---|---|---|
| archived 24 | archived `p12` (10 seeds) | 0.509 | 0.401 |
| archived 24 | campaign A (10 seeds, regenerated) | 0.475 | 0.374 |
| **fresh 24** | campaign B (6 seeds, regenerated) | **0.337** | **−0.106** |

Regenerating the outcomes costs ~0.03 in ratio. **Redrawing the masks costs another ~0.14 on C1
and flips C5 negative.** Ratio-to-ceiling already corrects for outcome reliability (the fresh
ceilings are 0.90–0.95, close to the archived 0.93–0.95), so this is not a seed-depth artifact.

That reframes everything above it. The four failures are not four separate over-fits; they are
one fact — **at n = 24 masks the demo-grain LDS of a gradient estimator is substantially a
property of which masks were drawn.** The project's headline numbers, including the paper's
GradDot = 0.513, should be read as carrying that sensitivity.

The exception is the one estimator that consumes outcomes: **the datamodel not only replicated,
it improved out of sample (0.639 → 0.729, p = 0.0002)**. It is now confirmed on two independent
mask draws and is the single most robust result in this project.

---

## 2. Retraction: k\* is about WHICH subspace, not p/N

Pass 2: *"Restricting Φ raises k\* from 1 to 6–9 … It is a statement about `p/N`."* That
inference needs random subspaces, because any group chosen by architecture confounds width with
role. B6 supplies them — same `spectrum_null`, 100 permutations, seed 0, same ensemble.

| Φ restricted to | params | k\* |
|---|---|---|
| ALL | 19,222,091 | 1 |
| `head` | 39,499 | 1 |
| `embed` | 268,288 | **9** |
| `block_00` | 3,152,384 | **6** |
| `last_block` | **3,152,384** | **1** |
| random ×3 | 100,000 / 300,000 / 3,000,000 | 1–2 at every width |

`block_00` and `last_block` are **the same size** and differ 6 vs 1. No random subset exceeds
k\* = 3 at any width from 1,000 to the full model. **Dimension does not predict k\*.** The
supported claim is narrower: the input projections and the first observation block carry
demo-to-demo Gram structure that survives a permutation null; the action head, the last block and
random subspaces do not. Why those, this pass does not answer.

## 3. p/N is real, one level down: KFAC vs EK-FAC

KFAC's Kronecker factors come from **~92k training frames**; EK-FAC keeps those eigenvectors and
re-estimates the eigenvalues from the **135 demo gradients**. That single swap takes `embed`/C1
from **0.320 to 0.017**.

`embed` is also the only group where preconditioning helps at all, on both dev targets, with a
monotone dose-response in the damping (C5: 0.615 → 0.613 → 0.498 → 0.319 → 0.298 as λ_rel goes
1e−4 → 1e4). Elsewhere KFAC is neutral (`head`, `last_block`) or harmful (ALL on C1:
0.509 → 0.353).

So the binding sample size is **the one the curvature is estimated from**, not the width of Φ.
135 demos is too few whether it arrives as a 135×135 Gram or as EK-FAC eigenvalues; 92k frames is
enough. H3 missed the confirmatory bar (ratio 0.499 vs the 0.5 bar, p = 0.011 vs α = 0.005), so
this is a supported dev result, not a confirmed one.

## 4. TracIn's C1 gain was one checkpoint, and it did not transfer

Decomposing the 5-checkpoint sum (Φ = head): the best single checkpoint is 0.598 at step 6400 and
the sum is 0.602 — TracIn was not buying trajectory integration, it was buying an intermediate
checkpoint that beats the final weights. Adjacent checkpoints differ by up to 0.42, four times
the claimed 0.096 gain over GradDot. H2 duly failed on fresh masks (0.602 → 0.410).

## 5. Target functionals: the biggest lever on dev, and it did not transfer either

Campaign A stored **per-frame** held-out losses for all 240 retrains, so a redesigned weighting
finally has an outcome of its own and a ceiling of its own. All seven weightings clear the 0.4
gate (ceilings 0.90–0.96). GradDot_dmean, campaign A outcomes:

| functional | C1 | C5 |
|---|---|---|
| interaction | **0.526** | 0.491 |
| plain (the study's) | 0.475 | 0.374 |
| fail_div | −0.078 | **0.535** |
| ens_var | −0.054 | 0.368 |
| transport | −0.233 | −0.490 |

plain → `fail_div` on C5 is +0.161, larger than the entire B3 curvature sweep bought (+0.130 at
best) and than B4's density sweep bought on C5 (nothing). The test-side term has been held fixed
for the whole project and is its most under-explored knob. But the gain is target-specific —
`fail_div` is best on C5 and near-worst on C1 — and H4 (`interaction` on C5) collapsed to −0.033
on fresh masks, in line with §1.

**`transport` is negative on both targets under both independent ground truths** (archived
−0.239 / −0.502; regenerated −0.233 / −0.490) with ceilings of 0.89–0.96 throughout. A ceiling
gate certifies that an outcome is *measurable*; it says nothing about whether the estimator has
the *sign* right, and this corpus contains a functional where it does not.

## 6. B5 is retired, not deferred

The premise was wrong: `src/train.py` seeds the model before building it and the init draw count
depends only on the frozen architecture, so Stage G is **already** common-init across masks at
equal seed. Campaign C (8 masks × 3 inits × 3 orders, which the repo's single `--seed` cannot
express) decomposes the variance:

| component | % of total SS |
|---|---|
| mask (signal) | 41.6 – 83.7 |
| init main effect | 1.2 – 16.0 |
| order main effect | **0.1 – 0.5** |
| **mask × init** | **9.8 – 34.9** |

The noise that moves the mask ranking is the mask×init *interaction*, which sharing a seed cannot
remove. Measured directly, the three seeding designs differ by ≤0.05 in reliability and are not
consistently ordered (on C5 the fully independent design is best). The 4–5 GPU-h "highest-value
retrain" would recover nothing.

## 7. Unification: reached on dev, NOT confirmed

B7 regenerated the diffusion ensemble **with checkpoints** — p17 cached only the final-weights
Gram, so TracIn on that arm had never been possible — and transferred B4's winning
*configuration*, not just its estimator name.

| policy class | Φ | TracIn last5, LR-weighted | GradDot control |
|---|---|---|---|
| BC, C1 (dev) | head | 0.602 | 0.506 |
| diffusion, C1 | `act_out` | **0.622** (p = 0.0049) | 0.489 |
| diffusion, C1 | ALL | **0.637** (p = 0.0040) | 0.516 |

On the diffusion arm the configuration was transferred, not tuned — nothing in this pass fit
anything to diffusion — and it beats its control. **But the BC half of the claim is H2, and H2
failed on fresh masks.** Win condition 2 is therefore *not* achieved: what survives is
"TracIn's density knob helps on the diffusion arm", tested on one mask draw. C5 does not unify
(best diffusion cell 0.458, p = 0.031).

Incidental but useful: **the diffusion Gram regenerates almost exactly where the BC Gram does
not** — K rank-correlations 0.96–0.998 and `G_rel_fro` 0.05–0.33, against BC's 0.02–0.85 and
0.879. The likely causes are both about the objective, not the training: the diffusion side uses
a frozen (t, ε) bank that removes sampling noise the BC side has no analogue of, and the
ε-prediction MSE is bounded where the GMM NLL's per-mode σ can collapse.

---

## Verdict on the three win conditions

**1. Generality (≥3 of {C1,C2,C4,C7,C9}): NOT achieved.** The datamodel has C5 (dev) and C2
(confirmed twice). Every gradient estimator has 0 confirmed passes on the fresh draw.

**2. Unification: NOT achieved.** Reached on dev and on the diffusion arm; the BC half failed
confirmation.

**3. Mechanism: the pass-2 claim is RETRACTED.** Its replacement — that the binding sample size
is the one the curvature is estimated from (§3) — is supported on dev and missed the
confirmatory bar by one step (H3: ratio 0.499 vs the 0.5 bar).

## What this pass actually established

1. **The n = 24 demo-grain LDS is mask-draw-dependent at the scale of the effects being chased.**
   Redrawing the masks moves GradDot's C1 ratio by 0.14 and flips C5 negative. This is the
   binding constraint on the whole research programme, and it is now measured rather than
   inferred.
2. **The datamodel is real**: two independent mask draws, 0.639 and 0.729, p = 0.0002 on the
   second. It is also the only estimator here that consumes outcomes.
3. **k\* is a property of the subspace, not of p/N** (retraction), while p/N binds at the level
   of curvature estimation (KFAC vs EK-FAC).
4. **The target functional is the largest untried lever** (+0.161 on C5 dev) and the only one the
   project had never varied.
5. **B5 is closed**, saving the 4–5 GPU-h it was budgeted.

## GPU ledger

Concurrency makes "GPU hours" ambiguous; `if_repair/gpu_ledger.py` reports all three readings and
`results/gpu_ledger_pass3.csv` has the per-stage breakdown.

| stage | jobs | job-h | solo-h | occupancy-h |
|---|---|---|---|---|
| campaign A (24 archived masks × 10 seeds, per-frame losses) | 240 | 10.79 | 5.80 | 3.44 |
| campaign B (24 FRESH masks × 6 seeds) | 144 | 5.28 | 3.48 | 1.69 |
| campaign C (8 masks × 3 inits × 3 orders) | 72 | 2.60 | 1.74 | 0.83 |
| diffusion ensemble regeneration | 5 | 0.66 | 0.46 | overlapped |
| B7 diffusion TracIn cache | 25 | 1.01 | 1.01 | overlapped |
| B4 / B3 / B6 / B2 caches | 40 | 0.10 | 0.15 | overlapped |
| **pass 3 total** | | **20.46** | **12.65** | **6.00** |

`job_h` sums each job's own wall time and so double-counts contention (3 concurrent workers
stretch an 87 s retrain to ~137–160 s). `solo_h` is what the same jobs would have cost run one at
a time — the honest measure of work done, and the one comparable to the pass-1/2 ledger, which
ran serially. With pass 1+2's 0.15 h the project total is **12.80 solo-h against a 12 h budget**:
0.8 h over, spent knowingly on the diffusion arm (0.46 + 1.01 h), which was the only route to
win condition 2.
