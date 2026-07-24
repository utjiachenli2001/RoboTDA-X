# FINDINGS — pass 3: Phase B done; after THREE mask draws, only the datamodel beats GradDot out of sample

> One-line status: generality NOT achieved, unification NOT achieved, mechanism RETRACTED and its
> replacement NOT confirmed. The datamodel is the sole estimator that beats GradDot on masks it was
> not selected against. The durable contribution is methodological — at n=24, only out-of-sample
> paired comparison can be trusted, and this pass shows three separate "wins" dissolve under it.


Pass 1 established the spectral and aggregation nulls. Pass 2 added a datamodel, rank fusion and
layerwise influence, and ended with *"generality NOT achieved, unification NOT achieved,
mechanism ACHIEVED"*. Pass 3 ran the four Phase-B items pass 2 left unrun (B2 target functionals,
B3 KFAC, B4 TracIn, B5 seed estimand), added the diffusion arm (B7), the width control (B6), the
structured-subspace probe (B9), a fresh-mask sampling study (B8), and — crucially — a **third,
independent mask draw** (the I-series) that no estimator was selected against.

**The arc of this pass is a case study in why out-of-sample validation is not optional.** Pass-3
dev found several apparent improvements over GradDot. The H-series (draw 2) killed most of them.
B8's paired analysis appeared to rescue two (KFAC-on-embed and the datamodel on C5, each beating
GradDot in ~100% of resampled draws) — but B8 pooled the G- and H-series, both of which the
estimators had by then been selected on. The I-series (draw 3) is the first clean test, and it is
humbling:

- **KFAC-on-embed on C5 does NOT replicate.** On fresh masks it scored 0.428 and *lost to
  GradDot* (0.490), paired Δρ −0.062. The B8 "100% of draws" result was an artefact of reusing the
  selection masks. The pass-2 mechanism, restated in §3, does not survive a clean draw.
- **The `fail_div` functional does NOT replicate.** Its +0.161 C5 dev lever shrank to +0.010.
- **The datamodel survives** — C5 ratio 0.882 on the I-series (absolute pass, p = 8e-8), beating
  GradDot by Δρ +0.358 — but even it does not clear the strict Bonferroni-3 paired bar
  (paired-p 0.030), and on C2 it weakened (0.400, an absolute fail on this draw).

The one durable conclusion across three independent mask draws: **the datamodel is the only
estimator that beats GradDot out of sample, and every gradient-side improvement this project found
was mask-draw overfitting.** That, and the methodological finding that at n = 24 no single-draw
improvement can be trusted, are the real results.

---

## 1. The confirmatory family: 2 of 5 (at matched seed depth)

24 fresh masks (the repo's own Stage-G generator at seed 4711, sharing no mask with the archived
24), Bonferroni α = 0.005. `PREREG` was frozen and committed while campaign B had zero runs
(`git log if_repair/confirm3.py`).

**Correction to the first write-up.** Campaign B was first run at 6 seeds while the dev numbers it
is compared against were at 10. That is not a neutral choice: holding masks, estimator and outcome
table fixed and varying only depth, GradDot_ALL/C1 scores 0.362 at depth 6 and 0.475 at depth 10
(and C5 moves the other way). Dividing by the ceiling does not remove it. Campaign B was extended
to 10 seeds and the family recomputed at matched depth; the depth-10 table supersedes, and the
protocol (match the archived depth, report both) was declared before looking. See BLOCKERS #17.

| hypothesis | dev ratio | fresh @ d6 | **fresh @ d10** | p (d10) | pass |
|---|---|---|---|---|---|
| H1 datamodel (LOO) on C2 | 0.639 | 0.729 | **0.626** | 0.0012 | **YES** |
| H3 KFAC embed λ=1e−4 on C5 | 0.615 | 0.499 | **0.563** | 0.0032 | **YES** |
| H2 TracIn head/last5/LR on C1 | 0.602 | 0.410 | 0.340 | 0.060 | no |
| H4 interaction functional on C5 | 0.474 | −0.033 | −0.055 | 0.60 | no |
| H5 GradDot head on C1 | 0.506 | 0.337 | 0.283 | 0.099 | no |

H3 sat exactly on the 0.5 bar at depth 6 (0.499) and cleared it at matched depth (0.563). It is
the §3 mechanism result — curvature from 92k frames on the one subspace with real Gram structure —
confirmed out of sample.

### The right statistic: paired differences on shared masks (B8)

The absolute ratio is unresolvable at this n. B8 pools the 24 archived (G-series) + 24 fresh
(H-series) masks — exchangeable: same generator, same 68-demo stratified design, same demo
universe, same held-out bank — and bootstraps 2000 random 24-mask subsets (matched 10 seeds). The
level of any estimator has a sampling sd of ~0.13–0.17. But every estimator is scored on the
*same* subset, so the draw cancels in a paired difference, and those are sharp:

| target | estimator vs GradDot | mean Δ | paired sd | beats GradDot in |
|---|---|---|---|---|
| C5 | KFAC on embed | **+0.48** | 0.15 | **100.0%** of draws |
| C5 | datamodel (LOO) | **+0.56** | 0.20 | 99.7% |
| C1 | TracIn head/last5/LR | +0.08 | 0.10 | 77.1% |
| C2 | datamodel | +0.12 | 0.23 | 70.9% |
| any | GradDot head vs GradDot ALL | ~0 | **0.005** | — |

On the pooled 48 (ceiling ~0.94), C5 KFAC-on-embed = 0.631 and datamodel = 0.684 both clear the
0.5 bar while GradDot sits at 0.137. Depth-6 and depth-10 bootstraps agree throughout.

On the pooled 48 this looked like two clean wins over GradDot on C5. **The I-series (§1b) shows one
of them was overfit.** What B8 *did* establish robustly, and the I-series does not touch, is the
paired-vs-level point itself and one exact identity:

**The action head IS the model, to four decimals.** GradDot on the 39,499-parameter head
reproduces GradDot on all 19.2M parameters with a paired sd of 0.005. B1's "0.2% of parameters
carry the C1 signal" is exact, not approximate. (This is an identity between two computations on
the same masks, not a mask-dependent claim, so it does not need a fresh draw.)

## 1b. The I-series — the third draw, and what it kills (`confirm_iseries.py`)

24 fresh masks, Stage-G generator seed 9973, disjoint from both G and H (`tests/test_iseries.py`),
10 seeds. `PREREG_I` frozen and committed at `3f5d978` with campaign I at zero runs. Two bars per
hypothesis: absolute (ratio ≥ 0.5, p < Bonf-3 = 0.0083) and paired (beats GradDot on the same
masks, one-sided mask-bootstrap).

| hypothesis | I-series ratio | GradDot on same masks | paired Δρ | paired p | abs pass | paired pass |
|---|---|---|---|---|---|---|
| J1 KFAC-embed, C5 | 0.445 | **0.510** | **−0.062** | 0.59 | no | no |
| J2 datamodel, C5 | **0.882** | 0.510 | +0.358 | 0.030 | **yes** | no |
| J3 datamodel, C2 | 0.400 | −0.234 | +0.598 | 0.013 | no | no |

**0 of 3 clear the strict paired bar; 1 of 3 clears the absolute bar (datamodel/C5).** The
headline is J1: KFAC-on-embed, which won 100% of B8's pooled draws, *loses to GradDot* on a draw
it was never selected against. B8 could not detect this because it resampled the very masks the
estimator was tuned on. The datamodel remains the best attributor — strongly positive on both
targets (Δρ +0.358, +0.598) — but the n = 24 paired test at Bonferroni-3 is too stringent for even
it to pass on a single fresh draw.

Item #3 (the redesigned functionals, on the I-series): `fail_div`, the biggest C5 dev lever at
+0.161, collapses to +0.010 over plain (0.519 vs 0.509). `ens_var` is worse than plain. Neither
model-derived functional replicates. And the whole of C1 goes negative on this draw (plain GradDot
−0.095), a raw demonstration of BLOCKERS #17 — mask-draw variance swamping every C1 comparison.

**Net across three draws:** only the datamodel beats GradDot out of sample, and no gradient-side
improvement does. The single-draw wins (KFAC-embed, TracIn, fail_div) were mask-draw overfitting,
which is exactly what the third draw exists to catch.

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
k\* = 3 at any width from 1,000 to the full model. **Dimension does not predict k\*.**

**B9 answers "why those?" — it is DEPTH, not side.** On the same cached full-width Φ, structured
subspaces of matched size: the six transformer blocks are identical in dimension (3.15M) yet
k\* falls monotonically with depth — block_00=6, block_01=5, block_02..05=1. `embed` has the
highest k\* (9), but so does block_00's attention (8) and even the first LayerNorm (6, on 1,024
params); the common factor is *early*, not *input-side*. Every structured subspace beats its
size-matched random control (k\* 5–9 vs 1–2), confirming the structure is real. **But k\* does not
predict where inverting helps**: mean gain-from-inverting for k\*≥3 vs k\*≤1 is +0.008 vs +0.001
(C1) and +0.093 vs +0.106 (C5) — indistinguishable. So even the estimable-curvature subspaces are
not where preconditioning pays off, which is the I-series result (§1b) seen from the mechanism
side.

## 3. p/N is real one level down (KFAC vs EK-FAC) — but the effect does not survive a fresh draw

KFAC's Kronecker factors come from **~92k training frames**; EK-FAC keeps those eigenvectors and
re-estimates the eigenvalues from the **135 demo gradients**. That single swap takes `embed`/C1
from **0.320 to 0.017**.

`embed` is also the only group where preconditioning helps at all, on both dev targets, with a
monotone dose-response in the damping (C5: 0.615 → 0.613 → 0.498 → 0.319 → 0.298 as λ_rel goes
1e−4 → 1e4). Elsewhere KFAC is neutral (`head`, `last_block`) or harmful (ALL on C1:
0.509 → 0.353).

On dev this looked like a real mechanism: the binding sample size is **the one the curvature is
estimated from**, not the width of Φ; 135 demos is too few whether as a 135×135 Gram or as EK-FAC
eigenvalues, 92k frames is enough. H3 (KFAC-embed/C5) then *passed* the H-series at matched depth
(0.563, p = 0.0032).

**But the I-series retires it.** On the third, clean draw KFAC-on-embed scored 0.428 and lost to
GradDot (0.490); the H-series pass and the B8 "100%" were both selection-mask artefacts. The
KFAC/EK-FAC *contrast* is still a genuine within-sample observation about where the estimable
signal lives (B9 confirms it is the early/input subspaces), but "KFAC-on-embed beats GradDot" is
not a claim that holds out of sample. The honest status of the mechanism is: **a plausible account
of the dev/H behaviour that a fresh draw does not support.**

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

### B10 completes the set: the rollout-visited functional is the WORST of the four

B2's fourth functional — weight held-out frames by the density of states the policy actually
visits when rolled out — was blocked in pass 3 (no sim stack). It is now installed (BLOCKERS #15)
and run (`b10_rollout.py`): roll the 3 regenerated members out on the 70 held-out tasks, build a
per-cluster kernel-density weighting over the visited-state cloud. Only the rollouts cost GPU
(~4 min); the outcome is free from the stored per-frame losses. The weighting clears the gate
(rollout ceilings 0.89–0.98). GradDot on it, vs plain:

| | plain | rollout |
|---|---|---|
| dev C5 | 0.374 | **−0.284** |
| I-series C5 | 0.509 | **0.002** |
| dev C1 | 0.475 | 0.088 |
| I-series C1 | −0.095 | 0.160 |

It is *worse* than plain everywhere the plain functional is resolvable — most sharply on C5, where
it flips negative. The reason is legible in the rollout data: **the policy scored 0 successes on
C5 and C6**, so the visited cloud there is entirely failed-trajectory states. Aiming attribution
at where the broken policy goes is worse than aiming it uniformly. (Caveat: C8/C9 rollouts failed
on a file-descriptor leak and fell back to uniform; C1–C7 use real clouds, so the C1/C5 result is
sound.)

**All four redesigned functionals — transport, ens_var, fail_div, rollout — fail to beat plain out
of sample.** The one that looked best on dev (`fail_div`, +0.161 C5) collapsed to +0.010 on the
I-series; the newest one is actively harmful. The target functional was the largest single-estimator
lever on dev and, like every other dev lever this pass, it does not survive.

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
failed on fresh masks at both depths (0.410 at d6, 0.340 at d10).** Win condition 2 is therefore
*not* achieved: what survives is "TracIn's density knob helps on the diffusion arm", tested on one
mask draw. C5 does not unify (best diffusion cell 0.458, p = 0.031).

Incidental but useful: **the diffusion Gram regenerates almost exactly where the BC Gram does
not** — K rank-correlations 0.96–0.998 and `G_rel_fro` 0.05–0.33, against BC's 0.02–0.85 and
0.879. The likely causes are both about the objective, not the training: the diffusion side uses
a frozen (t, ε) bank that removes sampling noise the BC side has no analogue of, and the
ε-prediction MSE is bounded where the GMM NLL's per-mode σ can collapse.

---

## Verdict on the three win conditions (after three mask draws)

**1. Generality (≥3 of {C1,C2,C4,C7,C9}): NOT achieved.** Out of sample, the only estimator that
beats GradDot at all is the datamodel, and it does so on C5 (strongly) and C2 (weakly) — not ≥3
targets. Every gradient estimator has zero out-of-sample passes. The framing is also suspect: the
absolute per-target bar is unresolvable at n = 24 (§1), so "generality" measured this way is partly
a measurement artefact.

**2. Unification: NOT achieved.** Reached on dev and on the diffusion arm; the BC half is H2,
which failed confirmation at both seed depths.

**3. Mechanism: RETRACTED, and its replacement is NOT confirmed either.** Pass 2's k\*/(p/N) claim
falls to B6/B9 (k\* tracks depth, not dimension). The replacement — curvature is estimable only
from the 92k-frame factors, so KFAC-on-embed should beat GradDot — passed dev and the H-series but
**failed the I-series** (0.428 vs GradDot 0.490). The KFAC/EK-FAC contrast remains a real
within-sample description of *where* the estimable signal lives (B9: early/input subspaces), but
the estimator claim built on it does not survive a clean draw.

## What this pass actually established

1. **Out-of-sample validation is the whole game at n = 24, and this pass demonstrates it three
   times over.** Dev found ~4 improvements over GradDot; the H-series killed most; B8's pooled
   paired test appeared to rescue two; the independent I-series killed KFAC-on-embed and left only
   the datamodel standing. Every gradient-side "win" was mask-draw overfitting — including ones
   that survived a bootstrap over the *selection* masks (B8), which is the subtle trap: resampling
   the masks an estimator was tuned on does not test generalisation.
2. **The datamodel is the only durable result** — it beats GradDot out of sample on the I-series
   (C5 ratio 0.882, Δρ +0.358; C2 Δρ +0.598) and is the only hypothesis confirmed on more than one
   draw (H1). It is also the only estimator that consumes outcomes, so it plays by different input
   rules. Even it does not clear the strict Bonferroni-3 paired bar on a single fresh draw.
3. **The absolute LDS ratio is unresolvable at n = 24 (sampling sd ~0.15); the paired comparison
   on shared masks is the right statistic** (B8) — but a paired win on the *selection* masks still
   has to survive a fresh draw (I-series). Report both: paired, and out-of-sample.
4. **k\* tracks DEPTH, not dimension or p/N** (B6 retraction + B9): the six equal-size blocks give
   k\* = 6,5,1,1,1,1, every structured subspace beats its random control, and k\* does not predict
   where inverting helps.
5. **The action head is the model to four decimals**: paired sd 0.005 between head-only and
   full-model GradDot (an identity, not a mask-dependent claim, so it needs no fresh draw).
6. **B5 is closed**, saving the 4–5 GPU-h it was budgeted.

## Corrections I made to my own write-up, in order

Three, each caught before the work was called done:

- **The depth-6/depth-10 confound.** Ran campaign B at 6 seeds, compared against depth-10 dev
  numbers. Confounded the confirmatory family with protocol (read 1/5; matched-depth answer 2/5).
  Fixed by extending B to 10 seeds.
- **BLOCKERS #17's attribution.** First version blamed a 0.14 C1 drop on the mask draw; at matched
  depth the mask sets differ by 0.025 and the rest was seed depth. Conclusion stronger, cause wrong.
- **The B8 "two estimators beat GradDot" claim (commit 34a3c83, and my report to the user).** Based
  on the pooled G+H masks, which the estimators had been selected on. The I-series refuted the
  KFAC-embed half: it loses to GradDot on a clean draw. The datamodel half survived. This is the
  most important correction — a paired bootstrap over selection masks is not out-of-sample, and I
  should have flagged that when I first reported B8 rather than after the third draw showed it.

## GPU ledger

Concurrency makes "GPU hours" ambiguous; `if_repair/gpu_ledger.py` reports all three readings and
`results/gpu_ledger_pass3.csv` has the per-stage breakdown.

| stage | jobs | job-h | solo-h |
|---|---|---|---|
| campaign A (24 archived masks × 10 seeds, per-frame losses) | 240 | 10.79 | 5.80 |
| campaign B (24 fresh H-series × 10 seeds) | 240 | 8.85 | 5.80 |
| campaign C (8 masks × 3 inits × 3 orders) | 72 | 2.60 | 1.74 |
| campaign I (24 fresh I-series × 10 seeds, out-of-sample) | 240 | ~8.8 | 5.80 |
| diffusion ensemble regeneration | 5 | 0.66 | 0.46 |
| B7 / B4 / B3 / B6 / B2 / B9 caches | ~70 | 1.14 | 1.2 |
| **pass 3 total** | | **~32.8** | **~20.9** |

`job_h` sums each job's own wall time and so double-counts contention (3 concurrent workers
stretch an 87 s retrain to ~137–160 s). `solo_h` is what the same jobs would have cost run one at a
time — the honest measure of work done, and the one comparable to the pass-1/2 ledger. With pass
1+2's 0.15 h the project total is **~21 solo-h**. The 12 h cap was explicitly lifted by the user
("no need to have the 12h cap"); the three retrain campaigns (A/B/I, ~17 solo-h between them) are
the bulk, and each earned its place — A gives every future functional a free outcome, B and I are
the two independent confirmation draws without which the KFAC-embed result would still be reported
as a win. Recorded in full; `results/gpu_ledger_pass3.csv` regenerates it.

---

# Passes 4-6: the first gradient-side estimators to beat GradDot out of sample

> One-line status: passes 1-3 found NO gradient estimator that beats GradDot out of sample (only
> the outcome-consuming datamodel did). Passes 4-6 find that a **leverage / self-influence
> correction** of GradDot's raw kernel beats GradDot out of sample **in direction, consistently
> across six mask draws** -- the first gradient estimators in the project to do so. But the
> magnitude is small: across the three preregistered fresh-draw confirmations (J, K, L), only C5 on
> campaign J cleared the absolute bar, and it did NOT replicate on campaign L (ratio 0.533 -> 0.227).
> No gradient estimator robustly clears the bar across draws. Generality is a property of the
> leverage FAMILY at best (per-target Phi), and a single fixed multi-view estimator fails to capture
> it. The binding limitation is n=24, not estimator design.

Pass 4 attacked demo attribution from four untried directions; pass 5 (planned with a Fable model,
executed with Opus) unified the winners into a two-parameter family and asked whether one member
generalizes; pass 6 tested a fixed multi-view estimator as the generality shot. The arc:

## 1. The three pass-4 winners are one family (P1)
Every gradient estimator that beats GradDot out of sample is a corner of
```
S_m(lam_rel, beta)[:, t] = diag(G_m)^(-beta) . (G_m + lam I)^(-1) . K_m[:, t]     lam = lam_rel * mean(diag G_m)
```
- (inf, 0) = GradDot ; (inf, 1) = RelatIF (W4) ; (0.1, 0) = TRAK on head-Phi (W3).
- The exact frozen-trunk surrogate-LOO (W2) is the nonlinear exact limit of the same idea: it
  computes the true leave-one-demo-out effect on the head's ridge fit by downdating the 512x512
  normal equations, and beats GradDot on C5 (Delta +0.22) but not C1/C2 -- so linearization error
  is real only on C5.

beta=1 (self-influence normalization) is the load-bearing ingredient; a mild Gram inversion
(lam_rel 0.3-3) adds breadth. Against the canonical GradDot_dmean bar (BLOCKERS #1/#23 -- never the
weaker unit-L2 GradDot, even when the challenger aggregates by unit-L2), the family beats GradDot
out of sample, sign-consistent across all three dev draws G/H/I, on FIVE targets with per-target
configs: C2 (+0.28), C4 (+0.15), C5 (+0.24), C7 (+0.32), C8 (+0.39). The winning Phi is
target-specific: C5/C7 live on the cached E=20 FULL-model Gram, C2/C8 on the regen E=5 HEAD Gram.

## 2. No single estimator generalizes (P1, P6.1)
Against the strict dmean bar, the best SINGLE config (regenE5-head/dmean/lam0.3/beta1) reaches only
2 targets (C2, C8). The "C2,C3,C8 generality" first observed held only against the weaker unit-L2
baseline and was retracted. Pass 6's fixed multi-view estimator (z/rank-average of the head-leverage,
cached-leverage, and exact-LOO views, judged against the MAX of both GradDot bars) qualifies on just
1 target (C7): averaging DILUTES, because the views help different Phi-specific targets, so for any
one target most views are noise. The P2 C5 ensemble works only because both its views target C5.
**Generality is a family property (per-target Phi selection), not a single-estimator property.**

## 3. The C5 winners are complementary (P2)
RelatIF (self-influence norm, cached Gram) and the exact surrogate-LOO (regen head) rank-correlate
only 0.20 over the 135 demos -- they see different aspects of C5's structure. Their z-score ensemble
reaches C5 pooled Delta +0.298 (p=0.0018), the best C5 number in the project. (TRAK-head and the
head-leverage config rank-correlate 0.81 -- same mechanism, as expected.)

## 4. Leverage correction complements the datamodel (P4)
Per-target leverage responsiveness correlates with the datamodel's per-target LDS (Spearman +0.66):
both exploit learnable demo structure. But the leverage family also wins on C7 and C8, where the
datamodel COLLAPSES to a constant (LDS NaN, BLOCKERS #8) -- so a zero-outcome gradient estimator
reaches targets the outcome-based datamodel cannot. The two classes are complementary, not the
datamodel strictly dominating. The self-influence-contamination diagnostic corr(|K[:,t]|,diagG) is
weak; datamodel-LDS is the best (imperfect) outcome-free predictor of responsiveness.

## 5. Out-of-sample confirmation
**Campaign J (pass 4, seed 20260723, PREREG_J frozen at zero runs).** Scored once:
- **RelatIF/C5: lds 0.516 vs GradDot 0.175, ratio 0.533, ABSOLUTE PASS (p=0.005)** -- the first
  gradient estimator to clear the half-ceiling bar out of sample. Paired Delta_rho +0.341 but the
  one-sided bootstrap p (0.073) narrowly misses 0.05 at n=24.
- surrogate-LOO/C5: lds 0.464, ratio 0.479 (just under 0.5), paired Delta +0.340 -- corroborates.
- TRAK-head/C2: does NOT confirm; the C2 outcome is negative for GradDot too on the J masks, so the
  draw cannot adjudicate C2 (BLOCKERS #17, the n=24 single-draw resolution limit).

**Campaign K (pass 5, seed 20260724, PREREG_K frozen at zero runs).** Scored once: **0/3 confirm.**
- leverage-head/C2: lds -0.166 vs GradDot -0.162 -- C2 outcome negative for both estimators on K
  too (unresolvable, as on J).
- leverage-head/C8: lds 0.420 vs GradDot 0.471, paired -0.051 -- the dev C8 win (+0.387) did NOT
  replicate; GradDot itself is strong on C8/K.
- leverage-cachedE20/C7: lds 0.416 vs GradDot 0.130, paired +0.286 (p=0.195) -- right sign, misses
  both bars at n=24.
So the pass-5 leverage FAMILY dev wins (5 targets) were largely mask-draw overfitting: on a fresh
draw C2/C8 do not survive, and only C7 keeps the right direction (underpowered). This is the same
n=24 lesson as pass 3 (BLOCKERS #17/#20), and it sharpens the verdict below.

**Campaign L (pass 6 capstone, seed 20260725, PREREG_L frozen at zero runs).** Second fresh draw
for the C5 self-influence win. Scored once: **0/3.**
- RelatIF/C5 and the C5 ensemble: lds 0.217 vs GradDot 0.143, ratio 0.227, paired +0.075 (p=0.36).
- surrogate-LOO/C5: lds 0.127 vs GradDot 0.212 -- loses to GradDot on this draw.
Campaign J gave RelatIF/C5 ratio 0.533 and paired +0.34; campaign L gives ratio 0.227 and paired
+0.075 for the SAME estimator. **So J's absolute-bar pass was substantially a favourable draw.**
Across two fresh draws the C5 effect is real in DIRECTION (RelatIF >= GradDot on both) but small and
draw-dependent; it does not robustly clear the half-ceiling bar.

## Negatives (clean, some preregisterable)
- **W1 unlearning-LOO:** ascent-unlearn ~= GradDot, finetune-forget dead. Killed on a 1-member pilot.
- **W5 mid-training GradDot:** single-checkpoint LDS swings +-1.0 between adjacent 400-step ckpts;
  pass-3 B4's step-6400 "signal" was one lucky draw. No mid-training effect.
- **W6 hybrid datamodel:** a gradient prior does NOT let the datamodel hit its 24-mask LDS with 12
  masks (win-condition-2 fails). It DOES regularize the datamodel at low mask counts (+0.43 on C5@12).
- **P6.1 multi-view:** a single fixed estimator does not generalize (see #2).

## What passes 4-6 established
1. A leverage / self-influence correction of GradDot (RelatIF, K/G_dd) is the first gradient-side
   estimator in the project to beat GradDot out of sample IN DIRECTION -- RelatIF >= GradDot on C5
   on every fresh draw (J +0.34, L +0.075) and leverage-cachedE20 >= GradDot on C7 (K +0.29). This
   directional consistency across six draws is real and new.
2. But no gradient estimator ROBUSTLY clears the absolute bar. RelatIF/C5 cleared it on campaign J
   (ratio 0.533) yet fell to 0.227 on campaign L -- J was a favourable draw. Dev over-promises at
   n=24: the family beat GradDot on 5 dev targets, but on fresh draws C5 is small/draw-dependent,
   C7 underpowered, and C2/C8 do not replicate (K). Same mask-draw overfitting caught every pass.
3. "Generality" is not achieved and is a FAMILY property at best: the effect is target-specific in
   both Phi and lambda, and a fixed multi-view estimator (P6.1) fails to capture it (1 target).
4. The C5 win rests on two near-orthogonal mechanisms (self-influence norm + exact frozen-trunk LOO,
   rank-corr 0.20), whose ensemble is the strongest single attributor on dev (+0.298) -- but campaign
   L showed even the ensemble does not robustly replicate (ratio 0.227, paired +0.075).
5. Leverage gradients and the outcome-consuming datamodel are complementary (P4): leverage wins on
   C7/C8 where the datamodel collapses, though neither C7 nor C8 survived their fresh draw.
6. The n=24 single-draw paired bar remains THE binding limitation: even the confirmed C5 effect
   (+0.34 over GradDot on J) clears only the absolute bar, missing the strict paired-p<0.05. Real
   progress on this corpus needs more masks per draw, not more estimators.
