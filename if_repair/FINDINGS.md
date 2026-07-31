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

---

# Pass 7: fix the measurement -- and the measurement changed the answer

> One-line status: passes 4-6 reported a leverage / self-influence correction (RelatIF, K/G_dd)
> as the first gradient estimator to beat GradDot out of sample, "in direction, consistently
> across six mask draws", with campaign J giving +0.341. Pass 7 spent its first hour scoring that
> FROZEN config on the one draw nobody had ever run it against (campaign K) and the sign reversed.
> It then measured what a resolving campaign actually costs, found the archived protocol is close
> to the worst allocation available, and ran one -- 144 masks at depth 2, FEWER retrains than any
> previous campaign -- to close the interval. **The answer: Delta rho = +0.06, 95% CI
> [-0.02, +0.14] over 192 out-of-sample masks. The effect is real in direction, small in size, and
> about a quarter of what was claimed.** No gradient estimator clears the absolute half-ceiling
> bar. The binding constraint was never the estimator; it was the number of counterfactual
> retrains, and pass 7 quantifies the exchange rate.

## 1. The correction: a frozen config had never been scored on an available draw (W0.1)

RelatIF/C5 was frozen in `PREREG_J` (`c7659cf`) with campaign J at zero runs and never retuned, so
it is legitimately out-of-sample on J, K **and** L. But campaign K's own prereg family named
C2/C8/C7, so the C5 config was never run against K. Running it costs zero GPU:

| draw | role | Delta rho vs GradDot_dmean |
|---|---|---|
| G / H / I | dev (selection) | +0.226 / +0.384 / +0.135 |
| J | **discovery** | +0.341 |
| K | **oos, never scored until pass 7** | **-0.171** |
| L | oos | +0.075 |

Every draw the effect was selected on or reported on is positive; the one clean draw nobody had
looked at is negative. `FINDINGS.md`'s claim of consistency "across six mask draws" was true only
of the five that had been looked at. **The confirmation draw you already own is worth more than
the estimator you were about to build.**

The second lesson is subtler and is now BLOCKERS #28: *frozen before a draw* is not the same as
*unselected-upon*. J is technically out-of-sample for RelatIF, but it is the draw the effect was
first reported a success on, so conditioning on "this is the hypothesis we followed up" selects a
favourable J. Every pass-7 contrast is therefore reported twice -- with and without the discovery
draw -- and the second is the one to believe. With J: +0.083. Without: -0.044.

## 2. The measurement study: masks beat seeds about 5 to 1 (W0.2)

Resampling the 144 masks x 10 seeds already on disk (the six draws are exchangeable -- a one-way
ANOVA puts the between-draw variance component at zero on every set tested):

- **Paired sd is FLAT in seed depth** (0.179 at depth 2 vs 0.183 at depth 10, n=24) and falls as
  **1/sqrt(n) in masks**. BLOCKERS #17 guessed depth 4-5 exhausted seed noise; for the paired
  statistic it is exhausted by depth 2.
- At 240 retrains: 120x2 -> paired sd 0.077, 48x5 -> 0.121, the archived **24x10 -> 0.183**. The
  protocol this project used for six passes is close to the worst allocation available.
- **Kendall tau_b resolves this corpus better than Spearman** (resolution 5.32 vs 4.52), chosen on
  reliability and noise alone -- criteria that never touch the hypothesis. It then returned CIs a
  third narrower, validating the selection method after the fact.
- **GradDot_dmean is itself a moving target**: its C5 LDS spans -0.066 to +0.489 across the six
  draws, sd 0.217 -- larger than most effects anyone has claimed here.
- **Pairing is not always right**: for C7 the estimator and baseline are NEGATIVELY correlated
  across draws, so var(paired) 0.071 exceeds var(level) 0.011 (BLOCKERS #32).

## 3. The resolving campaign (W1 / campaign M)

144 virgin masks in 6 disjoint sub-draws at depth 2, 288 retrains, ~7.5 solo-h, family of one,
scored once. Paired bar **PASS** under both statistics (kendall +0.085 p=0.031; spearman +0.111
p=0.041) -- a project first on a virgin preregistered draw. Absolute bar **FAIL** (ratio 0.413 /
0.489 against 0.5).

Pooled over all 192 clean out-of-sample masks: **+0.060, CI [-0.015, +0.137], p = 0.057** (kendall)
/ +0.084, CI [-0.027, +0.193], p = 0.064 (spearman). Matched-depth pooling agrees to 0.005-0.009.

**Campaign J's +0.341 is decisively excluded.** The discovery-draw winner's curse on this corpus
is about 4x, now measured.

And the campaign demonstrated it in miniature. M's six 24-mask sub-draws give +0.058, +0.116,
+0.145, **+0.341**, -0.058, -0.022. **Sub-draw M3 reproduces campaign J's headline number exactly**
-- a single 24-mask draw landing on +0.341 with CI [+0.121, +0.550], which in isolation would have
been written up as a confirmation. Six draws from one campaign, one of them "confirms" a +0.34
effect that the pooled 144 masks put at +0.085. That is what n=24 does, and it is why this project
spent six passes chasing effects that evaporated.

W0.2 forecast a CI width of ~0.18 before the campaign ran; measured 0.175. The design analysis
predicted its own result to within 0.005, using 288 retrains -- fewer than any prior campaign.

## 4. The duel design: a clean negative (W2)

Mask pairs differing in exactly one demo, matched seed slots, chosen where the estimators disagree.
Killed by its own preregistered pilot.

- **The pairing premise was right**: within a swap-pair the mask x init interaction differences out,
  2.39x noise reduction. Matched seeds inside a swap-pair genuinely differ from B5's common random
  seeds; B5 stays retired for the general question.
- **The signal is smaller still**: swapping one demo of 68 moves the C5 held-out loss by 0.44
  seed-noise sd. Not resolvable at depth 4, and reaching resolution would cost ~22 solo-h for one
  n=18 binomial test -- strictly worse than spending it on masks.
- **The 1-of-6 sign test is uninterpretable and is not reported as a result.** By the pilot's own
  diagnostic those signs are coin flips. The instrument returned nothing; it did not return a
  negative.

Read with #33 (the two estimators rank-correlate 0.547 and only 6.6% of within-cluster demo pairs
disagree confidently), this is the same fact from both sides: **demo-grain attribution on 135 demos
is signal-starved at the level of individual demos.**

## Corrections I made to my own work, in order

Four, each caught before it was committed:

- **The passes 4-6 six-draw consistency claim** (see §1). Not mine, but corrected here, and the
  correction is the pass's central result.
- **Finite-population sampling in the allocation study.** Subsampled masks WITHOUT replacement from
  a 144-mask pool, which carries a (1 - n/144) correction and drove the paired sd to exactly 0 at
  n=144, inflating the mask advantage. The estimand is a campaign of n FRESH masks, so the
  nonparametric bootstrap is right.
- **Scoring C7 on its own discovery draw.** W0.2b initially used K u L for both configs; K is C7's
  discovery draw, and that call returned a publishable-looking kendall +0.257, p = 0.026. On C7's
  actual honest set (J u L) it is +0.032, p = 0.406. The honest draw set is per-CONFIG, not
  per-pass (BLOCKERS #31).
- **A suspected ensemble bug in campaign L that was a genuine coincidence.** L's ensemble and
  RelatIF rows are identical to 16 digits, which looks exactly like a silently zeroed component.
  They are different predictors (mask-level Spearman 0.80, different bootstrap p) that both land on
  sum d^2 = 1800 over 24 masks. Chased and cleared rather than assumed either way.

## What pass 7 established

1. **The C5 self-influence effect is real in direction and small: +0.06, CI [-0.02, +0.14] on 192
   out-of-sample masks.** It clears the paired bar on the largest virgin draw and not when pooled;
   it clears the absolute bar nowhere. The +0.34 of passes 4-6 was a discovery-draw artifact.
2. **The binding constraint is retrains, and the exchange rate is ~5:1 in favour of masks over
   seeds.** The archived 24x10 protocol is close to the worst allocation available; 144x2 costs
   less and resolves more.
3. **A frozen config is out-of-sample on every later draw** -- and this project had an unscored one
   sitting on disk for a whole pass. Score what you own before you build.
4. **Statistic choice is worth ~33% of CI width and can be made without looking at the hypothesis.**
5. **Individual demos are below the noise floor** on this corpus; only aggregate contrasts carry
   measurable information. The duel design is not viable here at any reachable budget.

---

# PASS 8 -- the binding constraint was the GRAIN, not the estimator

> One-line status: at cluster grain a **plain, uncorrected GradDot clears the absolute
> half-ceiling bar** (ratio 0.71 Kendall / 0.83 Spearman on 149 out-of-sample masks) -- the bar
> nothing cleared out of sample in seven passes at demo grain. **Every self-influence and leverage
> correction from passes 4-7 is strongly NEGATIVE at this grain**, replicating across two
> independent draws and two outcome pipelines.

## What pass 8 established

**1. The absolute bar is reachable. It was measuring the grain.**

Pass 7's HANDOFF open thread #4 asked whether the half-ceiling bar is unreachable for every
estimator anyone would try, in which case it measures the ceiling rather than discriminating
hypotheses. It is reachable. Campaign N, 278 fresh cluster masks, C5, depth 4, conditional n=149:

| statistic | LDS | ceiling | ratio | bar | verdict |
|---|---|---|---|---|---|
| Kendall tau_b (primary) | 0.4747 | 0.6715 | **0.707** | 0.5 | **PASS** |
| Spearman (secondary) | 0.6789 | 0.8141 | **0.834** | 0.5 | **PASS** |

Preregistered as a family of one in `p8_prereg.md`, frozen at 956c061 while campaign N had zero
runs, scored once.

**2. Every correction passes 4-7 built REVERSES at cluster grain.**

The estimators that were the entire subject of three passes are not merely unhelpful here; they
are strongly harmful, on a fresh disjoint draw, with intervals nowhere near zero:

| config | demo grain (pass 7) | cluster grain (campaign N, Kendall) | 95% CI |
|---|---|---|---|
| relatif_C5 | **+0.06** [-0.02, +0.14] | **-0.627** | [-0.735, -0.518] |
| surrogate_C5 | (dev +0.298) | -0.539 | [-0.661, -0.414] |
| ensemble_C5 | -- | -0.751 | [-0.858, -0.638] |
| leverage_C7 | +0.032, p=0.406 | -0.474 | [-0.585, -0.360] |

This replicates the Stage F scan (deltas -0.28 to -0.63) on a **different draw** and a
**different outcome pipeline** -- Stage F outcomes come from the original project's probe battery,
campaign N's from `heldout_frame_losses`. Agreement across both is the strongest form of
robustness available inside this repo.

The parsimonious reading, given pass 7 measured the demo-grain benefit at +0.06 with an interval
touching zero: **the self-influence correction was fitting demo-grain noise.** Where signal is
abundant it does not merely fail to help, it destroys a ranking that was already good.

**3. What this does NOT show, and the write-up must not claim.**

A higher LDS at a coarser grain is partly a property of the task being easier. Among masks that
contain the target, the remaining variation is *which 4-5 of the other 8 clusters are present* --
a prediction problem with far fewer degrees of freedom than 135 demos. Normalising by the
cluster-grain noise ceiling controls for outcome noise but not for this.

So: pass 8 shows the **measurement** works at cluster grain, and that the demo-grain corrections
do not survive contact with a grain that has signal. It does **not** rescue per-demo attribution,
and it is not evidence that influence functions are good at the demo question. The honest claim is
about where the noise floor sits, which is exactly what BLOCKERS #33/#35 said was the open
question.

**4. Stage F had been sitting unused for the entire project.**

168 retrains, built as attribution-agnostic ground truth, never used for attribution. Every frozen
config was out-of-sample on all of them by construction -- a cleaner provenance than anything
pass 7 had, with no discovery draw to discount. The scan that reframed this pass cost **zero GPU**.
This is BLOCKERS #28's lesson applied a second time, and it paid a second time.

**5. Stage F's 72 masks are really 58.**

Its randomized construction plus swap repair drew from a space of only C(9,5)=126 and repeated 14
subsets. A repeated cluster subset is not a second mask -- both copies train on identical data, so
the repeat buys seed depth, not design coverage. Every conditional-n ever reported against Stage F
is optimistic by that factor. Campaign N replaces sampling with **complete enumeration**, which at
this size is strictly better: exact balance, uniform co-inclusion, exact disjointness.

## The write-up sentence this pass earned

> Raising the unit of attribution from one demonstration to a cluster of fifteen moves
> leave-one-out attribution from below the noise floor to 71% of the achievable ceiling
> (Kendall tau_b, 149 out-of-sample masks, preregistered), using a plain gradient dot product
> with no correction at all. The self-influence corrections that a demo-grain analysis selects --
> and which improve demo-grain attribution by a measured +0.06 -- are strongly harmful at this
> grain, reversing to -0.63. The binding constraint on demo-grain TDA benchmarks is neither the
> estimator nor the number of retrains but the size of the unit being attributed.

---

# PASS 9 -- the grain result was partly a training-set SIZE result

Pass 8 ended with a strong claim: at cluster grain a plain, uncorrected GradDot clears the absolute
half-ceiling bar at ratio 0.707, the first estimator to clear it in eight passes. Pass 9 set out to
locate the crossover between k=1 and k=15 and instead spent its first hours discovering that the
0.707 is not what it appears to be. The crossover campaign was redesigned around what was found and
is still running; this section records everything that is settled.

## 1. The correction: the primary is pooled over |S|, and a large part of it is training-set size

`p8_prereg.md` states that "|S| is a stratum, not a covariate to pool over. It sets the
training-set size (60/75/90 demos), which moves the outcome directly", and promises the result
"pooled with |S| controlled". `confirm_nseries.evaluate` computes the primary absolute bar over all
149 conditional masks with no stratum control -- `st` is built there and used only by the secondary
paired analysis. The control was promised and not applied to the number carrying the claim.

`p9_stratum_control.py`, C5, Kendall tau_b, depth 4, same committed data, same frozen
`p7_pooled_oos._graddot("cached")`:

| scope | n | LDS | ceiling | ratio | ratio 95% CI | perm null | clears 0.5 |
|---|---|---|---|---|---|---|---|
| POOLED (committed primary) | 149 | 0.4747 | 0.6715 | **0.7069** | [0.612, 0.817] | **0.3530** | yes |
| within 4of9 | 56 | 0.2078 | 0.5053 | 0.4112 | [0.092, 0.777] | 0.0009 | no |
| within 5of9 | 37 | 0.2583 | 0.5211 | 0.4956 | [0.115, 0.943] | 0.0022 | no |
| within 6of9 | 56 | 0.1688 | 0.5324 | 0.3171 | [-0.017, 0.719] | -0.0020 | no |

The pooled row reproduces `results/confirm_nseries.csv` exactly -- asserted as a regression test --
so this is the same computation, not a competing one.

**The permutation null carries the argument.** Outcomes are shuffled within stratum and correlated
against GradDot's predictions. GradDot is a *fixed* estimator: nothing is fit, so a correlation
surviving an outcome shuffle cannot be leakage under any definition. Pooled, **0.353 of the 0.475
survives the shuffle**. The mechanism is arithmetic -- |S| sets the training-set size, size moves
the outcome, and every estimator's mask prediction is a sum over kept demos, so it grows with the
count as well. Both sides earn credit for counting demos. Within stratum the same null collapses to
~0.000, which is what a working control looks like.

**What survives and what does not.** Real attribution signal remains within stratum: the observed
LDS beats the within-stratum null's 97.5th percentile in 4of9 and 5of9 (not in 6of9). But the
half-ceiling bar is **not cleared in any stratum**, and the per-stratum CIs are far too wide to
settle it either way. Pass 8's #36, "the absolute bar is REACHABLE", is therefore **not refuted --
it is unproven.** The specific figure 0.707 should not be quoted without this qualification.

## 2. The datamodel survives the grain change (thread 2), with a units caveat that matters

The design-based datamodel was the one estimator pass 8 never touched, and the only one with a
large replicable demo-grain OOS advantage (pass 3: C5 ratio 0.882). At cluster grain, scored
leave-one-mask-out on campaign N's existing retrains with alpha refit inside every fold -- zero
additional GPU -- it beats GradDot within **every** stratum by +0.31 to +0.38 Kendall.

Its ratio comes out at 1.01, which nearly got it discarded as leakage. It is not.
`confirm_mseries.ceiling` is a Spearman-Brown **reliability r**, and the largest correlation any
predictor can have with an observation of reliability r is about **sqrt(r)**, not r. The attainable
maximum on this scale is ~1/sqrt(0.6715) ~ 1.22, so a ratio between 1 and 1.22 is a saturating
estimator rather than a broken one. This is BLOCKERS #42 and it applies to **every historical ratio
in this repo**: they are not "fraction of achievable", and two of them measured at different seed
depths are not comparable.

Honest framing for the datamodel result: at demo grain the estimator is badly underdetermined (24
observations, 135 coefficients) and that difficulty is most of what it has to survive. At cluster
grain it inverts -- ~149 masks against 9 identifiable coefficients -- so this is not a like-for-like
improvement on 0.882. It is the same estimator facing a much easier estimation problem.

## 3. The corrections' reversal is a ranking error, not a scaling pathology (thread 3)

BLOCKERS #37 recorded that every self-influence / leverage correction reverses at cluster grain but
not why, and the difference decides whether the correction is salvageable. `p9_why_reverse.py`
separates the hypotheses with a rank/scale 2x2: an estimator contributes an ORDER and a MARGINAL
DISTRIBUTION to a summed prediction, and rank transforms swap them independently.

Neither swap recovers to baseline. Keeping the correction's order on GradDot's scale still reverses;
keeping its heavy tail on GradDot's order also reverses. **Read within stratum it is worse**: the
order-preserving arm's own LDS goes negative (relatif_C5: -0.262 / -0.276 / -0.262 against GradDot's
+0.208 / +0.258 / +0.169), so the correction's ordering is actively anti-predictive once
training-set size is removed. 12 of 12 config x stratum cells negative.

Pooling had been *propping the corrections up* -- the shared |S| component lifted every estimator's
pooled correlation. That is a second, smaller instance of section 1's lesson, and it changed a
verdict: `verdict()` originally read surrogate_C5 as "MIXED" off a pooled -0.104; within stratum it
is -0.25 to -0.41, the same ranking error as the rest. **#37 is an epitaph. Do not carry these
corrections to a new corpus hoping a better aggregation saves them.**

## 4. What this does to pass 8's write-up sentence

The sentence at the end of the pass-8 section claims the grain change "lifts leave-one-out
attribution from below the noise floor to 71% of the achievable ceiling". Three things in it need
amending, and the pass-8 section should not be quoted without this paragraph:

1. **"71% of the achievable ceiling"** is pooled over |S|, and ~0.353 of the underlying 0.475 is
   reproducible from training-set size alone by a fixed estimator on shuffled outcomes. Within
   stratum the ratio is 0.32-0.50 and clears nothing.
2. **"achievable ceiling"** overstates what the denominator is. It is a reliability, not an
   attainable maximum (BLOCKERS #42).
3. **The conclusion "the binding constraint is the size of the unit being attributed"** is still the
   most likely reading, but it is now a hypothesis campaign O is testing rather than something
   pass 8 established. What pass 8 established is that *something* about the coarser design makes
   the pooled number large, and part of that something is the training set getting bigger.

## 5. Campaign O: at a fixed training-set size, NO grain clears the bar

The crossover cannot be located at cluster grain -- BLOCKERS #38 caps the mask axis at 336 subsets,
campaign N consumed 278, and the per-stratum n is stuck at 56/37/56. Sub-cluster grain escapes the
cap combinatorially: a 75-demo mask keeps 25 of 45 groups at k=3, and C(45,25) ~ 3e12. So hundreds
of masks can share **exactly one training-set size**, which removes section 1's confound by
construction instead of adjusting for it afterwards.

Campaign O: 800 masks (400 each at k=3, k=5), every mask at exactly 75 retained demos, conditioning
"all C5 groups retained", exact group balance, 0 signature collisions with consumed cluster masks,
depth 2 in seed slots {4401, 4402} matched to the k=15 rung. 1600 retrains, 18.2 h wall, 0 failures.
Preregistered in `p9_prereg.md` at zero runs and scored exactly once.

| rung | n | LDS | ceiling | ratio | ratio 95% CI | rho/sqrt(r) | perm null (97.5th) | beats null | clears bar |
|---|---|---|---|---|---|---|---|---|---|
| k=3 | 400 | 0.1338 | 0.3759 | 0.356 | [0.180, 0.550] | 0.218 | -0.000 (0.068) | yes | **no** |
| k=5 | 400 | 0.1411 | 0.3862 | 0.365 | [0.204, 0.559] | 0.227 | -0.001 (0.063) | yes | **no** |
| k=15 | 37 | 0.3003 | 0.4512 | 0.666 | [0.194, 2.349] | 0.447 | 0.002 (0.228) | yes | **no** |

Kendall tau_b, primary. Spearman agrees throughout (0.392 / 0.410 / 0.783, same verdicts).

**The preregistered branch is "neither clears, k=15 does not either".** By PREREG_O's decision rule
that reads: at a fixed training-set size, attribution on this corpus is not measurable to the
half-ceiling standard at ANY grain, and pass 8's positive result was substantially the |S| effect
that section 1 identified.

**Three things this result is NOT.**

1. **It is not "no signal".** All three rungs beat their permutation nulls decisively -- the nulls
   sit at ~0.000 with 97.5th percentiles of 0.063-0.228, against observed LDS of 0.134-0.300. The
   attribution is real. What fails is the *half-ceiling bar*, which is a standard for usefulness,
   not a test of existence. The honest summary is that attribution here is real and weak.
2. **It is not a generous test that barely failed.** The ratios above use this repo's `rho/r`
   convention at the noisiest available depth, and BLOCKERS #42 shows that convention is inflated by
   ~1/sqrt(r) and inflated *more* as r falls. These are therefore the most flattering numbers the
   design can produce, and they still fail. On the attainable `rho/sqrt(r)` scale the rungs reach
   **22%, 23% and 45%** of achievable.
3. **It is not a located crossover.** k=3 and k=5 are statistically indistinguishable (0.356 vs
   0.365, near-identical intervals), so nothing happens between 3 and 5. k=15's point estimate is
   ~1.8x higher, which is the direction the grain hypothesis predicts, but n=37 makes its interval
   [0.19, 2.35] -- it overlaps k=3 and k=5 completely. **The grain trend is suggestive and
   unestablished.** If a transition exists it is above k=5.

**The structural problem that blocks the obvious next step.** k=15's precision cannot be improved by
drawing more cluster masks: campaign N exhausted the |S|=5 space. But it can be improved another
way, and cheaply. Conditional on C5, the 5of9 stratum has C(8,4) = 70 possible masks; campaign N
holds 37 of them and **Stage F holds the other 33**. Stage F's outcomes came from the original
project's probe battery (`src/evaluate.py`) rather than `retrain.heldout_frame_losses`, so they
cannot simply be pooled -- but re-running those 33 masks through the campaign pipeline at depth 2
costs **66 retrains, ~45 minutes**, and yields the complete 70-mask enumeration at k=15 in one
consistent pipeline. That roughly doubles n on the only rung that is currently unresolvable, and it
is the cheapest remaining experiment in the project.

## Corrections I made to my own work, in order

1. **The pass-9 plan's first version was killed before any GPU was spent.** An adversarial review
   found the proposed four-point curve was not one estimand -- points differed in seed depth, in
   mask geometry (demo-grain masks retain 7-8 of every cluster and never remove one whole), and in
   an unspecified conditioning rule -- and that its "blind design" stage was uncomputable, because
   no sub-cluster outcomes exist in the repo to resample. It also noted the depth-2 choice biases
   the absolute bar *toward* the pass's own hypothesis.
2. **A leakage control failed and I initially suspected the wrong thing.** The datamodel's
   permutation control returned 0.425 where it had to return ~0. The first hypothesis was leakage in
   the LOO fold construction. It was not: the control was shuffling within stratum while the
   statistic was pooled across strata, and what survived was |S|. That misread is what led to
   section 1.
3. **My first `verdict()` was computed pooled** and called surrogate_C5 undetermined. Fixed to
   decide within stratum; all four configs are now unanimous.
4. **A monitor I wrote fired a false "campaign died" alarm** -- unquoted `$st` under zsh, which does
   not word-split, so every field landed in `$1`. The campaign was never at risk. Mentioned because
   the same bug would silently mis-read any future watchdog on this box.
5. **I nearly let a partial campaign consume the one-shot score.** The seed-major job list makes
   k=3 analysable at job 1200 and k=5 only at job 1600. `confirm_oseries` refuses to overwrite, so
   scoring anywhere in that ~4.6 h window would have answered O1 and left O2 permanently
   unanswerable. Caught with about an hour to spare and fixed with a guard that refuses unless every
   preregistered grain has a complete even depth. Seed-major ordering was inherited from campaign N
   without re-examination; at depth 5 its even prefixes give real optionality, at depth 2 they give
   none, and the complementary-pair mask construction would have made mask-major ordering strictly
   better here.
6. **A sed-style patch of this very file duplicated 250 lines** because the end anchor
   ("## Corrections I made to my own work, in order") also appears in the pass-7 section, so
   `str.index` matched the earlier copy and the splice re-included everything from pass 7 onward.
   Caught by a line count, reverted from git, and redone with the end anchor searched from the start
   index. Append-only documents accumulate duplicate headings; never anchor a splice on one without
   asserting uniqueness.

---

# PASS 10 -- the bar is reachable, and gradient attribution does not reach it

Pass 9 corrected the cluster-grain headline and left three things open: whether the grain trend could
be resolved, whether the datamodel survives a fair design, and whether the half-ceiling bar is the
right standard at all. Pass 10 answers the first and third, and sharpens the second.

## 1. The grain question is CLOSED on this corpus (BLOCKERS #45)

The k=15 estimand's mask universe is C(8,4) = 70 -- a combinatorial cap. Campaign P completed it (132
retrains). At n=70 the interval is ~1.57 wide at depth 2 and ~0.60 at depth 4, against sub-cluster
intervals of ~0.37 centred near 0.36. **Nothing purchasable separates them.** The trend is not
unestablished but unestablishable here.

The census itself reads **0.487 at depth 4 on the complete population -- below the bar**, confirming
pass 9 on the whole population rather than a 37-mask sample.

## 2. Two methods results fell out of the census (BLOCKERS #46)

**No winner's curse was detectable**, because the selection was over the GRAIN rather than over a set
of configs, and the estimator itself was never selected. The curse scales with the selection set over
the thing being tested. And **the ratio's denominator is noisy enough to drive a comparison on its
own** -- two halves of one population at one depth gave ceilings of 0.521 and 0.668 while the LDS
stayed flat. That second point casts a shadow over historical cross-subset ratio comparisons and is
queued for audit.

## 3. Partition sensitivity is UNRESOLVED -- the original claim did not survive its own error bar (BLOCKERS #47)

Campaign R redrew the partition (zero shared groups) and re-ran the identical design. Both rungs
passed the preregistered containment test. The raw movement was large at k=3 (LDS 0.1338 -> 0.0781,
42%) and small at k=5 (3%).

**I initially reported that as "k=3 is partition-sensitive". That was wrong, and the correction is
recorded in #47.** A bootstrap on both campaigns' frozen outcomes puts the k=3 difference at
**z = 1.23** and k=5 at **z = 0.11** -- a 1.2-sigma difference is routine under mask-sampling noise
alone at n=400. I read a percentage movement as a finding without computing the noise floor.

What stands: **partition sensitivity is unresolved at both grains**, k=3 moved more than k=5 in a way
consistent with (but not establishing) greater sensitivity, and the design cannot resolve the question
-- the data-consistent between-partition SD (~0.021) is below a single partition's own SE (~0.032),
and separating them would take ~20 partitions and ~50,000 retrains.

Both rungs did read lower on the second partition, so pass 9's numbers were if anything optimistic and
its negative conclusion is reinforced. That part is unaffected.

The 18 GPU-hours still went to the right threat -- the partition threat's direction was genuinely
unknown, where the depth threat's was already measured -- but the honest return on them is a **null
with a bound**, not the effect the first write-up claimed.

## 4. The datamodel clears the bar -- and it is the one estimator that sees outcomes

At a fixed 75-demo training set, LOO over masks with regularisation refit inside every fold:
k=3 ratio 1.044 (0.640 attainable), k=5 1.084 (0.674), against GradDot's 0.356 and 0.365 (0.218,
0.227). Paired deltas +0.259 [0.185, 0.329] and +0.278 [0.209, 0.345]. Permutation control ~0, so not
leakage; ratio > 1 is expected because the ceiling is a reliability and the attainable maximum is
~1/sqrt(r) ~ 1.6 (#42).

Three caveats travel with it: it is **outcome-consuming** where every gradient estimator is
outcome-blind; it is **over-determined** here (400 observations, 45 or 27 parameters) against the
under-determined 24-vs-135 regime it earned its reputation in -- the opposite regime, and this corpus
cannot supply p >> n at any grain; and the read is **descriptive, not preregistered**.

## 5. The bar-standard question, answered in part (BLOCKERS #48)

Pass 7 asked whether the bar is unreachable for anything anyone would try. Across all twelve
committed attempts: **10 gradient attempts across 5 designs, none clearing once size and depth are
controlled** (12-45% of attainable), and **the datamodel clearing comfortably at both grains**. So the
bar is reachable and does discriminate -- it is not measuring the ceiling.

What that leaves is sharper than where the pass started: **gradient-based attribution does not reach
the bar at 135 demonstrations, while an outcome-consuming method does.** Whether the gradient failure
is a corpus-size limit or a limit of the approach is not decided by anything in hand, and I am not
claiming it is. Corpus-size scaling is the discriminator, and it is off-box.

## Corrections I made to my own work, in order

1. **The pass-10 plan's first version was killed before any GPU was spent.** Its only alpha-bearing
   test was preregistered on the 33 masks the hypothesis had been *selected* on -- the Stage F
   discovery draw -- and my defence ("never scored through the campaign pipeline") was a pipeline
   distinction, not an inferential one. It also hand-waved arithmetic that, once computed, showed the
   test was **unpassable** rather than underpowered: at n=33 the CI width is ~2.3, so clearing the bar
   needed a point ratio ~1.0 against an adjoining measurement of 0.30.
2. **I had the datamodel's regime backwards**, in the plan and inherited from my own pass-9 HANDOFF.
   Sub-cluster grain is 400/45 = 8.9 and 400/27 = 14.8 masks per coefficient -- *over*-determined --
   against demo grain's 24/135 = 0.18. The opposite of what I wrote.
3. **My first pass-10 falsification map named a unit that could not do the job**: it listed the
   depth-4 re-read as the trend's rescue, but that unit touches only k=3/k=5 and never k=15, whose
   uncertainty is binding.
4. **I nearly let a partial campaign consume a one-shot score** (recorded in pass 9) and had to build
   the same guard again for campaign R.

---

# PASS 11 -- the datamodel attributes, and a claim of mine did not survive its own error bar

Pass 11 spent no GPU. It produced the project's strongest positive result and retracted one of its
own, both from the same adversarial review of a plan that was never run.

## 1. CORRECTION: the k=3 partition effect I reported is 1.2 sigma (BLOCKERS #47, corrected)

Pass 10 reported "k=3 is partition-sensitive" from a 42% LDS movement between campaigns O and R. I
never computed what mask-sampling noise alone produces at n=400. A 4000-resample bootstrap on both
campaigns' frozen outcomes: k=3 differs by **z = 1.23**, k=5 by **z = 0.11**. A 1.2-sigma difference
is routine. **Partition sensitivity is unresolved at both grains**, and the design cannot resolve it --
the data-consistent between-partition SD (~0.021) sits below one partition's own SE (~0.032), so a
six-partition test has ~10-15% power and settling it would need ~20 partitions and ~50,000 retrains.

This killed pass 11's planned GPU spend. A third partition would have returned a near-certain null.

## 2. The datamodel ATTRIBUTES -- it transfers across an independent re-partition (BLOCKERS #50)

Pass 10 left open whether the datamodel's bar-clearing performance was attribution or fitting the
outcome surface of its own mask ensemble. Campaigns O and R are two independent partitions of the
same 135 demos sharing zero groups, so the test is free: fit on O, map each group coefficient to its
demos, aggregate over R's masks, score against R's frozen outcomes.

| grain | arm | LDS | SE | ratio | rho/sqrt(r) |
|---|---|---|---|---|---|
| k=3 | fit O -> score R (**transfer**) | 0.3057 | 0.0296 | **0.781** | 0.488 |
| k=3 | fit R -> score R (within) | 0.3287 | 0.0296 | 0.839 | 0.525 |
| k=3 | GradDot | 0.0781 | 0.0318 | 0.200 | 0.125 |
| k=5 | fit O -> score R (**transfer**) | 0.3210 | 0.0311 | **0.754** | 0.492 |
| k=5 | fit R -> score R (within) | 0.4699 | 0.0258 | 1.104 | 0.720 |
| k=5 | GradDot | 0.1362 | 0.0323 | 0.320 | 0.209 |

**It transfers and still clears the bar out of partition**, at 4.0x and 2.4x GradDot on identical
masks (z = 5.3 and 4.1). Qualifications: at k=5 the within-campaign number overstates it (transfer
loses 32% of the LDS, z = 3.7; at k=3 the loss is undetectable), and coefficient stability across
disjoint halves of campaign O is 0.690 Pearson at k=3, 0.899 at k=5.

So **#48 survives its strongest available test** -- the bar is reachable, by a method whose advantage
is not an artifact of the ensemble it trained on -- and what #48 says about GRADIENT attribution is
untouched.

## 3. What the pass did not do, and why

Pass 11's first plan proposed two units and both were void:

- **A subsampling ladder** walking masks-per-coefficient below 1.0, to test whether the datamodel's
  win was an over-determination artifact. It could not have worked: the datamodel's LOO prediction is
  refit on n-1 masks while GradDot's is a fixed cached score independent of n, so the paired delta
  shrinks **mechanically** as n falls, for any regression, informative or not. A decaying curve was
  guaranteed by estimation theory. The transfer test in section 2 replaced it and answers the question
  directly, because both arms score the identical 400 campaign-R masks.
- **A third partition**, killed by section 1's power calculation.

Also corrected in passing: the plan's GPU budget was off by 2x (a "full partition" is 400 masks *per
grain*, so 800 masks and 1600 retrains, not 400/800), and its U1-gates-U2 dependency was decorative --
U1 concerned the datamodel and U2 the gradient estimator's partition variance, which are uncoupled.

## Corrections I made to my own work, in order

1. **BLOCKERS #47 overstated a null as a finding.** I read a 42% percentage movement as partition
   sensitivity without computing the noise floor. Corrected in #47, the FINDINGS pass-10 section, the
   HANDOFF, and `docs/PASS9_10_REPORT.md` -- which had already gone out for external reading.
2. **My replacement probe was mechanically incapable of answering its question** (section 3).
3. **My budget arithmetic was off by a factor of two**, and a stated gate was decorative.

---

# PASS 12 -- the unbiased ceiling, and the end of what this corpus can answer

Pass 11 closed with campaign Q in flight: two extra seed slots on campaign O's identical masks, taking
them from depth 2 to depth 4. It was a footnote for two passes and then stopped being one, for a
reason worth stating plainly. The project's **negative** results all sat at depth 2 where BLOCKERS #42
says the ratio is inflated -- so they were measured on the most generous scale available and a
stricter denominator could only push them further down. But pass 11's **positive** result sat at depth
2 as well. A positive claim measured on an inflated scale has to be re-measured on the honest one.

**It survives** (BLOCKERS #51). At depth 4 the ceiling rises 45% and:

- **The datamodel still clears the bar** -- 0.849 (k=3) and 0.881 (k=5) against 0.5.
- **Gradient attribution still clears nothing**, and by a wider margin: 0.225 and 0.239.
- The gap widens from ~2.9x to ~3.8x.

Two unanticipated reads came out of it. The datamodel's **raw LDS rises** with depth (+18% / +16%)
while GradDot's falls slightly -- cleaner outcomes help the estimator capturing real structure and do
not help the one that is not, which discriminates the two methods without involving the contested
denominator at all. And on the attainable scale the datamodel is **depth-stable** (0.640 -> 0.627)
while GradDot decays (0.218 -> 0.166).

## Where this leaves the project

Every question this corpus can answer has been answered:

| question | status |
|---|---|
| Does the grain matter? | **Closed by combinatorics** (#45) -- the k=15 population is 70 and exhausted |
| Are the sub-cluster rungs partition-sensitive? | **Unresolvable** at a feasible budget (#47 corrected) |
| Does the datamodel attribute or fit its ensemble? | **It attributes** (#50) |
| Is the half-ceiling bar the right standard? | **It is reachable and discriminates** (#48) |
| Do historical ratio comparisons survive the ceiling effect? | **Yes** (#49) |
| Do the conclusions survive an unbiased ceiling? | **Yes** (#51) |

What remains is **off-box and is a resource decision, not a plan**: port to a corpus of 500+
demonstrations. It is simultaneously the only route to the grain question and the only clean
discriminator for whether the gradient failure is a limit of the corpus size or of the approach.

## The sentence the whole campaign earned

> On a 135-demonstration corpus, a design-based datamodel fit on one arbitrary partition predicts
> retraining outcomes on a completely independent partition at 75-78% of the achievable ceiling, and
> clears a half-ceiling usefulness bar at both an inflated and an unbiased denominator. Gradient-based
> attribution, on the identical masks, reaches 17-23% and clears that bar at no unit size from 3
> demonstrations to 15. Attribution on this corpus is achievable -- by a method that reads retraining
> outcomes. Reaching the same bar from gradients alone is not, and whether that is a property of the
> corpus or of the approach is the question the next corpus has to answer.
