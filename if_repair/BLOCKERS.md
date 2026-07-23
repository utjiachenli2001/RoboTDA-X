# BLOCKERS — stated assumptions that turned out to be wrong

## 1. (RESOLVED, but the brief is wrong) `scores_graddot` is **not** the 0.513 champion

The brief says the paper's headline `GradDot = 0.513` at E=20 is produced by
`phase3/src/p6_lambda_extend.py:scores_graddot(normalize_per_member=True)`, and warns that the
per-member `1/d_m` normalization is the subtlety to preserve.

Measured, on the exact triple the brief specifies (E=20 = p6+p11, `p12_outcomes_S10`
`neg_plain_loss`, ceiling 0.9506):

| estimator | C1 LDS | ratio |
|---|---|---|
| `scores_graddot(normalize_per_member=True)` — i.e. `K_m / mean(diag G_m)` | **0.5930** | 0.624 |
| per-member **unit-L2** normalization of `K_m`, then mean | **0.5130434782608695** | 0.540 |

The archived constant in `phase5/src/p16_analyze.py` is
`C1_GRADDOT_ARCHIVED = 0.5130434782608695`, described there as
*"P11 archived C1 GradDot_E20_normalized"*, and `p5lib.normalized_ensemble_scores` implements
that normalization as **unit L2** (`M / ‖M‖₂` per member column), not `1/d_m`.

So the repo contains **two** GradDot variants and the brief conflates them. Both are reproduced
here bit-for-bit against `p16_lds_table.csv` (`<1e-12`, all 7 non-focal targets).

**Impact / how it is handled.** The brief's `0.513` anchor is real and is reproduced exactly —
by `GradDot_unitL2`, implemented in `if_repair/anchors.py:graddot_unit_l2`. The brief's
*attribution of it to `scores_graddot`* is wrong. Both baselines are carried through every table,
and **the bar for "beats the champion" is the stronger of the two (`GradDot_dmean`, 0.5930 on C1)**
— not the 0.513 the brief names. This matters: several variants below beat 0.513 while losing to
0.5930, and calling those an improvement would be an artifact of picking the weaker baseline.

Note the brief's own secondary numbers ("unnormalized 0.397, normalized 0.504") *do* match
`scores_graddot` — but at the **E=10 `dev_s6`** triple (reproduced: 0.3965 / 0.5043), not E=20.

## 2. (RESOLVED, brief imprecise) "Cross-validated exact IF, C1" is tuned-on-C1, evaluated on C5

The brief asks for "cross-validated exact IF, C1, E=20 → ρ ≈ 0.40". A CV estimator *evaluated*
on C1 (tuned on C5) gives 0.3409 at E=10 / 0.5096 at E=20 — neither is 0.40. The repo's archived
`p6_lambda_sweep.json → MANDATORY_CROSS_VALIDATION` has exactly one 0.40:
`tune_on_C1__evaluate_on_C5 = 0.40347826086956523`. So "C1" names the **tuning** target.
Reproduced bit-for-bit at `dev_s6`/E=10 (diff 0.0e+00) and as 0.3939 at `bc_s10`/E=20 — both
within the required 0.03 of 0.40, and both FAIL the half-ceiling + p bar as the brief states.

## 3. (BLOCKING — Task 4 not attempted) No checkpoints in the repo

Task 4 requires per-layer Φ recomputed from `runs/stage_E/ens_s*/final.pt` via
`src/attribution.py:demo_gradient` / `build_targets`.

```
$ find . \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) | wc -l
0
```

`runs/stage_E/ens_s201..s210/` exist but contain only `demos.json`, `outcomes.json`,
`train_meta.json`, `train_log.jsonl` and markers — **no weights**. The published repo carries
code, configs, reports and small results only (see its initial-commit message); the 19.2M-param
checkpoints were never committed. Task 4 (subspace / per-layer, layer-ablation) is therefore
**not attempted**, exactly as the brief instructs. It is the only task that needed a GPU, so
nothing else is affected. Carried to `HANDOFF.md` as a GPU follow-up.

## 4. (RESOLVED, minor) Diffusion cache is E=5, and its tier needs the MEDIAN aggregator

`phase5/results/p17_diffusion_gram_cache.npz` has `G (5,135,135)`, `K (5,135,9)` — **5** members
(`dpens_s621..s625`), not the 10 of the BC arms. The brief does not state an E for it; noted so
that "E=20 = p6 ∪ p11" is not mistakenly applied here.

Separately, the diffusion tier must aggregate seeds by **median**, not mean: `p15_verdict.json →
seed_mean_brokenness_series` shows the mean-aggregated C1 ceiling collapsing to 0.201 vs 0.830
for the median. Using the mean would score predictions against a ceiling computed a different
way. Encoded in `data.py:TIERS["diff_s10"]["agg"] = "median"`.

## 5. (RESOLVED, environment) `src/bootstrap.py` hardcodes an absolute ROOT

`ROOT = "/mnt/sdb/ljc/RoboTDA-X"` — the original host's path — and every repo path constant
(`RESULTS`, `RUNS`, `P2_RESULTS`, `GK_CACHE`, …) derives from it. On any other machine the repo
modules are unimportable. Since the brief forbids editing repo files, this is handled **outside**
the repo by making the clone visible at the expected path:

```
sudo mkdir -p /mnt/sdb/ljc && sudo chown $USER /mnt/sdb/ljc
ln -s ~/code/RoboTDA-X /mnt/sdb/ljc/RoboTDA-X
```

`bootstrap.py` also pins `CUDA_VISIBLE_DEVICES` to one of GPUs 4–7 (a rule from the original
host). This box has one GPU (index 0), so the pin selects a nonexistent device — harmless here,
since Tasks 0–3 and 5 are pure NumPy and torch is only imported, never used for compute.
`if_repair` code itself derives its own ROOT from `__file__` and does not depend on the symlink.

---

# Pass 2 (methods sweep) — additional blockers

## 6. (RESOLVED with a caveat) Checkpoints are regenerable, but NOT reproducible

Pass 1 recorded (#3) that the repo ships no weights, blocking all of Phase B. That was only
half right. The training **data** is present (700 `.npz` under `data/proc`; `dataset.Bank`
builds fine), every member's seed + demo list + cfg is in `runs/stage_E/ens_s*/`, and a
member trains in **94 s** on the H200. So the checkpoints can be regenerated, which is what
unblocked B1.

They are **not the original weights**, and this was tested rather than assumed. Rebuilding
`(G,K)` for a regenerated `ens_s201` and comparing to its cached slice
(`if_repair/results/regen_verify_ens_s201.json`):

| quantity | value |
|---|---|
| `G` relative Frobenius difference | 0.879 |
| `K` relative Frobenius difference | 0.843 |
| per-target Spearman of `K` columns | 0.02 … 0.85 |
| median ratio of `diag(G)` regen/cached | **0.190** |

Gradient norms are ~5× smaller and the rank correlations are partial. The repo's determinism
gate (`p8b_determinism*.json`) established determinism of *rollouts on the original box*; it
does not imply bit-reproducibility across a different torch/CUDA/GPU stack, and this is a
direct measurement that it does not hold here (final loss −18.84 regen vs −18.70 archived).

**Consequence, applied throughout:** every B1 baseline is recomputed on the same regenerated
E=5 ensemble. B1's internal comparisons (group vs group, k vs k) are valid; B1's absolute
numbers are **not** comparable to the E=20 cached-Gram rows, and are labelled `regenE5`.

## 7. (RESOLVED) `demo_grain_lds` is circular for any estimator fit on mask outcomes

The demo-grain LDS forms its mask prediction as `sum_{d in mask} score_d`, which for a linear
datamodel is exactly `X @ beta` — the quantity the datamodel minimised. Scoring in-sample
coefficients therefore measures fit, not faithfulness, and duly returns ρ up to 0.986 with
**ratio-to-ceiling above 1.0** — impossible for a real estimator, since the ceiling is the
reliability of the outcome itself.

This bit twice: first in A4, then again in A5 when datamodel coefficients were fused into a
rank ensemble (it inflated C5 from an honest 0.43 to a fake 0.97). Both are now evaluated
**leave-one-mask-out**: each mask is predicted by a fit that excluded it. Gradient estimators
are unaffected — their scores never touch outcomes — and use the direct path.

Any future estimator that consumes outcome data must go through the LOO path in
`if_repair/confirm.py`, not `eval.evaluate`.

## 8. (NOT A BUG — real failure mode) The datamodel collapses to a constant on C7/C9

`A4_datamodel_lasso_LOO` returns NaN on hold-out targets C7 and C9. Cause: the
cross-validated `alpha` is selected at 0.1, which zeroes **every** coefficient, so the model
predicts the mean for all 24 masks and Spearman is undefined.

| target | selected alpha | non-zero coefs |
|---|---|---|
| C2 | 0.01 | 6 |
| C4 | 0.001 | 22 |
| C7 | 0.1 | **0** |
| C9 | 0.1 | **0** |

This is honest behaviour — CV decided the design explains nothing for those targets — but it
means the datamodel does not degrade gracefully: it either finds structure (C5, C2) or
returns nothing at all. NaN is scored as a non-pass, never dropped.

## 9. (NOT ATTEMPTED, not blocked) B2 / B3 / B4 were not run

Only B1 was executed from Phase B. This was a wall-clock/session limit, not a resource or
data limit: 0.15 of 12 GPU-h were used, and the checkpoint-regeneration path that unblocked
B1 unblocks the rest equally (including the 5 per-member checkpoints `ckpt_0..4.pt` that B4's
TracIn density needs). Carried to HANDOFF.md with the exact entry points.
---

# Pass 3 — additional blockers and corrections

## 10. (CORRECTED — pass 2's mechanism claim is not supported) k\* tracks WHICH subspace, not p/N

Pass 2's FINDINGS said: *"Restricting Φ raises k\* from 1 to 6–9, and only in that regime does
inverting beat not inverting. … It is a statement about `p/N`, not about influence functions."*

B6 breaks the width/role confound by restricting Φ to **random** parameter subsets. Same code
path (`spectral.spectrum_null`, 100 permutations, seed 0), same regenerated ensemble:

| Φ | params | k\* |
|---|---|---|
| `block_00` | 3,152,384 | 6 |
| `last_block` | **3,152,384** | **1** |
| `embed` | 268,288 | 9 |
| random | 100,000 / 300,000 / 3,000,000 | 1–2 at every width, 3 draws each |

`block_00` and `last_block` have **identical dimension** and differ 6 vs 1; a random subspace
*larger* than `embed` has k\* = 1 where `embed` has 9. Dimension does not predict k\*.

**Impact.** Every statement of the form "IF fails here because p/N makes curvature unestimable"
must be withdrawn as written. The supported statement is narrower: *particular* subspaces (the
input projections, the first observation block) carry demo-to-demo Gram structure that survives
a permutation null, and the late block, the action head, and random subspaces do not. Why those
subspaces is not answered here.

The p/N intuition is not dead, but it belongs one level down — see #11.

## 11. (REFINEMENT) The binding sample size is the one the CURVATURE is estimated from

B3 runs KFAC and EK-FAC, which differ in exactly one thing: KFAC's Kronecker factors come from
~92k training FRAMES; EK-FAC keeps those eigenvectors and re-estimates the eigenvalues from the
135 demo gradients. On `embed`/C1 that single swap takes ratio-to-ceiling from **0.320 to 0.017**.

So curvature estimated from 92k frames is usable and curvature estimated from 135 demos is not —
whether it arrives as a 135×135 Gram (Phase A) or as EK-FAC eigenvalues. That is the p/N claim
pass 2 wanted, at the level where it is actually true.

## 12. (NARROWED) BLOCKERS #6 applies to the GRAM, not to training

#6 measured that a regenerated member's Gram differs from its cached slice (`G_rel_fro` 0.879,
gradient norms ~5× smaller, K rank-correlations 0.02–0.85) and concluded that regenerated numbers
must never be mixed with archived ones. That conclusion stands for the Gram. It does **not**
extend to the outcomes.

Campaign A retrains the archived 24 masks at the archived seeds 401–410 under the archived
protocol (`retrain.train_one` reproduces `src/train.py` bit-for-bit when init == order; pinned by
`tests/test_pass3.py`). Matched (mask, seed) outcomes against `p12_outcomes_S10`:

| target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 |
|---|---|---|---|---|---|---|---|---|---|
| Spearman | 0.76 | 0.72 | 0.65 | 0.78 | 0.93 | 0.80 | 0.87 | 0.83 | 0.61 |
| Pearson | 0.85 | 0.82 | 0.81 | 0.82 | 0.92 | 0.90 | 0.81 | 0.84 | 0.74 |

Means agree to ~1%. **The environment sensitivity is localised to the gradient/Gram computation,
not to training.** Campaign A is therefore a valid regenerated ground truth, and campaign B's
fresh-mask outcomes can be trusted as such.

## 13. (BUG, FOUND AND FIXED) Restricting a preconditioner is not restricting Φ

B3's first implementation restricted KFAC to a parameter group but left Φ at full width. For
`head` that preconditions 0.2% of the score's coordinates, so every "head KFAC" cell was
reporting GradDot with a rounding error (all λ within 0.007 of the identity). Fixed: Φ and the
preconditioner are restricted together, and the control row (`method="none"`) is B1's GradDot on
the *same* restricted Φ. The B3 numbers in RESULTS.md are from the fixed version.

## 14. (BUG, FOUND AND FIXED BY A SMOKE TEST) A redesigned functional scored against the old outcome

`confirm3.py` initially built one ground truth (`weighting="plain"`) and scored every hypothesis
against it, including H4, whose whole content is that the **interaction-phase** functional beats
the plain one. Scoring it against the plain outcome gave 0.428 instead of its dev value 0.474 —
i.e. it measured the mismatch between two functionals, which is exactly the error B2's protocol
exists to prevent. Caught by a wiring smoke test run on the archived masks *before* campaign B
produced any data; `fresh_truth` is now keyed by weighting. `PREREG` was not touched.

## 15. (NOT ATTEMPTED, environment) Rollout-visited-state weighting

B2's third proposed functional needs the policy rolled out to collect visited states. Neither
`libero` nor `robosuite` is importable in this environment, and the functional would additionally
require re-rolling all 240 mask retrains, which is outside the GPU budget. Not attempted, and the
B2 conclusions are stated over the four functionals that were.

## 16. (OPERATIONAL) Three concurrent trainers saturate this H200; six is slower

Measured, not assumed. One trainer: 87 s/run. Three: 137 s/run (throughput 1.9×). Six: 314 s/run
(throughput 1.15× vs three, i.e. a net LOSS). The model is 19.2M parameters at batch 256, so the
bottleneck is kernel launch and scheduling rather than FLOPs, and oversubscription costs more
than it buys. Campaign workers are 3.

## 17. (MEASURED, and CORRECTED) The single-draw demo-grain LDS has a large sampling variance -- but the mask draw is a SHARED nuisance, so paired comparisons survive it

**First version of this entry was wrong about the cause, and the error was mine.** It read a
two-point gap -- GradDot_dmean on C1 scoring 0.509 on the archived 24 masks and 0.337 on the
fresh 24 -- and attributed ~0.14 of it to the mask draw. Two things were confounded into that
number, and B8 (`b8_maskdraw.py`) plus a seed-depth control (`/tmp/chk_seeddepth.py`, folded into
`test_pass3`) separate them.

**Confound 1: seed depth is not neutral.** Campaign B was first run at 6 seeds to save GPU, then
compared against dev numbers computed at 10. Holding the masks, the estimator and the outcome
table fixed and varying ONLY the number of seeds averaged into the outcome:

| seed depth | ceiling | GradDot_ALL C1 | GradDot_ALL C5 |
|---|---|---|---|
| 4 | 0.883 | 0.386 | 0.414 |
| 6 | 0.908 | 0.362 | 0.414 |
| 8 | 0.937 | 0.418 | 0.408 |
| 10 | 0.941 | 0.475 | 0.374 |

Dividing by the ceiling does NOT make the ratio invariant to depth, and it is not even monotone
(C1 rises with depth, C5 falls). So a depth-6 number is not comparable to a depth-10 one.
Campaign B was extended to 10 seeds and every fresh-mask number recomputed at matched depth.

**Confound 2: at matched depth the two mask sets barely differ.** At 6 seeds, G-series C1 = 0.362
and H-series C1 = 0.337 -- a 0.025 gap, not 0.14. Most of the original "mask-draw" effect was
seed depth (~0.11) plus outcome regeneration (~0.03).

**What IS true, and it is stronger.** B8 pools the 24 G + 24 H masks (exchangeable: same
generator, same 68-demo stratified design, same demo universe, same held-out bank) and bootstraps
2000 random 24-mask subsets. The sampling sd of the single-draw ratio is large -- ~0.15 for
GradDot on every target -- against a pass bar of 0.5. **The design cannot resolve an absolute
ratio at n = 24.** But the mask draw is a *shared* nuisance: every estimator is scored on the same
subset, so it cancels in a paired difference. The paired sd of (estimator - GradDot) is far
smaller than either level's sd, and the paired comparisons are decisive:

(2000-bootstrap, matched 10-seed depth; `results/b8_maskdraw_bootstrap_d10.csv`)

| target | comparison | level sd | paired sd | mean Δ | wins over draws |
|---|---|---|---|---|---|
| C5 | KFAC_embed - GradDot | 0.10 / 0.17 | 0.15 | +0.48 | **100.0%** |
| C5 | datamodel - GradDot | 0.14 / 0.17 | 0.20 | +0.56 | 99.7% |
| C2 | datamodel - GradDot | 0.14 / 0.19 | 0.23 | +0.12 | 70.9% |
| C1 | TracIn - GradDot | 0.13 / 0.14 | 0.10 | +0.08 | 77.1% |
| any | GradDot_head - GradDot_ALL | 0.13 | **0.005** | ~0 | -- |

On the pooled 48 (best point estimate, ceiling ~0.94), C5 KFAC_embed = 0.631 and datamodel =
0.684, both clearing the 0.5 bar, while GradDot sits at 0.137. Depths 6 and 10 agree throughout
(`results/b8_maskdraw_*_d6.csv` vs `_d10.csv`).

**Impact.** The project has been reporting the wrong statistic. Its bar is absolute
(ratio >= 0.5 and p < alpha), and the absolute ratio is unresolvable at n = 24 -- the paper's
GradDot = 0.513 carries a +-0.15 sampling sd. The *relative* question ("does estimator X beat
GradDot on these masks?") is answerable, and by that measure KFAC-on-embed beats GradDot on C5 in
99.6% of draws and the action head reproduces the full model to a paired sd of 0.004. Future work
should report paired differences on shared masks, not absolute ratios on one draw.

## 18. (CONFIRMED at matched depth) Two hypotheses replicate on fresh masks, not one

The confirmatory family was recomputed at matched 10-seed depth after confound 1 above. Result:
**2 of 5**, not the 1 of 5 the depth-6 run reported.

| hypothesis | depth-6 ratio | depth-10 ratio | p (d10) | verdict |
|---|---|---|---|---|
| H1 datamodel on C2 | 0.729 | 0.626 | 0.0012 | PASS (both depths) |
| H3 KFAC embed on C5 | 0.499 (fail) | **0.563** | **0.0032** | **PASS (d10 only)** |
| H2 TracIn head on C1 | 0.410 | 0.340 | 0.060 | fail |
| H4 interaction on C5 | -0.033 | -0.055 | 0.60 | fail |
| H5 GradDot head on C1 | 0.337 | 0.283 | 0.10 | fail |

H3 sat exactly on the 0.5 bar at depth 6 (0.499) and cleared it at matched depth (0.563,
p = 0.0032 < alpha = 0.005). This is the §3 mechanism result -- curvature from 92k frames on the
one subspace with real Gram structure -- confirmed out of sample. The depth-10 table
(`results/confirm3_fresh_masks_d10.csv`) supersedes the depth-6 one; the protocol (10 seeds
matches the archived dev depth; report both regardless of outcome) was declared before looking.

## 19. (CONSUMED) Both confirmatory families are now spent

The pass-2 hold-out (C2/C4/C7/C9 on the archived 24) and the pass-3 fresh-mask family (5
hypotheses on the H-series masks) have each been computed once. Neither can be re-used. The
G+H pool is now dev data too (B8 read it), so a genuinely new claim needs a THIRD mask draw;
`retrain.fresh_demo_masks(seed=...)` generates one at any seed, ~5.8 solo-h at 10 seeds.

## 20. (MEASURED) A paired bootstrap over the SELECTION masks is not out-of-sample

B8 resampled 24-mask subsets of the pooled G (dev) + H (confirm) masks and found KFAC-on-embed and
the datamodel each beating GradDot on C5 in ~100% of subsets. That reads like a decisive win, and
for the datamodel it held up -- but for KFAC-on-embed it did NOT. The I-series (a third draw,
disjoint from G and H, `confirm_iseries.py`) scores KFAC-on-embed at 0.428 vs GradDot 0.490: it
LOSES to the baseline out of sample.

The subtlety: every G and H mask had already been used to select or confirm KFAC-on-embed (it was
tuned on the G-series and passed on the H-series). Bootstrapping subsets of those masks tests
stability under resampling, not generalisation to new masks -- the estimator has effectively seen
all 48. The paired statistic (B8, BLOCKERS #17) is the right one for resolving a comparison at
n=24, but it must be computed on masks the estimator was NOT selected against.

**Consequence.** Two bars are now required for any estimator claim: PAIRED (beats GradDot on shared
masks, resolvable at n=24) AND OUT-OF-SAMPLE (on a fresh draw). Of everything this project tried,
only the datamodel clears both, and only on C5. The gradient-side improvements (KFAC-on-embed,
TracIn density, the fail_div/ens_var functionals) each cleared an in-sample bar and failed the
fresh draw. `if_repair/confirm_iseries.py` is the template; a fourth draw needs a new seed.

## 21. (RESULT) k* tracks depth, not dimension or side (B9)

B6 showed k* is not a function of Phi's dimension. B9 (`b9_structured.py`) identifies what it IS a
function of, on the same cached Phi: DEPTH. The six transformer blocks are identical in size
(3.15M) and give k* = 6,5,1,1,1,1 from first to last. `embed` (input) has k*=9 but so does
block_00's attention (k*=8) and the first LayerNorm (k*=6 on 1024 params) -- the common factor is
EARLY, not input-side. Every structured subspace beats its size-matched random control (k* 5-9 vs
1-2), so the structure is real, not a dimension artefact. And k* does NOT predict where inverting
helps: mean gain-from-inverting k*>=3 vs k*<=1 is +0.008/+0.001 (C1), +0.093/+0.106 (C5) --
indistinguishable. The subspaces richest in estimable curvature are not where preconditioning pays.
