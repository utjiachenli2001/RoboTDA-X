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

## 15. (RESOLVED -> DONE as B10) Rollout-visited-state weighting

Originally not attempted: `libero`/`robosuite` were not installed, and I wrongly believed the
functional would need re-rolling every mask retrain. Both were fixed. The install:

```
sudo apt-get install python3.12-dev libosmesa6 libglfw3 patchelf
.venv/bin/pip install robosuite==1.4.1 libero          # libero pins robosuite 1.4.0
.venv/bin/pip install mujoco==3.1.6                    # 3.10 breaks robosuite (mj_fullM sig)
```

The repo config (`configs/libero_cfg`) pointed at the original host's conda path; redirected
via a fresh `if_repair/runs/libero_cfg/config.yaml` (no repo file edited) + `LIBERO_CONFIG_PATH`.
LIBERO auto-downloads its assets to `~/.cache/libero` on first env build.

The re-rolling worry was wrong: the weighting is an ENSEMBLE property computed ONCE from rollouts
(like ens_var/fail_div), and the outcome side is free (per-frame campaign losses already on disk).
Only the rollouts cost GPU -- 3 members x 70 held-out tasks x 1 episode, ~4 min. `b10_rollout.py`
runs it. Two gotchas found: a file-descriptor leak in the robosuite env (raise `ulimit -n`; it
starved the last-processed clusters C8/C9, which fall back to uniform weighting -- C1-C7 have real
clouds), and per-step batch-1 GPU inference from many workers is pathologically slow (load the
model once per worker via a pool initializer, use <=6 workers). See FINDINGS §5 for the result:
the functional does NOT beat plain -- it is worse.

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

---

# Passes 4-6 -- new blockers and lessons

## 22. (METHODOLOGICAL) The kill rule works; killed W1 on a 1-member pilot
W1 (unlearning-LOO) was screened on a 1-member ascent/finetune-forget pilot showing no beat vs
GradDot (best paired Delta < +0.10 on C1/C5), and killed before the full 5-member sweep. The full
sweep would have cost ~4-5 GPU-h to confirm a negative already visible in the pilot. The kill rule
("one screen, stop if best paired Delta < +0.10 on both dev targets") is the reason pass 4 finished
inside budget. Do NOT run a full multi-member unlearning sweep unless a 1-member pilot shows signal.

## 23. (TRAP, redux of #1) The leverage family's unit-L2 aggregation invites the wrong baseline
The RelatIF/leverage family aggregates per-member scores by unit-L2 (the aggregation that won in
W4). It is tempting to benchmark a unit-L2-aggregated estimator against GradDot_unitL2 (0.513-class).
That is the BLOCKERS #1 trap wearing a new hat: the bar is ALWAYS GradDot_dmean (0.593-class),
recomputed on the same ensemble, regardless of how the challenger aggregates. Measured cost of the
mistake: the leverage family "generalizes to >=3 targets (C2,C3,C8)" ONLY against the unit-L2
baseline; against GradDot_dmean the best single config reaches 2 (C2,C8) and C3 collapses. p1's
generality scan was corrected to always pair against dmean (commit 5a5d554).

## 24. (RESULT) The leverage correction is target-specific in its Phi, not just its lambda
The family diag(G)^-beta (G+lam I)^-1 K beats GradDot_dmean OOS on 5 targets, but the winning Phi
differs by target: C5 lives on the cached E=20 FULL-model Gram (head-Phi is negative on C5, -0.22),
while C2/C8 live on the regen E=5 HEAD-Phi Gram. beta=1 (full self-influence normalization) is the
load-bearing ingredient across all of them; the Gram inversion (finite lambda) only adds breadth.
So there is no single (Phi, lambda, beta) that covers all wins -- "generality" is a family property,
not a single-estimator property, on this corpus.

## 25. (RESULT) Leverage correction complements the outcome-consuming datamodel
Per-target leverage responsiveness correlates with the datamodel's per-target LDS (Spearman +0.66),
but the leverage family additionally wins on C7 and C8 where the datamodel COLLAPSES to a constant
(LDS NaN, BLOCKERS #8). A zero-outcome gradient estimator therefore reaches targets the
outcome-based datamodel cannot -- the first time in the project the two estimator classes are shown
to be complementary rather than the datamodel strictly dominating.

## 27. (RESULT, humbling) Pass-5 dev wins mostly did not survive a fresh draw
The leverage family beat GradDot on 5 dev targets (pooled G/H/I paired). Fresh-draw confirmation:
campaign J confirmed RelatIF/C5 on the absolute bar (ratio 0.533, p=0.005); campaign K scored the
pass-5 configs 0/3 -- C2 unresolvable (outcome negative for GradDot too, as on J), C8's +0.387 dev
win did NOT replicate (GradDot itself is strong on C8/K), C7 kept its sign (+0.29) but missed both
bars. So 4 of the 5 dev "wins" were the mask-draw overfitting this project catches every pass; only
C5 is robust. Report leverage-family targets as DEV unless a fresh draw confirmed them (only C5 has).

## 26. (SEEDS) Campaign J, K and L draws
J = seed 20260723 (pass-4 confirmation), K = seed 20260724 (pass-5 confirmation). Both 24 masks,
10 seeds, archived protocol, disjoint from G(11)/H(4711)/I(9973) and from each other;
tests/test_jseries.py asserts pairwise disjointness of all five draws and freezes both seeds. Both
PREREG files (confirm_jseries.py, confirm_kseries.py) were committed with the respective campaign at
ZERO runs (git history), the same integrity mechanism as confirm3/confirm_iseries.

---

# Pass 7 -- new blockers and corrections

## 28. (CORRECTION, and the pass's central result) A frozen config is OUT-OF-SAMPLE on every later draw -- and nobody had scored RelatIF/C5 on campaign K

Passes 4-6 concluded that a leverage / self-influence correction (RelatIF, K/G_dd) is the first
gradient estimator in the project to beat GradDot out of sample "IN DIRECTION, consistently across
six mask draws". **That claim is false.** It was true only because one draw had never been scored.

The RelatIF/C5 config was frozen in `PREREG_J` (`c7659cf`) with campaign J at zero runs and never
retuned. It is therefore legitimately out-of-sample on J, K **and** L. But campaign K's own prereg
family named C2/C8/C7, so the C5 config was never run against K. Scoring it there (W0.1,
`p7_pooled_oos.py`) costs zero GPU and reverses the sign:

| draw | role | Delta rho vs GradDot_dmean |
|---|---|---|
| G | dev (selection) | +0.226 |
| H | dev (selection) | +0.384 |
| I | dev (selection) | +0.135 |
| J | **discovery** | +0.341 |
| K | **oos, never scored until pass 7** | **-0.171** |
| L | oos | +0.075 |

Every draw the effect was selected on or reported on is positive; the one clean draw nobody had
looked at is negative. Pooled over its winner's-curse-free set (K u L, 48 masks): **Delta rho
= -0.044, 95% CI [-0.264, +0.185], p = 0.659.** Ratio-to-ceiling 0.006 against a 0.5 bar.

**Two lessons, and the second is the subtle one.**

1. **The confirmation draw you already own is worth more than the estimator you were about to
   build.** Six draws existed on disk; the analysis that overturned the headline was a re-scoring,
   not an experiment. Before designing a new estimator, score every frozen config on every draw it
   is out-of-sample on.
2. **Frozen-before-the-draw is NOT the same as unselected-upon.** RelatIF/C5 was frozen before J,
   so J is technically out-of-sample. But J is the draw on which the effect was first *reported as
   a success*, and conditioning on "this is the hypothesis we followed up" selects for a favourable
   J. Hence every contrast in pass 7 is reported twice: full-information (all OOS draws) and
   without the discovery draw. **Believe the second.** With J the pooled estimate is +0.083; without
   it, -0.044. The winner's curse on a discovery draw is exactly that large.

## 29. (MEASURED) Seed depth buys almost nothing; masks buy everything. The exchange rate is ~5:1

`p7_design.py` resamples the existing 144 masks x 10 seeds. Sampling sd of the paired contrast
(C5, RelatIF vs GradDot_dmean), masks bootstrapped with replacement:

| paired sd | depth 2 | depth 5 | depth 10 |
|---|---|---|---|
| 24 masks | 0.179 | 0.172 | 0.183 |
| 48 masks | 0.127 | 0.121 | 0.129 |
| 72 masks | 0.103 | 0.107 | 0.098 |
| 144 masks | 0.078 | 0.070 | 0.069 |

**Depth is flat. Masks follow 1/sqrt(n) almost exactly.** BLOCKERS #17 guessed seed noise was
nearly exhausted by depth 4-5; for the PAIRED statistic it is exhausted by depth 2.

Iso-budget, 240 retrains spent three ways: 120x2 gives paired sd 0.077; 48x5 gives 0.121; the
archived **24x10 gives 0.183**. Same GPU, 2.4x smaller sd, ~5.7x the effective sample size. **The
archived depth-10 protocol is close to the worst allocation available on this corpus.**

Depth does buy the ceiling (0.809 at depth 2 -> 0.945 at depth 10), but the mean ratio-to-ceiling
is flat across allocations (0.405 vs 0.377) because rho and the ceiling fall together, so trading
depth for masks does not harm the absolute bar either.

**Sizing.** A 95% CI of width 0.20 needs paired sd 0.051: ~295 masks at Spearman, or ~112 masks
under the primary statistic (#30) -- at depth 2, ~224 retrains ~ 5.9 solo-h. The same resolution
under the archived protocol costs ~2950 retrains ~ 77 solo-h. **Thirteen times the GPU for the
same answer.** This is why six passes of 24-mask draws resolved nothing.

Methodological trap inside the trap: the first version of this table subsampled masks WITHOUT
replacement from the 144-mask pool, which carries a finite-population correction (1 - n/144) and
drives the sd to exactly 0 at n = 144. It inflated the mask advantage and was caught before commit.
The estimand is a campaign of n FRESH masks, so the nonparametric bootstrap (with replacement) is
the right model; seeds stay without replacement because a real campaign draws d distinct slots.

## 30. (RESULT) Kendall tau_b resolves this corpus better than Spearman, and the statistic can be chosen without looking at the hypothesis

Candidate LDS statistics scored on RELIABILITY (split-half of the outcome against itself) and NOISE
(spread of a random predictor) only -- criteria that never touch the estimator-vs-baseline contrast:

| statistic | reliability | null sd | resolution |
|---|---|---|---|
| **kendall_tau_b** | 0.763 | 0.145 | **5.32** |
| pearson_raw | 0.911 | 0.199 | 4.61 |
| spearman (incumbent) | 0.899 | 0.199 | 4.52 |
| quartile_gap | 2.278 | 0.596 | 3.76 |
| top6_overlap | 0.820 | 0.159 | 3.57 |

Same winner on C5 and C7. **The selection method validated itself:** chosen purely on
estimator-free criteria, Kendall then returned CIs about a third narrower than Spearman on the same
masks (C5 K u L width 0.306 vs 0.459). Spearman is retained as a mandatory secondary for continuity
with every historical number in the repo.

The C5 null is robust to this choice -- it is negative under all six statistics, including both
decision-quality ones. So the passes 4-6 result does not fail merely because Spearman is blunt.

Correction to the pass-7 brief: it lists "Pearson on ranks" as a candidate distinct from Spearman.
Pearson on ranks IS Spearman by definition.

## 31. (TRAP, redux of #20 one level down) The honest draw set is per-CONFIG, not per-pass

W0.2b initially scored both surviving configs on K u L. That is the honest set for RelatIF/C5,
whose discovery draw is J -- but **K is the discovery draw for leverage/C7**, so the same call
scored C7 on the very draw its effect was reported on. It returned Kendall +0.257, CI [0.000,
0.515], **p = 0.026** -- a publishable-looking out-of-sample win that is nothing of the kind. On
C7's actual honest set (J u L) the same statistic gives **+0.032, p = 0.406**.

Caught before commit. The honest set is now derived from `CONFIGS[...]["discovery"]` per config and
can never again be a per-pass constant. Different hypotheses in the same pass have different
discovery draws, and a single "the fresh draws are K and L" statement is wrong for at least one of
them.

## 32. (RESULT) Pairing is not always the right statistic -- it depends on the sign of the covariance

BLOCKERS #17 established that the paired contrast beats the level because the mask draw is a shared
nuisance that cancels. Measured across the six draws, that is true for C5 and **false for C7**:

| target | var(GradDot) | var(estimator) | cov | var(paired Delta) |
|---|---|---|---|---|
| C5 | 0.047 | 0.097 | **+0.052** | 0.041 (pairing removes 58% of the challenger's variance) |
| C7 | 0.043 | 0.011 | **-0.009** | 0.071 (**pairing is worse than the level**) |

When the estimator and the baseline are negatively correlated across draws, differencing them adds
variance instead of removing it. Nothing in the repo checked the sign before adopting the paired
statistic universally. Check `cov` before choosing.

Also worth recording: **GradDot_dmean is itself a moving target.** Its C5 LDS ranges from -0.066
(H) to +0.489 (I) across the six draws, sd 0.217 -- larger than most effects anyone has claimed on
this corpus. A "win over GradDot" on one draw is substantially a statement about which GradDot you
happened to draw.

## 33. (RESULT) RelatIF and GradDot mostly AGREE -- which is why the LDS cannot separate them

Building the W2 duel design surfaced the mechanism behind six passes of failure to discriminate.
The two estimators rank-correlate **0.547** over the 135 demos (0.566 within cluster). Of the 945
possible within-cluster demo pairs, only 6.6% disagree at a rank gap >= 4:

| min rank gap | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| disagreeing pairs | 108 | 62 | 37 | 23 | 11 | 3 | 1 |
| demo-disjoint duels | 29 | 18 | 12 | 11 | 5 | 1 | 1 |

On a random 68-of-135 mask the two summed scores nearly coincide, so almost all of the outcome
variation an LDS sees carries no information about which estimator is right. **This is independent
of sample size** -- it is why more masks help slowly (#29) and why the duel design exists at all.
It also caps the duel design: demo-disjointness is what makes the duels independent and the sign
test valid, and it limits the corpus to n = 18 usable duels, not the 32-40 the brief assumed.

## 34. (RESULT, and the pass's answer) The C5 effect is real, small, and about a quarter of what was claimed

Campaign M -- 144 virgin masks in 6 disjoint sub-draws at depth 2, 288 retrains, PREREG_M frozen
at zero runs (`bdab9af`), family of one, scored once.

| statistic | lds | GradDot | ceiling | ratio | paired Delta | 95% CI | p | bars |
|---|---|---|---|---|---|---|---|---|
| kendall_tau_b (primary) | 0.284 | 0.199 | 0.688 | 0.413 | **+0.085** | [-0.004, +0.171] | 0.031 | PAIRED pass, ABS fail |
| spearman (secondary) | 0.413 | 0.301 | 0.844 | 0.489 | **+0.111** | [-0.013, +0.231] | 0.041 | PAIRED pass, ABS fail |

**The first time any gradient-side estimator in this project has cleared paired p < 0.05 on a
virgin preregistered draw.** The absolute half-ceiling bar still fails (Spearman misses by 0.011).

**Pooled over every clean out-of-sample mask (K u L u M, 192 masks, 8 draws) -- quote this, not M
alone, which is the single most favourable slice:**

| pooling | Delta rho | 95% CI | p |
|---|---|---|---|
| mixed depth (10/10/2), kendall | **+0.060** | [-0.015, +0.137] | 0.057 |
| mixed depth, spearman | +0.084 | [-0.027, +0.193] | 0.064 |
| matched depth 2, kendall | +0.055 | [-0.020, +0.131] | 0.074 |

Depth mixing is not the driver: K u L is negative at BOTH depths (-0.094 at 2, -0.051 at 10) and
the mixed/matched pooled estimates agree to 0.005-0.009, as W0.2 predicted when it measured the
paired mean to be flat across allocations.

**Campaign J's +0.341 is decisively excluded** -- it lies far outside the upper bound of every
pooled interval. Passes 4-6 over-estimated the effect roughly 4x. That is the size of a discovery
draw's winner's curse on this corpus, now measured rather than feared.

Two reporting traps, both recorded because either alone would mislead: quoting only the pre-M
estimate (K u L, -0.044) gives "resolved negative"; quoting only M's p = 0.031 gives "confirmed".
The pooled interval is the answer, and it says **small, positive, not separable from zero at
alpha = 0.05.**

**The design analysis predicted its own result.** W0.2 forecast a 95% CI of width ~0.18 for a
144-mask depth-2 campaign before it ran; the measured width is 0.175. And it cost 288 retrains --
FEWER than any single prior campaign (A/B/I/J/K/L used 240 each at 24 masks) -- for an interval
less than half as wide.

## 35. (NEGATIVE DESIGN RESULT) One-demo counterfactuals are below this corpus's noise floor; the duel design is not viable

W2 built pairs of masks differing in exactly one demo, at matched seed slots, where the two
estimators disagree about which demo matters more. The mandatory pilot (6 duels, depth 4, 48
retrains) killed it.

**The pairing premise was correct.** Within a pair sharing 67 of 68 demos, the mask x init
interaction that B5 identified as the binding noise does difference out: mean paired sd 0.0365 vs
mean unpaired sd 0.0872, a **2.39x noise reduction**. So matched seeds inside a swap-pair really
are a different design from B5's common random seeds, and B5 stays correctly retired for the
general question. This is the measurement the brief asked for instead of an assertion.

**The signal is smaller than the noise anyway.** Swapping one demo out of 68 moves the C5 held-out
loss by only **0.44 seed-noise sd** on average (per-duel signal-to-paired-noise 0.09-1.41, mean
~0.81). Even after the 2.4x reduction the per-duel sign is not resolvable at depth 4.

**The sign test from a killed pilot is uninterpretable and must not be quoted.** RelatIF picked
the more influential demo in 1 of 6 duels (p = 0.98). That is not evidence against RelatIF -- by
the pilot's own diagnostic those six signs are close to coin flips. The instrument returned
nothing; it did not return a negative.

To resolve a duel sign would need ~(1.96/0.8)^2 ~ 6x the depth, i.e. depth ~24 per arm: 864
retrains ~ 22 solo-h for ONE binomial test at n = 18. Strictly worse than spending the same GPU
on masks (#29). **Do not attempt the duel design on this corpus again.**

Taken with #33 this is the same statement from both sides. The estimator side: the two estimators
rank-correlate 0.547 and rarely disagree. The outcome side: when they do disagree, the thing they
disagree about is too small to measure one demo at a time. Demo-grain attribution on 135 demos is
signal-starved at the level of individual demos; only aggregate many-demo contrasts carry
measurable information.

## 36. (RESULT, and the pass's answer) The absolute half-ceiling bar is REACHABLE -- it was measuring the grain

Seven passes failed the `ratio >= 0.5` bar out of sample at demo grain, and pass 7's HANDOFF
open thread #4 raised the possibility that it is unreachable for any estimator and therefore
measures the ceiling rather than discriminating hypotheses.

At cluster grain, **plain GradDot_dmean with no correction at all** reaches ratio **0.707**
(Kendall tau_b, LDS 0.4747 vs ceiling 0.6715) and **0.834** (Spearman, 0.6789 vs 0.8141) on 149
out-of-sample conditional masks from campaign N, depth 4. Preregistered as a family of one,
frozen while the campaign had zero runs, scored once.

The bar is fine. The unit was too small.

**Caveat that belongs next to the number every time it is quoted:** a coarser grain is partly an
easier prediction problem -- among masks containing the target, the variation is which 4-5 of the
other 8 clusters are present, which has far fewer degrees of freedom than 135 demos. Normalising
by the cluster-grain ceiling controls outcome noise, not this. Pass 8 shows the measurement works
at cluster grain; it does not rescue per-demo attribution.

## 37. (RESULT, and it overturns three passes of work) Every self-influence / leverage correction REVERSES at cluster grain

On campaign N (fresh, disjoint, 278 masks), Kendall tau_b, paired against GradDot_dmean:

    relatif_C5     -0.627  [-0.735, -0.518]      (demo grain, pass 7: +0.06 [-0.02, +0.14])
    surrogate_C5   -0.539  [-0.661, -0.414]
    ensemble_C5    -0.751  [-0.858, -0.638]
    leverage_C7    -0.474  [-0.585, -0.360]

and it replicates the independent Stage F scan (-0.28 to -0.63) on a **different draw** and a
**different outcome pipeline** (probe battery vs `heldout_frame_losses`).

Given pass 7 measured the demo-grain benefit of the same frozen config at +0.06 with an interval
touching zero, the parsimonious reading is that **the correction was fitting demo-grain noise**.
Where signal is abundant it does not merely fail to help -- it destroys a ranking that was
already good. Do not carry these corrections to a new corpus without re-testing them at the grain
that corpus actually supports.

## 38. (DESIGN) At cluster grain the mask axis is CAPPED, which inverts BLOCKERS #29

BLOCKERS #29 measured masks beating seeds ~5:1 and concluded "buy masks at depth 2". That was
measured where the mask supply is effectively unlimited: demo masks are 68-of-135 subsets.

The cluster space is finite and small:

    C(9,4) = 126     C(9,5) = 126     C(9,6) = 84     total 336

Stage F consumed 58 distinct 5-of-9 subsets; campaign N took all 278 that remain. **With the mask
axis exhausted, depth is the only axis left to buy.** This is a consequence of the combinatorics,
not a contradiction of #29 -- but it means the demo-grain allocation rule must not be applied at
cluster grain without re-deriving it.

Two measurement traps found while re-deriving it, both of which would have undersized the campaign:

- **Subsampling n of n is deterministic**, so its paired sd is identically 0. Including that point
  in a `sd ~ c/sqrt(n)` fit drags `c` down by (k-1)/k.
- **Seed depth cannot be measured from a 4-seed pool.** Sampling depth d without replacement from
  4 seeds shrinks toward the full mean by construction, hitting sd = 0 at d = 4. Only the 1->2
  step is interpretable. The apparent "seeds help a lot at cluster grain" is mostly this artifact.

## 39. (TRAP) The split-half noise ceiling returns NaN at ODD seed depth

`confirm_mseries.ceiling` -- the recipe PREREG_M and PREREG_N both name -- builds two DISJOINT
EQUAL halves (`h = S // 2`) and drops any split whose remainder is the wrong size. At odd S every
split is dropped and it returns **NaN**, silently. Verified: S = 2, 4, 6 give a ceiling; S = 3, 5
give NaN.

PREREG_N specified depth 5, so scoring at the achieved depth would have produced a NaN bar and
failed the primary for a mechanical reason. `confirm_nseries.analysis_depth()` analyses at the
largest EVEN depth, and applies it to **both** the LDS and the ceiling -- not the ceiling alone,
because Spearman-Brown extrapolates from half-depth to full S, so a 4-seed ceiling tested against
a depth-5 LDS is anti-conservative.

Found and fixed while campaign N was still running, with no result file written and no contrast
computed (commit 7937ca1). Pinned in `tests/test_p8.py` so it cannot regress silently.

**Any future prereg that names a seed depth should name an EVEN one.**

## 40. (INFRA) src/bootstrap.py disables CUDA on any box that is not the original 8-GPU machine

`ALLOWED_GPUS = (4, 5, 6, 7)`. On a 1-GPU box the `nvidia-smi -i 4,5,6,7` probe raises, the
`except` pins `CUDA_VISIBLE_DEVICES=4`, no device is visible, and `torch.cuda.is_available()`
becomes False -- surfacing much later as a confusing `torch.load` deserialization error rather
than as a GPU-selection failure.

`_pin_allowed_gpu` respects an already-set value, so the fix is environmental and needs no repo
edit: **export `CUDA_VISIBLE_DEVICES=0` on every command.** Do not "fix" bootstrap.py for one box.
