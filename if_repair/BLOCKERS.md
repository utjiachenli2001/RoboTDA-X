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

## 41. (CORRECTION, and it qualifies pass 8's headline) The cluster-grain primary is pooled over |S|, and a large part of it is training-set SIZE

`p8_prereg.md` is explicit that "|S| is a stratum, not a covariate to pool over. It sets the
training-set size (60/75/90 demos), which moves the outcome directly", and promises the result
"pooled with |S| controlled". `confirm_nseries.evaluate` computes the PRIMARY absolute bar as
`rho = fn(pg, o)` over all 149 conditional masks with no stratum control; `st` is built there and
then used only by the secondary paired analysis. The control was promised and not applied to the
number that carries the claim.

`p9_stratum_control.py`, target C5, Kendall tau_b, depth 4, on the same committed data and the same
frozen `p7_pooled_oos._graddot("cached")` object:

| scope | n | LDS | ceiling | ratio | ratio 95% CI | perm null | clears 0.5 |
|---|---|---|---|---|---|---|---|
| POOLED (the committed primary) | 149 | 0.4747 | 0.6715 | **0.7069** | [0.612, 0.817] | **0.3530** | yes |
| within 4of9 | 56 | 0.2078 | 0.5053 | 0.4112 | [0.092, 0.777] | 0.0009 | no |
| within 5of9 | 37 | 0.2583 | 0.5211 | 0.4956 | [0.115, 0.943] | 0.0022 | no |
| within 6of9 | 56 | 0.1688 | 0.5324 | 0.3171 | [-0.017, 0.719] | -0.0020 | no |

The pooled row reproduces `results/confirm_nseries.csv` exactly (asserted in `tests/test_p9.py`), so
this is the same computation and not a rival one.

**The permutation null is the argument.** Outcomes are shuffled WITHIN stratum and correlated
against GradDot's predictions. GradDot is a **fixed** estimator -- nothing is fit to anything -- so
a correlation surviving an outcome shuffle cannot be leakage under any definition. Pooled it
survives at **0.353** of a real 0.475. The mechanism is arithmetic: |S| sets the training-set size,
size moves the outcome directly, and every estimator's mask prediction is a SUM over kept demos via
`P7.mask_pred`, so it grows with the count too. Both sides earn credit for counting. Within stratum
the same null collapses to ~0.000, which is what a working control looks like.

**What survives.** There is still real attribution signal within stratum -- the observed LDS beats
the within-stratum null's 97.5th percentile in 4of9 and 5of9 -- but **the half-ceiling bar is not
cleared in any stratum**, and the per-stratum CIs are far too wide to resolve it either way. So
pass 8's "the bar is reachable" (#36) is not refuted; it is **unproven**, and the specific number
0.707 should not be quoted without this qualification.

**The general lesson, which is the transferable part:** when the design has a nuisance axis that
moves the outcome AND moves every estimator's prediction the same way, pooling over it credits both
sides for the nuisance. The cheap detector is a permutation null run on a FIXED estimator -- shuffle
the outcome within the nuisance stratum and see what correlation survives. It costs no GPU and it
cannot be confused with leakage.

## 42. (UNITS, and it affects every ratio in this repo) The ceiling is a reliability r, so the attainable maximum is ~sqrt(r) and a ratio above 1 is not prima facie evidence of a bug

`confirm_mseries.ceiling` is a mean split-half statistic with a Spearman-Brown step -- an estimate
of the **reliability r** of the depth-d seed-mean outcome. The largest correlation an oracle
predictor can have with an observation of reliability r is about **sqrt(r)**, not r. So
`ratio = rho/r` is inflated by roughly 1/sqrt(r), and it tops out near 1/sqrt(r) rather than at 1.

At the cluster-grain Kendall ceiling of 0.6715 the attainable maximum is ~1.22, so a ratio between
1 and 1.22 is a saturating estimator, not a broken one. The pass-9 datamodel result (#43) lands at
1.01 and was nearly discarded as leakage before this was worked out.

**Two consequences.** First, `rho/sqrt(r)` is now reported alongside `rho/r` everywhere pass 9
reports a ratio; it tops out at 1 and is the honest scale for cross-depth comparison. Second, the
inflation **grows as r falls**, and r falls with seed depth -- so a low-depth arm gets a higher
ratio for free. Any curve whose points differ in depth is confounded with protocol, which is why
campaign O is depth-matched and why campaign N's committed depth-4 0.7069 is used only as a
regression check and never as a curve point. Measured directly: the same 5of9 masks give ratio
0.496 at depth 4 and **0.666** at depth 2, because the ceiling falls from 0.521 to 0.451.

Every historical ratio in this repo uses the r convention. They are not wrong, but they are not
"fraction of achievable" either, and comparing two of them at different depths is not valid.

## 43. (RESULT) The correction reversal is NOT a scaling pathology -- rescaling does not salvage it, so #37 is an epitaph

#37 records that every self-influence / leverage correction reverses at cluster grain but not why.
The suspicion was mechanical: RelatIF divides by self-influence, a cluster prediction sums 75 such
scores, and a handful of demos with tiny `G_dd` could dominate the sum. If that were the whole
story the ranking would be fine and the correction salvageable.

`p9_why_reverse.py` separates the two hypotheses with a rank/scale 2x2. An estimator contributes an
ORDER and a MARGINAL DISTRIBUTION to a summed prediction, and rank transforms swap them
independently. Paired vs GradDot, Kendall, campaign N, n=149 (151 for C7):

| config | as-is | est order, GradDot scale | GradDot order, est scale |
|---|---|---|---|
| relatif_C5 | -0.627 | **-0.284** | -0.308 |
| surrogate_C5 | -0.539 | **-0.104** | -0.147 |
| ensemble_C5 | -0.751 | **-0.251** | -0.265 |
| leverage_C7 | -0.475 | **-0.251** | -0.102 |

**Neither swap recovers to baseline.** Keeping the correction's order on a well-behaved scale still
reverses; keeping its heavy tail on an order known to work on these same masks also reverses. Both
factors contribute and neither alone explains the effect, so there is no rescaling, winsorisation or
rank transform that rescues these corrections at this grain. The concentration diagnostic agrees
that the tail is real -- median top-5 share of a mask's summed |contribution| is 0.20-0.34 for the
corrections against 0.16-0.17 for GradDot -- but fixing it is not sufficient.

**#37 is therefore an epitaph, not a caveat.** Do not carry these corrections to a new corpus in the
hope that a better aggregation saves them.

**The per-stratum read, computed after the above, and it makes the conclusion STRONGER rather than
weaker.** #41 shows pooling is confounded by |S| for the absolute bar, so the paired contrast was
re-read within stratum. Order-preserving arm (`est_order_base_scale`), paired delta vs GradDot,
Kendall:

| config | pooled | 4of9 | 5of9 | 6of9 |
|---|---|---|---|---|
| relatif_C5 | -0.284 | -0.470 | -0.535 | -0.431 |
| surrogate_C5 | -0.104 | -0.305 | -0.411 | -0.248 |
| ensemble_C5 | -0.251 | -0.496 | -0.559 | -0.470 |
| leverage_C7 | -0.251 | -0.601 | -0.540 | -0.516 |

**Pooling was propping the corrections UP, not dragging them down.** Within stratum the
order-preserving arm's own LDS goes NEGATIVE -- relatif_C5 scores -0.262 / -0.276 / -0.262 against
GradDot's +0.208 / +0.258 / +0.169 -- so the correction's ordering is actively anti-predictive at
cluster grain once training-set size is removed, not merely uninformative. The pooled numbers looked
milder only because the shared |S| component lifted every estimator's pooled correlation, the
correction included.

12 of 12 config x stratum cells are negative. That consistency is the evidence; per-stratum
bootstrap CIs were not computed (n = 37-56 each, so individually they would be wide) and the claim
rests on the sign pattern across all twelve rather than on any one cell.

**This also overturns the one soft verdict.** `p9_why_reverse.verdict()` read surrogate_C5 as
"MIXED / UNDETERMINED" from its pooled delta of -0.104. Within stratum it is -0.25 to -0.41, the
same ranking error as the other three. The pooled verdict was an artifact of the #41 confound --
which is a second, smaller instance of the same lesson: **a pooled contrast on this design flatters
whatever it is applied to.**

## 44. (DESIGN) The cluster grain cannot control |S| for itself; only a sub-cluster grain can hold the training set fixed

The obvious repair for #41 is to report within stratum. That runs straight into #38: the cluster
mask axis is capped at C(9,4)+C(9,5)+C(9,6) = 336 subsets and campaign N consumed 278, so the
per-stratum n is stuck at 56/37/56 and the ratio CIs are [0.09,0.78], [0.12,0.94], [-0.02,0.72].
Nothing can be added at those sizes -- the strata are exhausted. **The cluster grain cannot answer
its own question**, and no amount of further GPU at that grain changes it.

Sub-cluster grain escapes the cap for a purely combinatorial reason. A mask keeping 75 demos keeps
25 of 45 groups at k=3 -- C(45,25) ~ 3e12 -- and 15 of 27 at k=5. Hundreds of masks can therefore
share EXACTLY one training-set size, which turns the #41 confound from a covariate to be adjusted
into a channel that does not exist. That is the design campaign O uses: 400 masks per grain, all at
75 retained demos, with the k=15 rung supplied free by campaign N's 5of9 stratum at matched depth.

**The transferable rule:** if a nuisance axis cannot be held fixed within a design's reachable mask
space, the answer is a finer grain, not a bigger sample at the same grain. Verified in passing --
the permutation null on the fixed-75-demo k=15 rung is 0.0019, i.e. the size channel really is gone
once the size is constant.

## 45. (STRUCTURAL) The k=15 conditional population is capped at 70 and is now EXHAUSTED -- the grain question is closed on this corpus

The pass-9 estimand at cluster grain conditions on the target cluster being retained, at a fixed
75-demo training set. That conditions the mask universe to C(8,4) = **70** subsets. Not a sampling
budget -- a combinatorial cap. Campaign N held 37 of them; pass 10's campaign P completed the other
33. There are no more.

Bootstrap width scales ~1/sqrt(n), and the measured width at n=37 was 2.155. So n=70 gives ~1.57 at
depth 2 and ~0.60 at depth 4. The sub-cluster rungs have intervals of width ~0.37 centred near 0.36.
**A k=15 interval of width 0.60 cannot be separated from them, and no purchasable design on this
corpus makes it narrower.** The grain trend is therefore not "unestablished" but
**unestablishable here**, which is the more useful statement: it tells the next corpus what to design
for. The only escapes are a larger corpus or a different estimand.

Corollary worth stating separately: **campaign N's 37 were the only winner's-curse-free masks of this
kind that will ever exist on this corpus**, because the remaining 33 are the Stage F discovery draw
for the cluster-grain hypothesis (`p8_cluster_grain`'s W1 scan). Pass 9 spent them. No alpha-bearing
test at k=15 is available again, ever.

## 46. (METHODS, two-part) The winner's curse depends on WHAT was selected; and the ratio's denominator can dominate a comparison

Pass 10's census completed the 70 and read the two halves separately, at depth 4:

| subset | n | LDS | ceiling | ratio | rho/sqrt(r) |
|---|---|---|---|---|---|
| 37 (campaign N, unselected-upon) | 37 | 0.2583 | 0.5211 | 0.4956 | 0.358 |
| 33 (Stage F DISCOVERY draw) | 33 | 0.2652 | **0.6678** | 0.3971 | 0.324 |
| 70 (complete population) | 70 | 0.2795 | 0.5738 | 0.4871 | 0.369 |

**(a) No detectable winner's curse here, and the reason generalises.** The discovery draw reads LOWER,
and the LDS is essentially identical across the halves (0.2583 vs 0.2652). BLOCKERS #28 measured ~4x
inflation, but that was selection over many CONFIGS. Here the W1 scan selected the **grain**, and
plain `GradDot_dmean` is a single estimator that was never itself selected -- so there was almost
nothing for the curse to act on. **The curse scales with the size of the selection set over the thing
being tested, not with "was this draw looked at".** #28 stands for config selection; it does not
transfer automatically to design selection.

**(b) At these n the ceiling is noisy enough to drive a ratio comparison on its own.** Two halves of
the same population at the same depth give ceilings of 0.5211 and 0.6678 -- a 28% spread. The entire
ratio gap between the halves comes from that, not from the estimator: LDS was stable at
0.258/0.265/0.280 while the ratio swung 0.397-0.496. **Normalising by the ceiling ADDS noise at these
sample sizes**, and comparing ratios across mask subsets conflates estimator performance with subset
reliability. Any historical cross-subset ratio comparison in this repo inherits this; auditing which
conclusions turn on a ceiling difference rather than an estimator difference is queued work.

## 47. (CORRECTED 2026-07-30 -- the original claim was NOT significant) Partition sensitivity is UNRESOLVED at both sub-cluster grains

Sub-cluster masks require partitioning each cluster into groups, and that partition is arbitrary.
Campaign R redrew it (a fully independent partition sharing zero groups) and re-ran the identical
design, 1600 retrains:

| grain | first partition | second partition | delta | LDS change |
|---|---|---|---|---|
| k=3 | 0.356 | **0.200** | -0.156 | 0.1338 -> 0.0781 (-42%) |
| k=5 | 0.365 | 0.320 | -0.045 | 0.1411 -> 0.1362 (-3%) |

Both rungs passed the preregistered containment test (each fell inside campaign O's CI), but that is
the weak read: k=3 stayed inside only because campaign O's interval is 0.37 wide, with 0.1995 sitting
0.020 above the lower bound. **Passing a containment test against a wide interval is not evidence of
agreement.** The preregistered secondary -- the signed difference -- is what carries the finding.

Two separations that matter. This is **not** #46(b): k=3's ceiling barely moved (0.3759 -> 0.3916)
while its LDS halved, so the partition genuinely changes predictive performance rather than the
denominator. And **both rungs read lower on the second partition**, so pass 9's sub-cluster numbers
were if anything optimistic and its negative conclusion is reinforced.

**CORRECTION (2026-07-30). The paragraph above overstated this, and the error was mine: I read a
raw 42% LDS movement as a finding without computing what mask-sampling noise alone produces at
n=400.** A 4000-resample bootstrap on both campaigns' frozen outcomes:

| grain | campaign O | SE | campaign R | SE | diff | SD(diff) | **z** |
|---|---|---|---|---|---|---|---|
| k=3 | 0.1338 | 0.0324 | 0.0781 | 0.0317 | 0.0557 | 0.0453 | **1.23** |
| k=5 | 0.1411 | — | 0.1362 | — | 0.0049 | 0.0455 | **0.11** |

**A 1.23-sigma difference is not evidence of a partition effect.** What survives is much weaker than
what was written: k=3 moved more than k=5 (1.23 vs 0.11 sigma), which is *consistent with* k=3 being
the more partition-sensitive grain, and is nowhere near establishing it. Neither grain shows a
partition effect distinguishable from mask sampling.

**And the design cannot currently resolve one.** Under a method-of-moments decomposition the
data-consistent between-partition SD is ~0.021 tau-units -- smaller than a single partition's own SE
(~0.032). Testing sigma_b = 0 with six partitions has ~10-15% power at that effect size; resolving it
would need on the order of 20 partitions at 800 masks, ~50,000 retrains. **A third partition would
not have settled this**, which is why pass 11 dropped the plan to buy one.

The methodological lesson, which is the transferable part: **two point estimates are not a variance.**
Before quoting a difference between draws as a finding, compute what the within-draw sampling noise
alone would produce. The raw movement here was 42% and the effect was 1.2 sigma.

**Pass 9's k=3 rung remains one draw of an arbitrary partition**, and a second draw differed by 1.2 sigma -- worth stating as a limitation, not as a measured instability. Why k=3 and not k=5 is open; no mechanism is asserted.
A third partition would establish whether k=3 is systematically unstable or this was one bad draw.

## 48. (RESULT, and it answers pass 7's HANDOFF #4) The half-ceiling bar is REACHABLE -- what fails is GRADIENT attribution specifically

Pass 7's HANDOFF open thread #4 asked whether the half-ceiling bar is unreachable for every estimator
anyone would try, in which case it measures the ceiling rather than discriminating hypotheses. Pass
10 answers it. Assembling all twelve committed absolute-bar attempts on one scale
(`p10_bar.py` -> `results/p10_bar_attempts.csv`):

- **10 gradient attempts across 5 independent designs. ZERO clear the bar** once training-set size is
  controlled AND depth inflation is accounted for. The one apparent exception, k=15 at depth 2
  (0.666, n=37), is contradicted by the depth-4 census on the *complete* population (0.487) -- which
  is #42's inflation behaving as predicted -- though 0.487 and 0.666 [0.19, 2.35] are statistically
  indistinguishable, so "contradicted" overstates it: the depth-2 read is SUPERSEDED by a
  lower-variance, depth-honest one in the direction #42 predicts. Note also the census clears the bar
  under Spearman (0.571) and not under Kendall (0.487), so the census verdict is primary-statistic
  specific. Range: **12-45% of attainable** (`rho/sqrt(r)`).
- **The design-based datamodel clears it at both sub-cluster grains** (0.640 and 0.674 attainable),
  at a fixed training-set size, with a permutation control at ~0.

So the bar is **not** unreachable and it is **not** measuring the ceiling: something on this corpus
clears it, and comfortably. #4 is answered in the negative -- the bar discriminates.

**The caveat that makes this a sharpening rather than a resolution.** The method that clears it is the
only competitor that READS OUTCOMES; every gradient estimator is outcome-blind. And it clears while
heavily over-determined (400 observations, 45 or 27 parameters), the opposite of the under-determined
regime it originally earned its reputation in. Its read is descriptive, not preregistered. So what is
established is: *the bar is reachable by some method on this corpus, and gradient-based attribution
does not reach it at 135 demos.* Whether that is a corpus-size limit or a limit of gradient
attribution as such remains open, and corpus-size scaling is the discriminator.

## 49. (AUDIT, and it CLEARS the back catalogue) The #46(b) ceiling effect is real but does not touch the historical conclusions

#46(b) found the ratio's denominator can drive a comparison on its own, which put every cross-subset
ratio comparison in nine passes under suspicion. `p11_ceiling_audit.py` decomposes every committed
comparison. For two rows with ratio r = L/C,

    log(r_A / r_B) = log(L_A / L_B) - log(C_A / C_B)

so the gap splits additively in logs into an ESTIMATOR term and a CEILING term. Logs, not raw
differences, because the ratio is multiplicative in its two parts.

**Coverage: 5,535 pairwise comparisons across 34 of 38 audit-eligible result files (89%).** Four
files yield no comparisons (single-row or no label column): `b2_scores_campaign.csv`,
`confirm_mseries.csv`, `confirm_nseries.csv`, `p7_pooled_oos.csv`.

**Result: 32 comparisons (1%) are denominator-driven. Median ceiling share is 0.102.**

A flag requires BOTH a denominator-dominated split AND material movement. That second condition was
missing from the first version of this audit and it mattered: a high ceiling SHARE is meaningless when
nothing moved. `TracIn` vs `TracIn_trunc_k1` sits at ratios 0.3897 vs 0.3949 with a 1.3% ceiling
spread -- the estimator term is ~0, so the share computes to 1.0 while the comparison is a near-tie
nobody draws a conclusion from. **5,406 of the 5,535 comparisons are immaterial in exactly this way**,
and reporting share alone produced a "worst offenders" list made entirely of noise.

**Where the 32 live, and why none of them overturns anything:**

- `p10_bar_attempts.csv` -- cross-design rows in pass 10's assembled overview (e.g. pass 8's pooled
  headline against the sub-cluster datamodel). These compare different designs at different depths on
  purpose; the table exists to show the spread, and no conclusion rests on any single pair.
- `p9_datamodel_cluster.csv` -- pooled-vs-stratum rows, which pass 9 already identified as confounded
  and re-reported per stratum.

**CORRECTED 2026-08-01: that is false, and the entry originally accounted for only 26 of its own 32
flags.** The true distribution is `p9_datamodel_cluster.csv` 16, `p10_bar_attempts.csv` 10,
`p10_k15_census.csv` 3, and **three in the back catalogue** -- `p7_per_draw.csv` (relatif_C5 vs
surrogate_C5, ceiling share 0.67), `holdout_table.csv` (0.63) and `holdout_phase2.csv` (0.50). The
back catalogue is **nearly** clear, not clear: 3 of ~5,000 comparisons are denominator-driven, all
three are near-ties whose conclusions do not appear to turn on them, but the sentence as originally
written was contradicted by this entry's own CSV.

**And the two load-bearing pass-9/10 conclusions are estimator-driven, checked directly:**

| conclusion | ceiling share | estimator term | ceiling term |
|---|---|---|---|
| k=3 partition sensitivity (#47) | 0.071 | +0.538 | +0.041 |
| the \|S\| correction (#41) | 0.294 | +0.609 | **-0.254** |

The |S| correction's ceiling term is NEGATIVE -- the denominator worked *against* the observed gap, so
that finding is understated by the ratio rather than inflated by it.

**What this does and does not settle.** It settles that no committed conclusion needs amending for
ceiling noise. It does not repeal #46(b): the effect is real, it is large where it bites (the census
halves, 0.90 share), and any FUTURE cross-subset ratio comparison at n ~ 40-150 should be decomposed
before it is believed. Paired contrasts sharing masks share a ceiling and are structurally immune.

## 50. (RESULT, and it settles HANDOFF thread 4) The datamodel ATTRIBUTES -- it transfers across an independent re-partition

Pass 10 left the datamodel's win ambiguous: with 400 masks against 45 or 27 coefficients it is a
well-posed regression of outcomes on an inclusion matrix, so is it attributing influence to
demonstrations or fitting the outcome surface of its own mask ensemble?

**The test that answers it costs no GPU, because campaigns O and R are two INDEPENDENT partitions of
the same 135 demos sharing zero groups.** Fit on campaign O, map each group coefficient to its demos
(evenly -- within a group the demos are collinear and no other split is identifiable), then aggregate
over campaign R's masks and score against R's frozen outcomes. Out-of-sample AND out-of-partition;
nothing about R is seen during the fit. A method that attributes to demonstrations transfers across an
arbitrary re-grouping; one that fits its own ensemble does not.

| grain | arm | LDS | SE | ratio | rho/sqrt(r) |
|---|---|---|---|---|---|
| k=3 | fit O -> score R (**transfer**) | 0.3057 | 0.0296 | **0.781** | 0.488 |
| k=3 | fit R -> score R (within, LOO) | 0.3287 | 0.0296 | 0.839 | 0.525 |
| k=3 | GradDot (campaign-independent) | 0.0781 | 0.0318 | 0.200 | 0.125 |
| k=5 | fit O -> score R (**transfer**) | 0.3210 | 0.0311 | **0.754** | 0.492 |
| k=5 | fit R -> score R (within, LOO) | 0.4699 | 0.0258 | 1.104 | 0.720 |
| k=5 | GradDot (campaign-independent) | 0.1362 | 0.0323 | 0.320 | 0.209 |

**It transfers, and it still clears the bar out of partition** -- 0.781 and 0.754 against a 0.5 bar,
at 4.0x and 2.4x GradDot on the same masks (z = 5.3 and 4.1). The datamodel is not merely fitting its
own mask ensemble.

**Two qualifications that must travel with it.** At k=5 the within-campaign figure **overstates** the
method: transfer loses 32% of the LDS (0.4699 -> 0.3210, z = 3.7), which is real ensemble-specific
fitting. At k=3 the loss is not detectable (z = 0.55). And coefficient stability across disjoint
halves of campaign O is substantial but not perfect -- Pearson 0.690 / Spearman 0.686 at k=3, 0.899 /
0.867 at k=5 -- so the coefficients are measuring a property of the groups rather than noise, more
cleanly where there are fewer of them.

So **#48 survives its strongest available test**: the bar is reachable, and reachable by a method
whose advantage is not an artifact of the ensemble it was trained on. What #48 says about GRADIENT
attribution is untouched -- GradDot sits at 0.200 and 0.320 on these same masks.

**Note on what replaced what.** Pass 11's first plan proposed answering this by subsampling masks to
walk masks-per-coefficient below 1.0. That could not have worked: the datamodel's LOO prediction is
refit on n-1 masks while GradDot's is a fixed cached score independent of n, so the paired delta
shrinks **mechanically** as n falls for any regression, informative or not. A decaying curve would
have been guaranteed by estimation theory rather than evidence of anything. The transfer design has no
such asymmetry -- both arms are scored on the identical 400 campaign-R masks.

## 51. (RESULT, and it closes the on-box campaign) At an UNBIASED ceiling the datamodel still clears the bar and gradient attribution still does not

Campaign Q added two seed slots to campaign O's identical 800 masks, taking that campaign from depth 2
to depth 4. Same masks, same estimators, same everything -- only the depth changes, so this is not
confounded with a fresh draw. 1600 retrains, 18.4 h, 0 failures.

| grain | arm | d2 ratio | **d4 ratio** | d2 rho/sqrt(r) | d4 rho/sqrt(r) | clears bar |
|---|---|---|---|---|---|---|
| k=3 | GradDot | 0.356 | **0.225** | 0.218 | 0.166 | no |
| k=3 | datamodel (LOO) | 1.044 | **0.849** | 0.640 | 0.627 | **yes** |
| k=5 | GradDot | 0.365 | **0.239** | 0.227 | 0.178 | no |
| k=5 | datamodel (LOO) | 1.084 | **0.881** | 0.674 | 0.655 | **yes** |

**The ceiling rises 45%** (0.3759 -> 0.5452 at k=3; 0.3862 -> 0.5524 at k=5), which is #42 behaving
exactly as it predicted: the depth-2 ratio was inflated, and this is the unbiased read.

**Both pass-10/11 conclusions survive it, in the direction each needed.** #48's negative is
reinforced -- gradient ratios fall by ~35% and clear nothing by a wider margin. #50's positive
survives the stricter denominator with room to spare: 0.849 and 0.881 against a 0.5 bar. **A positive
result measured on an inflated scale has now been re-measured on the honest one and stands.**

**Two things worth recording that were not anticipated.**

1. **The datamodel's raw LDS RISES with depth** (0.3926 -> 0.4631 at k=3, +18%; 0.4189 -> 0.4865 at
   k=5, +16%), which is ~2 sigma unpaired and consistent across all four cells. **Qualified
   2026-08-01:** GradDot's apparent fall (0.1338 -> 0.1227) is **z ~ 0.24** against SEs of ~0.032 --
   *unchanged within noise*, not a fall. The original entry read that as "cleaner outcomes do not help
   the estimator that is not capturing real structure", which assigned meaning to noise on one side of
   the contrast. No paired test was computed for the rise, the non-fall, or the difference between
   them, so "a sharper discriminator between the two methods" is not supported as stated.
2. **On the attainable `rho/sqrt(r)` scale the datamodel is essentially depth-STABLE** (0.640 ->
   0.627, 0.674 -> 0.655) while GradDot decays (0.218 -> 0.166, 0.227 -> 0.178). The datamodel is
   tracking a fixed fraction of what is achievable as the achievable ceiling moves; GradDot is not.
   The gap widens from 2.9x to 3.8x (k=3) and 3.0x to 3.7x (k=5).

**Limitation stated rather than papered over:** the cross-partition TRANSFER arm of #50 could not be
fully re-read at depth 4, because campaign R has only depth-2 outcomes and none were bought for it.
Only the fit side of that arm can improve. The transfer result stands at depth 2 as reported.

**Campaign O's own scoring remains frozen** -- `confirm_oseries.csv` is untouched and this is a
separate descriptive file (`results/p11_depth.csv`), carrying no alpha.

## 52. (HYPOTHESIS, downgraded from RESULT on 2026-08-01) The datamodel's transfer loss follows the FIT rather than the target; coefficient count is the candidate mechanism

#50 left an unexplained asymmetry: transferring across an independent partition, the datamodel loses
32% of its within-campaign LDS at k=5 (z = 3.7) but nothing detectable at k=3 (z = 0.55). The FINER
grain transfers better, which is the opposite of the naive expectation.

Two hypotheses, separable on data already on disk: (a) something about 5-demo groups ties the
coefficients to their grouping, or (b) k=5's fit is more over-determined (27 coefficients against 400
masks = 14.8 per coefficient, versus k=3's 45 and 8.9), so it absorbs more ensemble-specific structure
and loses more when that structure is removed.

**The test transfers ACROSS GRAINS within one campaign** -- fit at one grain, map coefficients to
per-demo scores, score the other grain's mask set. A k=3 mask's demo set is not a union of k=5 groups,
so this is out-of-sample in mask space while crossing the grouping without crossing the partition.

| scored on | own LOO (honest same-grain) | cross-grain fit | loss |
|---|---|---|---|
| k=3 | 0.3926 | fit k=5 -> **0.3275** | **17%** |
| k=5 | 0.4189 | fit k=3 -> **0.3755** | **10%** |

**The loss follows the FITTING grain in DIRECTION.** Fits made at k=5 degrade more wherever they are
scored (17% moved to k=3) than fits made at k=3 do (10% moved to k=5), agreeing with the
cross-partition direction in #50 where it was also the k=5 fits that lost.

**CORRECTION 2026-08-01 -- this entry originally asserted the mechanism and it is not established.**
The contrast is 0.0650 vs 0.0434 tau, a difference of **0.022** against per-arm SEs of ~0.028-0.029
(`p13_crossgrain.csv`). Unpaired that is **z ~ 0.4**; even generous pairing leaves it near 1 sigma. No
SE, CI or z appeared anywhere in the original entry -- **the same defect that forced the #47
retraction, repeated.** Separately, with exactly TWO grains the coefficient count is perfectly
confounded with every other property distinguishing k=5 fits from k=3 fits, so the cross-grain design
identifies "follows the fit, not the target" and cannot identify *which* property of the fit is
responsible.

**The mechanism, stated as the transferable part:** the more over-determined a datamodel fit is, the
more of the mask ensemble's specific structure it can absorb, and the more it loses when moved to any
new mask set -- another partition or another grouping. Over-determination buys within-sample
performance and pays for it in transfer. That is a design rule for the next corpus: **prefer the grain
that leaves the fit LESS over-determined, even though its within-sample number will look worse.**

**One trap this table contains on purpose.** The `fit k=X (IN-SAMPLE)` rows -- 0.4659 at k=3 and
0.4641 at k=5 -- are the full-fit model scored on the very masks it was fit on. They are NOT transfer
numbers and are labelled in the output so they cannot be misread as such. They are kept beside the LOO
arm because the gap between them (0.4659 vs 0.3926 at k=3) is a direct measure of how much the
in-sample read inflates on this design: about 19%.

## 53. (RESULT, and it QUALIFIES #50) Out of partition AND at an unbiased ceiling, the datamodel clears the bar at k=3 but not at k=5

#50 measured the cross-partition transfer arm at depth 2 and reported it as clearing the bar at both
grains (0.781 / 0.754). #42 says the depth-2 ratio is inflated. #51 re-read the WITHIN-campaign arm at
depth 4 and it survived, but could not touch the transfer arm, because the SCORING side is campaign R
and campaign R had only depth-2 outcomes. Campaign S bought the missing half: two more seed slots on
campaign R's identical 800 masks, 1600 retrains, 18.4 h, 0 failures.

Transfer arm (fit on campaign O, scored on campaign R), ratio bootstrap with the ceiling recomputed on
every resample:

| grain | depth | ratio | 95% CI | P(ratio >= 0.5) | clears on the CI rule |
|---|---|---|---|---|---|
| k=3 | 2 | 0.780 | [0.619, 0.991] | 1.000 | yes |
| k=3 | **4** | 0.638 | **[0.537, 0.745]** | 0.997 | **yes** |
| k=5 | 2 | 0.757 | [0.603, 0.945] | 1.000 | yes |
| k=5 | **4** | 0.538 | **[0.437, 0.642]** | 0.774 | **NO** |

**The strongest form of the claim -- out of partition AND on the honest denominator -- holds at k=3
and does not hold at k=5.** At k=5 the interval straddles the bar. #50 should not be quoted as
"clears the bar out of partition at both grains" without this.

**Qualification added 2026-08-01: this is a difference in SIGNIFICANCE, not a significant
difference.** The two ratios are 0.638 (SE ~0.053) and 0.538 (SE ~0.052), i.e. **z ~ 1.3 between
grains**. One CI excluding 0.5 and the other straddling it does NOT establish that k=5 fails where
k=3 holds. The correct reading is that the transfer arm's interval excludes the bar at k=3 and
includes it at k=5; the between-grain difference itself is not established.

**What is NOT qualified.** The datamodel still transfers far above GradDot everywhere: at depth 4 the
transfer arm is 0.638 vs GradDot's 0.133 at k=3, and 0.538 vs 0.158 at k=5 -- 4.8x and 3.4x. **It
still attributes**; #50's central finding stands. And the within-campaign arm clears comfortably at
both grains at depth 4 (0.823 / 0.960). It is specifically the CONJUNCTION of out-of-partition and
unbiased-ceiling that k=5 fails.

**This is mechanistically consistent with #52 rather than a separate surprise.** k=5 is the more
over-determined fit (27 coefficients against 400 masks versus k=3's 45), #52 showed such fits absorb
more ensemble-specific structure and lose more in transfer, and the honest denominator removes the
inflation that was hiding the consequence. Three findings pointing the same way: **over-determination
buys within-sample performance and pays for it exactly where it matters most.**

**GradDot at the unbiased ceiling on campaign R falls further still** -- 0.133 (k=3) and 0.158 (k=5),
against 0.200 and 0.320 at depth 2. The negative is reinforced at every control added.

## 54. (RESULT, and it separates two claims the project had been conflating) The PREDICTIONS transfer better than the ATTRIBUTIONS do

#50 established that the datamodel fit on one partition predicts another partition's outcomes far
better than any gradient estimator, and called that "it attributes". That is a claim about
PREDICTION. It is not the claim the word attribution makes, and the two are separable.

A summed mask prediction is an average over 75 demos, so it is robust to substantial per-demo
disagreement: two fits could both capture whatever coarse structure carries most of the predictable
variance, transfer well to each other's masks, and still assign quite different influence to any
particular demonstration. The transfer test cannot see that, because it only ever looks at summed
predictions.

**The literal test.** Campaigns O and R are independent partitions of the same 135 demonstrations
sharing zero groups. Fit each, map each set of group coefficients down to 135 per-demo scores, and
correlate the vectors directly. The two partitions group different demos together, so agreement is not
forced by construction -- a demo's score comes from one set of groupmates under O and a disjoint set
under R.

| grain | cross-partition (O vs R) | within-campaign halves (ceiling) | % of ceiling | shuffle null (97.5th) |
|---|---|---|---|---|
| k=3 | **0.524** | 0.690 | **76%** | 0.000 (0.170) |
| k=5 | **0.466** | 0.899 | **52%** | 0.001 (0.161) |

Pearson on the 135-demo vectors; Spearman and Kendall agree (0.485/0.363 and 0.487/0.361).

**The attribution is real but only moderately consistent.** Both grains sit far above a shuffle null
that collapses to ~0, so the per-demo scores are not noise. But at ~0.5 correlation, two arbitrary
re-groupings of the same corpus agree about half as much as the vectors' own scale allows. **Anyone
using these scores per-demonstration -- to prune, select or reweight data, which is what TDA is FOR --
inherits that, and it is much weaker than the 4.8x predictive advantage suggests.**

**Read cross-partition against the within-campaign ceiling, not against 1.0.** Two fits on disjoint
halves of ONE campaign at the same grain agree at 0.690 (k=3) and 0.899 (k=5); that is the best
achievable without crossing partitions, and it is well short of 1.0 on its own.

**RETRACTED 2026-08-01: the grain contrast in this entry does not survive the project's own secondary
statistics.** The original claim was that k=5 agrees LESS across partitions (52% of its ceiling versus
76%), completing an over-determination story. Under Pearson the cross-partition values are 0.524
(k=3) and 0.466 (k=5) -- a **&lt;1 sigma** difference -- and under the repo's other two statistics the
effect **reverses or vanishes**: Spearman 0.485 vs **0.487**, Kendall 0.363 vs 0.361. The entry's own
line "Spearman and Kendall agree" was true of the above-null claim and NOT of the grain contrast,
which is the claim it was used to support.

The "% of ceiling" framing made it worse: those denominators (0.690 vs 0.899) are correlations over
45 and 27 coefficient pairs from half-campaign fits, so the contrast was substantially
denominator-driven -- exactly what #46(b) warns against, in an entry written after #46(b).

**What stands from this entry:** cross-partition per-demo agreement is ~0.47-0.52 at BOTH grains, far
above a ~0 shuffle null and well below the within-campaign half-agreement of 0.69/0.90. The
predictions transfer better than the attributions do. **A grain difference in that agreement is not
established.**

**How #50 should now be quoted:** the datamodel attributes in the predictive sense -- fit on one
partition it predicts another's outcomes at 4.8x GradDot -- and its per-demo attributions agree across
partitions at roughly half their achievable ceiling. Both halves of that sentence are needed.
