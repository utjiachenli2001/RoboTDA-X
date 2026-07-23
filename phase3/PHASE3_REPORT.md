# RoboTDA-X — PHASE 3 REPORT

**What Phase 3 was for.** Phases 1–2 produced a strong null: data attribution is unfaithful at
demo grain, and the noise that hides it is *training-seed* noise, which gets worse with scale. An
external adversarial audit confirmed every headline number and found no critical error — but it
found latent defects, and it named one real gap: the ridge λ had never been tuned, so "attribution
was given its best shot" was an *assumption*. Phase 3 (A) fixes the defects and proves the fixes
change nothing, and (B) runs five follow-ups that close the remaining reviewer attack surfaces.

Every number below is read from an artifact file written by an actual run. Stages that did not run
say "did not run". Preregistration: `phase3/preregistration_phase3.json`, SHA-256
`efd515656a3226c0d6738a98ff304978288096f1bbdaef6fbc907a213f09e98b`, locked before any Phase-3
training, before the λ sweep, and before any Phase-3 read-out.

---

## 0. Headline: what changed, and what did not

| | |
|---|---|
| **The three latent defects are fixed and change ZERO reported numbers** | 125/125 headline statistics recompute bit-for-bit under the hardened readers (`p6_no_change.json`) |
| **The λ sweep partially REVISES Phase 2** | Phase 2's default λ was *not* attribution's best shot: C1's demo-grain LDS nearly doubles (+0.256 → +0.504) when λ is tuned. The **verdict survives** — cross-validated, no λ clears the bar — but the margin is far thinner than Phase 2 reported, and **oracle-tuned, C1 would have passed.** |
| **The exact preconditioner actively HURTS** | LDS rises monotonically with λ across 8 orders of magnitude and is maximized in the degenerate limit where the Fisher/Gram inverse is *switched off entirely*. A raw gradient dot product beats *exact* IF and *exact* TRAK at their default λ. |
| **Coarsening does not rescue attribution** | No grain g ∈ {1,3,5,15} clears half-ceiling on either focal target. The failure is not one of *resolution*. |

---

## 1. P6 — Audit hardening — **GATE PASSED**

### 1.1 The fixes

| # | defect | fix | verification |
|---|---|---|---|
| 6.1 | `rollout.py` writes `outcomes.json` **before** it raises on rollout errors and **before** it writes `probe.marker`; every Phase-1/2 reader ingests on *file existence* only. A partially-failed run could be silently ingested. | `phase3/src/p3lib.py`: writes are atomic (tmp + `os.replace`); `read_artifact()` **refuses** any artifact whose completion marker is absent. Every Phase-3 reader uses it. `rollout_diffusion.py` does not write the artifact at all on error. | **CLEAN** — 730 run dirs swept, 581 artifact+marker pairs, **0 violations** (`p6_marker_sweep.json`) |
| 6.2 | Stage-C's Q=490 training sets contain the Phase-2 per-task probe demos (`demo_45..49`). Harmless today, but nothing *prevented* attribution on such a model. | `p3lib.assert_no_probe_leak()` — a hard assertion in every Phase-3 attribution entry point; it refuses to compute attribution for any model whose `demos.json` intersects the probe set being used. | **CLEAN** — loaded gun **confirmed** (the 6 Q490 runs contain **300** probe-demo inclusions), but **0 leaking model-artifact pairs** across 8 influence artifacts (`p6_probe_leak_check.json`) |
| 6.3 | `analysis.py:32` hardcodes `N_EPISODES=30` for the logit clamp; `stage_d.py:85` hardcodes the literal 30. | `p3lib.logit_success_rowwise()` reads the row's own `n_episodes`. | **NO-OP CONFIRMED** (`p6_episode_count_check.json`) |
| 6.4 | On a *copy* of this project made to another machine, 41/96 `stage_G6` run dirs were zero-byte (a truncated transfer). | Verify on **this** host; record SHA-256 of the parquet and of a run-dir size manifest so future copies are checkable. | **INTACT** — 96/96 dirs, 1296/1296 rows re-derive from raw `outcomes.json`, 5/5 spot rows exact, max\|diff\| = 0.0 (`p6_g6_integrity.json`) |

### 1.2 The no-change proof

Every table was **re-ingested from the raw per-run artifacts through the marker-gated reader**
(not read from the archived parquet) and every headline statistic **recomputed** with the fixed
row-wise transform, then diffed against the archive. **125 checks, 0 mismatches**
(`p6_no_change.json`). Representative:

| quantity | archived | recomputed with fixed readers |
|---|---|---|
| Gate-1 best ρ (Stage D, IF) | 0.2517482517482518 | **0.2517482517482518** |
| Bonferroni-significant cluster-grain cells, L2 outcome | 8 / 27 | **8 / 27** |
| P1 6-seed ceiling (SB), C1 | 0.9327641408751334 | **0.9327641408751334** |
| P1 demo LDS, C1 / TracIn | 0.368695652173913 | **0.368695652173913** |
| P1 demo LDS, C5 / IF | 0.38 | **0.38** |

**The three fixes close real traps and move no reported number.**

### 1.3 Two bugs the machine-checks caught — one of them mine

Honesty requires reporting these, because both were caught by checks rather than by inspection:

1. **My own probe-leak guard passed vacuously.** `phase2_probe_ids()` initially walked only the
   top level of `per_task_probes.json` and returned an **empty set** — so
   `assert_no_probe_leak()` compared against nothing and "passed" everywhere. A guard that always
   passes is worse than no guard. Fixed; a post-condition now makes an empty or wrong-sized probe
   set **fatal**. Only after the fix did the check confirm the loaded gun is real.
2. **The episode-count check raised a false alarm.** Scoped over all tables, it flagged
   `p3_outcomes.parquet` (n_episodes = **200**, not 30). Re-scoped by *which reader consumes which
   table*: both hardcoding readers only ever consume n=30 tables, and the 200-episode table is
   consumed only by `p3_analyze.py`, which already reads `n_episodes` row-wise. The defect is
   genuinely latent.

### 1.4 Bonus: Phase-1 training is bit-reproducible across time

A fresh `src/train.py --seed 401` reproduces the **archived** Phase-1 checkpoint
`runs/stage_G/G000_s401/final.pt` **bit-for-bit** (81/81 tensors, max abs diff 0.0)
(`p8_bitcheck.json`). This was not asked for; it is a strong reproducibility result for the whole
study.

---

## 2. P6.5 — The λ ridge sweep — **PHASE-2'S VERDICT STANDS, ITS MARGIN DOES NOT**

**This is the one place Phase 3 revises Phase 2, and it must be stated plainly.**

**Zero retrains.** `G = ΦΦᵀ` and `K = ΦTᵀ` do not depend on λ, so ONE gradient pass over the E=10
ensemble yields the *exact* IF and TRAK scores at *every* λ in closed form. Self-validation: at the
default λ, the recomputation reproduces the archived `influence_table.parquet` (ρ = 1.000000, max
relative diff 7e-6) — so the sweep machinery is the same machinery that produced Phase 1's numbers.

Demo grain, held-out L2, 6-seed ground truth, focal targets:

| target | λ = 1e-2 (**Phase-2's default**) | **oracle-tuned max** | **cross-validated** | bar (½·ceiling) |
|---|---|---|---|---|
| **C1** | +0.256 (ratio 0.27, p=0.114) | **+0.504 (ratio 0.54, p=0.0060) — CROSSES THE BAR** | +0.397 (0.43, p=0.027) | 0.466 |
| **C5** | +0.380 (ratio 0.41, p=0.034) | +0.432 (0.47, p=0.018) | +0.341 (0.37, p=0.052) | 0.459 |

**What this means, stated without spin:**

* **Phase 2's claim that "attribution was given its best shot" was false at the default λ.** C1's
  demo-grain LDS nearly **doubles** when λ is tuned. Phase 2 *understated* its attributors.
* **Oracle-tuned, C1 would PASS Phase-2's preregistered criterion** (ratio 0.54 ≥ 0.5, p = 0.0060 <
  0.025). That is a real result and it is reported as one.
* **Cross-validated, it does not.** With λ and the attributor frozen on the *other* focal target,
  C1 reaches 0.43 (p = 0.027) and C5 reaches 0.37 (p = 0.052) — both below the bar. Tuning λ on the
  target you score it against is not a method; it is a 12-point multiple comparison. This is
  exactly why the cross-validation was preregistered, before any of these numbers existed.
* **Verdict: the FAIL survives, but the honest headline is now "no *cross-validated* λ reaches half
  of the seed-ensembled ceiling", not "attribution is nowhere close".** The margin is thin.

### 2.1 The substantive discovery: the exact preconditioner HURTS

Demo-grain LDS rises **monotonically** with the ridge across ~8 orders of magnitude
(C1: −0.226 at λ=1e-6 → +0.504 at λ=1e+2) and is **maximized in the degenerate limit where the
preconditioner is switched off entirely**:

> As λ → ∞, both `(G + λI)⁻¹K` and the Woodbury empirical-Fisher IF converge to `K/λ`. Spearman is
> scale-invariant, so **both estimators converge to the ranking of a raw gradient dot product** —
> no Fisher inverse, no Gram inverse, no preconditioning at all.

Convergence to that analytic limit is verified **exactly** (gap = 0.00e+00 on all 9 targets).

So the *exact* empirical-Fisher inverse and the *exact* Gram inverse — the machinery the field
considers principled, and which Phase 1 deliberately computed **exactly** rather than
approximately, precisely so that a failure could not be blamed on sketching error — **actively
degrade the ranking in this regime.** A naive gradient dot product beats both at their default λ.

*(A subtlety the convergence check caught, and which is worth stating because it is easy to get
wrong: because λ_m is set per-member as `ridge_rel · mean(diag(G_m))`, the λ→∞ limit of the
**ensemble-mean** score is the per-member **scale-normalized** dot product, not the plain mean of
K_m. The two differ materially — 0.504 vs 0.397 on C1 — so per-member scale normalization matters
more here than the preconditioner does.)*

Figure: `figures/lambda_sweep.png`.

---

## 3. P7 — The grain-resolution ladder — **NO GRAIN QUALIFIES**

**Zero retrains.** Grains g ∈ {1, 3, 5, 15}; each cluster's 15 training demos partitioned into 15/g
groups. Coarsened predictor: `group_score = Σ_{d∈G} score_d`, and
`predicted_mask_score(M) = Σ_G group_score(G) · |G ∩ M| / g` — what an attributor that can only
resolve *groups* would predict. Ground truth: the existing 24 Stage-G masks, 6-seed mean, held-out
L2; ceilings reused unchanged.

Endpoint checks pass, as they must: g=1 reproduces the Phase-2 demo predictor exactly, and the two
grouping rules coincide exactly at g=1 and g=15.

**Preregistered read-out (focal C1/C5, rule (a)): no g ∈ {1,3,5,15} reaches half-ceiling with
p < 0.025.** The smallest qualifying grain is **None** for both focal targets.

| target | g=1 | g=3 | g=5 | g=15 | bar |
|---|---|---|---|---|---|
| **C1** | +0.369 (0.40) | +0.331 (0.36) | +0.015 (0.02) | −0.032 (−0.03) | 0.466 |
| **C5** | +0.380 (0.41) | +0.242 (0.26) | +0.253 (0.28) | +0.361 (0.39) | 0.459 |

**Coarsening does not rescue attribution — so the failure is not one of RESOLUTION.** This is the
cleanest statement of the null available: it is not that the attributor is aimed at too fine a
grain and would work if we asked it a coarser question. Asked the coarser question, on the same
masks, it still cannot answer.

**Descriptive (not confirmatory):**
* 3 of 9 targets *do* qualify at some grain: **C2** at g=1,3 (ratio **0.65** at g=3), **C9** at
  g=1,3,5 (ratio **0.66** at g=5), **C4** at g=15. So the signal is not uniformly absent — it is
  absent *on the focal targets*, and present on some others. Phase 1 saw the same heterogeneity.
* **Group STRUCTURE carries signal.** DTW-clustered grouping beats random grouping at intermediate
  grains: **+0.062 mean LDS at g=3 (winning on 6/9 targets)** and +0.059 at g=5. At g=1 and g=15
  the two rules coincide by construction and the difference is exactly 0 — a self-consistency check
  that the contrast is real.

Figure: `figures/grain_ladder.png`.

---

## 4. P8 — Seed-noise anatomy

### 4.1 P8a — What IS the seed noise? **Parameter initialization, not data order.**

**48/48 retrains, 0 failures, 6.10 GPU-h** (ledger line 8). 12 Stage-G masks (rng(909)) × 2 init
seeds {701,702} × 2 order seeds {801,802}, fully crossed.

**Instrument check first.** `train_factorial(init=401, order=401)` is **bit-identical** to
`src/train.py --seed 401` (81/81 tensors, max abs diff 0.0), and the trainer is reproducible across
repeats. Without that equality the "decomposition" would be measuring an artifact of a rewrite;
with it, INIT and ORDER are genuinely two factors of the *same* trainer (`p8_bitcheck.json`).

The 2×2 sum-of-squares decomposition was itself validated on synthetic tables (pure-INIT,
pure-ORDER, pure-INTERACTION each recovered exactly; SS components sum to the total over 200 random
tables — `SYNTHETIC_p8a_decomposition_tests.json`, labelled SYNTHETIC).

| outcome | **INIT** (init + dropout) | **ORDER** (batch permutation) | INTER+RESID |
|---|---|---|---|
| **held-out L2** (the primary) | **0.718** [0.570, 0.816] | 0.151 [0.090, 0.252] | 0.130 [0.084, 0.199] |
| logit success | 0.454 [0.377, 0.537] | 0.222 [0.175, 0.277] | 0.324 [0.237, 0.408] |

*(bootstrap 95% CIs over masks)*

**Preregistered read-out.** On the **held-out L2 outcome, INIT crosses the 70% threshold declared
in advance and is NAMED the dominant source** — it is ~4.8× the data-order effect, and it dominates
on 7 of 9 targets individually (C6 is the exception, where ORDER leads at 0.53). **Stated honestly:
the point estimate (0.718) clears 70%, but the bootstrap CI [0.570, 0.816] straddles the threshold.**
So "dominant" is safe to read as *"much the largest component"*, and not as *"confidently above
70%"*.

On the **logit-success outcome no factor dominates** (INIT 0.454, ORDER 0.222, INTER+RESID 0.324) —
the preregistered fallback reading applies: *seed noise there is not attributable to a single RNG
factor.*

**Disclosed confound (preregistered, not discovered afterwards):** `torch.manual_seed` drives both
parameter initialization *and* the dropout mask stream, so INIT is "init + dropout", not init
alone. With one replicate per cell, interaction and residual are confounded and are reported as one
term.

**Why this matters.** Phases 1–2 said "training-seed variance dominates data-composition variance"
without naming a mechanism. It now has one: on the outcome the study actually attributes against,
the variance is overwhelmingly **where the weights started**, not **what order the data arrived
in**. That is the harder of the two to engineer away — you cannot fix it with a better sampler or a
fixed shuffle; it is the loss landscape's own basin lottery.

### 4.2 P8b — Do the cheap repairs work? **No. Neither of them.**

**0 retrains**, 60 model-evaluations (16,200 episodes), same 12 masks.

**Determinism gate — and an honest note about it.** The first-pass gate *passed but was
uninformative*: on C1 (near-floor) all three episodes failed at the full 600-step horizon, so the
step counts matched trivially (`[600,600,600]` twice). A model that fails everything is
bit-identical to itself for uninteresting reasons. We therefore re-ran the gate on the **most
discriminating** probe task available (`p8b_determinism_strong.json`), exactly as Phase 2 did when
confirming its 50-init-state defect:

* arm (i): 6/6 episodes **succeed**, step counts `[105, 103, 100, 106, 105, 112]` — replayed **exactly**
* arm (ii): mixed outcomes, step counts `[600, 111, 103, 241, 114, 600]` — replayed **exactly**

Both repaired policies are genuinely deterministic.

**The result** (split-half mask-ranking reliability over the 12 masks):

| estimator | held-out L2 | logit success | cost |
|---|---|---|---|
| baseline, single seed (S=1) | +0.593 | +0.336 | 1 model |
| **arm (i): checkpoint-averaged (last 3 ckpts)** | +0.596 (**+0.003**) | +0.238 (**−0.099**) | **FREE** — checkpoints already exist |
| baseline, outcome mean (S=3) | +0.819 | +0.495 | 3 models |
| **arm (ii): action-ensemble (S=3)** | +0.799 (**−0.020**) | +0.497 (+0.002) | 3 models — same as the baseline |

* **Checkpoint averaging buys nothing** on the primary outcome (+0.003) and actively **hurts** the
  success outcome (−0.099). Averaging along the training trajectory does not substitute for
  averaging across initializations — consistent with P8a: the variance lives in *where the run
  started*, and every checkpoint of a run shares that starting point.
* **Action-ensembling is no better than outcome-averaging at the same seed cost** (−0.020 / +0.002).
  Ensembling in action space and ensembling in outcome space buy the same thing, because they are
  averaging over the same underlying object: the seeds.

**There is no shortcut.** Phase-2 P4 showed reliability is bought with seeds, not episodes. P8b
closes the obvious follow-up: it cannot be bought with *cheap tricks* either. The seeds have to be
trained.

## 5. P9 — The attribution-side cost law — **the attribution itself is barely reproducible**

**10/10 retrains, 0 failures, 1.35 GPU-h** (ledger line 2). Seeds 211–220, full 135-demo corpus,
identical config to Phase-1's Stage E (seeds 201–210) → **E = 20** members. Ridge held at the
frozen Phase-1 default (the ridge question is P6.5's; changing both at once would confound them).

Split-half reliability of the **per-demo ranking** — two disjoint E/2-member sub-ensembles, Spearman
between their score vectors over the 135 demos, averaged over 200 splits and over the 9 targets:

| attributor | E=2 | E=5 | **E=10** (what Phase 1 used) | E=20 | **E\* for r = 0.8** (SB **EXTRAPOLATION**) |
|---|---|---|---|---|---|
| **IF** | +0.174 | +0.188 | **+0.199** | +0.254 | **≈ 235 models** |
| **TRAK** | +0.134 | +0.196 | **+0.298** | +0.419 | ≈ 111 models |
| **TracIn** | +0.326 | +0.449 | **+0.679** | **+0.831** | ≈ 16 — **already met at E=20** |

**This is a result about Phase 1's own headline numbers.** At the E = 10 ensemble Phase 1 actually
used, IF's per-demo ranking has a split-half reliability of **0.199**, and the *top-15 composition*
— the set the intrusion statistics are computed on — has a Jaccard of only **0.222** across
sub-ensembles. Two halves of the same ensemble, trained on the *identical* 135 demos, disagree
about which demos matter. The Phase-1 top-15 intrusion statistics therefore rest on a ranking that
is itself mostly noise for IF and TRAK.

**TracIn is the exception, and by a wide margin** (0.831 at E=20; ~16 members suffice for 0.8). It
averages over 5 checkpoints per member, which is itself a variance-reduction device — the same
trick P8b tests on the ground-truth side. This also explains why TracIn was the best attributor on
C1 in Phase 2: it is the only one whose scores are stable enough to correlate with anything.

**The two-sided cost statement** (with Phase-2 P4), both halves labelled EXTRAPOLATIONS:

> A trustworthy demo-grain study of this corpus costs, on the **ground-truth side**, ~12.5× the
> training seeds to reach a 0.8 reliability on a near-floor cluster (Phase-2 P4) — and, on the
> **attribution side**, ~**235 ensemble members for IF** / ~111 for TRAK / ~16 for TracIn to reach
> a 0.8 per-demo ranking reliability (P9). Phase 1 spent 10.

Figure: `figures/ensemble_cost_law.png`.

## 6. P10 — The diffusion replication

### 6.1 The policy is real, and it is *better* than the BC-Transformer

`diffusers` is not installed and Phase 3 does not install into the frozen env, so DDPM/DDIM are
implemented directly (`diffusion_policy.py`, **18.25M params**, inside the 10–30M target). **7/7
SYNTHETIC unit tests pass** (`SYNTHETIC_diffusion_unit_tests.json`) — and **test 6 caught a real
bug**: the raw cosine schedule gives `alpha_bar[T] = cos(π/2)² = 0`, so the DDIM x₀-estimate divides
by ~0 and amplifies float32 roundoff by ~10⁴. Fixed with the standard beta-clip at 0.999.

**Calibration** (identical protocol to Phase-1's BC calibration: 50 demos of
`libero_goal/open_the_middle_drawer_of_the_cabinet`, 20 rollouts), **7 of the 10-run cap used**:

| stage | result |
|---|---|
| learning rate | 1e-4 → 0.05, 3e-4 → 0.45, **1e-3 → 0.85** |
| step budget | 8000 → 0.85, **16000 → 0.95**, 24000 → 0.95 (no gain) |
| chunk length H | **8 → 0.95**, 4 → 0.95 (tie), 16 → 0.80 |
| DDIM steps (eval-only, free) | **5 → 1.00 @ 95 s**, 10 → 0.95 @ 149 s, 20 → 1.00 @ 204 s |

**Frozen config** (`p10_config_frozen.json`, sha `fac011e6…`, frozen **before** P10a/P10b launched):
lr = 1e-3, 16000 steps, H = 8, **DDIM = 5 steps**.

**Best single-task success 100%, versus the BC-Transformer's 70% on the identical sanity.** Far
above the 40% non-viability stop threshold → **VIABLE**. The DDIM step count was chosen by the
preregistered rule (smallest within 1 SE of the best), which also halved P10b's rollout bill.

**Determinism gate — PASSED and informative** (`p10_determinism.json`). A diffusion policy is
deterministic only if the sampler is DDIM with η=0 *and* the initial latent is FIXED; we do both
(fixed latent, seed 12345 — the analogue of BC's argmax). Replay of 6 episodes: 5 successes / 1
failure, step counts `[124, 216, 600, 133, 124, 123]`, **bit-identical on re-run**. Episodes 50–55
replay 0–5 exactly — Phase-2's 50-init-state wrap re-confirmed for this policy class.

### 6.2 P10a — Gate-0 sanity: the phenomenon exists for this class, more strongly

12/12 retrains, 3.20 GPU-h. Paired on seeds 611–613. **Descriptive, no pass/fail.**

| target | target-only (15 demos) | co-train (135) | **margin** |
|---|---|---|---|
| C1 | 44.7% | 54.3% | **+9.7 pts** |
| C5 | 19.5% | 27.1% | **+7.6 pts** |

*(BC-Transformer, Phase-1 Gate 0: C1 +1.0, C5 +5.0 pts.)*

The diffusion policy is far stronger in absolute terms (C1 target-only 44.7% vs the
BC-Transformer's 21.0%) **and** its co-train margins are larger. So the faithfulness test in P10b is
being run on a policy where data composition demonstrably *does* move the outcome — the test is not
vacuous.

### 6.3 P10b — the demo-grain LDS: **the preregistered test is INCONCLUSIVE, because the instrument broke**

**144/144 retrains, 0 failures** (96 at seeds 601–604 + 48 at seeds 605–606; 31.64 + 15.42 GPU-h).
Full defect writeup: **`PHASE3_DEFECT.md`**.

#### The preregistered verdict, reported verbatim

| S | target | measured ceiling | bar | best preregistered attributor | verdict |
|---|---|---|---|---|---|
| 4 (preregistered) | C1 | **+0.040** | 0.020 | TracIn −0.055 | FAIL |
| 4 | C5 | **+0.081** | 0.041 | TracIn −0.162 | FAIL |
| 6 (disclosed deviation) | C1 | **+0.079** | 0.039 | TracIn +0.086 | FAIL |
| 6 | C5 | **+0.137** | 0.069 | TracIn −0.041 | FAIL |

**This FAIL is uninformative and the preregistered symmetric interpretation is NOT invoked.** A
ceiling is what an *oracle* could achieve; Phase 2's FAIL meant something **precisely because its
ceiling was 0.93**. Here the ground truth cannot see at all:

* **4 of 9 measured ceilings are NEGATIVE** — impossible for a true reliability.
* **The Spearman–Brown consistency check fails by 0.61.** The 1-seed reliability (0.271) predicts a
  6-seed ceiling of **0.690**; we measure **0.079**. In Phase 2 the same check agreed to **0.014**.
* **A reliability that DROPS when you average more seeds is a broken aggregator, not noise.**

#### Why: the diffusion policy is deterministic but **not seed-stable**, and its L2 is heavy-tailed

Both policies are deterministic *given their weights* — verified bit-identically. But determinism
is not stability. BC's executed action is an argmax GMM mode: smooth in the weights. The diffusion
policy's executed action is a 5-step DDIM trajectory from a **fixed latent** through a **multimodal**
denoiser — so an occasional seed lands that latent in the **wrong basin** and its held-out L2 blows
up 3–5×. Mask G000 (C1) across the six seeds: `[0.77, 2.65, 0.65, 0.68, 0.72, 0.68]`.

| | between-mask sd (signal) | within-mask seed sd (noise) | **S/N** |
|---|---|---|---|
| BC-Transformer (C1) | 0.117 | **0.073** | **1.61** |
| **Diffusion (C1)** | 0.267 | **0.489** | **0.54** |

**This is Phase 1's own pathology, reborn one level up.** `src/policy.py:l2` records why Phase 1
abandoned the GMM-NLL functional: *"unbounded and heavy-tailed, so when the GMM's sigma collapses on
some seeds the held-out mean NLL swings 8–10× … while the MEDIAN barely moves."* Phase 1 escaped it
by switching to L2. The diffusion policy re-introduces the identical failure mode **inside** the L2 —
not in the per-frame loss, but in its seed-to-seed distribution. The **median** restores the
instrument completely:

| ceiling | C1 | C5 | range over all 9 targets |
|---|---|---|---|
| **mean** (preregistered) | 0.079 | 0.137 | −0.01 … 0.56 |
| **median** | **0.568** | **0.718** | **0.57 … 0.77** |

#### The median diagnostic — **POST-HOC, and MIXED**

With a working instrument we can finally *ask* the question. Criterion arithmetic identical; only
the seed aggregator changes (`p10_diagnostic_median.json`):

| target | ceiling (median) | bar | best attributor | ρ | ratio | p | |
|---|---|---|---|---|---|---|---|
| **C1** | 0.568 | 0.284 | TracIn | **+0.420** | **0.74** | **0.0205** | crosses |
| **C5** | 0.718 | 0.359 | TracIn | +0.060 | 0.08 | 0.390 | nothing |

**This is suggestive, not decisive, and it is not reported as a result.** One focal target of two;
a p-value a hair under the 0.025 threshold; and a **post-hoc aggregator**. The `any-focal-target`
criterion form would return PASS, but a claim resting on that would be exactly the kind of
seed/analysis lottery Phase 2 exposed in its own Stage-B arm.

#### One clean positive: the diffusion attribution is *far more reproducible*

The fixed paired (t, ε) noise bank did exactly what it was preregistered to do
(`p10_attr_stability.json`) — split-half reliability across ensemble members, at **E = 5**:

| attributor | BC-Transformer | **Diffusion** |
|---|---|---|
| IF | 0.188 | **0.521** |
| TRAK | 0.196 | **0.532** |
| TracIn | 0.449 | **0.927** |

So the two sides move in **opposite** directions: the diffusion policy's *attribution* is far more
stable, while its *ground truth* is far less. TracIn is also the only attributor with consistent
signal (C1 +0.42, C7 +0.39, C6 +0.30, C2 +0.28) — coherent with it being the only stable one.

#### What P10 actually licenses

1. A diffusion policy is **viable and stronger** here than the BC-Transformer (100% vs 70%
   single-task; **larger** co-train margins: C1 +9.7, C5 +7.6).
2. Its executed-action ground truth is **deterministic but not seed-stable, and heavy-tailed** — so
   the **seed-mean outcome that this entire literature uses is a broken instrument for it.** That is
   a methodological finding in its own right, and arguably the most transferable thing P10 produced.
3. On the repaired instrument the faithfulness evidence is **suggestive but mixed** (C1 0.74 of
   oracle, C5 0.08). **P10 does not settle the external-validity question.** A follow-up must
   *preregister* the median aggregator, S ≥ 6, and an **up-front ceiling-usability gate**.

## 7. Budget: actual vs ledger

All read from `phase3/logs/*_summary.json` and `phase3/results/budget_actual.json`.

| stage | retrains (ledger) | retrains (actual) | ok / fail | GPU-h (ledger) | GPU-h (actual) |
|---|---|---|---|---|---|
| P6 hardening + λ sweep | 0 | **0** | — | 1 | **0.10** |
| P7 grain ladder | 0 | **0** | — | 0.5 | **0.00** |
| P8 bit-identity check (instrument) | — | 4 | 4 / 0 | — | 0.60 |
| P8a factorial | 48 | **48** | **48 / 0** | 8 | **6.10** |
| P8b variance-reduction (eval-only, 60 evals) | 0 | **0** | — | — | **4.00** |
| P9 ensemble + attribution | 10 | **10** | **10 / 0** | 2 | **1.45** |
| P10 calibration | ≤10 | **7** | 7 / 0 | — | **1.50** |
| P10 ensemble | 5 | **5** | **5 / 0** | — | **1.45** |
| P10a Gate-0 | 12 | **12** | **12 / 0** | — | **3.20** |
| P10b (seeds 601–604) | 96 | **96** | **96 / 0** | — | **31.64** |
| **P10b (seeds 605–606) — the defect remedy** | **0** | **48** | **48 / 0** | **0** | **15.42** |
| P10 attribution | — | 0 | — | — | 0.70 |
| **TOTAL** | **183** | **230** | **219 / 0** | **56.5** (alert **85**) | **66.17** |

**Every orchestrated retrain succeeded: 219/219, zero failures.** **74,130 episodes.**
**66.17 GPU-h — under the 85 GPU-h alert; no alert was tripped and no budget pause was needed.**

The overrun against the 56.5 nominal is **entirely** the 48-retrain S=6 remedy for the P10 instrument
defect (15.42 GPU-h), which did not exist in the ledger because the defect was not foreseen. The
preregistered per-stage guard was run on the first retrains of every kind and never fired (P8a
projected 6.9 vs its 8 line; P10b projected 30.8 vs an actual 31.64).

### Cuts
**NONE.** No stage was cut, at any point. The preregistered global cut order was never reached.

### Deviations, in full
1. **P6.5-EXT (disclosed extension).** The preregistered 8-point λ grid returned its maximum at its
   right *edge*, which is a truncation artifact, so the boundary was closed analytically (four more
   grid points plus the exact λ→∞ limit). The preregistered 8-point read-out is preserved verbatim.
2. **P10 diffusion trains 16000 steps, not 8000.** Phase 1's fixed-step convention exists to stop
   dataset size confounding optimization *within* a policy class; it is not a cross-class
   constraint, and calibration is the mechanism the preregistration provides. All P10 runs share
   16000 steps, so every within-P10 comparison is clean.
3. **P10 S = 4 → 6 (disclosed deviation, `PHASE3_DEFECT.md`).** The preregistered S=4 ceiling
   collapsed. Two seeds were added to match the BC arm and to enable Phase-2's 3v3/10-split ceiling
   estimator. **The criterion was not changed** — only S. Both S=4 and S=6 are reported.
4. **P10 median aggregator (POST-HOC DIAGNOSTIC, never a verdict).** Reported as a diagnostic with
   its status stated plainly, because the mean-aggregated instrument is provably broken on a
   heavy-tailed outcome.

### Incidents (reported, not quietly fixed)
1. **My probe-leak guard passed vacuously.** `phase2_probe_ids()` returned an *empty set*, so
   `assert_no_probe_leak()` compared against nothing. A guard that always passes is worse than no
   guard. Fixed; an empty/wrong-sized probe set is now fatal. Only *after* the fix did the check
   confirm the Q490 loaded gun is real.
2. **A shell-quoting bug killed 3 calibration launches** (JSON config inside `bash -c '...'`). Zero
   GPU-h consumed — no model trained. Fixed by passing the config in a file; a guard now
   distinguishes "harness failure" from "policy non-viability" so an all-failed stage can never be
   misread as a scientific result. The 10-run calibration budget was intact (7 used).
3. **P8b's first determinism gate was uninformative** — it passed on a near-floor model where every
   episode failed at the full horizon, so step counts matched trivially. Re-run on the most
   discriminating task; both arms replay bit-identically with varied, information-bearing step
   counts.

---

## 8. What the combined three-phase evidence now licenses — and what it forbids

**Licensed.**

Across **three phases, 810 retrains and ~127 GPU-h**, on a state-based LIBERO corpus with exact
(not sketched, not factorized) attribution estimators and counterfactually-validated ground truth:

1. **Demo-grain data attribution is unfaithful for the BC-Transformer in this regime — and the
   claim now survives its own strongest counter-arguments.** It is not the noise floor (Phase 2:
   the ceiling rose to 0.93 and the LDS did not move). It is not an untuned ridge (P6.5: no
   *cross-validated* λ in eight orders of magnitude reaches half of oracle). It is not a
   *resolution* mismatch (P7: no grain in {1,3,5,15} clears the bar either — coarsening does not
   rescue it).
2. **The exact preconditioner is part of the problem, not the solution.** Demo-grain LDS rises
   monotonically as the Fisher/Gram inverse is turned *off*, and is maximized in the degenerate
   limit — a raw gradient dot product beats exact IF and exact TRAK at their default λ. Exactness
   was never the missing ingredient.
3. **The noise has a name: parameter initialization.** It is 71.8% of the seed variance on the
   primary outcome, ~4.8× the data-order effect (P8a) — the basin lottery, not the shuffle.
4. **And it cannot be bought off cheaply.** Not with episodes (Phase-2 P4), not with checkpoint
   averaging, not with action-ensembling (P8b). Only with trained seeds.
5. **A trustworthy demo-grain study is far more expensive than anyone is paying — on BOTH sides.**
   Ground truth: ~12.5× the seeds for a 0.8 reliability on a near-floor cluster (Phase-2 P4).
   Attribution: **~235 ensemble members for IF**, ~111 for TRAK, ~16 for TracIn (P9) — *Phase 1
   spent 10*, at which IF's per-demo ranking reliability is **0.199** and its top-15 set has a
   Jaccard of **0.222** across sub-ensembles. **The headline influence rankings in the literature
   are, at typical ensemble sizes, mostly noise.**
6. **A methodological warning that generalizes beyond this study:** for a diffusion policy, the
   *seed-mean* of a closed-loop/executed-action outcome — the convention this entire literature uses
   — is a **broken estimator**. The policy is deterministic given its weights yet not seed-stable in
   action space, its outcome is heavy-tailed, and averaging *more* seeds makes the mean-aggregated
   reliability **worse**. The median restores a 0.57–0.77 ceiling from a collapsed one.

**Forbidden.**

* **We may NOT say the null generalizes across policy class.** P10's preregistered test is
  **inconclusive** — the instrument, not the attributor, failed. On the repaired (post-hoc)
  instrument the evidence is **mixed**: C1 reaches 0.74 of oracle (p = 0.021) and C5 reaches 0.08.
  Anyone reading this as "diffusion is the same" or as "diffusion is different" is reading past the
  evidence.
* **We may NOT say attribution is hopeless in general.** Three of nine targets clear half-ceiling at
  some grain (P7); TracIn is stable and consistently positive; and the *cross-validated* margin at
  the best λ is thin, not cavernous (C1 0.43 vs a 0.5 bar). The honest headline is
  *"no cross-validated estimator reaches half of the seed-ensembled ceiling on the focal targets"* —
  **not** *"attribution is nowhere close."*
* **We may NOT generalize past this regime**: one benchmark, state-based observations, 9 clusters,
  ≤500 demos/task, two policy classes. Phase-2 P3 already showed the situation gets *worse* with
  scale within this range, not better — but that is a statement about *this* range.
* **We may NOT report the P10 median result as a finding.** It is a post-hoc aggregator on one of two
  focal targets with p a hair under threshold. It is a **hypothesis for a preregistered follow-up**,
  and it is labelled as one everywhere it appears.

**The one-paragraph statement.** This study set out to ask whether data attribution can tell you
which robot demonstrations matter, and it can now say: *not at demo grain, not in this regime, and
not for want of a better estimator — we gave attribution the exact Fisher inverse, the tuned ridge,
a coarser question, and a seed-ensembled oracle to aim at, and it still fell short of half of what
that oracle can do.* But the deeper finding is not about attribution at all. It is that **the ground
truth this literature optimizes against is a far weaker instrument than anyone has checked**: its
reliability is bought with training seeds rather than rollouts, it costs an order of magnitude more
than is being spent on both the ground-truth and the attribution side, its dominant noise source is
the initialization lottery, and — for the diffusion policies the field is converging on — the
standard seed-mean estimator of it is **not merely noisy but structurally broken**. Before the field
asks for a better attributor, it should check whether it can measure the thing the attributor is
supposed to predict.

---

## 9. Artifact index

| artifact | contents |
|---|---|
| `preregistration_phase3.json` | locked before any Phase-3 training (sha `efd51565…`) |
| **`PHASE3_DEFECT.md`** | **the P10 instrument defect: collapsed ceiling, heavy-tailed outcome, remedy** |
| `results/p6_marker_sweep.json` | 730 run dirs, 0 artifact-without-marker violations |
| `results/p6_probe_leak_check.json` | Q490 loaded gun confirmed (300 inclusions); 0 leaking pairs |
| `results/p6_episode_count_check.json` | the hardcoded clamp is provably inert (scoped by reader) |
| `results/p6_g6_integrity.json` | 96/96 dirs intact; SHA-256 of parquet + size manifest |
| `results/p6_no_change.json` | **125/125 headline numbers bit-identical under the fixed readers** |
| `results/p6_lambda_sweep.json` | the preregistered 8-point λ grid, verbatim |
| `results/p6_lambda_sweep_extended.json` | boundary closed analytically; λ→∞ limit; cross-validation |
| `results/p7_grain_ladder.json` / `.csv` | no grain qualifies; DTW-vs-random contrast |
| `results/p8_bitcheck.json` | factorial trainer ≡ Phase-1 trainer, bit-for-bit |
| `results/p8a_variance_decomposition.json` | INIT 0.718 vs ORDER 0.151 on held-out L2 |
| `results/p8b_variance_reduction.json` | neither cheap repair works |
| `results/p8b_determinism_strong.json` | the *informative* determinism gate (discriminating task) |
| `results/p9_ensemble_cost_law.json` / `.csv` | IF needs ~235 members; at E=10 its reliability is 0.199 |
| `results/p10_calibration.json`, `p10_config_frozen.json` | 7/10 runs; frozen config (sha `fac011e6…`) |
| `results/p10_determinism.json` | DDIM η=0 + fixed latent ⇒ bit-identical replays |
| `results/p10a_gate0.json` | diffusion co-train margins (larger than BC's) |
| `results/p10_verdict_S4_PREREGISTERED.json` | **the preregistered S=4 verdict, archived verbatim** |
| `results/p10_verdict_S6.json` | the S=6 disclosed deviation — still inconclusive |
| `results/p10_diagnostic_median.json` | **POST-HOC DIAGNOSTIC** — mixed (C1 0.74, C5 0.08) |
| `results/p10_attr_stability.json` | diffusion attribution is *more* reproducible than BC's |
| `results/budget_actual.json` | 230 retrains, 66.17 GPU-h, 74,130 episodes, 0 failures |
| `results/report_verification.json` | **64/64 report numbers verified against artifacts** |
| `results/SYNTHETIC_diffusion_unit_tests.json` | **SYNTHETIC** — 7/7 DDPM/DDIM algebra tests |
| `results/SYNTHETIC_p8a_decomposition_tests.json` | **SYNTHETIC** — 2×2 decomposition validation |
| `figures/lambda_sweep.png` | the preconditioner hurts; LDS maximized where it is off |
| `figures/grain_ladder.png` | the preregistered P7 deliverable |
| `figures/seed_anatomy.png` | what the seed noise IS, and that the cheap repairs fail |
| `figures/ensemble_cost_law.png` | the attribution-side cost law |
| `figures/diffusion.png` | the broken instrument, the heavy tail, and the mixed result |
