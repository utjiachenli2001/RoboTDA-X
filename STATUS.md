# RoboTDA-X — STATUS

Append-only stage log. Every number is read from an artifact file written by an actual run.
GPU policy: only indices **4,5,6,7** are ever used; 0–3 belong to another user and are never touched.

---

## Stage 0 — environment + corpus (setup, no training)

- **Environment**: conda env `robotda_x`, Python 3.12.13, torch 2.11.0+cu130, robomimic 0.2.0,
  robosuite 1.4.0, mujoco 3.8.1, hf_libero 0.1.4, dattri 0.3.0. Pinned in `requirements.txt`
  (173 packages); full record in `ENV.md`.
- **Datasets**: all 5 LIBERO suites present (46 GB), `data/libero/download.done.marker`.
- **Install verification**: env reset + 5 random actions on one task per suite; action dim = 7;
  env obs exposes `object-state`, `robot0_eef_pos/quat`, `robot0_gripper_qpos`, `robot0_joint_pos`.
- **Key finding**: LIBERO hdf5 files do **not** store `object-state`. It is reconstructed by
  replaying each demo's stored sim states through the env, which also makes the offline and
  online featurizations byte-identical (same code path).
- **Correctness checks** (these are what license every later number):
  - open-loop replay of 3 demos' own actions → **success on all 3** (validates action execution,
    init-state setting, and success detection);
  - the constructed env's controller config **matches the demo-collection config exactly**
    (OSC_POSE, kp=150, output limits ±0.05/±0.5, control_freq 20).
- **Corpus** (`results/corpus_manifest.json`): 9 clusters — C1 goal, C2 spatial, C3 object,
  C4 long(10), C5–C9 = 5 largest libero_90 scene groups (KITCHEN_SCENE2/10/4/9/1, 7/6/6/6/5 tasks).
  **135 training demos** (15/cluster), **90 held-out** (10/cluster), disjoint; every probe task
  is covered by the training pool. state_dim = 128 (16 proprio + 112 padded object-state).
- **Extraction**: 700 demos cached (train + held-out + the full libero_goal Stage-C reserve),
  **0 errors**.
- **Phase segmentation**: transport fraction over the 135 training demos = **0.707 ± 0.113**
  (per-cluster means 0.58–0.79). The spec's under-identification flag (std < 0.05) does **not**
  fire → the phase contrast is identifiable. All 225 demos have a non-degenerate gripper range
  (min 0.0116) and ≥1 open/close crossing.
- **Mask designs** (frozen, audited): Stage F = 72 masks × 5-of-9 clusters (75 demos), every
  cluster in exactly 40 masks, pairwise co-inclusion in **[17,23]**; 12 noise-ceiling masks with
  each cluster in 6–8. Stage G = 24 masks × 68 demos, per-demo inclusion **[12,13]**,
  within-cluster stratification 8/7.

## Stage A — synthetic component tests

- **29 / 29 PASS** (`results/STAGE_A_SYNTHETIC_tests.json`, all labelled SYNTHETIC).
- Includes the required planted-signal test: the LDS scorer recovers true utilities at
  **ρ = 0.996** and finds no signal in random scores (**ρ = −0.095**); the noise-ceiling
  estimator returns ~1.0 on noiseless replicates and ~0 on pure noise.
- No synthetic number is mixed into any real result.

## Calibration (C1 only; spec §3 "tune briefly on C1, then freeze") — not a study result

`results/calibration.json`. Probe = C1 probe tasks, 3 × 10 rollouts, seed 101.

| run | demos | steps | success |
|---|---|---|---|
| single task, all 50 demos | 50 | 7.7k | **70.0%** ← architecture ceiling test |
| target-only (C1 pool) | 15 | 8k | 10.0% |
| target-only (C1 pool) | 15 | 20k | 10.0% |
| co-train (all 135) | 135 | 8k | 13.3% |
| co-train (all 135) | 135 | 20k | 6.7% |

- The 70% single-task result proves the pipeline (featurization → training → rollout → success
  detection) is sound end-to-end, so low success on 15-demo pools is **data scarcity, not a bug**
  (the 135-demo pool gives each of C1's 10 tasks only 1–2 demos).
- 20k steps is **not** better than 8k and costs 2.4× more → **frozen at 8000 gradient steps**
  (`configs/policy.yaml`), 19.22M-param BC-Transformer, GMM(5) head.
- **Protocol decision**: every run in the study gets the same *total gradient steps*, not the same
  *epochs*. Gate 0 and Stage C compare datasets of different size; at fixed epochs the larger
  dataset would silently receive proportionally more optimization, confounding "more data helps"
  with "more gradient steps". Inside Stages F/G every retrain has identical data size, so the two
  conventions coincide there.

## Preregistration

`preregistration.json` frozen **before Stage B**: all seeds, probe tasks, phase thresholds, gate
criteria, focal targets (C1, C5), statistical thresholds, mask-design hashes, budget ledger and
cut order. Only the Stage-A synthetic tests and the C1-only calibration above precede it; neither
is a study result.

---

## Stage B — GATE 0 (target C1)   [STOP/GO]

- runs: 10/10 ok, 0 failed. Wall-clock 19.5 min, **1.07 GPU-h**. 2,000 rollout episodes
  (10 models × 10 goal tasks × 20 rollouts). Artifacts: `runs/stage_B/*/cluster_eval.json`,
  `results/stage_B_gate0.json`.

| seed | target-only | co-train | paired margin |
|---|---|---|---|
| 101 | 9.5% | 37.5% | **+28.0** |
| 102 | 30.5% | 15.0% | −15.5 |
| 103 | 23.0% | 17.5% | −5.5 |
| 104 | 29.5% | 23.5% | −6.0 |
| 105 | 21.5% | 25.5% | +4.0 |

- **mean margin = +1.00 pts (SD 16.59), t = 0.135, one-sided p = 0.4497**
- criterion (margin ≥ +5 pts AND p < 0.05): **FAIL**

**Why it failed — the gate is underpowered, and this is the finding.** The margin SD across
training seeds is 16.6 points, so a 5-seed paired t-test has only **11% power** to detect the
+5-point margin the gate is testing for (power reaches 0.85 only at ~80 seeds). Target-only
success alone ranges 9.5%–30.5% across seeds on 200 episodes each — far beyond the ~3-point
binomial sampling SE, so this is *training-seed* variance, not rollout noise. The gate therefore
cannot distinguish "co-training does not help" from "co-training's effect is smaller than the
seed noise at n=5". Both readings are consistent with these numbers; the data do not separate them.

Per the spec's own fallback, the same paired design is now run for **C2 and C5** before any halt.

## Stage B — GATE 0 fallback (targets C2, C5) → **GATE 0 PASSES; pipeline PROCEEDS**

- runs: 20/20 ok, 0 failed. C2 wall 23.4 min / 1.27 GPU-h; C5 similar. 3,400 further episodes
  (C2: 10 tasks × 20 × 10 models = 2,000; C5: 7 tasks × 20 × 10 models = 1,400).

| target | per-seed margins (pts) | mean margin | SD | t | one-sided p | verdict |
|---|---|---|---|---|---|---|
| C1 goal | +28.0, −15.5, −5.5, −6.0, +4.0 | +1.00 | 16.59 | 0.135 | 0.4497 | **FAIL** |
| C2 spatial | +2.0, +20.0, +30.0, −4.5, +23.0 | **+14.10** | 14.66 | 2.151 | **0.0489** | **PASS** |
| C5 kitchen-scene2 | +9.3, +2.1, +2.1, +4.3, +7.1 | **+5.00** | 3.15 | 3.545 | **0.0120** | **PASS** |

**2 of 3 targets pass ⇒ proceed** (spec §4: "If some pass → proceed, noting heterogeneity is
itself a finding"). The heterogeneity is stark and is a headline result: co-training helps
libero_spatial a lot and KITCHEN_SCENE2 consistently (all 5 seeds positive), but does nothing
measurable for libero_goal.

**Implementation fix disclosed (not a change of criterion).** C5's exact mean margin is
`(65/7 + 15/7 + 15/7 + 30/7 + 50/7)/5 = 175/7/5 = 5` — *exactly* the +5-point threshold. Success
rates are rationals (k/20), so the mean margin is an exact rational; evaluated in binary floating
point the sum came to 4.999999999999999 and the `>= 5.0` test returned False, reporting C5 as
FAIL. The comparison is now performed in exact rational arithmetic (`fractions.Fraction`), which
flips C5 to PASS. The preregistered criterion is unchanged — only its evaluation was defective.
This was found by checking the knife-edge value, not by searching for a way to pass.

**Caveats carried forward, to be repeated in the report:**
- C2's p = 0.0489 is *marginal* (the criterion is p < 0.05).
- C1's failure is not evidence that co-training is useless there: with a seed-margin SD of 16.6
  pts, the 5-seed test has **11% power** to detect +5 pts. Underpowered ⇒ inconclusive, not null.
- C1's per-task decomposition (5 seeds × 20 rollouts = 100 episodes per task) shows co-training
  **consistently helps 4 tasks** (put_the_bowl_on_the_stove +22, put_the_bowl_on_top_of_the_cabinet
  +23, put_the_wine_bottle_on_the_rack +21, put_the_cream_cheese_in_the_bowl +10) and
  **consistently hurts others** (turn_on_the_stove −22, put_the_bowl_on_the_plate −19,
  push_the_plate_to_the_front_of_the_stove −11). The cluster-level ~0 margin is **cancellation of
  opposing per-task transfer**, not absence of transfer — which is precisely the thing a per-demo
  attribution study should explain.

---

## Stage E — TRAK ensemble (10 models on all 135 demos)

- 10/10 ok, 0 failed. Wall 21.9 min, **1.21 GPU-h**, 2,700 episodes.
- Probe-battery success by cluster (mean over the 10 seeds): C1 5%, C2 31%, C3 38%, C4 10%,
  C5 18%, C6 20%, C7 46%, C8 36%, C9 37%. Good dynamic range for the LDS outcomes.

## Attribution (full corpus)

- `results/influence_table.parquet`: 10,935 rows = 3 attributors × 3 functionals × 9 targets ×
  135 demos, averaged over the E=10 ensemble; per-member scores kept for the jackknife CIs.
- Estimators are the **exact** TRAK dual form and the **exact** Woodbury empirical-Fisher IF
  (not dattri's JL-projected TRAK / EK-FAC). Justification: with N=135 demos vs p=19.2M params,
  TRAK's k×k Gram is singular for any useful k, and EK-FAC exists only to approximate a Fisher
  inverse that is available in closed form at N=135. Both identities are verified to ~5e-15
  against dense brute force (Stage A tests 30–31). This gives attribution its **best possible
  shot**, so a Gate-1 failure cannot be blamed on sketching or factorisation error.

## ⚠ INSTRUMENT DEFECT FOUND AT GATE 1 — the loss functional was broken

Gate 1 initially returned **FAIL** (best |ρ| = 0.21, all CIs straddling 0). Before accepting
"attribution is unfaithful", I checked whether the **ground truth itself was predictable** —
i.e. whether two seeds of the *same* mask agree. They did not:

| mask | held-out GMM-NLL (seed 101) | (seed 102) |
|---|---|---|
| D00 | 23.6 | **187.0** |
| D04 | 64.2 | **286.1** |
| D10 | 27.1 | **239.3** |

An 8–10× swing on **identical data**. Mechanism (`results/` + Stage-A evidence): the *median*
per-frame NLL is nearly unchanged across those seeds (27.6 → 32.4 → 36.0), but 34–36% of frames
exceed NLL 100 in the bad runs — the GMM's σ collapses toward its 1e-4 floor, and NLL is
**unbounded** where the model is confidently wrong. The mean NLL was therefore measuring
σ-collapse tail noise, not data quality. **No attributor can predict an outcome that
unreliable**, so the original Gate-1 FAIL indicted the metric, not the attributors.

**Fix (deviation, disclosed):** the spec allows either loss — *"plain action loss (L2 or GMM NLL
— pick one, freeze)"*. I had frozen GMM NLL. The **evaluation/attribution functional** is now
**L2 on the executed action** (bounded, behaviourally meaningful); the **training objective is
unchanged** (still GMM NLL, which gives 70% on the single-task ceiling test). **No retraining
was required** — losses are recomputed from saved checkpoints by forward passes. The old NLL
numbers are retained in every `outcomes.json` under `*_nll`, and the original NLL-based Gate-1
verdict is archived at `results/stage_D_gate1_GMMNLL_ORIGINAL.json` (FAIL: IF +0.077,
TRAK +0.105, TracIn −0.210), so the effect of the change is fully auditable.

Effect on the pathology — mask D00's two seeds now read **L2 = 1.0654 vs 1.0695** (they read
23.6 vs 187.0 under NLL). L2 spans a sane 0.65–1.32 across masks with no explosions.

**This change was made after seeing a gate result, which is exactly what preregistration exists
to prevent — so it is flagged here, in the report, and in the deviations list.** The
justification is a demonstrable instrument defect (medians stable, means exploding, σ→floor),
established independently of any gate outcome; it was not a search for a way to pass.

## Stage D — GATE 1 (attributor sanity)   [STOP/GO] → **FAIL**

- 24/24 retrains ok (12 masks × 8-of-15 C1 demos × 2 seeds), 720 episodes.
- Verdict with the **fixed L2 functional**: best ρ = **+0.252** (IF) vs required 0.50 → **FAIL**.
  TRAK +0.189, TracIn −0.042. Full write-up: `GATE1_FAIL.md`.
- **Noise ceiling** (2-seed, Spearman–Brown): held-out L2 = **+0.570**; success = **−0.932**.
  So IF captures ~44% of the achievable signal, and the 0.50 bar is 88% of the ceiling — a
  near-oracle demand. Closed-loop success carries **no** reproducible rank information at demo
  grain (its two seeds are anti-correlated), so it cannot serve as an LDS outcome for C1.
- Consequence (per spec §5): Stages E–G still run — the retrains are attribution-agnostic ground
  truth — and Stage H's LDS numbers are computed **as evidence**, but the study does **not** make
  per-demo attribution claims as if attribution were trustworthy. Every downstream attribution
  statistic carries this caveat.

## Stage C — in-target quantity sweep (primary moderator)

- 18/18 ok, 0 failed. Wall 30.1 min, **1.81 GPU-h**, 3,600 episodes (each model on the full
  10-task goal suite × 20 rollouts). Artifacts: `results/stage_C_quantity.json`,
  `results/stage_C_intrusion.json`.

| Q (in-target goal demos) | target-only | co-train (Q + 120 outsiders) | **margin** |
|---|---|---|---|
| 15 | 21.0% ± 10.6 | 23.3% | **+2.3** |
| 50 | 48.5% ± 8.8 | 49.3% | **+0.8** |
| 490 | **93.8% ± 0.6** | 90.7% | **−3.2** |

**The co-train margin decays monotonically with in-target quantity and turns NEGATIVE.** With
15 demos the 120 outsiders are worth ~+2 points; by 490 in-target demos they *cost* ~3 points.
Outsider data helps only while the target is data-starved, then becomes interference. This is
the cleanest quantitative result in the study.

**It also independently validates the whole pipeline**: at Q=490 the policy reaches **93.8%
success (SD 0.6)** on the full goal suite. The low absolute numbers everywhere else are
therefore data scarcity, not a broken policy/rollout stack — the same conclusion the single-task
ceiling test (70%) reached.

**Outsider intrusion (TracIn → C1, plain L2 functional)** — fraction of the 120 outsiders scoring
above the median insider: Q=15 → 0.578 ± 0.101; Q=50 → 0.414 ± 0.164; Q=490 → 0.542 ± 0.159.
All are near 0.5 (chance) with no trend in Q: TracIn ranks outsider demos about as highly as the
target's own demos at every quantity. Consistent with the Gate-1 finding that attribution
carries little reliable per-demo signal here.

Q=490 is the spec's "Q=500 (full suite)" minus C1's 10 held-out probe demos, which must stay
unseen — the spec itself reserves "the full suite minus C1's held-out 10" for this stage.

## Stage F — cluster-grain ground-truth corpus (the main compute)

- **168/168 retrains ok, 0 failed.** Wall 319.4 min (5h19m), **21.17 GPU-h**
  (ledger allowed 84 — well under). **45,360 rollout episodes** (168 models × 27 probe tasks ×
  10). Artifact: `results/stage_F_outcomes.parquet` (1,512 rows), collected from 168/168 runs
  with none missing.
- Design executed as preregistered: 72 balanced masks (5-of-9 clusters = 75 demos; every cluster
  in exactly 40 masks; pairwise co-inclusion 17–23) × seeds 301,302, plus the 12 noise-ceiling
  masks × seeds 303,304.

Ground truth is well-behaved and confirms the need for the **conditional** LDS:

| target | success (target EXCLUDED) | success (INCLUDED) | held-out L2 (excl → incl) |
|---|---|---|---|
| C1 | 0.0% | 11.8% | 1.210 → 0.474 |
| C2 | 0.0% | 41.0% | 1.322 → 0.424 |
| C3 | 0.0% | 42.1% | 2.194 → 0.492 |
| C4 | 0.0% | 9.6% | 1.641 → 0.688 |
| C5 | 0.0% | 18.5% | 1.150 → 0.370 |
| C6 | 2.0% | 34.9% | 1.038 → 0.350 |
| C7 | 1.4% | 34.6% | 0.984 → 0.497 |
| C8 | 0.1% | 30.4% | 1.218 → 0.375 |
| C9 | 0.4% | 38.0% | 1.069 → 0.401 |

Excluding a cluster drives its success to ~0. A full-72-mask LDS would therefore score high
merely by predicting "is the target in the mask?" — which is trivial and says nothing about
*which outsiders matter*. This is exactly the inflation the spec warns about, and it is why the
primary metric is the conditional LDS over only the 40 target-included masks (the full-72 value
is reported as secondary, labelled inflated).

## Stage G — demo-grain corpus

- **48/48 ok, 0 failed.** Wall 92.8 min, **6.10 GPU-h**, 12,960 episodes.
  24 masks × 68 demos (within-cluster stratified 8/7; every demo in 12–13 masks) × 2 seeds.

## Stage H — attribution analysis (LDS vs noise ceiling)

Computed as **evidence**, not as validated attribution (Gate 1 FAILED).

**Cluster-grain conditional LDS (n=40 target-included masks), plain functional:**

| outcome | Bonferroni-significant cells (p < 0.0056) | best examples |
|---|---|---|
| held-out **L2** | **8 / 27** | C8 TRAK **+0.650** (ceiling 0.825), C7 TracIn **+0.607** (0.843), TRAK C5 +0.568 (0.564) |
| **success** | 3 / 27 | erratic, many negative |

The **conditional** ceiling (0.08–0.84) is far below the **all-12** ceiling (0.78–0.94). Using the
all-12 ceiling — which the spec's literal wording implies — would have judged every attributor
against an unattainable bar, because masks that *exclude* the target are trivially reproducible
(success → 0 for every seed). Both are reported; the conditional one is the honest comparator.

**So attribution is NOT uniformly useless**: at cluster grain, against the stable L2 outcome, it
reaches 40–100% of its noise ceiling for several targets. It is at **demo** grain that it fails.

**Demo grain (24 masks, n=24):** on the preregistered confirmatory outcome (success) **both focal
targets fail** (C1 best p = 0.056, C5 p = 0.711) → **the preregistered downgrade rule fires: all
per-demo claims downgrade to cluster grain.** (On the *secondary* loss outcome C1 would have
passed, p = 0.019 < 0.025, and C5 nearly, p = 0.042 — reported for completeness. The outcome was
**not** switched post-hoc to manufacture a pass.)

**Headline statistics** (best attributor per target, plain functional, delete-1 jackknife SE over
the E=10 ensemble) — **cluster-grain licence only**:

| target | attributor | intrusion > median insider | jk SE | > p75 insider | outsiders in top-15 | insider-adv. AUC |
|---|---|---|---|---|---|---|
| C1 | TRAK | 0.192 | 0.125 | 0.033 | 9/15 | 0.705 |
| C2 | TRAK | 0.550 | 0.316 | 0.108 | 11/15 | 0.458 |
| C3 | TRAK | **0.033** | 0.075 | 0.008 | 7/15 | 0.588 |
| C4 | IF | 0.158 | 0.185 | 0.033 | 9/15 | 0.702 |
| C5 | TRAK | 0.100 | 0.081 | 0.033 | 9/15 | 0.654 |
| C6 | TRAK | 0.567 | 0.272 | 0.158 | 12/15 | 0.509 |
| C7 | TracIn | 0.358 | 0.079 | 0.125 | 12/15 | 0.721 |
| C8 | TRAK | **0.733** | 0.457 | 0.075 | 11/15 | 0.487 |
| C9 | TracIn | 0.383 | 0.113 | 0.192 | **15/15** | 0.584 |

**9 of 9 targets have ≥1 outsider ranked above their 75th-percentile insider.** C9's fifteen
most-influential demos contain **none** of C9's own. C3 — the one cluster the DTW matrix flags as
a trajectory-space outlier (0.49–0.53 from everything else vs 0.051 within itself) — is also the
one with almost no intrusion (0.033). Several jackknife SEs are large (C8 0.457, C2 0.316): the
ensemble members disagree substantially, which is itself a caution.

**Best attributor per target is selected on the L2-outcome LDS, not the success LDS**, because the
success outcome's conditional ceiling is low and sometimes *negative* (C2 0.078, C7 0.117,
C8 −0.095) — ranking attributors on an outcome an oracle cannot predict would be ranking them on
noise. Both rankings are saved (`results/best_attributor_selection.json`).

**RQ3 (phase contrast)** — conditional LDS by attribution functional (mean over 9 targets,
L2 outcome): TracIn plain +0.199 / transport +0.196 / interaction +0.187; TRAK +0.303 / +0.249 /
+0.191; IF +0.057 / +0.066 / +0.085. **Essentially flat** — the phase-masked functionals do not
change predictive validity. This is a genuine null, not an under-identified one: the transport
fraction is 0.707 ± 0.113 across the 135 demos, far above the spec's std < 0.05 flag.

**RQ2 (moderators, trajectory-space only, NO image features)** — insider-advantage AUC regressed
on similarity: mean-outsider DTW r = **+0.011** (p = 0.98), MMD r = +0.098 (p = 0.80), bddl overlap
r = +0.412 (p = 0.27), within-target redundancy r = +0.482 (p = 0.19); OLS R² = 0.251 (n = 9).
**No moderator is significant** — trajectory-space similarity does not explain who intrudes.
n = 9 targets, so this is descriptive and underpowered, and is reported as such.

## Stage I — RQ4 (data selection at budget B=15)

- **24/24 ok, 0 failed.** 2.57 GPU-h, 1,440 episodes.

| condition | C1 | C5 |
|---|---|---|
| target-only | 12.8% ± 16.4 | **17.2% ± 12.1** |
| influence-top-15 | **13.3% ± 15.3** | **1.1% ± 1.0** |
| random-15 | 0.0% | 13.9% ± 14.2 |
| similarity-top-15 | 0.0% | 3.3% ± 4.4 |

Paired margins: on **C1** influence beats random and similarity by **+13.3 pts** and ties
target-only (+0.6). On **C5** influence is **−16.1 pts worse than simply using your own data**.
**Split verdict, reported plainly**: influence-based selection is *not* reliably better than the
trivial baseline — the expected outcome given Gate 1. (DTW-similarity selection is also poor, so
this is not evidence that similarity beats influence; both lose to in-domain data.)

---

# FINAL — all stages complete

- **322 retrains, 0 failures, ≈37.8 GPU-h** (ledger ~154 → **4× under budget**),
  **72,210 rollout episodes** (counted from artifact files; ledger ~70,500).
- **No budget cut was taken** — the pre-declared cut order was never invoked.
- Gates: **Gate 0 PASS** (C2, C5; C1 fails/underpowered) → proceed. **Gate 1 FAIL**
  (best ρ = 0.252 vs 0.50) → `GATE1_FAIL.md`; per-demo claims downgraded to cluster grain.
- Deliverables: `REPORT.md`, `GATE1_FAIL.md`, `ENV.md`, `preregistration.json`, 8 figures/tables
  in `figures/` + `results/`, all 16 headline numbers cross-checked against their artifacts.
