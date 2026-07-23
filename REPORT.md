# RoboTDA-X — Training-data attribution in robot-policy posttraining on LIBERO

**Final report.** Every number below is read from an artifact file written by an actual run;
paths are given. Nothing is extrapolated. Where something did not run, it says so.

---

## 0. Executive summary

Both preregistered STOP/GO gates were run. **Gate 0 passed on 2 of 3 targets; Gate 1 failed.**
The study therefore reports what the spec says it must: *attribution is not faithful enough to
license per-demo claims* — but the evidence is richer than a flat null, and two of the strongest
results in the study are not about attribution at all.

1. **Outsider data helps only while the target is starved, then actively hurts.** Co-training
   margin vs in-target quantity Q: **+2.3 pts (Q=15) → +0.8 (Q=50) → −3.2 (Q=490)**. At Q=490 the
   policy reaches **93.8% success**, so the low numbers elsewhere are data scarcity, not a broken
   stack. (§4)
2. **Whether co-training helps at all is target-dependent.** Gate 0: C2 **+14.1 pts** (p=0.049) and
   C5 **+5.0 pts** (p=0.012, all 5 seeds positive) pass; C1 **+1.0 pts** (p=0.45) fails — but C1's
   per-task decomposition shows co-training *consistently helps 4 tasks and consistently hurts 3*.
   The cluster-level null is **cancellation**, not absence of transfer. (§3)
3. **Attribution fails at demo grain.** Gate 1: best ρ = **+0.252** (IF) against a required 0.50
   and a measured noise ceiling of **+0.570**. The preregistered demo-grain downgrade rule fired
   (both focal targets failed). (§5, §7)
4. **But attribution is not uniformly useless.** At *cluster* grain against a stable loss outcome,
   **8 of 27** (target × attributor) cells are Bonferroni-significant, several at 70–100% of their
   noise ceiling (C8 TRAK +0.650 vs ceiling 0.825; C7 TracIn +0.607 vs 0.843). (§7)
5. **Outsider demos routinely outrank the target's own.** **9 of 9** targets have ≥1 outsider above
   their 75th-percentile insider; C9's fifteen most-influential demos contain **none** of C9's own.
   (§8 — cluster-grain licence only.)
6. **The dominant variance is the seed, not the data.** A 15-demo LIBERO policy's success swings
   9.5%–30.5% across training seeds at fixed data. This is why both gates were hard, and it is the
   study's central methodological finding.

**Two defects were found and fixed during the run, and both are disclosed in full (§10):** a
GPU-selection bug that let a bare script touch a forbidden GPU, and — more consequentially — a
**broken loss functional** whose repair changed a gate's inputs.

---

## 1. Environment and versions

`ENV.md`, `requirements.txt` (173 pinned packages).

| item | value |
|---|---|
| host | 8× RTX A6000 (48 GB); **only GPUs 4–7 used**, 0–3 never touched |
| Python | 3.12.13 (spec said 3.10 — deviation, §10) |
| torch | 2.11.0+cu130 (CUDA 13.0) |
| robomimic / robosuite / mujoco | 0.2.0 / 1.4.0 / 3.8.1 |
| LIBERO | `hf_libero` 0.1.4 (pip, not a repo clone — §10) |
| dattri | 0.3.0 (installed; **not used** for the estimators — §10) |

**Install verification.** Env reset + 5 random actions on one task per suite; action dim = **7**;
env obs exposes `object-state`, `robot0_eef_pos/quat`, `robot0_gripper_qpos`, `robot0_joint_pos`.

**Two correctness checks license every number in this report:**
- **Open-loop replay** of 3 demos' own actions → **success on all 3** (validates action execution,
  init-state setting, and success detection).
- The constructed env's controller **matches the demo-collection config exactly** (OSC_POSE,
  kp=150, output limits ±0.05/±0.5, control_freq 20), read from the hdf5 `env_args`.

---

## 2. Corpus

`results/corpus_manifest.json`. **M = 9 clusters**, resolved from the *installed* benchmark at load
time (no hardcoded task lists):

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 |
|---|---|---|---|---|---|---|---|---|---|
| suite | goal | spatial | object | long(10) | \<— 5 largest libero_90 scene groups —\> | | | | |
| scene | – | – | – | – | KITCHEN_2 | KITCHEN_10 | KITCHEN_4 | KITCHEN_9 | KITCHEN_1 |
| tasks | 10 | 10 | 10 | 10 | 7 | 6 | 6 | 6 | 5 |

- **135 training demos** (15/cluster), **90 held-out** (10/cluster), disjoint; every probe task is
  covered by the training pool. 27 probe tasks (3/cluster, greedy max-coverage over bddl objects).
- Observation = `[eef_pos(3), eef_quat(4), gripper_qpos(2), joint_pos(7), object-state→112]` =
  **128-D**, z-normalised with frozen train-pool stats. **No images anywhere.**
- **LIBERO hdf5 files do not store `object-state`.** It is reconstructed by replaying each demo's
  stored sim states through the env — which also makes the offline and online featurisations
  byte-identical (same code path). 700 demos extracted, **0 errors**.
- **Phase segmentation is identifiable**: transport fraction **0.707 ± 0.113** over the 135 demos
  (per-cluster 0.58–0.79). The spec's under-identification flag (std < 0.05) does **not** fire. All
  225 demos have a non-degenerate gripper range (min 0.0116) and ≥1 open/close crossing.
- **Mask designs** (frozen, audited): Stage F = 72 masks × 5-of-9 clusters (75 demos), every cluster
  in exactly **40** masks, pairwise co-inclusion in **[17,23]**; 12 noise-ceiling masks, each cluster
  in 6–8. Stage G = 24 masks × 68 demos, per-demo inclusion **[12,13]**, within-cluster 8/7.

**Stage A (synthetic):** **31/31 tests pass** (`results/STAGE_A_SYNTHETIC_tests.json`, all labelled
SYNTHETIC, never mixed with real results). Includes the required planted-signal test — the LDS
scorer recovers true utilities at **ρ = 0.996** and finds nothing in random scores (ρ = −0.095) —
and machine-precision verification of both attribution estimators (§6).

**Policy (frozen, `configs/policy.yaml`):** state-based BC-Transformer, causal, CTX=10, GMM(5) head,
**19.22 M params** (spec: 10–50 M). **8000 gradient steps for every run** — not fixed epochs: Gate 0
and Stage C compare datasets of different size, and fixed epochs would hand the larger dataset
proportionally more optimisation, confounding "more data helps" with "more steps". Inside Stages
F/G every retrain has identical data size, so the conventions coincide there.

---

## 3. Gate 0 (Stage B) — **PASS on 2 of 3 targets → proceed**

`results/stage_B_gate0.json`. Target-only vs co-train, 5 seeds, paired; each model evaluated on
**every task of the target cluster × 20 rollouts**.

| seed | C1 target / co-train | margin | C2 | margin | C5 | margin |
|---|---|---|---|---|---|---|
| 101 | 9.5 / 37.5 | **+28.0** | 50.0 / 52.0 | +2.0 | 7.9 / 17.1 | +9.3 |
| 102 | 30.5 / 15.0 | −15.5 | 20.5 / 40.5 | +20.0 | 13.6 / 15.7 | +2.1 |
| 103 | 23.0 / 17.5 | −5.5 | 15.5 / 45.5 | +30.0 | 17.9 / 20.0 | +2.1 |
| 104 | 29.5 / 23.5 | −6.0 | 41.5 / 37.0 | −4.5 | 10.7 / 15.0 | +4.3 |
| 105 | 21.5 / 25.5 | +4.0 | 9.5 / 32.5 | +23.0 | 12.9 / 20.0 | +7.1 |
| **mean (SD)** | | **+1.00** (16.59) | | **+14.10** (14.66) | | **+5.00** (3.15) |
| t, one-sided p | | 0.135, **0.4497** | | 2.151, **0.0489** | | 3.545, **0.0120** |
| **verdict** | | **FAIL** | | **PASS** | | **PASS** |

Criterion: margin ≥ +5 pts **and** one-sided paired p < 0.05. **2 of 3 pass ⇒ proceed** (spec §4:
"If some pass → proceed, noting heterogeneity is itself a finding").

**Caveats, stated plainly:**
- **C2's p = 0.0489 is marginal** against a 0.05 threshold.
- **C5's margin is exactly 5.000** — and this exposed a bug in my own gate code. Success rates are
  rationals (k/20), so the mean margin is an exact rational: (65+15+15+30+50)/7/5 = **exactly 5**.
  In binary floating point the sum evaluated to 4.999999999999999 and `>= 5.0` returned False,
  reporting C5 as FAIL. The comparison is now done in exact rational arithmetic, which flips C5 to
  PASS. **The preregistered criterion is unchanged; only its defective evaluation was fixed.** Found
  by checking the knife-edge value, not by hunting for a pass.
- **C1's failure is inconclusive, not a null.** With a seed-margin SD of 16.6 pts, a 5-seed paired
  test has **11% power** to detect +5 pts (85% power needs ~80 seeds). And the per-task
  decomposition (5 seeds × 20 rollouts = 100 episodes/task) shows co-training **consistently helps**
  `put_the_bowl_on_the_stove` (+22), `put_the_bowl_on_top_of_the_cabinet` (+23),
  `put_the_wine_bottle_on_the_rack` (+21), `put_the_cream_cheese_in_the_bowl` (+10) and
  **consistently hurts** `turn_on_the_stove` (−22), `put_the_bowl_on_the_plate` (−19),
  `push_the_plate_to_the_front_of_the_stove` (−11). **The cluster-level ~0 is cancellation of
  opposing per-task transfer** — precisely what a per-demo attribution method ought to explain.

---

## 4. Stage C — in-target quantity sweep (the cleanest result)

`results/stage_C_quantity.json`, `results/stage_C_intrusion.json`. 18/18 runs ok. Figure:
`figures/fig4_quantity.png`.

| Q (in-target goal demos) | target-only | co-train (Q + 120 outsiders) | **margin** | TracIn outsider intrusion |
|---|---|---|---|---|
| 15 | 21.0% ± 10.6 | 23.3% | **+2.3** | 0.578 ± 0.101 |
| 50 | 48.5% ± 8.8 | 49.3% | **+0.8** | 0.414 ± 0.164 |
| 490 | **93.8% ± 0.6** | 90.7% | **−3.2** | 0.542 ± 0.159 |

**The co-train margin decays monotonically with in-target quantity and turns negative.** The 120
outsider demos are worth ~+2 points to a starved target and **cost ~3 points** to a well-fed one.

**This also independently validates the whole pipeline**: at Q=490 the policy reaches **93.8%
success (SD 0.6)** on the full 10-task goal suite. Every low number elsewhere in this study is
therefore genuine data scarcity, not a defective policy/rollout stack. (The single-task ceiling
test reached 70% with 50 demos of one task, same conclusion.)

Intrusion is flat at ~0.5 (chance) across all Q: TracIn ranks outsiders about as highly as the
target's own demos at every quantity.

*Q=490 is the spec's "Q=500 (full suite)" minus C1's 10 held-out probe demos, which must stay
unseen — the spec itself reserves "the full suite minus C1's held-out 10" for this stage.*

---

## 5. Gate 1 (Stage D) — **FAIL**

`GATE1_FAIL.md`, `results/stage_D_gate1.json`. 24/24 retrains ok (12 masks × 8-of-15 C1 demos ×
2 seeds). Attributors computed from the five full-C1 models (Stage B), predicted mask score = sum
of per-demo attributions.

| attributor | ρ (held-out **L2**, PRIMARY) | p | ρ (success, secondary) |
|---|---|---|---|
| **IF** | **+0.252** | 0.215 | −0.327 |
| **TRAK** | +0.189 | 0.278 | −0.359 |
| **TracIn** | −0.042 | 0.552 | +0.056 |

**Criterion: any attributor ρ > 0.50 → FAIL** (best 0.252).

**Noise ceilings** (Spearman between the two seeds' outcome vectors over the 12 masks,
Spearman–Brown-corrected to the 2-seed mean the LDS predicts):

| outcome | 1-seed ρ | **2-seed ceiling** |
|---|---|---|
| held-out L2 | +0.399 | **+0.570** |
| success (30 episodes) | −0.318 | **−0.932** |

**What this means, honestly.** IF captures **≈44% of the achievable rank signal** and its CI
includes 0; TracIn is indistinguishable from noise. That is a real failure. But the 0.50 bar is
**88% of the measured ceiling** — Gate 1 as written demands near-oracle performance, so "IF carries
genuine but weak signal" is the fairer reading than "attribution is worthless".

**Closed-loop success is unusable as a demo-grain outcome, and that is itself a finding**: its
ceiling is **negative** (−0.93) — the two seeds of the same mask produce *anti-correlated* success
vectors. At 30 episodes with C1 near the floor, success carries no reproducible rank information.

---

## 6. Attributors — exact estimators (deliberate deviation)

The spec asks for dattri's TRAK / TracIn / EK-FAC IF. dattri 0.3.0 is installed but was **not**
used, because with **N = 135 demos vs p = 19.22 M parameters** both of its approximations are
unnecessary *and* ill-posed here:

- **TRAK** projects gradients to *k* dims with a JL sketch; with N ≪ k the k×k Gram (ΦᵀΦ) is
  **singular** for any useful k, and the sketch only injects noise. We use the exact **N×N dual
  (kernel) form** — the same estimator without the sketch.
- **EK-FAC** exists to *approximate* a Fisher inverse that is intractable when N and p are both
  large. At N = 135 the exact empirical-Fisher inverse is available in closed form by **Woodbury**.

Both identities are verified to **~5e-15** against dense brute-force p×p inversion (Stage A tests
30–31). **This gives attribution its best possible shot** — so the Gate-1 failure cannot be blamed
on sketching or factorisation error. `results/influence_table.parquet` = 10,935 rows
(3 attributors × 3 functionals × 9 targets × 135 demos), plus per-member scores for jackknife CIs.

---

## 7. LDS vs noise ceilings (Stage H)

Computed **as evidence**, not as validated attribution (Gate 1 failed).
`results/lds_cluster_grain.parquet`, `results/noise_ceilings.json`. Figure:
`figures/fig3_lds_vs_ceiling.png`.

**Primary = conditional LDS** over only the **40 target-included** masks. Stage F shows why: with
the target excluded, success collapses to ~0 for *every* seed (see table below), so a full-72-mask
LDS would score high merely by predicting "is the target in the mask?" — trivial, and silent about
which *outsiders* matter. Full-72 values are in the parquet, labelled inflated.

**A methodological correction to the spec's ceiling recipe.** The spec computes the ceiling over
all 12 replicate masks. But target-*excluded* masks are trivially reproducible across seeds, so
that ceiling (0.78–0.94) is inflated by the same exclusion effect the conditional LDS is forbidden
to exploit. Judging a conditional LDS against it would compare against an unattainable bar. We
therefore compute and overlay the **conditional ceiling** (target-included replicate masks only,
n = 6–8); both are reported.

**Cluster-grain conditional LDS, plain functional:**

| outcome | Bonferroni-significant cells (p < 0.05/9 = 0.0056) | strongest |
|---|---|---|
| **held-out L2** | **8 / 27** | C8 TRAK **+0.650** (ceiling 0.825) · C7 TracIn **+0.607** (0.843) · C5 TRAK +0.568 (0.564) · C6 TRAK +0.473 (0.124) |
| **success** | 3 / 27 | erratic, many negative; conditional ceilings are low (0.08–0.56) and one is negative |

**So attribution is not uniformly unfaithful.** At cluster grain, against a stable loss outcome, it
reaches **40–100% of its noise ceiling** for several targets and clears Bonferroni in 8 of 27 cells.
It is at **demo** grain that it breaks down.

**Demo grain (Stage G, 24 masks, n = 24):** on the **preregistered confirmatory outcome (success)**,
**both focal targets fail** — C1 best p = 0.056, C5 p = 0.711 (Bonferroni-2, α = 0.025 one-sided).
⇒ **The preregistered downgrade rule fires: all per-demo claims downgrade to cluster grain.**
`results/demo_grain_verdict.json`.

*For completeness (and explicitly NOT used to claim a pass):* on the **secondary** loss outcome the
focal targets look better — C1 TracIn ρ = +0.424 (p = 0.019 < 0.025) and C5 IF ρ = +0.361
(p = 0.042). The partial Spearman controlling for in-target demo count (7 vs 8) barely moves them
(C1 TracIn +0.459; C5 IF +0.470), so they are not an artifact of how many of the target's own demos
each mask contains. **The outcome was not switched post-hoc to manufacture a pass** — the
preregistered confirmatory outcome is success, it failed, and the downgrade stands.

---

## 8. Headline statistics — **cluster-grain licence only**

`results/headline_stats.json`, `figures/fig2_insider_outsider.png`, `figures/fig1_transfer_matrix.png`.
Best attributor per target, plain functional; ± = **delete-1 jackknife SE over the E=10 ensemble**.

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

- **(c) 9 of 9 targets have ≥1 outsider above their 75th-percentile insider.**
- **C9's fifteen most-influential demos contain none of C9's own.**
- **C3 is the clean exception** (intrusion 0.033) — and it is exactly the cluster the DTW matrix
  flags as a trajectory-space outlier (0.49–0.53 from every other cluster; 0.051 within itself).
- Several jackknife SEs are large (C8 0.457, C2 0.316): **ensemble members disagree substantially**,
  an additional caution on top of the Gate-1 failure.
- **Every per-demo statistic in this table is licensed at cluster grain only** (demo-grain
  downgrade, §7), and all of it is subject to the Gate-1 verdict that no attributor is validated.
- **Best attributor is selected on the L2-outcome LDS, not the success LDS**, because the success
  conditional ceiling is low and sometimes negative (C2 0.078, C7 0.117, C8 −0.095) — ranking
  attributors on an outcome an oracle cannot predict would be ranking them on noise. Both rankings
  are saved (`results/best_attributor_selection.json`).

**Transfer matrix** (`results/fig1_transfer_matrix.csv`): the diagonal is *not* reliably the
strongest entry — C2's self-influence is **negative** in column z-score (−1.5) while C5→C2 is +2.0.
Consistent with the intrusion result.

---

## 9. RQ2 / RQ3 / RQ4

### RQ2 — moderators (trajectory-space only; **no image/DINO features anywhere**)
`results/rq2_moderators.csv`. Insider-advantage AUC regressed on similarity, n = 9 targets:

| predictor | Pearson r | p |
|---|---|---|
| mean-outsider **DTW** | **+0.011** | 0.98 |
| object-state **MMD** | +0.098 | 0.80 |
| **bddl** object overlap | +0.412 | 0.27 |
| within-target **redundancy** | +0.482 | 0.19 |

OLS (AUC ~ DTW + redundancy): R² = 0.251, n = 9. **No moderator is significant.** Trajectory-space
similarity does **not** explain who intrudes. With n = 9 this is descriptive and underpowered, and
is reported as such — not as a demonstration of absence. Quantity's moderator evidence comes from
Stage C (§4), not from this regression, exactly as the spec directs.

### RQ3 — phase contrast (a **contrast of functionals**, never an additive decomposition)
`results/rq3_phase_contrast.csv`. Mean conditional LDS over the 9 targets (L2 outcome):

| attributor | plain | transport-masked | interaction-masked |
|---|---|---|---|
| TracIn | +0.199 | +0.196 | +0.187 |
| TRAK | +0.303 | +0.249 | +0.191 |
| IF | +0.057 | +0.066 | +0.085 |

**Essentially flat** — which phase you attribute toward barely changes predictive validity. This is
a **genuine null, not an under-identified one**: the transport fraction is **0.707 ± 0.113** across
the 135 demos (spec's flag is std < 0.05), so the contrast *was* identifiable. Masks at ±25% of both
thresholds (window 8/12, percentile 22.5/37.5) were precomputed as specified; the reported LDS uses
the frozen base thresholds.

### RQ4 — data selection at budget B = 15 (`results/stage_I_rq4.json`)
24/24 runs ok. 4 conditions × 3 seeds × 2 focal targets; evaluated on the target's 3 probe tasks ×
20 rollouts. **A negative result is reported plainly — it is a finding, not a failure.**

| condition | **C1** | **C5** |
|---|---|---|
| target-only (the 15 insiders) | 12.8% ± 16.4 | **17.2% ± 12.1** |
| **influence-top-15** | **13.3% ± 15.3** | **1.1% ± 1.0** |
| random-15 | 0.0% ± 0.0 | 13.9% ± 14.2 |
| similarity-top-15 (DTW) | 0.0% ± 0.0 | 3.3% ± 4.4 |

Paired-by-seed margins of influence vs each alternative:

| | C1 | C5 |
|---|---|---|
| influence − target-only | +0.6 (SD 3.8) | **−16.1** (SD 11.1) |
| influence − random-15 | **+13.3** (SD 15.3) | −12.8 (SD 13.4) |
| influence − similarity-top-15 | **+13.3** (SD 15.3) | −2.2 (SD 5.4) |

**Split verdict.** On C1, influence-selection matches target-only and **beats random and similarity
by +13.3 pts** (both of which score 0.0%). On C5 it is **catastrophically worse than simply using
your own data (−16.1 pts)**. Influence-top-15 picks 9 outsiders for both targets; for C5 (7 tasks,
15 demos) that leaves too few insiders to cover the target's tasks. **Influence-based selection is
not reliably better than the trivial baseline**, and given Gate 1 this is the expected outcome.
DTW-similarity selection is *also* poor (it picks 12/11 outsiders), so this is not evidence that
similarity beats influence — both are worse than just using in-domain data.

---

## 10. Deviations from the spec, and defects found

**Every deviation is listed. Two are material.**

1. **MATERIAL — task-language conditioning.** The spec's state-only policy is *provably degenerate*
   on LIBERO: the 10 `libero_goal` tasks share one scene and **byte-identical reset states**
   (verified directly). An unconditioned state policy cannot represent three different goals from
   the same state, so *both* arms of Gate 0 would have measured label noise. The policy is
   conditioned on a **frozen** MiniLM embedding of the task's LIBERO language string (metadata, not
   perception). The encoder is never trained. **No image features are used anywhere**, including in
   the RQ2 moderators, which remain trajectory-space as specified.

2. **MATERIAL — the loss functional was changed after seeing a gate result.** Gate 1 first ran with
   the preregistered **GMM-NLL** functional and failed (best |ρ| = 0.21; archived at
   `results/stage_D_gate1_GMMNLL_ORIGINAL.json`). Before accepting that, I checked whether the
   *ground truth* was predictable — whether two seeds of the **same** mask agree. They did not:

   | mask | GMM-NLL (seed 101) | (seed 102) | → L2 (101) | L2 (102) |
   |---|---|---|---|---|
   | D00 | 23.6 | **187.0** | 1.0654 | 1.0695 |
   | D04 | 64.2 | **286.1** | 0.6543 | 0.8191 |
   | D10 | 27.1 | **239.3** | 0.8535 | 0.8459 |

   An 8–10× swing on **identical data**. Mechanism: the *median* per-frame NLL barely moves
   (27.6 → 32.4 → 36.0) but **34–36% of frames exceed NLL 100** — the GMM's σ collapses toward its
   1e-4 floor and NLL is **unbounded** where the model is confidently wrong. The mean NLL was
   measuring σ-collapse tail noise, not data quality; **no attributor can predict that**. The spec
   permits either loss ("L2 or GMM NLL — pick one"), so the **evaluation/attribution functional** is
   now **L2 on the executed action** (bounded). The **training objective is unchanged** (GMM NLL).
   **No retraining was needed.** NLL numbers are retained under `*_nll` in every `outcomes.json`.

   **This is exactly the post-hoc change preregistration exists to prevent, so it is flagged
   everywhere.** Mitigations: the justification is an instrument defect established *independently*
   of any gate outcome; the original verdict is archived; and **the gate still fails after the fix**
   (0.252 < 0.50). The fix bought a *meaningful* failure, not a pass.

3. **Exact estimators instead of dattri's** (TRAK dual form; Woodbury empirical-Fisher IF instead of
   EK-FAC). Justified in §6; verified to ~5e-15. Strictly more accurate than the approximations.

4. **Conditional noise ceiling** overlaid on the conditional LDS, in addition to the spec's all-12
   ceiling (§7). The all-12 ceiling is inflated by the target-exclusion effect.

5. **Fixed gradient-step budget (8000) rather than fixed epochs**, so that Gate 0 and Stage C do not
   confound "more data" with "more optimisation" (§2).

6. **Stage C Q = 490, not 500** — the full goal suite minus C1's 10 held-out probe demos, which must
   stay unseen. The spec itself reserves "the full suite minus C1's held-out 10" for this stage.

7. **Gate 0 margin criterion evaluated in exact rational arithmetic** — a float-comparison bug had
   reported C5's *exactly*-5.0 margin as below 5.0 (§3). Criterion unchanged; evaluation fixed.

8. **Best attributor selected on the L2-outcome LDS, not the success LDS** (§8), because the success
   conditional ceiling is too low (sometimes negative) to rank on.

9. Python 3.12 (not 3.10); LIBERO as the `hf_libero` pip package (not a repo clone) — benchmark
   membership still resolved from the installed benchmark at load time, so no task list is hardcoded.

10. **GPU-selection defect (found and fixed).** `attribution.py` runs outside the orchestrator, so
    `CUDA_VISIBLE_DEVICES` was unset and torch defaulted to **cuda:0 — a forbidden GPU**. It OOM'd
    immediately and allocated nothing (GPUs 0–3 stayed at exactly 48,459 MiB throughout), but it
    should never have looked there. `bootstrap.py` now pins any un-pinned process into {4,5,6,7}
    **before torch is imported**, making the hard rule structural rather than a thing each caller
    must remember.

**Cuts taken: NONE.** The budget cut order (rollouts 10→7; demo-corpus probes restricted to C1+C5;
noise-ceiling 12→8 masks; K 72→63) was **never invoked** — the study came in far under budget (§11).

---

## 11. Corpus completion and budget (actual vs ledger)

Read from `logs/*_summary.json` and the run artifacts.

| stage | runs | failures | GPU-h |
|---|---|---|---|
| B Gate 0 (C1) | 10 | 0 | 1.07 |
| B Gate 0 fallback (C2) | 10 | 0 | 1.27 |
| B Gate 0 fallback (C5) | 10 | 0 | 1.18 |
| C quantity sweep | 18 | 0 | 1.81 |
| D Gate 1 | 24 | 0 | 1.45 |
| E ensemble | 10 | 0 | 1.21 |
| F cluster corpus | **168** | **0** | **21.17** |
| G demo corpus | 48 | 0 | 6.10 |
| I RQ4 | 24 | 0 | 2.57 |
| H attribution + analysis | — | 0 | ~0.5 |
| **TOTAL** | **322** | **0** | **≈ 37.8** |

| | ledger | actual |
|---|---|---|
| retrains | ~302 | **322** (the extra 20 are the Gate-0 fallback on C2 and C5, which the spec mandates on a C1 failure) |
| GPU-h | ~154 (120–190) | **≈ 37.8** — **4× under** |
| rollout episodes | ~70,500 | **72,210** (counted from artifact files) |

**Zero failed runs across all 322 retrains.** No `BUDGET_ALERT.md` was needed: a 75-demo Stage-F
retrain took ~0.13 GPU-h against the 0.8 GPU-h pause threshold.

---

## 12. Figures and tables

| # | file | content |
|---|---|---|
| 1 | `figures/fig1_transfer_matrix.png` | 9×9 transfer matrix, diverging scale, per-column conditional-LDS badges |
| 2 | `figures/fig2_insider_outsider.png` | per-target insider/outsider influence + intrusion with jackknife SE and **grain-licence labels** |
| 3 | `figures/fig3_lds_vs_ceiling.png` | conditional LDS bars vs **noise-ceiling reference lines**, both outcomes |
| 4 | `figures/fig4_quantity.png` | co-train margin **and** intrusion vs Q — **two panels, never a dual axis** |
| 5 | `results/rq2_moderators.csv` | moderator regression table |
| 6 | `results/rq3_phase_contrast.csv` | phase-contrast table + transport-fraction check |
| 7 | `figures/fig7_influence_vs_dtw.png` | influence-vs-DTW scatter per target |
| 8 | `results/table_demo_grain_lds.csv` | demo-grain LDS, focal (confirmatory) vs exploratory |

The data behind every figure is saved alongside it in `results/`.

---

## 13. What did not run

- **Nothing in the spec was skipped.** All stages A–I completed.
- dattri's TRAK / TracIn / EK-FAC implementations were **not** used (exact estimators instead, §6).
- The ±25% phase-threshold sensitivity masks were **computed** but the headline LDS is reported at
  the frozen base thresholds only; a full LDS re-run under the low/high variants **did not run**.
- No budget cut was taken.

## 14. Bottom line

The attribution methods this study set out to audit **cannot be trusted to say which individual
demonstration matters** for a LIBERO policy: Gate 1 fails at 0.252 against a 0.50 bar, the demo-grain
downgrade rule fires, and influence-based data selection loses badly to simply using in-domain data
on one of the two focal targets. At the coarser grain of *which cluster* matters, and against a
stable loss outcome, the same methods do carry real signal (8/27 cells Bonferroni-significant,
several at 70–100% of their noise ceiling).

The deeper reason sits underneath both gates and is the study's most transferable result: **at this
data scale the training seed moves the outcome more than the data does.** A 15-demo policy's success
swings 9.5%–30.5% across seeds at fixed data; two seeds of the same 8-demo mask produce
anti-correlated success vectors. Any attribution method — and any data-selection method built on one
— is trying to explain a signal smaller than the noise it is measured in. That is not a property of
TracIn, TRAK or influence functions. It is a property of the regime, and it should be measured
before attribution is trusted, not after.
