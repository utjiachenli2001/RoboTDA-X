# RoboTDA-X — PHASE 2 REPORT

**Question Phase 2 exists to answer:** Phase 1 found data attribution failing its counterfactual
(LDS) sanity gate at demo grain. Was attribution *unfaithful*, or was the *measurement regime too
noisy* to detect faithfulness?

**Answer: attribution is unfaithful.** Seed-ensembling the ground truth raised the noise ceiling
from 0.57 to **0.93** — and the attribution did not move (**+0.37**). It is not the noise floor.

Everything below is read from an artifact file written by an actual run. Stages that did not run
say "did not run". Preregistration: `phase2/preregistration_phase2.json`, locked before any
Phase-2 training. Deviations: §7. Defect: `PHASE2_DEFECT.md`. Budget: `BUDGET_ALERT.md`.

---

## 1. Intake audit (P0) — **PASS**

Two Phase-1 headline numbers were recomputed **from the raw per-run artifacts**, not from the
summary JSONs (`phase2/results/p0_intake_audit.json`):

| # | quantity | recomputed | REPORT.md | verdict |
|---|---|---|---|---|
| a | Gate-1 best ρ | **+0.251748** (IF) from `runs/stage_D/*/outcomes.json` + `stage_D_influence_C1.parquet` | +0.252 | **MATCH** |
| b | Stage-C margins | **7/3, 5/6, −19/6** = +2.333 / +0.833 / −3.167 pts (exact rationals) from `runs/stage_C/*/cluster_eval.json` | +2.3 / +0.8 / −3.2 | **MATCH** |

The raw Stage-D mask outcomes reproduce the archived block to `max|Δ| = 0.0`. Q=490 target-only =
93.83% (REPORT: 93.8%). Checkpoints verified intact: stage_B=30, stage_E=10, stage_F=168,
stage_G=48, all with `train.marker` + `final.pt`.

A **third, unplanned** confirmation fell out of P2: recomputing Gate 0 from raw per-task rollouts
gives C1 **+1.00**, C2 **+14.10**, C5 **+5.00** pts — exactly REPORT's +1.0 / +14.1 / +5.0.

**Environment:** env reset + **open-loop replay of 1 demo → SUCCESS at step 127/138**; 50-step
training run finite (loss −0.196 → −2.388). `phase2/results/p0_env_check.json`.

---

## 2. P1 — Seed-ensembled demo-grain LDS — **PREREGISTERED VERDICT: FAIL**

96/96 retrains, **0 failures**, **12.13 GPU-h**. The *existing* 24 Stage-G masks (not resampled) ×
4 new seeds (403–406) → **S=6** per mask. Attribution reused unchanged from Phase 1's
`influence_table.parquet`. Artifacts: `stage_G6_outcomes.parquet` (1,296 rows),
`p1_demo_grain.json`, `p1_seed_ladder.json`.

**Criterion (preregistered):** on held-out L2, 6-seed mean, focal C1/C5 — any attributor
ρ ≥ 0.5 × measured 6-seed ceiling, one-sided p < 0.025 (Bonferroni-2).

| target | 1-seed rel. | 6-seed ceiling **predicted** (SB) | 6-seed ceiling **measured** | bar = ½·ceil | best attr | ρ | ratio | p₁ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **C1** | 0.750 | 0.947 | **0.933** | 0.466 | TracIn | **+0.369** | 0.40 | 0.038 | **FAIL** |
| **C5** | 0.648 | 0.917 | **0.918** | 0.459 | IF | **+0.380** | 0.41 | 0.034 | **FAIL** |

Both focal targets fail **both** arms (ratio < 0.5 *and* p₁ > 0.025).

### Predicted-vs-measured ceiling check — the instrument worked exactly as designed
The preregistration predicted a ~0.80 six-seed ceiling from a ~0.40 one-seed reliability. The
measured one-seed reliability on these masks is higher (0.750 / 0.648), and Spearman–Brown predicts
**0.947 / 0.917** against a **measured 0.933 / 0.918** — agreement to within 0.014. The ceiling rose
from Phase-1's 0.570 to **0.93**. *The experiment succeeded; the attributor did not.*

### The seed ladder — the decisive evidence (`figures/p1_seed_ladder.png`)

| S seeds | C1 ceiling | C1 best LDS | C5 ceiling | C5 best LDS |
|---|---|---|---|---|
| 1 | 0.750 | +0.333 | 0.648 | +0.334 |
| 2 | 0.837 | +0.361 | 0.789 | +0.373 |
| 3 | 0.874 | +0.361 | 0.848 | +0.391 |
| 6 | **0.933** (SB) | **+0.369** | **0.918** (SB) | **+0.380** |

Six times the ground-truth seeds moves the **ceiling by +0.18** and the **LDS by +0.04**. The LDS
*plateaus while the ceiling climbs.* The residual gap is **attribution error, not measurement
noise** — the preregistered symmetric interpretation reads: *"FAIL at a ceiling of ~0.9 ⇒
attribution is genuinely unfaithful at demo grain in this regime — a clean, strong null."*

All 9 targets are reported in `p1_lds_table.csv` (only C1/C5 are confirmatory). Descriptively, two
non-focal targets do clear half-ceiling: C2 (ρ=+0.464, p=0.011) and C9 (ρ=+0.462, p=0.012).

---

## 3. P2 — Per-task transfer-sign prediction — **PREREGISTERED VERDICT: FAIL**

**0 retrains, 0 new episodes.** The brief budgeted ≤11k episodes to re-evaluate 30 Stage-B
checkpoints *if* per-task rollouts had not been saved. They **were** saved — every
`runs/stage_B/*/cluster_eval.json` already carries `per_task_success` at 20 rollouts/task. This is a
saving, not a cut.

Phase-1's cancellation is confirmed: C1's cluster margin is +1.00 pts, but **4 tasks are helped
(up to +23.0)** and **5 hurt (down to −22.0)**. Does attribution know which is which?

| ensemble | attributor | pooled ρ (n=27) | p₁ | sign agreement |
|---|---|---|---|---|
| **Stage-E — PREREGISTERED PRIMARY** | **IF** | **+0.163** | 0.209 | 17/27 (63%, binom p=0.248) |
| Stage-E | TracIn | +0.226 | 0.128 | 14/27 (52%) |
| Stage-E | TRAK | −0.040 | 0.578 | 15/27 (56%) |
| Stage-B (secondary) | IF | +0.345 | 0.039 | 17/27 (63%) |
| Stage-B (secondary) | TRAK | +0.318 | 0.053 | 18/27 (67%) |

Critical ρ ≈ 0.32 at n=27. **The primary fails.** Sign agreement is not significant.

### The Stage-B "PASS" is a seed lottery (`p2_attribution_stability.json`)
Stage-B co-train and Stage-E are trained on the **identical 135-demo pool** — verified: their
`demos.json` sets are *equal*. They differ **only in training seed**. Yet:

* the two ensembles' per-task predictions agree only weakly — IF **+0.774**, TRAK **+0.389**,
  TracIn **−0.103** (i.e. TracIn's predictions from two same-data ensembles are *anti*-correlated);
* across disjoint 5-member sub-ensembles of Stage-E the pooled statistic ranges
  **−0.086 … +0.381** (IF; range **0.468**, sd 0.089) and only **4%** would clear the critical value.

The headline statistic moves further on **seed choice alone** than the effect being claimed, and the
Stage-B result does not survive Bonferroni-6 (α=0.0083) either. This extends Phase 1's central
finding — *training-seed variance dominates data-composition variance* — **to the attribution
itself**. Figure: `figures/p2_transfer_sign.png`.

---

## 4. P3 — Regime boundary — **BOTH PREREGISTERED TESTS: FAIL**

159/159 retrains, **0 failures**, **16.54 GPU-h**. Q ∈ {15, 50, 150} on C1's Goal suite; 12 masks/Q
at α=0.6 (9/30/90 demos, per-demo inclusion balanced 7–8); 4 seeds/mask; 5 full-Q models per Q as
that scale's own attribution ensemble. The Q-ladder is **asserted to reproduce Phase-1's
`stage_c.goal_demo_ladder()` bit-for-bit** at Q=15/50/490 — nesting is verified, not claimed.

| Q | L2 ceiling (4-seed) | **success ceiling** (4-seed) | best LDS (L2) | LDS/ceiling |
|---|---|---|---|---|
| 15 | 0.891 | +0.690 | +0.476 (TracIn) | 0.53 |
| 50 | 0.915 | +0.753 | +0.406 (TRAK) | 0.44 |
| **150** | 0.861 | **−0.561** | **−0.042** (TRAK) | **−0.05** |

**Preregistered test (i)** — success ceiling rises with Q: Page's L=32, exact one-sided
**p = 0.968 → FAIL**.
**Preregistered test (ii)** — best LDS/ceiling ratio rises with Q: Page's L=31, exact one-sided
**p = 0.995 → FAIL**.

(The exact Page test was validated on synthetic input first — perfect monotone increase gives
L=42, p = 1/216 = 0.00463, the known minimum for k=3, n=3. `SYNTHETIC_page_test_validation.json`,
labelled SYNTHETIC and never mixed with results.)

### The trend is not flat — it *reverses* (descriptive, not preregistered)
Both preregistered tests are one-sided for an **increase**; reporting only "no increase" would hide
the direction. Re-running the identical machinery in the decreasing direction:
LDS/ceiling ratio **decreases** with Q (exact p = **0.032**); success ceiling decreases
(exact p = 0.088). Secondary Spearman of statistic vs Q: best-LDS/ceiling **−1.0**.

### Why (`figures/regime_boundary.png`, panel c)
A ceiling is a **signal-to-noise ratio**. As Q grows the policy saturates (success 6–27% → 62–69%),
so deleting 40% of the demos moves it *less* — the data-composition signal collapses while seed
noise does not:

| Q | between-mask sd (**signal**) | within-mask seed sd (**noise**) | signal/noise | success range across masks |
|---|---|---|---|---|
| 15 | 0.0627 | 0.0557 | **1.13** | 0.061 – 0.270 |
| 50 | 0.0580 | 0.0656 | 0.88 | 0.334 – 0.517 |
| 150 | **0.0273** | 0.0738 | **0.37** | 0.616 – 0.693 |

**Prescription (the opposite of the one anticipated).** There is no data scale, within this range,
above which demo-level attribution or closed-loop ground truth becomes trustworthy. The negative
result **generalizes beyond the 15-demo regime and worsens with scale**, because the counterfactual
effect of removing a demo shrinks faster than the seed noise that hides it. Note the held-out **L2**
ceiling stays high (0.86–0.92) throughout — L2 keeps discriminating masks; it is *closed-loop
success* that becomes unusable, and *attribution* that degrades regardless.

---

## 5. P5 — RQ4 coverage fix on C5 (diagnostic) — **COMPLETE**

3 retrains, 0.34 GPU-h. Coverage-constrained selection (top-15 by influence subject to ≥1 demo per
C5 task) vs Phase-1's unconstrained top-15. Paired at seeds 501–503; 12/15 demos shared.

| arm | C5 success |
|---|---|
| coverage-constrained (**new**; 6/15 outsiders, all 7 tasks covered) | **1.67%** |
| unconstrained influence-top-15 (Phase 1; 9/15 outsiders) | 1.11% |
| random-15 | 13.89% |
| target-only | **17.22%** |

* coverage-fixed − unconstrained = **+0.56 pts** (p = 0.74) — **indistinguishable**
* coverage-fixed − target-only = **−15.56 pts** (Phase-1 unconstrained: −16.1)

**The coverage constraint does not rescue influence selection.** Phase-1's RQ4 catastrophe was the
**influence scores themselves**, not the missing coverage constraint. Influence-based selection is
beaten by *random* selection by ~12 points. (n=1 target; diagnostic only — do not generalize.)

---

## 6. P4 — Closed-loop success reliability — **DESCRIPTIVE (no pass/fail), and the most actionable result**

**0 retrains.** 144 models re-evaluated (24 masks × 6 seeds), **0 errors**, 43,200 episodes,
15.45 GPU-h. Per-episode outcomes are stored, so 10/30/50 rollouts-per-task are formed by
*subsampling the same rollouts* — and the 10-episode subsample is exactly Phase-1's Stage-G episode
set, so the ladder is nested. Artifacts: `p4_success_reliability.json`, `p4_reliability.csv`,
`figures/success_reliability.png`.

Split-half reliability of the closed-loop success outcome across the 24 masks
(`*` = Spearman–Brown **extrapolation** — a split-half of an S-seed mean needs 2S seeds, so S=6
would need 12 and is never directly measurable):

**C1 (near-floor success)**

| rollouts/task | distinct episodes | S=1 | S=2 | S=3 | S=6* |
|---|---|---|---|---|---|
| 10 | 30 | +0.185 | +0.316 | +0.423 | +0.595* |
| 30 | 90 | +0.260 | +0.376 | +0.475 | +0.644* |
| 50 | 150 | +0.242 | +0.378 | **+0.502** | +0.668* |

**C2 (mid-range success)**

| rollouts/task | distinct episodes | S=1 | S=2 | S=3 | S=6* |
|---|---|---|---|---|---|
| 10 | 30 | +0.601 | +0.737 | +0.797 | +0.887* |
| 30 | 90 | +0.602 | +0.732 | +0.795 | +0.886* |
| 50 | 150 | +0.576 | +0.736 | **+0.797** | +0.887* |

### The finding: reliability is bought with SEEDS, not EPISODES

| cluster | 5× the **episodes** (30 → 150, S=1) | 3× the **seeds** (S=1 → S=3, at 150 episodes) |
|---|---|---|
| C1 | **+0.057** | **+0.259** |
| C2 | **−0.025** (i.e. *nothing*) | **+0.221** |

Rolling out the same model five times as often buys essentially **zero** reliability — on C2 it is
flat to slightly negative. Training more seeds buys ~4–10× more per unit of the same budget. Read
across the rows of either table: they are almost constant. Read down the columns: they climb.

**Why:** the mask-ranking signal is a property of the *trained model*, and re-rolling one model
cannot average away the variance introduced by its *training seed*. Beyond a modest episode count
the estimator is no longer episode-limited, it is seed-limited — the same seed-dominance Phase 1
identified, now measured on the ground-truth instrument itself.

### The hard instrument wall
Spearman–Brown extrapolations (**labelled extrapolations, not measurements**), from the measured
1-seed reliability at 150 episodes: C1 needs **~3.1×** the seeds to reach a 0.5 ceiling and
**~12.5×** to reach 0.8; C2 needs **~0.7×** and **~2.9×**. But note §7.1: past **50 episodes/task**
there are *no more distinct episodes to buy* — LIBERO supplies 50 init states. The episode axis
does not merely have poor returns; **it terminates.** Additional closed-loop ground truth must come
from more tasks, more seeds, or new initial states.

### What this means for the curation literature
Closed-loop return is the functional CUPID-style demonstration-curation methods optimize against,
**without** validating it as ground truth. Measured here, that ground truth at a typical evaluation
budget (30 episodes, 1 seed) has a reliability of **+0.185** on a near-floor cluster — and the way
to fix it is *not* the axis practitioners actually spend on.

---

## 7. Deviations, cuts, and the defect

### 7.1 Instrument defect — found BEFORE P4 ran, nothing contaminated (`PHASE2_DEFECT.md`)
The brief's **"90 rollouts per probe task" is impossible.** Every LIBERO probe task has exactly
**50 initial states**; `rollout.py` selects them with `init_states[ep % 50]`; and both the policy
(`act` = argmax-mode mean, no sampling) and the env (fixed seed, `set_init_state`) are
deterministic. Episodes 50–89 are therefore **bit-identical replays** of episodes 0–39 — a nominal
"90-episode" estimate is really 50 distinct episodes with 40 double-counted.

Confirmed end-to-end on a **discriminating** (~50%-success) checkpoint — a near-floor model fails
every episode at the full horizon and would have matched *trivially*: episodes 50–59 reproduce 0–9
exactly, **including the step counts (154, 158) at which the two successes occur**.

**Amendment (P4 has no preregistered pass/fail criterion, so no verdict changed):** ladder
10/30/**50** rollouts/task = 30/90/150 distinct episodes per cluster estimate.
**No Phase-1 result is affected** — every Phase-1 rollout count (10, 20, 30) is ≤ 50.

**This is itself a finding.** Past 50 episodes/task, no further *independent* closed-loop samples
exist for that task. More must come from **more tasks, more seeds, or new initial states** — you
cannot buy them by rolling out the same task again. That is a hard constraint on the CUPID-style
closed-loop-return curation objective.

### 7.2 Budget (`BUDGET_ALERT.md`)
P4's ledger line (1 GPU-h) was mis-estimated — it implies 0.14 s/episode, but a 600-step-horizon
episode costs ~8–13 s and *failing* episodes run the full horizon. P4a projected **5.6×** its line,
which tripped the preregistered per-stage guard: the stage was **PAUSED at 2/48**, the alert
written, and then **RESUMED** by documented decision, because the *global* budget — the number that
governs — had ample headroom. **Nothing was cut.**

### 7.3 Disclosed deviations
1. **Marker names.** The brief says `done.marker`; Phase 1 uses `train.marker` / `probe.marker` /
   `clustereval.marker`. Phase 2 **retains Phase-1's names** so the existing skip/resume logic is
   unchanged rather than regressed. Atomicity and resumability are preserved.
2. **P2 probe-set scoping** (fixed in the preregistration, *before* running). The brief asks for
   probe demos "never used in any Phase-1 training pool". Taken to include Stage-C's Q=490 pool this
   is **unsatisfiable** — Q=490 is by construction *every* libero_goal demo except C1's 10 held-out,
   so C1 would have zero eligible probes. "Training pool" was therefore scoped to the pools of the
   models whose gradients define the attribution (the 135-demo corpus) plus every held-out set.
   Stage-C's Q=490 models are used nowhere in P2. The corpus provably uses only demo_0…demo_4, so
   the chosen probes (demo_45…demo_49) are disjoint by a wide margin — machine-checked.
3. **P2 needed 0 retrains / 0 episodes** (Stage-B already stored per-task success) — a saving.
4. **P5 needed 3 retrains, not 6** — the unconstrained comparator already exists at the same seeds.
5. **Phase-2 writes only under `phase2/`.** New demo features live in a `phase2/data/proc`
   **symlink overlay** over Phase-1's `data/proc`; Phase-1 artifacts are untouched and normalization
   uses Phase-1's frozen `norm_stats.npz` (never refit).

### 7.4 Cuts
**None.** P4b was budget-conditional in the preregistration; P1 and P3 both finished *under* their
lines, so the condition was met and **P4b was run**.

### 7.5 Zero post-hoc criterion changes
Every preregistered criterion was evaluated exactly as written, including the P1 multiplicity rule
(max over 3 attributors uncorrected, Bonferroni-2 over the 2 focal targets) — with the stricter
Bonferroni-6 reported alongside as robustness. The one place a criterion *could* have been bent —
P2's Stage-B arm crossing α=0.05 — is reported as a **FAIL of the preregistered primary**, with the
seed-lottery analysis that explains it away.

---

## 8. Synthesis — the three questions Phase 2 exists to answer

### Q1. Does seed-ensembled ground truth rescue demo-grain attribution? **No.**
The rescue was given every chance: the ceiling was raised from 0.57 to **0.93** (Spearman–Brown
predicted 0.947, measured 0.933 — the instrument behaved exactly as modelled), the *exact*
estimators were used, and the attribution was reused unchanged. The LDS moved **+0.04** while the
ceiling moved **+0.18**, and never reached half of oracle. Phase 1's Gate-1 failure was **not the
noise floor**. Attribution is genuinely unfaithful at demo grain in this regime — a clean, strong
null.

### Q2. Does attribution predict the direction of per-task transfer, where cluster means cancel? **No.**
Pooled ρ = **+0.163** (p=0.209) for the preregistered estimator; sign agreement 17/27 (p=0.248).
And the one arm that *did* cross α=0.05 is a **seed artefact**: two ensembles trained on the
*identical 135 demos* produce per-task predictions correlating as low as **−0.103**, and the
statistic swings across a range of **0.47** on seed choice alone. Attribution cannot tell you which
tasks co-training will help and which it will hurt.

### Q3. At what data scale do ground truth and attribution become reliable? **Not within this range — and the trend runs backwards.**
**Scale axis (P3).** Both preregistered monotone-increase tests fail (p=0.97, p=0.995); the
*decrease* in the LDS/ceiling ratio is what is significant (p=0.032). The mechanism is measured,
not speculated: as Q grows the policy saturates, so the counterfactual signal of deleting demos
shrinks (sd 0.063 → 0.027) while seed noise grows (0.056 → 0.074) — signal/noise **1.13 → 0.37**.
At Q=150 the closed-loop success ceiling is **negative**: no reproducible rank signal at all.

**Budget axis (P4).** The reliability of closed-loop ground truth is bought with **seeds, not
episodes**: 5× the episodes buys **+0.057** (C1) and **−0.025** (C2); 3× the seeds buys **+0.259**
and **+0.221**. And the episode axis does not merely have poor returns — it **terminates** at 50
episodes/task, because that is every initial state LIBERO has (§7.1). At a typical evaluation budget
(30 episodes, 1 seed) the ground truth the curation literature optimizes against has a reliability of
**+0.185** on a near-floor cluster.

So the answer to "what would it take?" is: **not more rollouts.** More training seeds — and the
extrapolated cost of reaching a *0.8* ceiling on a near-floor cluster is ~12.5× the seeds.

**The transferable claim.** Phase 1 found that training-seed variance dominates data-composition
variance at the 15-demo scale. Phase 2 shows this is not a small-data artefact: it is **structural,
it gets worse with scale, and it contaminates the attribution itself**. Any method that curates
robot demonstrations against a closed-loop-return objective — the CUPID setting — is ranking on a
signal that, in this regime, is smaller than the seed noise it is measured against, and no
attributor tested recovers it. The correct methodological response is not a better attributor but
**seed-ensembled, counterfactually-validated ground truth reported with its noise ceiling** — which
is precisely what this two-phase study is an instrument for.

---

## 9. Budget: actual vs ledger

All figures read from the stage summaries (`phase2/logs/*_summary.json`), the per-run artifacts,
and `phase2/results/episode_ledger.json`.

| stage | retrains (ledger) | retrains (actual) | ok / fail | GPU-h (ledger) | GPU-h (actual) | episodes (actual) |
|---|---|---|---|---|---|---|
| P0 intake | 0 (+1 tiny) | 1 tiny | 1 / 0 | <0.5 | ~0.01 | 1 |
| P1 demo-grain S=6 | 96 | **96** | 96 / **0** | 13 | **12.13** | 25,920 |
| P2 per-task signs | 0 | **0** | — | 2–4 | **0.26** | **0** |
| P3 regime boundary | 159 | **159** | 159 / **0** | 25–30 | **16.54** | 31,800 |
| P4 success reliability | 0 | **0** | 144 / **0** (eval-only) | 1 | **15.45** | 43,200 |
| P5 coverage fix | 3–6 | **3** | 3 / **0** | 0.5–1 | **0.34** | 180 |
| attribution (P2 + P3) | — | 0 | — | — | **0.30** | 0 |
| **TOTAL** | **~258** | **258** | **258 / 0** | **45–50** (alert 75) | **44.76** | **101,101** |

**Every retrain succeeded: 258/258, zero failures.** Total **44.76 GPU-h** — inside the 45–50
nominal and well under the 75 GPU-h alert. P1, P3, P2 and P5 all came in *under* their lines; P4's
overrun (15.45 vs 1) is the mis-estimated line documented in `BUDGET_ALERT.md` and is absorbed
several times over. 101,101 episodes vs a ~95k projection.

**Nothing was cut.** P4b was budget-conditional and the condition was met, so it ran.

---

## 10. Artifact index

| artifact | contents |
|---|---|
| `preregistration_phase2.json` | locked before any Phase-2 training |
| `PHASE2_DEFECT.md` | the 50-init-state instrument defect + end-to-end confirmation |
| `BUDGET_ALERT.md` | P4 line breach, pause, and documented resume |
| `results/p0_intake_audit.json`, `p0_env_check.json` | P0 recomputations, env replay |
| `results/stage_G6_outcomes.parquet` | P1 merged S=6 outcomes (1,296 rows) |
| `results/p1_demo_grain.json`, `p1_lds_table.csv`, `p1_seed_ladder.json` | P1 verdict, all 9 targets, seed ladder |
| `results/p2_transfer_sign.json`, `p2_attribution_stability.json` | P2 verdict + seed-lottery analysis |
| `results/per_task_probes.json`, `per_task_influence_*.parquet` | P2 probe sets (disjointness machine-checked) + 27-task attribution |
| `results/p3_regime_boundary.json`, `p3_lds_table.csv`, `p3_outcomes.parquet` | P3 verdicts, Page tests, signal/noise |
| `results/p4_success_reliability.json`, `p4_reliability.csv` | P4 grid + episodes-vs-seeds |
| `results/p4_determinism_check.json` | the defect's end-to-end proof |
| `results/p5_coverage_fix.json`, `p5_selection.json` | P5 diagnostic |
| `results/SYNTHETIC_page_test_validation.json` | **SYNTHETIC** — Page-test validation, never mixed with results |
| `figures/p1_seed_ladder.png` | ceiling climbs, LDS does not |
| `figures/regime_boundary.png` | the regime-boundary figure |
| `figures/p2_transfer_sign.png` | per-task prediction + seed lottery |
| `figures/success_reliability.png` | reliability curve + instrument wall |
