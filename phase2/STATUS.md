# RoboTDA-X Phase 2 — STATUS

Project root: `/mnt/sdb/ljc/RoboTDA-X`. Phase 2 writes only under `phase2/`.
GPUs 4–7 only; every launch gated on `mem < 1000 MiB AND util < 10%`.

---

## P0 — INTAKE AUDIT — **PASS** (2026-07-11)

Artifact: `phase2/results/p0_intake_audit.json`, `phase2/results/p0_env_check.json`

**Fingerprint verified.** `REPORT.md`, `preregistration.json`, `results/`, `runs/`, `data/`, `src/`
all present. Checkpoint counts: stage_B=30, stage_E=10, stage_F=168, stage_G=48 (all with
`train.marker` + `final.pt`; 0 missing). Stage-G mask manifest: K=24, 68 demos/mask, seed 11.

**Two Phase-1 numbers recomputed from RAW per-run artifacts** (not from the summary JSONs):

| # | quantity | recomputed | source | REPORT.md | verdict |
|---|---|---|---|---|---|
| a | Gate-1 best ρ | **+0.251748** (IF) | `runs/stage_D/*/outcomes.json` + `stage_D_influence_C1.parquet` | +0.252 | **MATCH** |
| b | Stage-C margins (pts) | **+7/3, +5/6, −19/6** = +2.333 / +0.833 / −3.167 | `runs/stage_C/*/cluster_eval.json`, exact rational arithmetic | +2.3 / +0.8 / −3.2 | **MATCH** |

The raw Stage-D mask outcomes reproduce the archived `mask_outcomes` block to `max|Δ| = 0.0`.
Q=490 target-only = 93.83% (REPORT: 93.8%). Per-attributor Gate-1: IF +0.2517, TRAK +0.1888,
TracIn −0.0420.

**Environment** (`robotda_x`, py3.12.13, torch 2.11.0+cu130): env reset + **open-loop replay of
1 demo → SUCCESS at step 127/138**; 50-step training run finite, loss −0.196 → −2.388. PASS.

**Preregistration** `phase2/preregistration_phase2.json` written and LOCKED before any Phase-2
training run.

### Marker-name deviation (disclosed)
The brief specifies `done.marker`. Phase 1's orchestrator uses `train.marker` / `probe.marker` /
`clustereval.marker`. Phase 2 **retains Phase-1's names** — adopting them keeps the existing
resumability/skip logic identical rather than regressing it. Atomicity and skip-on-restart are
unchanged.

---

## P1 — SEED-ENSEMBLED DEMO-GRAIN LDS — RUNNING

96 retrains = the **existing** 24 Stage-G masks (not resampled) × 4 new seeds 403–406, giving
S=6 per mask when merged with Phase-1's 401–402. Probe battery identical to Phase-1 Stage G
(9 clusters × 3 probe tasks × 10 rollouts + held-out L2/transport/interaction) = 270 episodes
per model, 25,920 total.

**Budget guard (first 4 retrains timed):** 466 / 485 / 496 / 490 s → mean **484 s**.
Projected stage cost = 96 × 484 s = **12.9 GPU-h** vs ledger **13 GPU-h** → ratio **0.99**,
well inside the 1.5× alert threshold. No alert. Projected wall ≈ 3.2 h on 4 GPUs.

Attribution is **reused unchanged** from `results/influence_table.parquet` — no new attribution.

---

## Work prepared while P1 occupies the GPUs (CPU-only)

- **P2 probe sets** — `phase2/results/per_task_probes.json`. Rule: the 5 highest-indexed demos of
  each of the 27 tasks (demo_45…demo_49). Machine-checked disjointness: the Phase-1 135-demo
  corpus and all 9 held-out sets use **only demo_0…demo_4** (max index 4 ≪ 45). 135 probe demos,
  135/135 load. Features extracted for C2/C5 (libero_goal was already fully extracted by
  Stage-C's Q=490 pool) into a `phase2/data/proc` **symlink overlay** — Phase-1 `data/` untouched;
  normalization uses Phase-1's frozen `results/norm_stats.npz` (never refit).
- **P2 per-task attribution** — `phase2/src/p2_attr.py`, Phase-1's exact estimators (TracIn/TRAK
  dual/Woodbury IF) retargeted from 9 cluster functionals to 27 per-task L2 functionals.
- **P3 masks + jobs** — `phase2/results/p3_mask_manifest.json`, `p3_jobs.json`. Q ∈ {15,50,150};
  the Q-ladder is re-expressed and then **asserted to reproduce Phase-1's `stage_c.goal_demo_ladder()`
  bit-for-bit at Q=15/50/490**, so nesting is verified, not claimed. 12 masks/Q at α=0.6
  (9/30/90 demos), per-demo inclusion balanced to 7–8 (target 7.2). 144 mask retrains + 15 full-Q
  ensemble models = 159.

### P2 budget saving (not a cut)
The brief budgeted ≤11k episodes to re-evaluate 30 Stage-B checkpoints *if* per-task rollouts had
not been saved. They **were** saved: every `runs/stage_B/*/cluster_eval.json` already contains
`per_task_success` at 20 rollouts/task. P2 therefore needs **0 retrains and 0 new episodes** for
its measured margins.

---

## P1 — SEED-ENSEMBLED DEMO-GRAIN LDS — **COMPLETE. PREREGISTERED VERDICT: FAIL**

96/96 retrains, **0 failures**, **12.13 GPU-h** (ledger 13 → 0.93×), wall 3.05 h.
Artifacts: `stage_G6_outcomes.parquet` (1296 rows, 24 masks × 6 seeds), `p1_demo_grain.json`,
`p1_lds_table.csv`, `p1_seed_ladder.json`.

**The ceiling rose exactly as intended — attribution did not follow.**

| target | 1-seed rel. | 6-seed ceiling *predicted* (SB) | 6-seed ceiling *measured* | bar = ½·ceiling | best attributor | ρ | ratio | p₁ |
|---|---|---|---|---|---|---|---|---|
| **C1** (focal) | 0.750 | 0.947 | **0.933** | 0.466 | TracIn | **+0.369** | 0.40 | 0.038 |
| **C5** (focal) | 0.648 | 0.917 | **0.918** | 0.459 | IF | **+0.380** | 0.41 | 0.034 |

Both focal targets **FAIL** on both arms of the criterion (ratio < 0.5 *and* p₁ > 0.025).
The Spearman–Brown prediction check is near-exact (0.947 vs 0.933; 0.917 vs 0.918).

**Seed ladder — the direct answer to "attribution, or the noise floor?"** (same 24 masks):

| S seeds | C1 ceiling | C1 best LDS | C5 ceiling | C5 best LDS |
|---|---|---|---|---|
| 1 | 0.750 | +0.333 | 0.648 | +0.334 |
| 3 | 0.874 | +0.361 | 0.848 | +0.391 |
| 6 | 0.933 (SB) | **+0.369** | 0.918 (SB) | **+0.380** |

Six times the ground-truth seeds moves the ceiling by **+0.18** and the LDS by **+0.04**.
The LDS *plateaus while the ceiling climbs* ⇒ the residual gap is **attribution error, not
measurement noise**. Phase 1's Gate-1 failure was **not** merely the noise floor.
(Non-focal, descriptive: C2 ρ=+0.464 and C9 ρ=+0.462 do exceed half-ceiling.)

## P2 — PER-TASK TRANSFER-SIGN PREDICTION — **COMPLETE. PREREGISTERED VERDICT: FAIL**

**0 retrains, 0 new episodes** (Stage-B already stored `per_task_success`).
Measured margins independently reproduce Phase-1 Gate 0: C1 **+1.00**, C2 **+14.10**, C5 **+5.00** pts
(REPORT: +1.0 / +14.1 / +5.0), and C1's cancellation is confirmed (4 tasks helped, up to +23.0;
hurt down to −22.0).

| ensemble | attributor | pooled ρ (n=27) | p₁ | sign agreement |
|---|---|---|---|---|
| **Stage-E (PREREGISTERED PRIMARY)** | **IF** | **+0.163** | 0.209 | 17/27 (63%, binom p=0.248) |
| Stage-E | TracIn | +0.226 | 0.128 | 14/27 |
| Stage-B (secondary) | IF | +0.345 | 0.039 | 17/27 |

Critical ρ ≈ 0.32 at n=27. The primary **FAILS**. The Stage-B arm crosses α=0.05 uncorrected, but
it is **not** the preregistered estimator and does not survive Bonferroni-6 (α=0.0083).

**Why the Stage-B "PASS" is a seed lottery** (`p2_attribution_stability.json`): Stage-B co-train and
Stage-E are trained on the **identical 135-demo pool** (verified: their `demos.json` sets are equal)
— they differ *only* in training seed. Yet:
- the two ensembles' per-task predictions agree only weakly: IF **+0.774**, TRAK **+0.389**, TracIn **−0.103**;
- across disjoint 5-member sub-ensembles of Stage-E the pooled statistic ranges **−0.086 … +0.381**
  (IF; range **0.468**, sd 0.089), and only **4%** would cross the critical value.

So the headline statistic moves further on *seed choice alone* than the effect being claimed. This
extends Phase 1's central finding — seed variance dominates data-composition variance — **to the
attribution itself**.

## P5 — RQ4 COVERAGE FIX (C5, diagnostic) — **COMPLETE**

3 retrains, 0.35 GPU-h. Coverage-constrained selection (6/15 outsiders, all 7 C5 tasks covered)
vs Phase-1's unconstrained (9/15 outsiders); 12/15 demos shared; seeds 501–503, paired.

| arm | C5 success |
|---|---|
| coverage-constrained (new) | **1.67%** |
| unconstrained influence-top-15 (Phase 1) | 1.11% |
| random-15 | 13.89% |
| target-only | **17.22%** |

coverage-fixed − unconstrained = **+0.56 pts** (p=0.74) — *indistinguishable*.
coverage-fixed − target-only = **−15.56 pts** (Phase-1 unconstrained was −16.1).

**The coverage constraint does not rescue influence selection.** Phase-1's RQ4 catastrophe was the
**influence scores themselves**, not the missing coverage constraint. (n=1 target; diagnostic only.)

## Instrument defect — see `PHASE2_DEFECT.md` (found BEFORE P4 ran; nothing contaminated)

The brief's "90 rollouts/probe task" is **impossible**: every LIBERO task has exactly **50** initial
states, `rollout.py` indexes them `ep % 50`, and the policy (`argmax` GMM mode) and env are
deterministic — so episodes 50–89 are **bit-identical replays** of 0–39. Confirmed end-to-end on a
*discriminating* (~50%-success) checkpoint: episodes 50–59 reproduce 0–9 exactly, **including the
step counts (154, 158) at which the two successes occur**. P4's ladder is amended to **10/30/50**
rollouts/task (30/90/150 distinct episodes per cluster estimate). **No Phase-1 result is affected** —
every Phase-1 rollout count (10/20/30) is ≤ 50.

## Budget alert — see `BUDGET_ALERT.md`

P4a projected **5.6 GPU-h vs a 1 GPU-h ledger line (5.6×)** → the preregistered per-stage guard
fired; P4a was **PAUSED at 2/48**, the alert written, and then **RESUMED** by documented decision.
The ledger line implied ~0.14 s/episode; real episodes cost ~8–13 s (600-step horizon, and failing
episodes run the full horizon). Global projection **≈36 GPU-h vs the 75 GPU-h alert** — P3, P2 and P1
all came in at/under their lines and absorb the overrun. **Nothing cut.** P4b remains
budget-conditional (as preregistered) and is not run; its cell is a labelled extrapolation.

## RUNNING

- **P3** (159 retrains, GPUs 5–7): mean 402 s/job → projected **17.8 GPU-h** (ledger 25–30). 0 failures.
- **P4a** (48 models, eval-only, GPU 4): resumed; 14,400 episodes at 50 rollouts/task.

---

## P3 — REGIME BOUNDARY — **COMPLETE. BOTH PREREGISTERED TESTS: FAIL**

159/159 retrains, **0 failures**, **16.54 GPU-h** (ledger 25–30), wall 5.5 h.

| Q | L2 ceiling | success ceiling | best LDS (L2) | LDS/ceiling |
|---|---|---|---|---|
| 15 | 0.891 | +0.690 | +0.476 | 0.53 |
| 50 | 0.915 | +0.753 | +0.406 | 0.44 |
| **150** | 0.861 | **−0.561** | **−0.042** | **−0.05** |

Test (i) success ceiling rises with Q: Page L=32, exact p=**0.968 → FAIL**.
Test (ii) LDS/ceiling ratio rises with Q: Page L=31, exact p=**0.995 → FAIL**.
Descriptive reverse direction: the LDS/ceiling ratio **decreases** with Q (exact p=**0.032**).

**Mechanism (measured):** as Q grows the policy saturates, so the data-composition signal collapses
while seed noise does not — between-mask sd **0.063 → 0.027**, within-seed sd **0.056 → 0.074**;
signal/noise **1.13 → 0.88 → 0.37**. The negative result **worsens with scale**.

## P4 — CLOSED-LOOP SUCCESS RELIABILITY — **COMPLETE** (descriptive)

144 models re-evaluated (24 masks × 6 seeds; P4a 48 + P4b 96), **0 errors**, 43,200 episodes,
15.45 GPU-h. P4b's condition (P1+P3 under budget) was **met**, so it ran — nothing cut.

**Reliability is bought with SEEDS, not EPISODES:**

| cluster | 5× the episodes (30→150, S=1) | 3× the seeds (S=1→3, at 150 ep) |
|---|---|---|
| C1 (near-floor) | **+0.057** | **+0.259** |
| C2 (mid-range) | **−0.025** (nothing) | **+0.221** |

At a typical eval budget (30 episodes, 1 seed) the ground truth the curation literature optimizes
against has reliability **+0.185** (C1). Extrapolated (labelled): C1 needs ~3.1× the seeds for a 0.5
ceiling, ~12.5× for 0.8. And the episode axis **terminates at 50/task** — LIBERO has no more init
states.

---

## FINAL — all stages complete

**258/258 retrains succeeded, 0 failures. 44.76 GPU-h** (ledger 45–50; alert 75). **101,101 episodes.**
Every preregistered criterion evaluated exactly as written; **zero post-hoc criterion changes**;
**nothing cut**. Report: `PHASE2_REPORT.md`. Numbers machine-verified against artifacts by
`src/verify_report.py` (**45 checks, 0 failures**).

| stage | verdict |
|---|---|
| P0 intake | **PASS** (both Phase-1 numbers reproduce from raw artifacts) |
| P1 seed-ensembled demo-grain LDS | **FAIL** — ceiling 0.57→0.93, LDS +0.04. Not the noise floor. |
| P2 per-task transfer sign | **FAIL** — ρ=+0.163; the one "significant" arm is a seed lottery |
| P3 regime boundary | **FAIL** (both tests) — reliability *declines* with scale |
| P4 success reliability | descriptive — seeds, not episodes; episode axis terminates at 50/task |
| P5 coverage fix | coverage does **not** rescue influence selection |
