# RoboTDA-X — PHASE 3 STATUS

Preregistration: `phase3/preregistration_phase3.json`
SHA-256: `efd515656a3226c0d6738a98ff304978288096f1bbdaef6fbc907a213f09e98b` (locked 2026-07-12,
before any Phase-3 training, before the P6.5 λ sweep, and before any Phase-3 read-out).

Hard rules in force: GPUs 4–7 only (idleness verified before every launch); Phase-1/2 artifacts
read-only; Phase 3 writes only under `phase3/`; every number read from an artifact file; failed
runs reported as failed; synthetic tests labelled SYNTHETIC and never mixed with results.

---

## P6 — AUDIT HARDENING — **COMPLETE, GATE PASSED**

All four sweeps clean; the no-change proof passes 125/125 checks. No Phase-3 experiment started
until this gate passed.

| check | artifact | verdict |
|---|---|---|
| 6.1 marker-gated ingestion | `p6_marker_sweep.json` | **CLEAN** — 730 run dirs scanned, 581 artifact+marker pairs, **0 violations** |
| 6.2 Q490 probe-leak guard | `p6_probe_leak_check.json` | **CLEAN** — loaded gun CONFIRMED (6 Q490 runs contain 300 probe-demo inclusions) but **0 leaking model-artifact pairs** across 8 influence artifacts |
| 6.3 hardcoded episode count | `p6_episode_count_check.json` | **NO-OP CONFIRMED** |
| 6.4 stage_G6 integrity | `p6_g6_integrity.json` | **INTACT** — 96/96 dirs, 1296/1296 rows re-derive, 5/5 spot rows exact |
| no-change proof | `p6_no_change.json` | **125/125 IDENTICAL** |

**Two bugs were caught by the machine-checks — one of them in my own guard.**

1. `phase2_probe_ids()` initially walked only the top level of `per_task_probes.json` and returned
   an **empty set**, which made `assert_no_probe_leak()` pass **vacuously**. Fixed, and a
   post-condition now makes an empty/wrong-size probe set fatal. The Q490 loaded gun is real: each
   of the 6 Q490 runs contains 50 probe demos. No influence artifact derives from them.
2. The episode-count check was initially scoped too broadly and raised a false alarm
   (`p3_outcomes.parquet` has n_episodes = **200**, not 30). Re-scoped by *which reader consumes
   which table*: the two HARDCODING readers (`analysis.py:44`, `stage_d.py:85`) only ever consume
   tables with n = 30, and the 200-episode table is consumed only by `p3_analyze.py`, which
   already reads `n_episodes` row-wise. The fix is genuinely latent.

**Bonus reproducibility result:** a fresh `src/train.py --seed 401` reproduces the **archived**
Phase-1 checkpoint `runs/stage_G/G000_s401/final.pt` **bit-for-bit** (81/81 tensors, max diff 0.0).

### P6.5 λ ridge sweep — **PHASE-2 VERDICT STANDS, BUT ITS MARGIN IS MUCH THINNER THAN REPORTED**

0 retrains: `G = ΦΦᵀ` and `K = ΦTᵀ` do not depend on λ, so ONE gradient pass over the E=10
ensemble yields the exact IF/TRAK scores at **every** λ in closed form. Self-validation: at the
default λ the recomputation reproduces `influence_table.parquet` (ρ = 1.000000, max rel diff 7e-6).

Demo grain, held-out L2, 6-seed ground truth (`p6_lambda_sweep.json`, `..._extended.json`):

| target | λ=1e-2 (**Phase-2 default**) | oracle-tuned max | **cross-validated** | bar (½·ceiling) |
|---|---|---|---|---|
| C1 | +0.256 (ratio 0.27) | **+0.504 (ratio 0.54, p=0.006) — CROSSES** | +0.397 (0.43, p=0.027) | 0.466 |
| C5 | +0.380 (ratio 0.41) | +0.432 (0.47, p=0.018) | +0.341 (0.37, p=0.052) | 0.459 |

* **The default λ was NOT attribution's best shot.** C1's LDS nearly doubles when λ is tuned.
* **Oracle-tuned, C1 would PASS Phase-2's criterion.** Cross-validated (λ frozen on the *other*
  focal target), it does not. That is why the cross-validation was preregistered.
* **The preconditioner actively hurts.** LDS rises monotonically with λ over 8 orders of magnitude
  and is maximized in the **degenerate limit where the Fisher/Gram inverse is switched off
  entirely**. A raw (per-member scale-normalized) gradient dot product beats *exact* IF and *exact*
  TRAK at their default λ. Convergence to the analytic λ→∞ limit verified exactly (gap = 0.00e+00).
* A naive λ→∞ limit was **wrong** and the convergence check caught it: because λ_m is per-member
  adaptive, the limit is the per-member *scale-normalized* dot product, not the plain mean of K_m
  (0.504 vs 0.397 on C1).

---

## P7 — GRAIN-RESOLUTION LADDER — **COMPLETE: NO GRAIN QUALIFIES**

0 retrains. g ∈ {1,3,5,15}; primary grouping random-within-cluster (seed 707); secondary
complete-linkage on the Phase-1 DTW matrix. Ground truth = the 24 Stage-G masks, 6-seed mean,
held-out L2. Endpoint checks pass: g=1 reproduces the Phase-2 demo predictor; rules (a) and (b)
coincide exactly at g=1 and g=15, as they must.

**Preregistered read-out (focal C1/C5): no g ∈ {1,3,5,15} reaches half-ceiling with p<0.025.**
Coarsening does not rescue attribution — so the failure is not one of *resolution*.

Descriptive: 3 of 9 targets DO qualify at some grain (C2 at g=1,3 — ratio 0.65 at g=3; C9 at
g=1,3,5 — ratio 0.66 at g=5; C4 at g=15). DTW-structured grouping beats random grouping at
intermediate grains (**+0.062 mean LDS at g=3, winning on 6/9 targets**; +0.059 at g=5), which is
real evidence that group *structure* carries signal.

---

## P10 — DIFFUSION — **COMPLETE: PREREGISTERED TEST INCONCLUSIVE (instrument defect)**

See **`PHASE3_DEFECT.md`**. 144/144 retrains (96 @ seeds 601-604 + 48 @ 605-606), 0 failures.

**The policy is real and STRONGER than the BC-Transformer.** Calibration (7/10 runs): lr=1e-3,
16000 steps, H=8, DDIM=5 → **100% single-task success vs BC's 70%**. Config frozen (sha `fac011e6…`)
before P10a/P10b. Determinism gate passed informatively. Gate-0 co-train margins **larger** than
BC's (C1 +9.7, C5 +7.6 vs +1.0, +5.0).

**But the ground-truth instrument broke.** The preregistered seed-MEAN ceiling collapsed
(C1 0.079, C5 0.137; **4 of 9 negative**; SB consistency off by **0.61**). Cause, measured: the
diffusion policy is **deterministic but not seed-stable** — its executed DDIM action comes from a
fixed latent through a *multimodal* denoiser, so an occasional seed lands in the wrong basin and its
held-out L2 blows up 3–5×. Within-mask seed sd **0.489 vs BC's 0.073**; S/N **0.54 vs 1.61**.
**This is Phase 1's GMM-NLL pathology reborn inside the L2.** The **median** restores the ceiling
(0.079 → **0.568** on C1; all 9 targets land in 0.57–0.77).

**Median diagnostic (POST-HOC, not a verdict): MIXED.** C1 TracIn ρ=+0.420, ratio **0.74**, p=0.0205
(crosses); C5 ρ=+0.060, ratio 0.08 (nothing). **P10 does NOT settle external validity.**

**One clean positive:** the diffusion attribution is *far more* reproducible than BC's (E=5
split-half: TracIn **0.927** vs 0.449; IF 0.521 vs 0.188) — the fixed paired (t,ε) noise bank works.

---

## P8 — SEED-NOISE ANATOMY — **COMPLETE**

**P8a (48/48 ok, 0 failed, 6.10 GPU-h vs 8 line).** Variance decomposition:

| outcome | INIT (init+dropout) | ORDER (batch permutation) | INTER+RESID |
|---|---|---|---|
| **held-out L2** | **0.718** [0.570, 0.816] | 0.151 [0.090, 0.252] | 0.130 |
| logit success | 0.454 | 0.222 | 0.324 |

**INIT crosses the preregistered 70% threshold on the primary outcome and is NAMED dominant**
(~4.8× ORDER; leads on 7/9 targets). Stated honestly: the point estimate clears 70% but the
bootstrap CI straddles it. On logit-success **no factor dominates** — the preregistered fallback
reading applies.

**P8b (0 retrains, 60 evals).** **Neither cheap repair works:** checkpoint-averaging buys +0.003 on
L2 and *−0.099* on success; action-ensembling is −0.020 / +0.002 vs outcome-averaging at the same
3-seed cost. **There is no shortcut — the seeds must be trained.**

Determinism gate: the first pass PASSED but was **uninformative** (near-floor C1: all episodes
failed at the 600-step horizon, so step counts matched trivially). Re-run on the most
discriminating task (`p8b_determinism_strong.json`): arm (i) 6/6 successes at steps
`[105,103,100,106,105,112]`, arm (ii) mixed at `[600,111,103,241,114,600]` — both replayed exactly.

---

## P9 — ATTRIBUTION-SIDE COST LAW — **COMPLETE**

10/10 retrains, 1.35 GPU-h. E=20. Split-half per-demo ranking reliability:

| attributor | E=10 (Phase 1's) | E=20 | E* for r=0.8 (EXTRAPOLATED) |
|---|---|---|---|
| IF | **0.199** | 0.254 | **≈235 models** |
| TRAK | 0.298 | 0.419 | ≈111 |
| TracIn | 0.679 | **0.831** | ≈16 (met) |

**At the E=10 ensemble Phase 1 used, IF's per-demo ranking reliability is 0.199 and its top-15
Jaccard is 0.222** — the headline intrusion statistics rest on a ranking that is mostly noise for
IF and TRAK. TracIn is the exception.

---

## Budget so far (actual)

| stage | retrains | GPU-h | status |
|---|---|---|---|
| P6 hardening + λ sweep | 0 | ~0.1 | complete |
| P7 grain ladder | 0 | 0 | complete |
| P8 bitcheck (instrument) | 4 | 0.6 | complete |
| P8a factorial | 48 | **6.10** | complete, 0 failed |
| P8b variance-reduction (eval-only) | 0 | ~4 | complete, 60 evals |
| P9 ensemble + attribution | 10 | 1.35 + 0.1 | complete, 0 failed |
| P10 calibration | 7 | ~1.5 | complete (7 of 10 cap) |
| P10 ens + a + b | 113 | 36.29 | complete, 0 failed |
| P10b seeds 605/606 (defect remedy) | 48 | 15.42 | complete, 0 failed |
| P10 attribution | 0 | 0.70 | complete |
| **TOTAL** | **230** | **66.17** | vs 56.5 nominal / **85 alert — not tripped** |

## Cuts

**None.**

## Deviations / incidents

0. **P10 INSTRUMENT DEFECT (`PHASE3_DEFECT.md`).** The preregistered S=4 diffusion ceiling
   collapsed, making the verdict uninformative. Remedy: +2 seeds (S=6, matching the BC arm) — the
   criterion was NOT changed, only S. Still collapsed, because the outcome is heavy-tailed and the
   MEAN is a broken aggregator. Reported as INCONCLUSIVE; the median analysis is a labelled
   POST-HOC DIAGNOSTIC, never a verdict.
1. **Calibration harness bug (my own), cost 0 GPU-h.** The first three calibration launches passed
   the config as a JSON string inside `bash -c '...'`, so the outer shell ate the quotes and
   `train_diffusion.py` died on `json.loads`. No model was trained (0 GPU-h). Fixed by passing the
   config in a FILE (`--cfg_file`), and a guard now raises a clear "harness failure, not a policy
   result" error rather than letting an all-failed stage be misread as non-viability. The three
   runs were re-launched; the 10-run calibration budget is intact (7 used).
2. **P8b's first determinism gate was uninformative** (degenerate near-floor match). Strengthened
   and re-run on a discriminating task; both arms pass. Reported in full rather than quietly
   re-run.
