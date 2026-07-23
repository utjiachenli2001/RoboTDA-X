# PHASE-3 INSTRUMENT DEFECT — the P10 diffusion ground truth is too noisy at S=4 to test against

**Found:** 2026-07-12, after P10b's 96 retrains completed (96/96 ok, 0 failures) and the
preregistered analysis ran.
**Status:** documented BEFORE any interpretation is published, per the Phase-3 hard rules.
**Nothing is contaminated.** The defect is in the *ground-truth instrument* for P10, not in any
Phase-1/2 artifact and not in any other Phase-3 stage.

---

## 1. What the preregistered analysis returned

`p10_verdict.json`, primary outcome `neg_plain_loss` (held-out L2 on the executed DDIM action),
4-seed mean, focal C1/C5:

| target | measured 4-seed ceiling | bar (½·ceiling) | best preregistered attributor | verdict |
|---|---|---|---|---|
| C1 | **+0.040** | 0.020 | TracIn, ρ = −0.055 | FAIL |
| C5 | **+0.081** | 0.041 | TracIn, ρ = −0.162 | FAIL |

Taken at face value this is a FAIL, and the preregistered symmetric interpretation would read
*"the null generalizes across policy class."*

**That reading would be WRONG, and it is not being made.**

## 2. Why the verdict is uninformative

**The ceiling collapsed.** A ceiling is the reliability of the ground truth itself — what an
*oracle* could achieve. Phase 2's FAIL was meaningful **precisely because its ceiling was 0.93**:
the instrument could see, and the attributor still could not. Here the instrument cannot see at
all.

Three independent symptoms, all pointing the same way:

1. **Four of the nine measured ceilings are NEGATIVE** (C3 −0.014, C6 −0.127, C8 −0.206,
   C9 −0.350). A true reliability cannot be negative. These are noise.
2. **The Spearman–Brown consistency check fails badly.** From the measured 1-seed reliability,
   SB predicts the 2-seed ceiling; on C1 it predicts **+0.387** and we measure **+0.020**. In
   Phase 2 the same check agreed to within **0.014**. The 2v2 estimator here — 3 pairings over
   24 masks — has enormous sampling error.
3. **Several LDS values EXCEED their own ceiling** on the denoising outcome (TracIn C1
   ρ = +0.464 vs ceiling 0.263). An attributor cannot beat the oracle; this says the ceiling is
   *underestimated*, not that attribution is superhuman.

The per-seed agreements are visibly erratic — C1's six pairwise 1-seed Spearmans are
`+0.042, +0.422, +0.238, +0.139, +0.062, +0.537`.

## 3. The mechanism (measured, not speculated)

The diffusion policy's held-out L2 is **far more seed-variable** than the BC-Transformer's, on the
*same 24 masks*:

| | between-mask sd (**signal**) | within-mask seed sd (**noise**) | **signal/noise** |
|---|---|---|---|
| BC-Transformer (C1) | 0.117 | **0.073** | **1.61** |
| **Diffusion (C1)** | 0.267 | **0.489** | **0.54** |
| BC-Transformer (C5) | 0.065 | 0.047 | 1.38 |
| **Diffusion (C5)** | 0.304 | 0.538 | 0.56 |

The seed noise is **~7× larger in absolute terms** and swamps a data signal that is itself larger
than the BC-Transformer's.

**Why.** Both policies are *deterministic given their weights* — that was verified end-to-end
(`p10_determinism.json`: bit-identical replays, varied step counts). But determinism is not
stability. The BC-Transformer's executed action is the argmax GMM mode: a smooth, single-valued
function of the weights. The diffusion policy's executed action is the output of a 5-step DDIM
trajectory launched from a *fixed* latent — and a diffusion model is *multimodal*. Two seeds that
learned the same task can put that fixed latent in **different basins**, so the executed action
jumps between modes across seeds even though each model is individually deterministic.

So: **the diffusion policy is deterministic but not seed-stable in action space.** That is a real
finding, and it is arguably the most interesting thing P10 produced — but it means the S=4
executed-action ground truth cannot support a ratio-to-ceiling test.

## 4. This is the study's own central finding, applied to itself

Phase-2 P4: *reliability is bought with seeds.* We gave the BC arm **6** seeds and the diffusion
arm **4** — and the diffusion arm needed **more**, not fewer, because its seed noise is ~7× larger.
The P10 design under-provisioned exactly the axis this study exists to warn about.

## 5. Remedy (decided and executed; both results reported)

1. **The preregistered S=4 analysis is reported EXACTLY as computed** — FAIL — and **flagged as
   uninformative**, with this document cited. The preregistered symmetric interpretation is
   **NOT** invoked, because it presupposes a ceiling the instrument did not deliver.
2. **DISCLOSED DEVIATION:** two further seeds (**605, 606**) are trained on the same 24 masks
   (48 retrains, ~15 GPU-h), raising the diffusion arm to **S = 6** — matching the BC arm exactly,
   and enabling the **same 3v3 / 10-distinct-split ceiling estimator Phase-2 P1 used** instead of
   the 3-pairing 2v2 estimator that failed here.
   * The **criterion is unchanged**: focal C1/C5, primary held-out L2, any preregistered attributor
     ρ ≥ 0.5 × the measured ceiling, one-sided p < 0.025 (Bonferroni-2).
   * Only **S** changes, 4 → 6. This is precisely the move Phase 2 made on Phase 1 (raise the
     ceiling, do not lower the bar), and it is affordable: Phase-3 total goes to ~64 GPU-h against
     an 85 GPU-h alert threshold.
   * **Both** the S=4 and the S=6 results are reported. If S=6 still yields a collapsed ceiling,
     P10 is reported as **INCONCLUSIVE on the executed-action outcome** — a measurement failure,
     not evidence for either arm of the symmetric interpretation.
3. The **diffusion-native denoising-loss outcome** is reported in full alongside, with the same
   caveat about its ceiling.

## 6. What is NOT affected

* No Phase-1 or Phase-2 artifact or number. P10 is a new, self-contained arm.
* No other Phase-3 stage. P6, P7, P8a, P8b and P9 all rest on the **BC-Transformer** ground truth,
  whose ceilings are high and whose SB consistency check passes (Phase-2 P1: predicted 0.947 vs
  measured 0.933).
* The 96 P10b retrains are **not wasted** — they are reused unchanged; the two new seeds are added
  to them.
