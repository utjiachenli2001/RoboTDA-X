# GATE 1 — FAIL (attributor sanity)

Preregistered criterion (`preregistration.json` → `gates.gate1_stage_D`):

> PASS iff **any** attributor (TracIn, TRAK, EK-FAC IF) reaches **Spearman > 0.50** against the
> retrained outcome (held-out plain loss primary, success secondary) over the 12 masks.

**Verdict: FAIL.** Best attributor ρ = **+0.252** (IF), against a required 0.50.

Consequence, per spec §5: Stages E–G still run (retrains are attribution-agnostic ground truth),
Stage H's LDS numbers are still computed **as evidence**, but the study does **not** make
per-demo attribution claims as though attribution were trustworthy. Every downstream
attribution statistic in `REPORT.md` carries this caveat.

---

## 1. The numbers

Design: C1 only. K=12 masks, each exactly 8 of C1's 15 training demos (each demo in 6 or 7
masks, seed 23) × 2 seeds (101, 102) = **24 retrains**. Attributors computed from the five
full-C1 models (Stage B `C1_target_s101..s105`, 5 checkpoints each) toward the C1 held-out
functional. Predicted mask score = sum of per-demo attributions over the mask's 8 demos.

Sign convention: scores are oriented so a faithful attributor gives a **positive** Spearman
(loss reported as negative loss = "utility").

| attributor | ρ (held-out **L2**, PRIMARY) | p (one-sided) | ρ (success, secondary) |
|---|---|---|---|
| **IF** (exact Woodbury empirical-Fisher) | **+0.252** | 0.215 | −0.327 |
| **TRAK** (exact dual form, E=10) | +0.189 | 0.278 | −0.359 |
| **TracIn** (5 ckpts × 5 models) | −0.042 | 0.552 | +0.056 |

**Noise ceilings** (Spearman between the two seeds' outcome vectors across the 12 masks,
Spearman–Brown-corrected to the 2-seed mean that the LDS actually predicts):

| outcome | 1-seed ρ | **2-seed ceiling** |
|---|---|---|
| held-out L2 | +0.399 | **+0.570** |
| success (30 episodes) | −0.318 | **−0.932** |

## 2. What the failure does and does not mean

**It is a real failure, and it is not close.** IF reaches +0.252 against a ceiling of +0.570 —
i.e. it captures **≈44% of the achievable rank signal**, and its 95% CI includes 0. Two of the
three attributors are statistically indistinguishable from noise; TracIn is slightly negative.
None is usable for confident per-demo claims.

**But the bar is near-oracle.** The preregistered threshold of 0.50 is **88% of the measured
ceiling (0.570)**. Gate 1 as specified therefore demands that attribution be almost as good as a
perfect oracle. A milder reading of the same data — "IF carries genuine but weak signal" — is
also consistent with these numbers, and the report says so rather than overclaiming a clean null.

**The success outcome is unusable at this grain, and that is a finding.** Its ceiling is
**negative** (−0.93): the two seeds of the same mask produce *anti-correlated* success vectors.
With C1 success pinned near the floor (0–11.7% over 30 episodes) and 8-of-15-demo masks, closed-
loop success carries no reproducible rank information at demo grain. Any LDS computed against it
would be measuring noise. This is why the primary outcome is the loss.

## 3. An instrument defect was found and fixed BEFORE this verdict was accepted

Gate 1 first ran with the preregistered **GMM-NLL** functional and returned FAIL with best
|ρ| = 0.21 (archived: `results/stage_D_gate1_GMMNLL_ORIGINAL.json` — IF +0.077, TRAK +0.105,
TracIn −0.210). Rather than accept that, I checked whether the *ground truth itself* was
predictable — whether two seeds of the **same** mask agree. They did not:

| mask | held-out GMM-NLL (seed 101) | (seed 102) | → L2 (seed 101) | L2 (seed 102) |
|---|---|---|---|---|
| D00 | 23.6 | **187.0** | 1.0654 | 1.0695 |
| D04 | 64.2 | **286.1** | 0.6543 | 0.8191 |
| D10 | 27.1 | **239.3** | 0.8535 | 0.8459 |

Mechanism: the **median** per-frame NLL is nearly unchanged across those seeds (27.6 → 32.4 →
36.0), but **34–36% of frames exceed NLL 100** in the bad runs. The GMM's σ collapses toward its
1e-4 floor and NLL is **unbounded** where the model is confidently wrong, so the *mean* NLL was
measuring σ-collapse tail noise rather than data quality. No attributor can predict an outcome
that unreliable.

**Fix:** the spec permits either loss — *"plain action loss (L2 or GMM NLL — pick one, freeze)"*.
The **evaluation/attribution functional** is now **L2 on the executed action** (bounded; actions
lie in [-1,1]). The **training objective is unchanged** (still GMM NLL, which yields 70% success
on the single-task ceiling test). No retraining was needed — losses are recomputed from saved
checkpoints. The NLL numbers are retained under `*_nll` in every `outcomes.json`.

**This change was made after seeing a gate result — exactly what preregistration exists to
prevent — so it is disclosed here, in `STATUS.md`, and in the report's deviations list.** The
justification is a demonstrable instrument defect (medians stable, means exploding, σ→floor),
established independently of any gate outcome. It was not a search for a way to pass: **the gate
still fails after the fix** (0.252 < 0.50). The fix moved the best ρ from 0.077 → 0.252 and made
the verdict *interpretable* rather than *vacuous*.

## 4. Why attribution is hard here (context, not excuse)

- **The attributors were given their best possible shot.** We compute the **exact** TRAK dual
  form and the **exact** Woodbury empirical-Fisher IF rather than dattri's JL-projected TRAK and
  EK-FAC-approximated IF, because with N=135 demos vs p=19.2M parameters the projected Gram is
  singular and the exact Fisher inverse is available in closed form. Both are verified to ~5e-15
  against dense brute force. The failure therefore **cannot** be blamed on sketching or
  factorisation error.
- **The ground truth is intrinsically noisy at this grain.** Removing 7 of 15 demos from a
  15-demo cluster changes the outcome by less than training-seed noise does (2-seed ceiling
  0.57, i.e. even an oracle would score only 0.57).
- This is itself the headline scientific content: **in this regime, which demos you keep matters
  less than which seed you draw.**
