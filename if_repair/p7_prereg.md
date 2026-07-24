# Pass 7 preregistration — frozen before any campaign has a single run

Written after W0 (zero GPU, commits `1d706d2`, `6a1e9ad`, `0666081`, `ff00075`) and before any
new outcome exists. Everything in §1–§4 is fixed. §5 records the branch verdict; §6 is the
conditional campaign spec and is **not authorised to run** until the user selects an option.

---

## 1. The branch verdict — Branch C

The branch rule was frozen in the pass-7 brief before W0.1 was computed, keyed on the without-J
pooled contrast for RelatIF/C5 on K ∪ L:

| branch | condition | measured | verdict |
|---|---|---|---|
| A — resolved positive | paired p < 0.01 **and** ratio ≥ 0.5 | p = 0.659, ratio 0.006 | no |
| B — unresolved | Δρ > 0 with CI including 0 | Δρ = **−0.044** | no |
| C — resolved negative/null | Δρ ≤ 0, or CI excluding +0.15 from above | Δρ = −0.044 ≤ 0 | **TRIGGERED** |

**Branch C.** Campaign J was a favourable draw; the C5 self-influence effect is at best
negligible out of sample. The brief's instruction for this branch: do not spend 10 GPU-hours
chasing it, put the main budget on W2, and write the pass up as a resolved negative with the CI
as the headline.

### What W0.1 actually overturned

`FINDINGS.md` (passes 4–6) claims the effect is sign-consistent across all six draws — *"RelatIF
≥ GradDot on C5 on every fresh draw (J +0.34, L +0.075)"*. **That claim is false, and was only
ever true because campaign K had never been scored.** RelatIF/C5 was frozen in `PREREG_J`
(`c7659cf`) and never retuned, so K is legitimately out-of-sample for it; K's own prereg family
named C2/C8/C7, so nobody ever ran the C5 config against it.

| draw | role | Δρ vs `GradDot_dmean` |
|---|---|---|
| G | dev | +0.226 |
| H | dev | +0.384 |
| I | dev | +0.135 |
| J | **discovery** | +0.341 |
| K | **oos, never scored until now** | **−0.171** |
| L | oos | +0.075 |

Every selection draw is positive; the effect reverses on the one clean draw nobody had looked at.

**Correction required in `FINDINGS.md` and `RESULTS.md`:** the passes 4–6 headline
("the first gradient-side estimator to beat GradDot out of sample, in direction, consistently
across six mask draws") does not survive. It should read: consistent across the five draws it was
selected on or reported on, and reversed on the sixth.

---

## 2. Primary statistic — frozen

**Primary: Kendall τ_b. Mandatory secondary: Spearman** (continuity with every past number in the
repo). Chosen in `p7_design.py --stage statistic` on **reliability and noise only** — split-half
reliability of the outcome against itself, and the spread of a random predictor — criteria that
never touch the RelatIF-vs-GradDot contrast. Committed (`0666081`) before the contrast under any
candidate was computed (`ff00075`).

Validation of that method, after the fact: Kendall returns CIs ~⅓ narrower than Spearman on the
same masks (C5 K∪L width 0.306 vs 0.459), exactly as the estimator-free criterion predicted.

Correction to the brief: it lists "Pearson on ranks" as a candidate distinct from Spearman.
Pearson on ranks **is** Spearman by definition; the small gap in the selection table is
Monte-Carlo noise between independent draws.

## 3. Allocation — frozen

**A retrain spent on a new mask is worth about five spent on a new seed.** Paired sd is flat in
depth (0.179 → 0.183 from 2 to 10 seeds at n=24) and falls as 1/√n in masks. Any pass-7 campaign
uses **depth 2** and spends the rest of its budget on masks.

At a fixed 240 retrains: 120×2 gives paired sd 0.077; 48×5 gives 0.121; the archived 24×10 gives
0.183 — the archived protocol is close to the worst allocation available. Mean ratio-to-ceiling is
flat across allocations (0.405 vs 0.377), so trading depth for masks does not harm the absolute
bar even though the ceiling falls (0.945 → 0.809).

Per the brief §2, any number compared to a project-historical one is **also** reported at depth 10,
recomputed from existing per-seed data.

## 4. Bars, α, CI method — frozen

- Two bars unchanged: ABSOLUTE (ratio ≥ 0.5, one-sided p < 0.05/|family|) and PAIRED (beats
  `GradDot_dmean` on the same masks, one-sided mask-bootstrap p < 0.05).
- **Every primary contrast is reported as an effect size with a 95% CI.** The CI is the
  deliverable. A hypothesis whose CI excludes +0.15 is *resolved negative*.
- CI method: percentile bootstrap over masks, 5000 resamples, **stratified within draw**.
- The honest draw set is **per config**, derived from `CONFIGS[...]["discovery"]`, never hardcoded
  per pass. Different configs have different discovery draws (J for RelatIF/C5, K for
  leverage/C7), and scoring one on the other's set manufactures a false out-of-sample win — this
  actually happened during W0.2b and was caught before commit.
- Any estimator consuming outcomes goes through leave-one-mask-out (BLOCKERS #7). Not applicable
  to anything in §6, all of which is zero-outcome.
- Ceilings via the archived `split_half_ceiling` recipe; the pooled-union form is asserted
  identical to it in `tests/test_p7.py`.

---

## 5. What W0 changed about the brief's own assumptions

Two of the brief's premises are now measured and wrong, and both make the remaining decision
cheaper than it looked:

1. **Win condition 1 is unachievable as written.** It asks for ≥72 out-of-sample masks with a 95%
   CI width ≤ 0.20. At 72 masks the CI width is ~0.40 *at every depth*. No allocation of a 72-mask
   campaign reaches 0.20.
2. **But the resolving campaign is ~13× cheaper than the brief assumed.** With the primary
   statistic and depth 2, CI width 0.20 needs ~112 masks ≈ 224 retrains ≈ **5.9 solo-h** — not the
   ~77 solo-h the archived protocol would have required. The brief sized W1 at 72 masks × depth 5
   (9.4 solo-h) for a CI that would still have been ~0.40 wide.

This is why §6 needs a decision rather than an automatic execution: Branch C says "do not spend 10
GPU-hours chasing C5", and that instruction was written on the assumption that a resolving
campaign costs ~10 GPU-hours. It costs ~6, and it would convert a wide null into a tight one.

---

## 6. Conditional campaign spec — NOT AUTHORISED TO RUN

No campaign may start until the user selects. Whichever is selected, the family below is frozen as
written here and scored **once**.

### Option 1 — W2 duel design only (Branch C as literally written), ~9–12 solo-h

Mandatory pilot first: 4–6 duels at depth 3–5 (~0.6–1.6 solo-h). **Kill rule:** if the within-duel
outcome difference is under 1 seed-noise sd, kill W2, write the pilot up as a negative design
result, and return the budget. Full arm only if the pilot passes: ~32–40 duels × 2 masks × depth.
Preregister `n_duels`, the demo-pair selection rule and its seed, the depth, and the test
(one-sided sign test on discordant pairs, α = 0.05) before any duel is trained.

### Option 2 — the resolving campaign only, ~7.5 solo-h

Campaign **M**, one draw of **144 fresh masks at depth 2** (288 retrains), seed 20260726, prefix
`"M"`, wired into `retrain.py` and `export_outcomes.py`; `tests/test_mno.py` asserts all seven
draws are pairwise disjoint at mask level with correct 8/8/8/8/8/7/7/7/7 stratification.
Predicted paired sd ~0.045 → **CI width ~0.18**, which achieves win condition 1.

`PREREG_M`, family of **1** (so α_abs = 0.05, stated per the brief):
- **M1 — RelatIF/C5** (cached E=20, `unitl2_then_mean`; the exact `L2` config from
  `confirm_lseries.py`), paired vs `GradDot_dmean`-cached, primary statistic Kendall τ_b.

C7 does **not** enter: on its honest set (J∪L) it is +0.032, p = 0.406, and the brief admits it
only inside a high-power protocol, which a single-hypothesis campaign is not.

### Option 3 — both, ~17–19 solo-h (inside the 22 soft cap) — RECOMMENDED

Option 2 first (it is the cheaper and the surer deliverable, and it closes the question of
record), then Option 1 with the remainder. Satisfies win conditions 1 and 3 together.

### Option 4 — stop here, 0 solo-h

W0 already answers the pass's question at the resolution the existing data supports. The write-up
is a resolved negative with Δρ = −0.05, CI [−0.20, +0.11] on 48 clean out-of-sample masks, plus
the exchange-rate result. Defensible, and free.

---

## 7. Stopping rule

Whatever is selected is scored **once**, at the prespecified stopping point, with every
preregistered cell reported regardless of outcome. **No optional stopping and no depth top-up.**
The group-sequential option the brief allowed is explicitly declined: depth top-ups buy almost
nothing (§3), so there is no reason to keep the option open and every reason to close it.

W3 (wider / second-order exact LOO) is **not run**. Its gate was pooled dev Δρ ≥ +0.10 above the
current C5 champion, and the champion is now measured at ≈ 0 out of sample, so the gate no longer
identifies anything worth confirming.
