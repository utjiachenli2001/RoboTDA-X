# PASS 4 PLAN — new estimator classes (unlearning, exact surrogate-LOO, TRAK, timing, hybrids)

> Status at start of pass 4: passes 1-3 merged to `main` (791a46b). Across three independent
> mask draws (G/H/I) the **only** estimator that beats GradDot out of sample is the
> outcome-consuming datamodel; every gradient-side idea was mask-draw overfitting. Pass 4 attacks
> from four untried directions under a strict two-bar out-of-sample protocol.

Read `FINDINGS.md`, `RESULTS.md`, `BLOCKERS.md`, `HANDOFF.md`, `MAP.md` first — they are ground
truth. Where this plan and those files disagree on a number, trust the files.

## The bar to beat (BLOCKERS #1)

**`GradDot_dmean`** (`p6_lambda_extend.scores_graddot(normalize_per_member=True)` = `K_m / mean(diag G_m)`):

| | C1 | C5 |
|---|---|---|
| GradDot_dmean, E=20 cached Grams | LDS 0.593 (ratio 0.624) | 0.390 (0.416) |

Never call a win against the paper's 0.513 (`GradDot_unitL2`). On regenerated ensembles recompute
the GradDot_dmean baseline **on the same ensemble** (regenerated Grams are not comparable to cached;
outcomes ARE reusable, Spearman 0.61-0.93 — BLOCKERS #6/#12).

## Win conditions (any one)

1. A **zero-outcome** estimator (W1/W2/W3/W4/W5) beats GradDot_dmean on a **virgin mask draw
   (campaign J)** under BOTH bars on C1 or C5.
2. W6 lets the datamodel reach its 24-mask LDS with **≤12 masks**, confirmed on the virgin draw.
3. A **clean preregistered negative** — all workstreams dev-screened, surviving ≤3 confirmed-failed
   on J under the two-bar protocol. **This is publishable; do not torture estimators to avoid it.**

Budget: **15 solo-GPU-h soft, 20 hard.** Track with `gpu_ledger.py`.

## Evaluation protocol (non-negotiable)

- **Dev**: G (seed 11), H (seed 4711), I (seed 9973) — all consumed, all legitimate for
  development. Prefer pooled paired analysis (`b8_maskdraw.py`).
- **Confirmation**: campaign **J**, `retrain.fresh_demo_masks(seed=<new>, prefix="J")`, 24 masks ×
  10 seeds, archived protocol; disjointness test vs G/H/I (pattern `tests/test_iseries.py`).
- **Two bars per confirmatory hypothesis** (BLOCKERS #20):
  - ABSOLUTE: ratio-to-ceiling ≥ 0.5, p < 0.05/|family| (Bonferroni; family ≤ 3 → α = 0.0167).
  - PAIRED: beats GradDot_dmean on the SAME masks, one-sided mask-bootstrap p < 0.05.
- **Order**: dev on G/H/I → select ≤3 hypotheses → write `confirm_jseries.py` with a frozen
  `PREREG_J` → **commit while campaign J has zero runs** → launch J → score once → report every cell.
- Seed depth always **10**. Ceilings via `functionals.split_half_ceiling`. Any outcome-consuming
  estimator (W6) scored **leave-one-mask-out**, prior strength tuned inside the LOO loop.
- A **wiring smoke test** on archived masks for every scorer BEFORE it sees campaign J.

## Kill rule (every workstream)

After the first full dev evaluation, if the best cell's paired Δρ vs GradDot_dmean (pooled G/H/I)
is < **+0.10** on BOTH C1 and C5 → stop tuning. One configuration sweep, two tuning iterations max.

## Workstreams

Run W4/W6 (zero/near-zero GPU) while W1/W2/W3/W5 caches build.

### W4 — Φ-hygiene rescoring (`b14_rescoring.py`) — ~0.5h GPU, rest CPU
RelatIF (`K[d,t]/√G[d,d]`, `K[d,t]/G[d,d]`) — **runnable on cached E=20 Grams for free, the only
pass-4 idea evaluable against the true 0.593; do that first.** Plus per-query aggregation,
frame-robust demo gradients (winsorize / unit-norm / top-k-loss), σ-clamped NLL gradients. Only the
single best dev cell enters the prereg family, and only if paired Δ ≥ +0.15.

### W6 — Hybrid gradient-prior datamodel (`b16_hybrid.py`) — ZERO GPU at dev
GradDot as a prior on the datamodel: (i) adaptive-LASSO `w_d ∝ 1/(|s_d|+ε)^γ`; (ii) kernel ridge
`αĜ+(1−α)I`; (iii) prior-mean ridge toward `c·s`. All inside the LOO machinery, prior strength tuned
in-loop. Dev: subsample K ∈ {6,9,12,16,20,24}, LOO-LDS of hybrid vs plain datamodel vs GradDot,
per target. Deliverable: LDS-vs-K curve. Prereg gate: hybrid@12 ≥ plain@24 on C2 or C5.

### W2 — Exact surrogate-LOO on the head (`b12_headloo.py`) — ~0.5h, diagnostic
Head is `nn.Linear(512, 75)` (+ `ln_f`). Freeze trunk, cache features φ(s) for ~92k train + held-out
frames; fit ridge head predicting the executed action (MSE surrogate), λ by GCV. Per demo, remove its
frame block by Woodbury; score_d = Δ held-out L2 on the target. **Interpretation is the deliverable
even if it loses**: if exact frozen-trunk LOO also fails, linearization error is exonerated.

### W1 — Unlearning-LOO (`b11_unlearn.py`) — headline, ~3-4h
From each regen E=5 `final.pt`, per demo d: (A) k ascent steps on d's frames; (B) finetune-forget on
train-minus-d; (C) SCRUB-lite (ascent on d interleaved with replay descent). score_d = Δ held-out L2
per target cluster. Grid k∈{10,50,200}, η∈{0.1×,1×}. Scores never touch outcomes ⇒ direct LDS.

### W3 — TRAK (`b13_trak.py`) — ~1-2h, due diligence
Exact dual TRAK on head-Φ (`(G+λI)⁻¹K`, reuse `attribution.py`), λ sweep, ensemble-mean. Close the
"did you try TRAK?" question with a number. Prior honestly low (N=135 caps kernel rank).

### W5 — Mid-training GradDot (`b15_ckptgrid.py`) — ~0.2-1.5h, loose end
One member, ckpt every 400 steps, GradDot_dmean per ckpt on C1+C5. If the curve is spiky
(adjacent swings ≳0.2) close as ckpt noise. Step 2 (all 5 members, GradDot@τ*) only on a smooth
plateau, τ* chosen on C1 dev only.

## Confirmation — campaign J
Select ≤3 hypotheses total. Freeze `PREREG_J` (estimator config, target, both bars, α) in
`confirm_jseries.py`; commit with J at zero runs; launch 24×10 (~5.8 solo-h); extend
`export_outcomes.py` so the parquet gains J rows; score once, report all cells. W6 J-eval is LOO
within J vs plain datamodel@K on same masks.

## Deliverables
1. Code `b11`-`b16` + `confirm_jseries.py` + tests (J disjointness; wiring smoke per scorer).
2. Results CSVs under `results/` (committed); J parquet rows committed; npz caches machine-local.
3. Docs to pass-3 standard: FINDINGS/RESULTS/BLOCKERS/HANDOFF pass-4 sections; gpu_ledger CSV.
4. Every FINDINGS claim states which mask draws it used and whether dev or confirmed. "Beats
   GradDot" reserved for J-series results.
5. Commit + push to `main` incrementally; then `sudo poweroff` and remind user to confirm STOPPED.

## Honest priors
W1 and W6 have a real chance; W2 is cheap and decisive as a diagnosis; W3/W4 due diligence; W5 a
loose end. A complete negative with three virgin-draw confirmations behind it is a real result.
