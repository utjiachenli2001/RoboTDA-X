# FINDINGS — confirmatory hold-out (computed once) and cross-policy transfer

Run once, with `if_repair/configs/frozen.yaml` (`frozen: true`), via
`python -m if_repair.run holdout --config if_repair/configs/frozen.yaml`.
The config was fixed on C1/C5 alone before this command was executed, and has not been
edited since. Hold-out targets C2, C4, C7, C9 were not evaluated at any earlier point.

## What was frozen, and why

Tasks 1–3 produced **no** estimator that beats the strongest dev baseline on C1
(`GradDot_dmean`, LDS 0.5930). So:

- **PRIMARY — `GradDot_dmean` + mean.** The strongest dev performer; equals `truncated_if`
  at k=0 in the dmean convention, i.e. the true λ→∞ limit of ensemble-mean IF/TRAK.
- **SECONDARY — `GradDot_unitL2` + Hodges–Lehmann.** The best genuinely *new* variant from
  Tasks 1–3: the only one that improved on its own baseline on C1 (0.5130 → 0.5678).

The published champion (`GradDot_unitL2` + mean) is **not** part of the tested family — its
hold-out values are already archived in `phase5/results/p16_lds_table.csv`, so re-testing
them would inflate the correction for no new information. They are quoted as the reference.

Family size = 2 estimators × 4 targets = **8**. Bonferroni α = 0.025 / 8 = **0.003125**.
PASS requires ratio ≥ 0.5 **and** p < 0.003125.

## Hold-out table (`results/holdout_table.csv`)

| estimator | target | LDS | ceiling | ratio | bar | p | α_bonf | n | PASS |
|---|---|---|---|---|---|---|---|---|---|
| GradDot_dmean + mean | C2 | 0.2922 | 0.9270 | 0.315 | 0.4635 | 0.0830 | 0.003125 | 24 | ✗ |
| GradDot_dmean + mean | C4 | 0.2165 | 0.8839 | 0.245 | 0.4420 | 0.1548 | 0.003125 | 24 | ✗ |
| GradDot_dmean + mean | C7 | 0.3757 | 0.9492 | 0.396 | 0.4746 | 0.0352 | 0.003125 | 24 | ✗ |
| GradDot_dmean + mean | C9 | 0.2200 | 0.8946 | 0.246 | 0.4473 | 0.1508 | 0.003125 | 24 | ✗ |
| GradDot_unitL2 + HL | C2 | 0.4061 | 0.9270 | 0.438 | 0.4635 | 0.0245 | 0.003125 | 24 | ✗ |
| GradDot_unitL2 + HL | C4 | **0.4896** | 0.8839 | **0.554** | 0.4420 | 0.0076 | 0.003125 | 24 | ✗ |
| GradDot_unitL2 + HL | C7 | 0.3617 | 0.9492 | 0.381 | 0.4746 | 0.0412 | 0.003125 | 24 | ✗ |
| GradDot_unitL2 + HL | C9 | 0.3887 | 0.8946 | 0.434 | 0.4473 | 0.0302 | 0.003125 | 24 | ✗ |

**Nothing passes.** One cell crosses the half-ceiling bar — `GradDot_unitL2` + HL on C4,
ratio 0.554 — but at p = 0.0076 it does not clear the Bonferroni-8 threshold of 0.003125.
It would clear an uncorrected α = 0.025. It is a single crossing out of eight preregistered
tests, which is what the correction exists to discount; it is reported, not claimed.

## The one durable signal: dmean and unitL2 swap places off C1

Against the archived champion (`GradDot_E20_normalized`, `p16_lds_table.csv`):

| target | champion LDS | champion ratio | unitL2+HL LDS | unitL2+HL ratio | Δ ratio | dmean+mean ratio |
|---|---|---|---|---|---|---|
| C2 | 0.3678 | 0.397 | 0.4061 | 0.438 | **+0.041** | 0.315 |
| C4 | 0.4304 | 0.487 | 0.4896 | 0.554 | **+0.067** | 0.245 |
| C7 | 0.3313 | 0.349 | 0.3617 | 0.381 | **+0.032** | 0.396 |
| C9 | 0.3539 | 0.396 | 0.3887 | 0.434 | **+0.038** | 0.246 |

Two things worth stating plainly:

1. **Hodges–Lehmann beat the mean on all four hold-out targets** within the champion's
   unit-L2 normalization, by a consistent +0.03…+0.07 in ratio. Four out of four with
   uniform sign is not proof at these n, but it is the only effect in this project that
   pointed the same way on every unseen target. It is the one thing worth a confirmatory
   retrain (see `HANDOFF.md`).

2. **The PRIMARY choice was wrong, and dev could not have told me.** `GradDot_dmean`
   dominates on C1 (0.593 vs 0.513) and loses to `unitL2` on three of four hold-out targets
   (C2, C4, C9; it wins only on C7). The dmean advantage is a C1 phenomenon, not a property
   of the estimator. Selecting the primary on C1+C5 picked the variant that generalizes
   *worse*. This is a direct, measured caution about how much the two focal targets can
   carry — and it is exactly why the hold-out was held.

## Cross-policy-class transfer: diffusion (`results/diffusion_table.csv`)

Tier `diff_s10` — the phase-5 diffusion Gram (E=5, `p17`), `p15_outcomes_S10`, 10-seed
**median** (the mean-aggregated outcome is broken for this arm; see `MAP.md`).
Family size = 2 × 2 = 4, Bonferroni α = 0.00625.

| estimator | target | LDS | ceiling | ratio | bar | p | PASS |
|---|---|---|---|---|---|---|---|
| GradDot_dmean + mean | C1 | 0.4200 | 0.8298 | 0.506 | 0.4149 | 0.0205 | ✗ |
| GradDot_dmean + mean | C5 | 0.1861 | 0.8444 | 0.220 | 0.4222 | 0.1920 | ✗ |
| GradDot_unitL2 + HL | C1 | 0.4200 | 0.8298 | 0.506 | 0.4149 | 0.0205 | ✗ |
| GradDot_unitL2 + HL | C5 | 0.1861 | 0.8444 | 0.220 | 0.4222 | 0.1920 | ✗ |

Both estimators cross the half-ceiling bar on C1 (ratio 0.506) and neither clears
Bonferroni-4. C5 is far below. The `GradDot_dmean` C1 value reproduces the archived
`p17_exploratory.json → GradDot_diffE5_dmean` ratio of 0.5062 — an independent check that
the diffusion tier is wired to the right triple.

The two estimators coincide exactly here. That is not a bug (they differ on `bc_s10`, and
`tests/test_estimators.py::test_frozen_estimators_are_actually_different_on_bc` asserts it):
with only M=5 members the Hodges–Lehmann of 15 Walsh averages induces the same ranking over
these 24 masks as the mean. It does mean the diffusion run carries only one effective
estimator, and the Bonferroni-4 correction is conservative there.

## Verdict

**No influence-score variant tested here is more faithful than the current champion, on the
held-out targets.** GradDot remains unbeaten. The specific claims:

- **Robust aggregation (Task 1):** no new maximum on dev. On hold-out, Hodges–Lehmann
  improved the champion's own normalization on 4/4 targets, but no cell survives correction.
- **Spectral truncation (Task 2):** the headline. Its curve peaks at **k=0** on C1 — the
  identity preconditioner. The interior peak on C5 is a width-1 spike that inverts sign
  under cross-validation. **Truncating the inverse does not repair IF; not inverting at all
  is what works.**
- **Shrinkage (Task 3):** no help; destructive toward cluster means.

The mechanism the k-sweep exposes is stronger than the one the brief hypothesised. It is not
that IF fails because the *small* eigendirections are noise-dominated and want truncating —
if that were so, an interior k would win somewhere and transfer. It is that with 135 demos
against a 19.2M-parameter model, the Gram carries **k\* ≈ 1** direction distinguishable from
random demo pairing, so essentially the whole preconditioner is noise. Under that condition,
the best available inverse is the identity — which is precisely GradDot, and precisely why
the paper's null has been so hard to move. The fix, if there is one, is more demos or a
lower-dimensional Φ (per-layer restriction — Task 4, blocked), not a better solve.
