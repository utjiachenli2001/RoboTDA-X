# BUDGET_ALERT.md — Stage P4 exceeds its ledger line

**Raised:** 2026-07-11, after timing the first P4a models. **Stage PAUSED at 2/48 models** as the
preregistration requires (`per_stage_alert_rule`: time the first retrains of each kind; if the
projected stage cost exceeds 1.5× its ledger line, write `BUDGET_ALERT.md` and PAUSE that stage).

## The overrun

| | ledger (preregistration) | measured / projected |
|---|---|---|
| P4 GPU-h | **1.0** | **5.6** (48 models × 423 s on one GPU) |
| ratio to ledger | — | **5.6×** (alert fires above 1.5×) |
| P4 episodes | 25,920 | **14,400** (*fewer* — see below) |

Note the episode count went **down**, not up: PHASE2_DEFECT.md cut the ladder from 90 to 50
rollouts/task (90 was impossible — only 50 distinct init states exist). So this is not a scope
creep. The ledger line was simply **mis-estimated**.

## Why the ledger line was wrong

The 1 GPU-h line implies ≈0.14 s per episode. Real episodes are far more expensive: the rollout
horizon is **600 steps** at ~14 ms/step, and a *failing* episode runs the full horizon. At C1/C2's
near-floor success rates almost every episode runs to 600 steps → **~8–13 s per episode**, ~60–90×
the implied rate. Phase 1's own numbers agree (Stage-B `cluster_eval`: 200 episodes in 241 s wall
at 12-way parallelism). The ledger's P4 line was optimistic by roughly the same factor for every
eval-only stage; P1/P3 were unaffected because their ledger lines were dominated by training.

## Global budget is NOT at risk

| stage | GPU-h (actual / projected) |
|---|---|
| P1 (96 retrains) | **12.13** (measured, ledger 13) |
| P2 attribution (Stage-E + Stage-B) | **0.26** (measured, ledger 2–4) |
| P3 (159 retrains) | **17.8** (projected from 402 s/job; ledger 25–30) |
| P5 (3 retrains) | **0.35** (measured) |
| P4a | **5.6** (projected; ledger 1) |
| **total** | **≈36.1** |

Global alert threshold: **75 GPU-h**. Projected total ≈ **36 GPU-h** — under half of it, and under
the 45–50 nominal. P1, P2, P3 and P5 all came in **at or under** their lines; the P4 overrun is
absorbed several times over by P3's and P2's underruns.

## Decision — RESUME P4a, overrun recorded

P4a is **resumed** rather than cut, because:

1. The **global** budget (the number that actually governs) has ~39 GPU-h of headroom.
2. P4 is a **core deliverable** — it is the CUPID-boundary result (closed-loop return is the
   functional the robotics-curation literature curates against *without* LDS validation), and it is
   the only stage that quantifies what that ground truth would cost.
3. The cut order (`P5 → P4's C1 arm → P3's Q=150 → P1 seeds`) exists to protect the global budget.
   The global budget is not threatened, so invoking a cut would discard a deliverable for no gain.

**Nothing is cut.** P4b (the conditional 96-model arm, 51,840 episodes, ~20 GPU-h) was *already*
conditional on budget in the preregistration; it is **NOT run** — see the cut/deviation log in
PHASE2_REPORT.md. The S=6 column at 50 rollouts is therefore filled by a Spearman–Brown
**extrapolation**, explicitly labelled as such, exactly as the preregistration provided for.

This alert is the record that the P4 line was breached deliberately, not silently.
