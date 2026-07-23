# Phase 5 — STATUS (COMPLETE)

Preregistration locked: `phase5/preregistration_phase5.json`
SHA-256 `b4883cb1a7048aee1a47410291573c3e27401c7e08be7852f67c58f42f2922b8` (+ `.sha256`).
Locked BEFORE any Phase-5 training/attribution/analysis on real data.

Final report: `phase5/PHASE5_REPORT.md`. Verification: **78/78 checks pass**
(`results/report_verification.json`).

| stage | state | result |
|---|---|---|
| SYNTHETIC unit tests | **DONE** | 15/15 pass |
| P15 diffusion S=10 (48 retrains) | **DONE** | **PASS (C1)** on a sound instrument. Stricter gate all-pass (ceiling 0.830; S4→10 0.070; S8→10 **0.010**). Champion ρ=+0.479 (≈ P13's 0.477), ratio 0.58, p=0.0089. |
| P16 GradDot breadth (0 retrains) | **DONE** | **k = 0 of 7** → C1-specific. Self-val C1 GradDot=+0.5130434782608695 EXACT; probe-leak 20/20. |
| P17 exploratory diffusion GradDot (0 retrains) | **DONE** | EXPLORATORY: TracIn (0.479) > GradDot (0.414) on C1 — reverse of BC. Not a verdict. |

## Verdicts, verbatim interpretation
- **P15**: *"the diffusion C1 result carries full weight: demo-grain faithfulness is
  policy-class-dependent and the BC null does not generalize — now replicated at two seed depths on a
  sound instrument."*
- **P16**: *"the achievability result is C1-specific on this corpus; the paper must scope it that
  way."*
- **P17**: EXPLORATORY, no interpretation licensed.

## Budget (actual vs ledger)
48 retrains (48/0 ok/fail), **8.46 GPU-h** vs 16 nominal / 25 alert, 1,440 episodes. No alert, no
cut, no incident. `results/budget_actual.json`.
