# RoboTDA-X — PHASE 4 STATUS

Preregistration `phase4/preregistration_phase4.json`, SHA-256 `fca37f54804c0173f5a9029e99b220de3db2a97ca2af7b3cf3c1c1a5372ecb80` — locked before any Phase-4 training, attribution, or verdict.

| stage | what | retrains (ledger) | retrains (actual) | ok/fail | GPU-h (ledger) | GPU-h (actual) | episodes | status |
|---|---|---|---|---|---|---|---|---|
| P12 | BC ground truth S=10 | 96 | 96 | 96/0 | 13 | 11.95 | 25920 | complete |
| P13 | diffusion S=8 + verdict | 48 | 48 | 48/0 | 17 | 15.35 | 12960 | complete |
| P14 | fresh heterogeneity corpus | 64 | 64 | 64/0 | 9 | 5.18 | 3840 | complete |
| P11 | attribution (Gram pass, 0 retrains) | 0 | 0 | — | 1 | 0.02 | 0 | complete |
| **TOTAL** | | **208** | **208** | **208/0** | **40** | **32.50** | **42720** | |

**GPU-h alert threshold: 60. Not tripped.**

Cuts: NONE

## The three preregistered verdicts

| stage | verdict | numbers |
|---|---|---|
| **P11** champion test (BC) | **CHAMPION FAIL** — preregistered **SECONDARY PASSES on C1** | champion TracIn E=20: C1 ρ=+0.409 (ratio 0.43, p=0.0237), C5 ρ=+0.304 (0.32, p=0.0741). Secondary GradDot (λ→∞): **C1 ρ=+0.513, ratio 0.54, p=0.0052 < 0.0125 (Bonf-4) → PASS** |
| **P13** diffusion settlement | **PASS on C1** | C1 ρ=+0.477, ratio **0.84**, p=**0.0092**, median ceiling 0.569. C5 FAIL (0.11, p=0.369). C2 INSTRUMENT-UNUSABLE |
| **P14** fresh heterogeneity | **FAIL on all 4 cells, both families** | C2@g1 +0.29, C2@g3 −0.23, C9@g1 +0.01, C9@g5 −0.19 (ratios). Fresh ceilings 0.837/0.851 |

## Instrument gates

| gate | result |
|---|---|
| P12 SB-consistency (BC S=6→10) | **PASS 9/9** — C1 miss 0.017, C5 0.010 |
| P13 ceiling usability (diffusion) | C1 **USABLE by 0.003** (0.147 vs 0.15); C5 USABLE (0.014); **C2 UNUSABLE (0.221)** |
| P14 fresh-mask coincidence | **PASS** — 0/16 coincide with Stage G, 0 seed increments |
| P11 probe-leak guard | **PASS** — 10 members, 90 probe ids, 0 intersections |
| P11 Gram self-validation | **PASS** — 180/180 vs archived P9, ρ=1.000000, max rel diff 2.8e-10 |
| SYNTHETIC unit tests | **32/32 PASS** |
| Report verification | **159/159 numbers verified, 0 mismatches** |

Deviations: none to any criterion (two disclosed design additions, both locked before any number).
Incidents: **none.**
