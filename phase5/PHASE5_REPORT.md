# RoboTDA-X — PHASE 5 REPORT (PRE-PAPER DE-RISK)

**What Phase 5 was for.** Phase 4 flipped the study's conclusion — demo-grain attribution IS
achievable on target C1, in both policy classes, with qualifiers. Phase 5 de-risks **exactly the
two sentences the paper now leans on hardest**, and one exploratory cross-check. Nothing else: no
C5 re-tests, no P14 re-litigation, no estimator sweeps.

Preregistration: `phase5/preregistration_phase5.json`, SHA-256
`b4883cb1a7048aee1a47410291573c3e27401c7e08be7852f67c58f42f2922b8`, locked **before any Phase-5
training, before any Phase-5 attribution, and before any Phase-5 analysis code ran on real data** —
including the zero-retrain P16, whose GradDot breadth test reads existing artifacts and where the
temptation to peek was highest.

Every number below is read from an artifact file and machine-verified
(`results/report_verification.json`, **78/78 checks pass**). **48/48 retrains succeeded, 0 failures,
8.46 GPU-h, 1,440 episodes. No stage was cut. No budget alert was tripped. No incidents.**

---

## 0. Headline: the two de-risks and the cross-check

| stage | preregistered result | one-line |
|---|---|---|
| **P15** — diffusion C1 replication at S=10, stricter gate | **PASS on C1, on a SOUND instrument** | P13's marginal-instrument caveat is **resolved**: all three stricter gate checks pass with margin (the tightest, S=8→10, by 0.010 against a 0.10 tolerance), and the champion ρ is **unchanged** (0.479 vs P13's 0.477). |
| **P16** — GradDot breadth on the 7 non-focal BC targets | **k = 0 of 7 pass → C1-SPECIFIC** | The paper's strongest positive sentence must be **scoped to C1**. GradDot's demo-grain faithfulness does not generalize across BC targets on this corpus. |
| **P17** — exploratory diffusion GradDot | EXPLORATORY (never a verdict) | One data point **consistent** with the policy-class-by-estimator hypothesis: for diffusion, TracIn (0.479) edges GradDot (0.414) on C1 — the **reverse** of BC. Still a hypothesis, not a finding. |

The net effect on the paper: **one leaned-on sentence gets stronger and loses its qualifier
(diffusion C1); the other gets narrower and gains one (BC GradDot is C1-specific).** Both moves are
what a de-risking phase is supposed to produce.

---

## 1. P15 — DIFFUSION C1 REPLICATION AT S=10 — **PASS on a sound instrument**

**48/48 diffusion retrains, 0 failures, 8.35 GPU-h, 1,440 episodes.** Frozen config
`p10_config_frozen.json` untouched. Seeds 601–608 (Phases 3–4) + **609, 610** → **S = 10**.

### 1.1 The evaluation economy (preregistered, disclosed) — 9× cheaper, and it touched nothing the verdict uses

The verdict outcome is held-out executed-action L2, computed by a **forward pass**
(`evaluate_diffusion.heldout_losses`) — no closed-loop simulation. So each new model produced
held-out L2 for **all 9 targets** with no rollouts; closed-loop **success** rollouts were run for
**C1's 3 probe tasks only** (30 episodes/model, 1,440 total) as a descriptive record. This cut
simulation ~9× versus the full 27-task battery. The verdict path (L2) is untouched; the economy is
disclosed here because it is disclosed in the preregistration.

### 1.2 The stricter instrument gate — computed BEFORE any LDS, and it PASSES with margin

P13's C1 gate passed by **0.003** (0.147 vs 0.15), and a descriptive variant would have **failed**.
Phase 5's gate is stricter in two ways — a deeper ceiling (**126** distinct 5v5 median split-halves,
vs P13's 35 4v4) and a **third** check with a **tighter 0.10** tolerance. C1 is USABLE iff **all
three** hold:

| check | definition | C1 value | tolerance | pass |
|---|---|---|---|---|
| **(i)** ceiling | median SB-corrected ceiling (126 5v5 splits, SB×2) | **0.830** | ≥ 0.40 | **✓** |
| **(ii)** S=4→10 | \|SB(r1[601–604], 10) − ceiling\| | **0.070** | ≤ 0.15 | **✓** |
| **(iii)** S=8→10 | \|SB(r1[601–608], 10) − ceiling\| | **0.010** | ≤ 0.10 | **✓** |

The descriptive all-seed variant (the check that would have failed P13) now reads **0.016** — it
*passes* too. **The instrument that sat at the edge of usability in P13 is sound at S=10, on every
definition.** (`results/p15_verdict.json`, gate computed before any attribution.)

### 1.3 The preregistered verdict

Champion **identical to P13, bit-for-bit**: TracIn, E=5 diffusion ensemble (dpens_s621–625),
frozen (t, ε) bank (sha `61aadccfef2f…`, verified), per-member unit-L2 normalization. Median
aggregator, n = 24, exact rational criterion.

| target | ceiling (median, SB10) | bar | ρ | ratio | p | verdict |
|---|---|---|---|---|---|---|
| **C1** | 0.830 | 0.415 | **+0.479** | **0.58** | **0.0089** | **PASS** |

**P15 VERDICT: PASS (C1), on a sound instrument.** Preregistered interpretation, stated verbatim:

> *"the diffusion C1 result carries full weight: demo-grain faithfulness is policy-class-dependent
> and the BC null does not generalize — now replicated at two seed depths on a sound instrument."*

**The replication is genuine, and the honest nuance is stated.** The champion ρ is **0.479, all but
identical to P13's 0.477** at S=8 — the signal did not move. The **ratio fell (0.58 vs P13's 0.84)
because the deeper S=10 ceiling ROSE (0.830 vs 0.569)**, i.e. the oracle bar got higher, not the
attributor weaker. The verdict still clears the half-ceiling bar (0.58 > 0.50) and significance
(p = 0.0089 < 0.025). A reader should read this as *"the same effect, now measured against a sounder,
higher ceiling, and it still passes"* — which is strictly stronger evidence than P13's marginal-gate
pass.

### 1.4 The seed-MEAN is still broken at S=10 (descriptive series)

Extending the S=6/8/10 series the study has tracked, the **mean**-aggregated ceilings remain broken
while the **median** restores the instrument:

| C1 ceiling | S=6 (Phase 3, measured) | S=8 (Phase 4, SB) | S=10 (Phase 5, SB) |
|---|---|---|---|
| **MEAN** | 0.079 | **−0.113** | **+0.201** |
| **MEDIAN** | 0.568 | 0.569 | **0.830** |

C9's mean ceiling is **still negative at S=10 (−0.022)**. **Deeper seeds do not repair the mean; only
the median does.** This confirms, now at three seed depths, that for a diffusion policy the seed-mean
of an executed-action outcome is structurally broken — the methodological finding holds.

`results/p15_verdict.json`, `results/p15_outcomes_S10.parquet`, `results/p15_lds_table.csv`.

---

## 2. P16 — GRADDOT BREADTH ON THE 7 NON-FOCAL BC TARGETS — **k = 0, the result is C1-specific**

**Zero retrains.** GradDot (P11's secondary, frozen bit-for-bit) computed on the existing S=10 BC
ground truth (`phase4/results/p12_outcomes_S10.parquet`, held-out L2, seed mean; ceilings from
`p12_ceilings.json`). Targets C2, C3, C4, C6, C7, C8, C9 (C1/C5 excluded as spent).

### 2.1 Self-validation (before any verdict)

- **Probe-leak guard: 20/20 members clean** (90 held-out probe ids, 0 intersections).
- **C1 reproduction: GradDot on the cache path used here gives ρ = +0.5130434782608695, matching the
  P11 archived value EXACTLY** (exact-rational equality, not approximate). The estimator is the same
  one that passed on C1, not a lookalike.

### 2.2 The preregistered breadth test

Bonferroni-7, one-sided p < 0.05/7 = 0.00714. **At n = 24 the critical ρ is ≈ 0.493 — commensurate
with the half-ceiling bar**, so a target can miss on either component. Both are reported; a PASS
needs both.

| target | ceiling | bar (½·ceil) | ρ | ratio | p | ≥ bar? | p < 0.00714? | **PASS** |
|---|---|---|---|---|---|---|---|---|
| C2 | 0.927 | 0.464 | +0.368 | 0.40 | 0.0385 | no | no | fail |
| C3 | 0.755 | 0.377 | −0.075 | −0.10 | 0.636 | no | no | fail |
| C4 | 0.884 | 0.442 | +0.430 | 0.49 | 0.0179 | no | no | fail |
| C6 | 0.961 | 0.480 | +0.157 | 0.16 | 0.233 | no | no | fail |
| C7 | 0.949 | 0.475 | +0.331 | 0.35 | 0.0569 | no | no | fail |
| C8 | 0.946 | 0.473 | −0.178 | −0.19 | 0.798 | no | no | fail |
| C9 | 0.895 | 0.447 | +0.354 | 0.40 | 0.0449 | no | no | fail |

**PRIMARY: k = 0 of 7 pass.** The secondary family (GradDot_dmean, Bonferroni-14) also passes **0 of
7**. The nearest misses (C4 ratio 0.49, C2/C9 ratio 0.40; C4 p = 0.018) fall short on **both** the
bar and significance — this is not a marginal near-miss dressed as a fail.

**P16 interpretation, stated as preregistered for k = 0:**

> *"the achievability result is C1-specific on this corpus; the paper must scope it that way."*

`results/p16_verdict.json`, `results/p16_lds_table.csv`.

---

## 3. P17 — EXPLORATORY DIFFUSION GRADDOT — **labelled EXPLORATORY, never a verdict**

**Zero retrains; one gradient pass (412 s) over the 5 diffusion members** (probe-leak guard clean;
frozen (t, ε) bank verified). GradDot for the diffusion policy = the raw gradient dot product at each
member's final checkpoint on the denoise functional (the champion's test side), E=5, unit-L2
normalized, against P15's S=10 median ground truth.

**EXPLORATORY — no criterion, no verdict.**

| target | ceiling | GradDot ρ | ratio | p | GradDot_dmean ρ | **TracIn (S8)** | **TracIn (S10)** |
|---|---|---|---|---|---|---|---|
| C1 | 0.830 | +0.414 | 0.50 | 0.022 | +0.420 | +0.477 | **+0.479** |
| C5 | 0.844 | +0.186 | 0.22 | 0.192 | +0.186 | +0.072 | +0.117 |

**One data point, consistent with Phase 4's open hypothesis.** For the diffusion policy on C1,
**TracIn (0.479) edges out GradDot (0.414)** — the *reverse* of the BC-Transformer, where GradDot
beat TracIn (P11: 0.513 vs 0.409). That is coherent with "the winning estimator differs by policy
class (BC → GradDot, diffusion → TracIn)." **It remains a hypothesis for a future preregistration,
not a finding** — it is one target, one exploratory pass, no criterion.

`results/p17_exploratory.json`, `results/p17_diffusion_gram_cache.npz`.

---

## 4. Instrument gates and validation — summary

| gate / check | stage | criterion | result |
|---|---|---|---|
| Stricter ceiling-usability (diffusion, median, C1) | P15 | ceiling ≥ 0.40 AND S4→10 ≤ 0.15 AND S8→10 ≤ 0.10 | **USABLE** — 0.830 / 0.070 / **0.010**; all pass with margin |
| Probe-leak guard (BC) | P16 | no attributed model trained on a probe demo | **PASS** — 20 members, 90 probe ids, 0 intersections |
| GradDot self-validation (C1) | P16 | reproduce P11's archived C1 GradDot | **PASS** — ρ = +0.5130434782608695, EXACT |
| Probe-leak guard (diffusion) | P17 | " | **PASS** — 5 members, 0 intersections |
| Frozen (t, ε) bank integrity | P15, P17 | sha == `61aadccfef2f…` | **PASS** |
| SYNTHETIC unit tests | all | new Phase-5 statistics on known answers | **15/15 PASS** |

The SYNTHETIC suite (`results/SYNTHETIC_p5_unit_tests.json`, **labelled SYNTHETIC, never mixed**)
validated: exact-rational criterion arithmetic (correctly **fails** a value one ULP below the bar);
balanced split enumeration (**126** at S=10, 35 at S=8, 3 at S=4); Spearman–Brown recovery within
0.05; **the median beating the mean by +0.5 on planted heavy-tail contamination** (the premise the
diffusion instrument rests on); the **stricter three-condition gate** (a target passing (i)+(ii) but
failing (iii) is ruled UNUSABLE); per-member normalization scale-invariance; and the Bonferroni-7/14
thresholds with the **commensurate critical ρ ≈ 0.493 at n = 24**.

---

## 5. Budget: actual vs ledger

| stage | retrains (ledger) | retrains (actual) | ok / fail | GPU-h (ledger) | GPU-h (actual) | episodes |
|---|---|---|---|---|---|---|
| P15 diffusion S=10 + verdict | 48 | **48** | **48 / 0** | 15.5 | **8.35** | 1,440 |
| P16 GradDot breadth | 0 | **0** | — | 0.2 | **0.00** | 0 |
| P17 exploratory cross-check | 0 | **0** | — | 0.2 | **0.11** | 0 |
| **TOTAL** | **48** | **48** | **48 / 0** | **16** (alert **25**) | **8.46** | **1,440** |

**Every retrain succeeded: 48/48, zero failures. Well under the 16 GPU-h nominal and the 25 GPU-h
alert.** The per-stage 1.5× guard never fired. **Cuts: NONE** (the cut order P17 → P15 rollouts was
never reached). P15 came in far under its 15.5 GPU-h line because the preregistered eval economy cut
simulation ~9×. `results/budget_actual.json`.

---

## 6. Deviations and incidents

### Deviations from the brief — **all disclosed BEFORE the lock, none to any criterion**

1. **Two SB-consistency checks had to be made concrete** and were, in the locked file: the stricter
   gate's "S=4→10" base is the single-seed reliability over the **original** seeds 601–604, and its
   "S=8→10" base is the single-seed reliability over 601–608; both are aggregator-free (the correct
   SB base for the median arm), each extrapolated to depth 10. Tolerances 0.15 and 0.10 as the brief
   specified.
2. **P16's secondary "stricter within-stage correction" was set to Bonferroni-14** (2 families ×
   7 targets), declared in the preregistration.
3. **The evaluation economy** (C1-only rollouts + all-9 held-out L2) was preregistered and disclosed;
   it touches nothing the verdict uses.

**Zero post-hoc criterion changes. No gate was moved after seeing a number.**

### Incidents

**None.** No failed retrain, no harness failure, no corrupted artifact, no marker-gate violation.

### Material observations that are not deviations, and must not be lost

* **C8 was ruled INSTRUMENT-UNUSABLE at S=10** (gate (ii) S4→10 = 0.153 > 0.15) — but C8 is **not a
  verdict target**; it appears only in the descriptive all-9 gate table. It does not affect the C1
  verdict and is reported for completeness.
* **P15's ceiling ROSE from 0.569 (S=8) to 0.830 (S=10).** This lowers the *ratio* while the *ρ* is
  unchanged; §1.3 states this explicitly so the ratio drop is not misread as a weakening.

---

## 7. After Phase 5, the paper may claim X — and may not claim Y

**This is the version the paper will be written from. Each licensed sentence is stated as it should
print, with its numbers and qualifiers inline.**

### May claim

1. **On the diffusion policy, demo-grain attribution is faithful on C1 — replicated at two seed
   depths on a sound instrument.** TracIn on the frozen (t, ε) noise bank reaches **ρ = +0.479 = 0.58
   of a 0.830 median-SB ceiling, p = 0.0089** at S=10 (P15, preregistered), with **every** cell of a
   stricter three-part usability gate passing with margin (ceiling 0.830; S4→10 consistency 0.070;
   S8→10 consistency 0.010 against a 0.10 tolerance). The champion ρ is **unchanged from P13's 0.477
   at S=8** — the effect replicated; the deeper ceiling merely raised the bar. **The P13 "marginal
   instrument" caveat is resolved and should no longer be attached to this result.**

2. **On the BC-Transformer, GradDot's demo-grain faithfulness is C1-specific on this corpus.** Of the
   7 non-focal BC targets, **0 pass** a preregistered half-ceiling + Bonferroni-7 test (P16); the
   best, C4, reaches ratio 0.49 (p = 0.018) but clears neither the bar nor significance. **The paper
   must scope the BC positive to C1** (where GradDot reached 0.54 of a 0.951 ceiling, p = 0.0052,
   P11): "attribution done right is demo-grain faithful **on C1**", not "on BC targets in general".

3. **For a diffusion policy the seed-MEAN of an executed-action outcome is structurally broken, and
   deeper seeds do not repair it — now shown at three seed depths.** The mean-aggregated C1 ceiling
   reads 0.079 (S=6), −0.113 (S=8), +0.201 (S=10) — still far below the median (0.83) and, for C9,
   **still negative at S=10 (−0.022)**. Only the median restores the instrument.

4. **The self-validation holds exactly:** GradDot recomputed on the archived Gram caches reproduces
   P11's C1 value to the last bit (ρ = +0.5130434782608695), so the estimator carried into P16 is the
   same one that passed on C1.

### May NOT claim

* **We may NOT say demo-grain attribution is faithful across targets in either policy class.** In BC
  it passes on **C1 only** (k = 0 of the other 7, P16); in diffusion it passes on **C1 only** (C5
  fails on a sound instrument, P13, settled). The achievability result is a **C1 result**, in both
  classes — one target, robustly, not a method that generalizes across this corpus's targets.

* **We may NOT present the diffusion/BC winning-estimator difference as a finding.** P17 is
  exploratory: one target, one gradient pass, no criterion. It shows TracIn (0.479) > GradDot (0.414)
  for diffusion on C1, the reverse of BC — **consistent with** the policy-class-by-estimator
  hypothesis, but it remains a hypothesis for a future preregistration.

* **We may NOT weaken the diffusion C1 result by pointing at its ratio.** The ratio fell to 0.58 only
  because the S=10 ceiling rose to 0.830; the ρ (0.479) and the pass are intact, on a sounder
  instrument than P13 had.

* **We may NOT generalize past this regime:** one benchmark, state-based observations, 9 clusters,
  ≤500 demos/task, two policy classes.

### The one-paragraph statement

Phase 5 asked two questions of the two results the paper leans on. The diffusion C1 pass — which in
Phase 4 rested on an instrument sitting 0.003 inside its usability gate — **replicated at S=10 on an
instrument that now passes a strictly harder gate with room to spare, at an unchanged effect size**;
its qualifier is gone. The BC GradDot pass — the paper's strongest positive sentence — **did not
generalize: zero of seven other targets clear the same preregistered bar**, so the sentence must name
C1 rather than "attribution done right." And an exploratory pass added one data point, no more,
suggesting the winning estimator flips between the two policy classes. So the study's final position
is sharper than Phase 4 left it: **demo-grain data attribution is achievable on this corpus on C1 —
in both policy classes, against a ground truth measured far more carefully than the field measures it
— and Phase 5 removed the one instrument caveat that hung over the diffusion half while showing the
BC half is C1-specific, not general.**

---

## 8. Artifact index

| artifact | contents |
|---|---|
| `preregistration_phase5.json` (+ `.sha256`) | locked before all Phase-5 training, attribution, and analysis (`b4883cb1…`) |
| `results/p15_verdict.json` / `p15_lds_table.csv` | **PASS on C1** (0.479, ratio 0.58, p=0.0089); stricter 3-part gate all-pass; seed-mean brokenness series |
| `results/p15_outcomes_S10.parquet` | the 24×10 diffusion ground truth |
| `results/p16_verdict.json` / `p16_lds_table.csv` | **k = 0 of 7** non-focal BC targets pass GradDot; C1 self-validation exact |
| `results/p17_exploratory.json` / `p17_diffusion_gram_cache.npz` | **EXPLORATORY** diffusion GradDot vs TracIn, C1/C5 |
| `results/SYNTHETIC_p5_unit_tests.json` | **SYNTHETIC** — 15/15 on the Phase-5 statistics |
| `results/budget_actual.json` | 48 retrains, 8.46 GPU-h, 1,440 episodes, **0 failures** |
| `results/report_verification.json` | **78/78** — every number in this report machine-verified against artifacts |
