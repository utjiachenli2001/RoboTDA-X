# RoboTDA-X — PHASE 4 REPORT (SETTLEMENT)

**What Phase 4 was for.** Phase 3 closed with three open verdicts, each flagged in its own report
as requiring a preregistered follow-up. Phase 4 gave each **one confirmatory shot** — no sweeps,
no tuning, no second bites.

Preregistration: `phase4/preregistration_phase4.json`, SHA-256
`fca37f54804c0173f5a9029e99b220de3db2a97ca2af7b3cf3c1c1a5372ecb80`, locked **before any Phase-4
training, before any Phase-4 attribution, and before any Phase-4 analysis code was run on real
data** — including the zero-retrain P11, whose verdict reads existing data and where the
temptation to peek was highest.

Every number below is read from an artifact file. **208/208 retrains succeeded, 0 failures,
32.50 GPU-h, 42,720 episodes. No stage was cut. No budget alert was tripped.**

---

## 0. Headline: the three verdicts

| stage | preregistered verdict | one-line |
|---|---|---|
| **P11** — champion test, BC-Transformer | **CHAMPION FAILS** — but the preregistered **SECONDARY PASSES on C1** | We froze the wrong champion, and the stage's own secondary says so. The estimator that clears the bar is the one with **no preconditioner at all**. |
| **P13** — diffusion settlement | **PASS on C1** (ratio 0.84, p = 0.0092) | Phase-3's post-hoc median hypothesis **replicated and strengthened** under preregistration — but C1's instrument gate passed by 0.003, and that qualifies it. |
| **P14** — heterogeneity on fresh ground truth | **FAIL on all 4 cells, both families** | The P7 C2/C9 heterogeneity **did not replicate out-of-sample**. The point estimates collapsed, not merely the p-values. |

---

## 1. P12 — the instrument (96 retrains, 11.95 GPU-h) — **GATE PASSED**

The 24 Stage-G masks now carry **10 training seeds** (401–406 from Phases 1–2, **407–410** new).
96/96 retrains, 0 failures, 25,920 episodes.

**The preregistered instrument gate.** From the S=6 data *only*, the mean pairwise single-seed
reliability `r1` predicts a 10-seed ceiling by Spearman–Brown. That prediction is compared with
the ceiling actually measured on the S=10 data (all **126** distinct 5v5 split-halves, SB-corrected
×2). Tolerance 0.10 on both focal targets. **This is the check that failed by 0.61 for the
diffusion policy in Phase 3** — it is the reason a ratio-to-ceiling test can be trusted at all.

| target | r1 (from S=6 only) | predicted 10-seed (SB) | measured 5v5 | **measured 10-seed (SB)** | \|diff\| | bar = ½·ceiling | gate |
|---|---|---|---|---|---|---|---|
| **C1** | 0.750 | 0.968 | 0.906 | **0.951** | **0.017** | 0.475 | **PASS** |
| **C5** | 0.648 | 0.948 | 0.883 | **0.938** | **0.010** | 0.469 | **PASS** |

**All 9 targets pass** (worst miss 0.043, C9). The BC ground truth at S=10 is a sound instrument.

Note the direction: the deeper ceiling **raises** P11's bar (0.475 vs Phase-2's 0.466). Phase 4
raised the ceiling; it did not lower the bar.

`results/p12_ceilings.json`, `results/p12_outcomes_S10.parquet`.

---

## 2. P11 — THE CHAMPION CONFIRMATORY TEST — **the champion FAILS; the preregistered secondary PASSES**

**Zero retrains.** One gradient pass (58 s) to extend the Gram cache to members 211–220.

Phases 2–3 never tested the configuration their own evidence said was best. The **champion** was
frozen in the preregistration from prior-phase evidence only:

> **TracIn, E = 20 (members 201–220), each member's 135-demo score vector normalized to unit L2
> before averaging.**
> Justification on record: TracIn is the only reproducible attributor (split-half 0.831 at E=20 vs
> IF 0.254, P9); per-member scale normalization is what the λ→∞ analysis showed matters (0.504 vs
> 0.397 on C1, P6.5). Neither fact was derived from the statistic this test computes.

**SECONDARY** (also preregistered, Bonferroni-4 within stage): the exact **λ→∞ limit estimator** —
the raw gradient dot product `K`, per-member unit-L2 normalized, same E=20.

### 2.1 The preregistered verdict, reported verbatim

Ground truth: 24 Stage-G masks, **S=10**, held-out L2, seed mean. n = 24.

| target | ceiling | bar | **CHAMPION** ρ | ratio | p (α=0.025) | | **SECONDARY** ρ | ratio | p (α=0.0125) | |
|---|---|---|---|---|---|---|---|---|---|---|
| **C1** | 0.951 | 0.475 | **+0.409** | **0.43** | **0.0237** | **fail** | **+0.513** | **0.54** | **0.0052** | **PASS** |
| **C5** | 0.938 | 0.469 | +0.304 | 0.32 | 0.0741 | fail | +0.329 | 0.35 | 0.0584 | fail |

**CHAMPION VERDICT: FAIL.** Preregistered interpretation, stated verbatim as required:

> *"the null now covers the best configuration the evidence could assemble — champion-tested, not
> just default-tested."*

**And that sentence is partly falsified by this stage's own preregistered secondary — which is
exactly why the secondary was preregistered.** It must be said plainly:

* On **C1**, the **raw gradient dot product** reaches **ρ = +0.513 = 0.54 of a 0.951 ceiling,
  p = 0.0052**, clearing both the half-ceiling bar and a **stricter** Bonferroni-4 threshold than
  the champion had to clear. This is a **preregistered confirmatory PASS**, not a post-hoc rescue.
* So **the evidence *could* assemble a configuration that is demo-grain faithful on C1. We simply
  did not name it champion.** The honest headline is not "the null covers the best configuration"
  — it is **"the null covers the champion we froze, and the configuration that beats it is the one
  with no preconditioner at all."**

### 2.2 Why we froze the wrong champion — a design lesson, stated against ourselves

The champion composed two prior-phase facts. One transferred; one did not.

* **TracIn's stability was real** — but stability is not faithfulness. TracIn is the *most
  reproducible* attributor; it is not the *most accurate* one.
* **The normalization justification did not transfer to TracIn.** P6.5's 0.504-vs-0.397 contrast
  was measured on the *λ→∞ ensemble-mean score*, where per-member gradient scales differ wildly.
  On TracIn, normalization is nearly inert: **C1 normalized +0.409 vs unnormalized +0.419**
  (descriptive). We imported a correction into an estimator that did not need it.
* **P6.5's actual substantive discovery was that the exact preconditioner HURTS** — and the
  estimator that *embodies* that discovery is GradDot, not TracIn. Phase 4's own secondary is the
  one that carries Phase 3's real finding forward, and it is the one that passes.

**Descriptive (never a verdict):** the literal P6.5 λ→∞ form (`K_m / mean(diag(G_m))`, the exact
limit of the ensemble-*mean* score) scores **C1 ρ = +0.593, ratio 0.62, p = 0.0011** — stronger
still. It is reported here as descriptive only, because the preregistered secondary is the
unit-L2 form.

Per-attributor detail: `results/p11_verdict.json`, `results/p11_lds_table.csv`.

### 2.3 The attribution input, and its self-validation

The champion needed **zero** new GPU (members 211–220 already had per-member `plain` scores from
P9). Only the secondary needed a new gradient pass, because the raw `K` was never stored for those
members and cannot be inverted back out of the archived IF/TRAK solves.

The probe-leak guard passed on all 10 members before any gradient was taken (90 held-out probe
ids, 0 intersections). **Self-validation: TRAK and IF recomputed from the new Gram cache at the
frozen ridge reproduce the archived P9 per-member scores — 180/180 cells, minimum Spearman
1.000000, worst max relative difference 2.8e-10** (`results/p11_gram_selfvalidation.json`). The
cache is the same estimator, not a lookalike.

---

## 3. P13 — THE DIFFUSION SETTLEMENT — **PASS on C1**, with a marginal instrument

**48/48 diffusion retrains, 0 failures, 15.35 GPU-h, 12,960 episodes.** Frozen config
`p10_config_frozen.json` untouched. Seeds 601–606 (Phase 3) + **607, 608** → **S = 8**.

Phase 3 §6.3 demanded exactly three things of a follow-up. All three were preregistered:
the **median aggregator**, **S ≥ 6**, and an **up-front ceiling-usability gate**.

### 3.1 The up-front gate — computed BEFORE any LDS

Per target: proceed only if (a) the median SB-corrected ceiling ≥ 0.40 **and** (b) the S=4→8
SB-consistency check holds within 0.15. Otherwise the target is **INSTRUMENT-UNUSABLE — not FAIL**.

| target | median ceiling (SB) | mean ceiling (SB) | SB-consistency \|diff\| | usable |
|---|---|---|---|---|
| **C1** | **0.569** | **−0.113** | **0.147** | **USABLE** (by 0.003) |
| **C5** | **0.664** | +0.021 | **0.014** | **USABLE** |
| C2 | 0.576 | −0.049 | **0.221** | **UNUSABLE** |
| C9 | 0.669 | −0.023 | 0.075 | usable |

**Two things this table settles.**

1. **The Phase-3 defect is confirmed at deeper S, not outgrown.** The *mean*-aggregated ceilings at
   S=8 are still broken — **C1 −0.113, C2 −0.049, C9 −0.023, all negative**, which is impossible
   for a true reliability. Two more seeds did not repair the mean. **Only the median does.** The
   methodological warning generalizes: for a diffusion policy, the seed-mean of an executed-action
   outcome is not merely noisy, it is structurally broken.
2. **The gate did real work.** **C2 was ruled INSTRUMENT-UNUSABLE** despite showing ρ = +0.310,
   ratio 0.54 — which would have read as a near-pass had we not gated it. That is precisely the
   failure mode the gate exists to catch, and it caught one.

### 3.2 The preregistered verdict

Champion: TracIn, E=5 diffusion ensemble, frozen paired (t, ε) noise bank
(sha `61aadccfef2f…`), per-member unit-L2 normalization. Median aggregator, n = 24.

| target | ceiling (median, SB) | bar | ρ | ratio | p | 95% CI | verdict |
|---|---|---|---|---|---|---|---|
| **C1** | 0.569 | 0.285 | **+0.477** | **0.84** | **0.0092** | [0.109, 0.711] | **PASS** |
| **C5** | 0.664 | 0.332 | +0.072 | 0.11 | 0.369 | [−0.341, 0.482] | **FAIL** |

**P13 VERDICT: PASS on C1.** Preregistered interpretation, stated verbatim:

> *"demo-grain faithfulness is policy-class-dependent, and the BC null does not generalize."*

**This is a genuine confirmatory replication, not a re-run of the same number.** Phase-3's *post-hoc*
median diagnostic gave C1 ratio 0.74 at p = 0.0205 on S=6. The *preregistered* test at S=8 gives
**ratio 0.84 at p = 0.0092**. The hypothesis strengthened when it was tested properly.

### 3.3 The C1 gate is MARGINAL — stated plainly, because it qualifies the verdict

C1's ceiling-usability gate passed **by 0.003** (0.147 against a 0.15 tolerance).

Worse — and this is reported because it cuts against the result: the **descriptive** variant of the
same check, SB-extrapolated from all 8 seeds rather than from the original 4, gives **0.215 for C1
and would have failed the gate.** C5's gate, by contrast, passes comfortably on **both** definitions
(0.014 and 0.093).

**The preregistered definition is binding and it has not been touched.** But the consequence must be
stated: **C1's PASS rests on an instrument sitting at the very edge of usability, while C5's FAIL
rests on a sound one.** A reader is entitled to weight those two differently, and should.

### 3.4 The C5 diagnosis — EXPLORATORY, not a verdict

Why did C5 carry no signal? **Not a success floor.**

| | between-mask sd (**signal**) | within-mask seed sd (**noise**) | **S/N** | mean success |
|---|---|---|---|---|
| **C1** | **0.136** | 0.537 | **0.253** | 0.229 |
| **C5** | **0.092** | 0.580 | **0.158** | **0.268** |

C5's mean success rate is *higher* than C1's (0.268 vs 0.229) — it is not failing because the policy
cannot do the task. It is failing for **lack of data signal**: its between-mask outcome spread is
**33% smaller** than C1's against slightly *larger* seed noise, giving an S/N of 0.16 vs 0.25. There
is simply less for an attributor to predict. This is instrument characterization, and it is labelled
exploratory everywhere it appears.

`results/p13_verdict.json`, `results/p13_outcomes_S8.parquet`, `results/p13_lds_table.csv`.

---

## 4. P14 — CONFIRMATORY HETEROGENEITY ON FRESH GROUND TRUTH — **FAIL on all four cells**

**64/64 retrains, 0 failures, 5.18 GPU-h, 3,840 episodes.**

**The fresh corpus.** 16 new masks (68/135 fixed-alpha, within-cluster 7–8 stratification, **new
seed 1104**), × 4 new seeds (411–414). Probes restricted to C2's and C9's clusters.
**Mandatory coincidence check: 0 of 16 masks coincide with any of the 24 Stage-G masks** (0 seed
increments needed). **No analysis had ever touched this ground truth.**

Fresh 4-seed ceilings are healthy: **C2 0.837, C9 0.851** — the instrument is sound, so a FAIL here
is a FAIL of the *attributor*, not of the measurement.

| cell | ceiling | bar | **PRIMARY** (champion, TracIn) ρ | ratio | p | | **SECONDARY** (P7 qualifier) ρ | ratio | p |
|---|---|---|---|---|---|---|---|---|---|
| C2 @ g=1 | 0.837 | 0.418 | +0.241 | 0.29 | 0.184 | fail | TracIn +0.241 | 0.29 | 0.184 |
| C2 @ g=3 | 0.837 | 0.418 | −0.191 | −0.23 | 0.761 | fail | **IF** +0.224 | 0.27 | 0.203 |
| C9 @ g=1 | 0.851 | 0.426 | +0.006 | 0.01 | 0.491 | fail | TracIn +0.006 | 0.01 | 0.491 |
| C9 @ g=5 | 0.851 | 0.426 | −0.162 | −0.19 | 0.725 | fail | TracIn −0.162 | −0.19 | 0.725 |

**P14 VERDICT: FAIL on all four cells, in both preregistered families.** Interpretation, verbatim:

> *"the P7 heterogeneity is flagged as likely exploration noise."*

### 4.1 The FAIL is stronger than the power limit alone would license

The preregistration stated the power limit **up front** and it is restated here: at n = 16 the
critical ρ is **0.557** (primary) against a bar of ~0.42, so **significance binds, not the ratio.
The test is CONSERVATIVE: a FAIL is WEAK evidence and a PASS would be STRONG.**

But the honest reading is not merely "underpowered". **The point estimates collapsed**, which is
not what a power failure looks like:

| cell | P7 ratio (exploration) | **P14 ratio (fresh)** |
|---|---|---|
| C2 @ g=1 (TracIn) | +0.51 | **+0.29** |
| C2 @ g=3 (**IF**) | +0.65 | **+0.27** |
| C9 @ g=1 (TracIn) | +0.52 | **+0.01** |
| C9 @ g=5 (TracIn) | +0.66 | **−0.19** |

An effect that is real but under-powered keeps its point estimate and loses its p-value. These lost
both — C9@g=5 went from **+0.66 to −0.19**. That is the signature of a **selection artifact**: P7
reported the max over 3 attributors × 4 grains × 9 targets on ground truth already used for
exploration, and the winners did not survive contact with fresh masks.

### 4.2 A disclosed correction to the brief, made before the lock

Phase-4's brief described P7's qualifying cells as if a single champion had produced them. Reading
`phase3/results/p7_grain_ladder.csv` directly showed otherwise: **C2@g=3 was qualified by IF
(ratio 0.652), not TracIn (0.277)**. Freezing the primary estimator to the champion therefore made
that one cell adversarial. **This was disclosed in the preregistration** (§P14
`DISCLOSED_TENSION_IN_THE_PRIMARY_DESIGN`), and rather than swapping estimators per cell — the exact
multiple-comparison move P6.5 exposed — a **second preregistered family** was declared, using each
cell's actual P7-qualifying attributor at a wider Bonferroni-8 correction. **Both families fail.**
The heterogeneity is not being killed by an estimator mismatch.

`results/p14_verdict.json`, `results/p14_mask_manifest.json`, `results/p14_lds_table.csv`.

---

## 5. Instrument gates — summary

| gate | stage | criterion | result |
|---|---|---|---|
| SB-consistency (BC, S=6 → S=10) | P12 | \|pred − meas\| ≤ 0.10 on C1/C5 | **PASS** — C1 0.017, C5 0.010; all 9 targets pass |
| Ceiling usability (diffusion, median) | P13 | ceiling ≥ 0.40 **and** SB-consistency ≤ 0.15 | **C1 USABLE by 0.003**; C5 USABLE; **C2 UNUSABLE** (0.221) |
| Fresh-mask coincidence | P14 | 0 of 16 masks equal any Stage-G mask | **PASS** — 0 coincidences, 0 seed increments |
| Probe-leak guard | P11 | no attributed model trained on a probe demo | **PASS** — 10 members, 90 probe ids, 0 intersections |
| Gram-cache self-validation | P11 | new cache reproduces archived P9 scores | **PASS** — 180/180, ρ = 1.000000, max rel diff 2.8e-10 |
| SYNTHETIC unit tests | all | new Phase-4 statistics on known answers | **32/32 PASS** |

The SYNTHETIC suite (`results/SYNTHETIC_p4_unit_tests.json`, **labelled SYNTHETIC, never mixed**)
validated: exact rational criterion arithmetic (correctly *fails* a value one ULP below the bar,
where an epsilon comparison would wrongly pass); split enumeration (126 at S=10, 35 at S=8, 10 at
S=6, 3 at S=4); SB recovery on data with known reliability; **the median beating the mean by +0.48
on planted heavy-tail contamination** — the premise the entire P13 remedy rests on; per-member
normalization invariance; and the g=1 endpoint identity of the coarse predictor.

---

## 6. Budget: actual vs ledger

| stage | retrains (ledger) | retrains (actual) | ok / fail | GPU-h (ledger) | GPU-h (actual) | episodes |
|---|---|---|---|---|---|---|
| P12 BC S=10 | 96 | **96** | **96 / 0** | 13 | **11.95** | 25,920 |
| P11 champion test (attribution) | 0 | **0** | — | 1 | **0.02** | 0 |
| P13 diffusion S=8 | 48 | **48** | **48 / 0** | 17 | **15.35** | 12,960 |
| P14 fresh heterogeneity | 64 | **64** | **64 / 0** | 9 | **5.18** | 3,840 |
| **TOTAL** | **208** | **208** | **208 / 0** | **40** (alert **60**) | **32.50** | **42,720** |

**Every retrain succeeded: 208/208, zero failures.** **Under the nominal 40 GPU-h and well under the
60 GPU-h alert.** The per-stage 1.5× guard never fired.

**Cuts: NONE.** The preregistered cut order (P14 → P12 seeds 4→2 → P13 C5 arm) was never reached.

The P11 attribution line came in at **0.02 GPU-h against a 1 GPU-h estimate** — the Gram pass over
10 members took 58 s, because only the final checkpoint is needed (no TracIn checkpoint sweep).

---

## 7. Deviations and incidents

### Deviations from the brief — **all disclosed BEFORE the lock, none to any criterion**

1. **P14 gained a second preregistered family** (§4.2). The brief named only the champion. Because
   C2@g=3's P7 qualification came from IF and not from the champion, a champion-only test would
   have been adversarial on that cell for a reason unrelated to the hypothesis. A second family —
   each cell's actual P7-qualifying attributor, frozen from the Phase-3 artifact, at Bonferroni-8 —
   was declared in the preregistration. It changed nothing: both families fail on all four cells.
2. **Two preregistered definitions had to be made concrete**, and were, in the locked file: P13's
   "S=4→8 SB-consistency check" (the SB base is the aggregator-free single-seed reliability over
   the *original* seeds 601–604), and P11's "same normalization" for the secondary (unit-L2
   per member, with the literal P6.5 `K/d̄` form reported descriptively alongside).

**Zero post-hoc criterion changes. No gate was moved after seeing a number.**

### Incidents

**None.** No failed retrain, no harness failure, no corrupted artifact, no marker-gate violation.

### Material caveats that are not deviations, and must not be lost

* **P13/C1's gate passed by 0.003**, and a descriptive variant of the same check would have failed
  it (§3.3). The verdict stands as preregistered; its instrument is marginal.
* **P11's champion FAIL and secondary PASS are both preregistered results and both are reported.**
  Neither is averaged into the other, and the stage's `VERDICT` field records the champion's FAIL.

---

## 8. After Phase 4, the paper may claim X — and may not claim Y

### May claim

1. **Demo-grain attribution CAN be faithful on this corpus — on C1, for BOTH policy classes — but
   only with the right estimator, and the right estimator differs by class.**
   * BC-Transformer: the **raw gradient dot product** (the exact λ→∞ limit, E=20, per-member
     normalized) reaches **0.54 of a 0.951 ceiling, p = 0.0052** (P11 secondary, preregistered,
     Bonferroni-4).
   * Diffusion policy: **TracIn** on the frozen noise bank reaches **0.84 of a 0.569 median
     ceiling, p = 0.0092** (P13 champion, preregistered) — with the instrument caveat of §3.3.
2. **The Phase-2/3 null was a statement about attribution AS PRACTICED, and it does not survive
   contact with the best preregistered configuration on C1.** It was never a statement about
   attribution's ceiling. Phase 3 already suspected this; Phase 4 measured it.
3. **The exact preconditioner is not merely unhelpful — turning it OFF is what makes attribution
   work.** P6.5 discovered the monotone λ trend; P11 now shows the λ→∞ endpoint is the estimator
   that **passes a preregistered confirmatory test** where the field-standard exact IF/TRAK, and
   even the most *reproducible* attributor, do not.
4. **Stability is not faithfulness.** TracIn is by far the most reproducible attributor (0.831
   split-half at E=20) and it is *not* the most faithful one on the BC-Transformer (0.43 of ceiling
   vs GradDot's 0.54). A study that selects an attributor on reproducibility alone selects wrongly.
   We did exactly that, in the open, and our own preregistered secondary caught us.
5. **For a diffusion policy the seed-MEAN of an executed-action outcome is structurally broken, and
   deeper seeds do not repair it.** At S=8 the mean-aggregated ceilings are still **negative**
   (C1 −0.113). Only the **median** restores the instrument. This now rests on a preregistered
   aggregator, not a post-hoc one.
6. **The P7 C2/C9 heterogeneity was exploration noise.** On 16 fresh masks no analysis had touched,
   all four preregistered cells fail in both estimator families, and the point estimates **collapse**
   (C9@g=5: +0.66 → −0.19).
7. **The BC ground truth is a sound instrument at S=10** (SB-consistency miss 0.017 on C1), and the
   deeper ceiling **raised** the bar rather than lowering it.

### May NOT claim

* **We may NOT say demo-grain attribution is faithful in general.** It passes on **C1 only**, in
  both classes. **C5 fails in both** — with a *sound* instrument in the diffusion arm (§3.4), so
  that FAIL is real, not a measurement artifact. One of two focal targets is not a method.
* **We may NOT present the P11 champion's FAIL as "the null covers the best configuration".** Its own
  preregistered secondary refutes that sentence on C1. The correct statement is the narrower one:
  *the null covers the champion we froze.*
* **We may NOT lean hard on P13's C1 PASS without its caveat.** The ceiling-usability gate passed by
  **0.003**, and a descriptive variant of the same check would have failed it. The result is
  preregistered and it stands — but it is a result whose instrument is at the edge of usable, and
  it should be replicated at greater S before it carries weight.
* **We may NOT claim the diffusion/BC difference is a clean policy-class effect.** The two arms
  differ in aggregator (median vs mean), ceiling height (0.57 vs 0.95), and winning estimator
  (TracIn vs GradDot) simultaneously. That C1 passes in both is the robust part; *why* the winning
  estimator differs is a **hypothesis for a future preregistration**, not a finding.
* **We may NOT say the P14 FAIL kills target-dependence in general.** It kills *the specific C2/C9
  cells P7 surfaced*, at n=16 with a conservative critical ρ of 0.557. The collapse of the point
  estimates makes "exploration noise" the best-supported reading, but a genuinely target-dependent
  effect of moderate size would also have been missed here.
* **We may NOT generalize past this regime**: one benchmark, state-based observations, 9 clusters,
  ≤500 demos/task, two policy classes.

### The one-paragraph statement

Phase 4 settled all three open verdicts, and two of them went against the plan. The heterogeneity
Phase 3 hoped to confirm **evaporated on fresh ground truth** — a clean demonstration of what
exploration on already-used ground truth buys you. The champion we assembled from Phase 3's own
evidence **failed**, while the secondary we preregistered beside it **passed** — because we had
selected our champion for *reproducibility* when Phase 3's real discovery was about
*preconditioning*, and the estimator carrying that discovery was sitting in the secondary slot the
whole time. And the diffusion hypothesis that Phase 3 could only whisper post-hoc **replicated and
strengthened under preregistration**, on an instrument that only works because the median replaced a
seed-mean that is still, at eight seeds, returning negative reliabilities. So the study's closing
position is not the null it started defending. It is this: **demo-grain data attribution is
achievable on this corpus — on some targets, with the preconditioner switched off, against a ground
truth measured far more carefully than the field measures it — and every one of those three
qualifiers was invisible until it was preregistered and tested.**

---

## 9. Artifact index

| artifact | contents |
|---|---|
| `preregistration_phase4.json` (+ `.sha256`) | locked before all Phase-4 training, attribution, and verdicts (`fca37f54…`) |
| `results/p12_ceilings.json` | S=10 ceilings; **SB-consistency instrument gate, 9/9 PASS** |
| `results/p12_outcomes_S10.parquet` | the 24×10 BC ground truth |
| `results/p11_verdict.json` / `p11_lds_table.csv` | **champion FAIL; preregistered secondary PASS on C1** |
| `results/p11_gram_cache_new_members.npz` | G, K for members 211–220 (the λ→∞ input) |
| `results/p11_gram_selfvalidation.json` | **180/180 vs archived P9 scores, ρ = 1.000000** |
| `results/p13_verdict.json` / `p13_lds_table.csv` | **PASS on C1** (0.84, p=0.0092); C5 FAIL; C2 INSTRUMENT-UNUSABLE; C5 diagnosis |
| `results/p13_outcomes_S8.parquet` | the 24×8 diffusion ground truth |
| `results/p14_mask_manifest.json` | 16 fresh masks, seed 1104, **0 Stage-G coincidences** |
| `results/p14_verdict.json` / `p14_lds_table.csv` | **FAIL on all 4 cells, both families**; power limit |
| `results/SYNTHETIC_p4_unit_tests.json` | **SYNTHETIC** — 32/32 on the new Phase-4 statistics |
| `results/budget_actual.json` | 208 retrains, 32.50 GPU-h, 42,720 episodes, **0 failures** |
| `results/report_verification.json` | every number in this report, machine-verified against artifacts |
