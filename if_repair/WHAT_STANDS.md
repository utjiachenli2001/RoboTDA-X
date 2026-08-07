# What stands — RoboTDA-X `if_repair`, passes 9–16

**Authoritative as of 2026-08-01.** BLOCKERS and FINDINGS are append-only and now carry one
retraction, one withdrawal, one downgrade, four qualifications and two factual corrections layered in
place. This file is the single statement of what survives. **Where this document and an older entry
disagree, this document wins.**

Corpus: 135 robot imitation-learning demonstrations, 9 clusters × 15. LDS measures how well an
estimator's per-demo influence scores predict the outcome change when demos are removed and the model
retrained. Compute: **6,532 retrains / ~74.7 occupancy-hours** on one H200 across passes 9-13 (campaigns O 1600, P 132, R 1600, Q 1600, S 1600; pass 11 spent none). Verified against `results/gpu_ledger_pass*.csv` -- a first draft of this file said 8,132 / ~93h, which double-counted a campaign. That arithmetic slip has now occurred four times in this project and is the reason the figure is cited with its source.

---

## 1. Established

**Gradient-based attribution does not clear the usefulness bar at any unit size.**
12–45% of attainable (`ρ/√r`) across ten attempts and five designs; at the unbiased depth-4 ceiling,
0.133 (k=3) and 0.158 (k=5) on the `ρ/r` convention. Reinforced by every control added — |S|,
partition, depth. *(#48, #51, #53)*

**The published cluster-grain success was substantially a training-set-size artifact.**
A *fixed* estimator scores Kendall 0.353 pooled on outcomes shuffled within stratum, against a real
0.475 — which cannot be leakage. Within stratum the null collapses to ~0 and the bar clears in no
stratum. *(#41)*

**The bar is reachable and therefore discriminates.** It is not measuring the ceiling: a
design-based datamodel clears it. *(#48)*

**The datamodel attributes — it transfers across an independent re-partition.**
Fit on one partition, scored on another sharing zero groups: 4.0–4.8× the gradient estimator on
identical masks (z = 5.3, 4.1). A method fitting only its own mask ensemble would collapse. *(#50)*

**But at the unbiased ceiling it clears the bar at k=3 only.**
Out of partition *and* at depth 4: k=3 ratio 0.638, CI [0.537, 0.745]; k=5 0.538, CI [0.437, 0.642],
P(clear) = 0.77. The between-grain difference is itself only ~1.3σ. *(#53)*

**The transfer penalty is ~20–23%, stable, and a property of the method.**
`transfer ÷ within` = 0.75–0.84 at both grains and flat across a 6× range of fit sizes. Roughly a
fifth of within-partition performance does not cross to an independent partition, regardless of how
much data the fit saw. **This is the figure a next-corpus design should budget for.** *(#55, #56)*

**Predictions transfer better than attributions do.**
Two independent partitions agree on *per-demo* influence at only ~0.47–0.52 (against a ~0 shuffle
null; within-campaign half-agreement is 0.69/0.90). **Any downstream use of these scores
per-demonstration — pruning, selection, reweighting, which is what TDA is for — inherits the ~0.5,
not the 4.8×.** *(#54, surviving portion)*

**The passes-4–7 self-influence corrections are dead.** Neither rank nor scale swap recovers them;
within stratum their ordering is actively anti-predictive. *(#37, #43)*

**The grain question is closed by combinatorics.** The k=15 conditional population is C(8,4) = 70 and
is exhausted. No purchasable design tightens that rung below a CI width of ~0.6. *(#45)*

---

## 2. Retracted, withdrawn or downgraded — do not quote these

| claim | status | why |
|---|---|---|
| "k=3 is partition-sensitive" (42% movement) | **retracted** | z = 1.23. A percentage read as a finding without a noise floor. *(#47)* |
| "k=5 transfer loses 32%, k=3 nothing" | **withdrawn** | Denominator artifact — divided by a fit on the *other* campaign, whose k=3 draw ran 1.2σ low. Apples-to-apples: 0.779 vs 0.766, **no asymmetry**. *(#56)* |
| "k=5 agrees less across partitions" | **retracted** | Pearson-only, <1σ, and *reverses* under Spearman (0.485 vs 0.487). *(#54)* |
| "the loss is about coefficient count" | **downgraded to hypothesis** | With two grains it is confounded with everything else; fit-held-fixed difference is −0.104, CI [−0.308, +0.100]. *(#52)* |
| "over-determination buys within-sample and pays in transfer" — **and the design rule from it** | **unsupported** | Tested at fixed grain across 6× masks-per-coefficient: slopes −0.010 and −0.001, both CIs containing zero. *(#55)* |
| "zero flagged comparisons in passes 1–8" | **corrected** | Three exist (`p7_per_draw`, `holdout_table`, `holdout_phase2`); the entry accounted for 26 of its own 32 flags. *(#49)* |
| "the datamodel reaches 75–78% of attainable" | **corrected** | Units error — that is `ρ/r`. Attainable is ~47–49%. *(advisor report)* |

---

## 3. Unresolved, and why it stays that way here

- **Whether attribution improves with unit size.** Unresolvable: the k=15 population is capped at 70
  and exhausted (#45).
- **Whether partition draw matters.** Unresolvable at feasible cost: the data-consistent
  between-partition SD (~0.021) sits below one partition's own SE (~0.032); separating them needs
  ~20 partitions, ~50,000 retrains (#47).
- ~~**Whether the gradient failure reflects the corpus or the approach.**~~ **RESOLVED by campaign U
  (pass 18), on a second corpus.** See `p18_prereg.md` and `results/confirm_useries.{csv,json}`.
  Holding the retained training set at 25 demonstrations and growing the candidate pool
  50 -> 100 -> 200 -> 370 on `libero_goal`, with 11,386 retrains:

  | arm | tau at pools 50/100/200/370 | ceiling | fraction of attainable |
  |---|---|---|---|
  | gradient (GradDot) | 0.057 / 0.036 / 0.130 / 0.032 | 0.47-0.57 | **5.6-23.1%** |
  | design-based datamodel | 0.128 / 0.203 / 0.215 / 0.226 | 0.47-0.57 | **27-40%** |

  **Gradient attribution clears no useful bar at any reachable pool size, and its curve is
  NON-MONOTONE** (interior deviation 0.065 against a 0.0497 margin), so the preregistered branch
  precedence quotes no slope for it -- what stands is the level. **The datamodel attributes, and
  its advantage is FLAT in pool size** (weighted slope +0.0182, CI [-0.0047, +0.0409], inside the
  TOST margin). More candidate data does not rescue gradient attribution.

  **Scope, and it is narrower than the question as originally posed.** Campaign U never trains an
  evaluation model on more than 25 demonstrations; what grows is the candidate pool. The claim is
  therefore **for subset selection**, unconditionally -- not "the failure is not a corpus-size
  artifact" in general. Two preregistration defects are recorded as amendments in
  `p18_prereg.md` (a ceiling/band scale mismatch, and a sign-convention error in scoring that
  briefly produced a false "gradient improves with data" headline).

  **What killed the first attempt, and is worth carrying forward.** Campaign T removed 50% of each
  pool and was preregistered, implemented and nearly launched before a 32-retrain variance pilot
  measured its ceiling at **exactly zero** at the top rung: training on 185 of 370 demos puts the
  model past the point where WHICH demos it received still matters, so the primary statistic was
  0/0. Had it run, it would have reported "attribution degrades with corpus size" when the truth
  was "the outcome stopped moving". **Retained count, not pool size, governs measurability.**

---

## 4. Methodological lessons, in order of how much they cost to learn

1. **A ratio inherits the noise of its denominator — and a denominator built from a different fit on a
   different draw inherits that draw's luck.** Four entries chased a mechanism for what was campaign
   R's k=3 draw coming in 1.2σ low. The fix: hold the fit fixed, vary only the target. *(#56)*
2. **Two point estimates are not a variance.** Compute the within-draw sampling noise before quoting a
   difference between draws. The raw movement was 42%; the effect was 1.2σ. *(#47)*
3. **The ceiling is a reliability `r`, so the attainable maximum is ~`√r`.** `ρ/r` is inflated, and
   inflated *more* at low depth. Two ratios at different depths are not comparable. *(#42)*
4. **A nuisance axis that moves the outcome *and* every estimator's prediction credits both sides for
   the nuisance.** Detector: a permutation null on a *fixed* estimator — it cannot be confused with
   leakage. *(#41)*
5. **When a nuisance axis cannot be held fixed inside the reachable design space, go finer, not
   bigger.** *(#44)*
6. **Having a written rule against a failure mode did not prevent repeating it four times.** #42 and
   #47 were both on the books when #48–#54 were written. What caught them was an adversarial external
   audit, not the rule.

---

## 5. What the next corpus needs

**Port to 500+ demonstrations.** It is the only route to the grain question and the only clean
discriminator for the gradient failure. Design against these figures:

- Budget for a **~20% transfer penalty** — a property of the method, not of fit size.
- If the goal is per-demonstration use, design against **~0.5 cross-partition agreement**, not the
  predictive comparison.
- Preregister the **conditioning rule and stratification** before drawing masks; both bit this campaign.
- Use **even seed depth** (the split-half ceiling returns NaN at odd depth) and report `ρ/√r`
  alongside `ρ/r`.
- Do **not** carry forward the self-influence corrections, or the withdrawn over-determination design
  rule.

---

## 6. Provenance

Every campaign preregistered before any run existed; O and R scored exactly once under a
never-overwrite guard. Campaigns P, Q, S and all datamodel analyses are descriptive and carry no
alpha. 151 tests pass. Entry points: `if_repair/BLOCKERS.md` (#41–56), `if_repair/HANDOFF.md`
(pass-9 through pass-12 sections), `docs/PASS9_10_REPORT.md` (advisor-facing).

**Reading order for someone new:** this file, then the HANDOFF's most recent section, then BLOCKERS
only for a claim you intend to quote.
