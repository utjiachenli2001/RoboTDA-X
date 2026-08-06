---
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
origin: if_repair/WHAT_STANDS.md §5 + §3
created: 2026-08-04
plan_type: feat
---

# feat: The strong version — does gradient attribution fail because the corpus is small, or because it is gradient attribution?

**Target repo:** RoboTDA-X. **Target box:** `h100-1` (8× H100, idle). **Status:** planning only, no
compute committed.

---

## Summary

The campaign's honest position (`WHAT_STANDS.md`) is that gradient-based attribution clears a
usefulness bar at no unit size on a 135-demo corpus, a design-based datamodel does clear it and
genuinely attributes, and **the single unresolved question is whether the gradient failure reflects
the corpus size or the approach.** Every other question is answered or shown combinatorially closed.

Two facts discovered while scoping change what is reachable:

1. **The bigger corpus is already on disk.** 700 processed demos exist; the campaign uses 225. In
   particular **`libero_goal` is 10 tasks × exactly 50 demos = 500**, of which the campaign has
   touched 25. Nothing needs collecting.
2. **`h100-1` is idle with 8 GPUs.** The campaign ran on one H200 at ~88 retrains/hour. Eight cards
   should give roughly 8× that, turning a ~68-hour ladder into ~9 hours.

That makes the decisive experiment affordable: **a corpus-size ladder at a fixed task distribution.**

---

## Problem Frame

**Why the size question is the one that matters.** Every negative in this project is qualified by a
single corpus. A reviewer's first question — and the honest answer is *unknown* — is whether 135
demonstrations is simply too few for any gradient method to work. If attribution quality climbs with
N, the result is "TDA needs more data than robot corpora usually have", which is actionable. If it
stays flat from 50 to 500, the result is "gradient TDA does not work on this kind of data", which is
a much stronger and more publishable claim. **Both outcomes are worth the compute; that is what makes
this the right experiment.**

**Why `libero_goal` alone, rather than growing the existing mixed corpus.** Growing the current
135 by adding whatever is available would confound size with composition: the unused pool is 71%
`libero_goal`, so a bigger corpus would also be a *differently shaped* corpus, and any trend could be
task-mix rather than size. `libero_goal` is 10 tasks × 50 demos, so every rung of the ladder can hold
the task distribution **exactly** fixed — 5, 10, 20 or 50 demos per task, all ten tasks present at
every N. Size becomes the only thing moving.

**It is also a second corpus.** The 135-demo corpus spans 5 suites; this one is a single suite with a
different task mix. So the ladder delivers generalisation and size-scaling from the same runs.

**What the campaign's own scars require of the design.** Each of these is a rule this project paid
for, and the ladder must respect all of them:

| lesson | requirement here |
|---|---|
| #41 — a nuisance axis that moves outcomes *and* predictions credits both sides | every mask at a rung keeps **exactly the same number of demos**; no pooling over training-set size |
| #42 — the ceiling is a reliability, `ρ/r` is inflated and depth-dependent | report `ρ/√r` beside `ρ/r`; **identical seed depth at every rung**, or the ladder measures depth |
| #39 — the split-half ceiling is NaN at odd depth | even depth everywhere |
| #47 — two point estimates are not a variance | every rung-to-rung comparison quoted with a CI, never as a percentage |
| #54 — per-demo attributions agree across partitions at only ~0.5 | **two partitions at every rung**, so agreement is itself a curve in N |
| #56 — a ratio inherits its denominator's luck | hold the fit fixed and vary only the target; never divide by a fit from another draw |

---

## Requirements

- **R1.** Measure attribution quality for both a gradient estimator and the datamodel at
  N ∈ {50, 100, 200, 500} demonstrations from `libero_goal`, task distribution held exactly fixed,
  training-set size held fixed within each rung, identical seed depth across rungs.
- **R2.** Report, per rung: LDS, ceiling, `ρ/r`, `ρ/√r`, and a bootstrap CI on the ratio with the
  ceiling recomputed per resample.
- **R3.** Run **two independent partitions at every rung** and report cross-partition per-demo
  agreement as a function of N.
- **R4.** State whether attribution quality is flat or rising in N, with a CI on the trend — not a
  percentage between endpoints.
- **R5.** **The downstream test.** At the largest rung, drop the bottom-*k* demonstrations by
  influence, retrain, and compare against dropping *k* at random and against dropping the top-*k*.
  This is what TDA is *for*, and no result in this project has tested it.
- **R6.** Preregister before any run; score once; write up with the same discipline as passes 9–16.

**Definition of done:** the ladder scored once against its prereg; the agreement curve reported; R5
answered; a stated position on size-vs-approach with its CI; everything pushed.

---

## Key Technical Decisions

**KTD1 — `libero_goal` only, all 10 tasks at every rung.** Size is the only moving part (Problem
Frame). The 25 demos already used by the old campaign are **excluded** from the fresh pool to keep
this corpus unselected-upon (#28, #31).

**KTD2 — Fixed retained-set size within each rung.** Directly from #41. Masks at rung N keep a
constant fraction — proposed 55% — so no mask differs from another in training-set size.

**KTD3 — Identical seed depth (2) at every rung.** #42 makes the ratio depth-dependent, so a ladder
with varying depth would measure depth, not size. Depth 2 is even (#39) and allocation-optimal (#29).

**KTD4 — Group grain, scaled with N to hold masks-per-coefficient roughly constant.** Otherwise the
estimation regime drifts along the ladder and confounds size (this is #52/#55's territory — the
regime effect was *not supported* when tested, but the design should not reintroduce the confound).
Proposed groups-of-5, giving 10/20/40/100 coefficients at the four rungs.

**KTD5 — Two partitions at every rung, sharing zero groups.** #54's agreement figure becomes a curve,
which is itself a publishable quantity: *does per-demo attribution become more reproducible with more
data?* Nobody reports this.

**KTD6 — R5's pruning test uses a matched random control and a top-*k* arm.** Dropping low-influence
demos must beat dropping random ones to mean anything; the top-*k* arm bounds how much *could* be
lost. Without both controls the experiment is uninterpretable.

**KTD7 — Verify the cost model before committing.** `retrain.py` uses `total_steps` (8000), which
suggests retrain cost is **independent of corpus size**. The entire budget rests on that. **One timed
retrain at N=500 versus one at N=135 settles it, and must run before anything else is launched.** If
cost instead scales with N, the ladder's top rung dominates and the design needs re-sizing.

---

## Implementation Units

### U0. Verify the cost model and stage the box  *(~30 min, 2 retrains)*

**Files:** none (operational).
Time one retrain at a 275-demo training set against one at 75. Clone the repo on `h100-1`, rsync
`data/proc` (and note that `if_repair/runs/` is 6.7 GB, gitignored and **not** on GitHub — it must
come from `h200-1` by rsync while that box is up). Confirm 8-GPU worker placement and that
`CUDA_VISIBLE_DEVICES` is set per worker (#40 — `bootstrap.py` pins `ALLOWED_GPUS=(4,5,6,7)`, which
on an 8-GPU box is *correct* but must still be checked rather than assumed).

**Verification:** measured seconds-per-retrain at both sizes, and a per-GPU throughput figure that
replaces the ~88/hour assumption.

### U1. The corpus and the ladder  *(zero GPU)*

**Files:** `if_repair/p18_corpus.py`, `if_repair/p18_masks.py`, tests.
Build the `libero_goal` pool excluding the 25 already-used demos; define rungs N ∈ {50,100,200,500}
with exactly N/10 demos per task; build two independent partitions per rung; draw masks at a fixed
retained fraction with exact group balance and exact signature disjointness across rungs and
partitions.

**Test scenarios:** every rung has all 10 tasks in equal number; every mask at a rung keeps the same
demo count; the two partitions at a rung share zero groups; no mask signature repeats within or
across rungs; group inclusion balance spread is 0.

### U2. Prereg  *(zero GPU, frozen at zero runs)*

**Files:** `if_repair/p18_prereg.md`.
Hypothesis of record on the **trend** (R4), family size and alpha stated, the statistic fixed by
continuity (Kendall), the stopping rule outcome-blind, scored-once enforced, and a decision rule
covering flat / rising / non-monotone before any data exists.

### U3. The ladder campaign  *(GPU — the main spend)*

**Files:** `if_repair/retrain.py` (campaign `T`), `if_repair/confirm_tseries.py`, tests.
Estimated **~6,000 retrains**: roughly 800 / 1,200 / 1,600 / 2,400 across the four rungs (masks ×
depth 2 × 2 partitions). At the campaign's measured single-GPU rate that is ~68 h; on 8 idle H100s,
**~9 h** — to be replaced by U0's measured figure.

### U4. The downstream pruning test  *(GPU, ~100–200 retrains)*

**Files:** `if_repair/p18_prune.py`, tests. R5/KTD6.

### U5. Analysis and write-up  *(zero GPU)*

Ratio-vs-N with CIs, agreement-vs-N, the pruning result, FINDINGS/BLOCKERS/HANDOFF, and a refresh of
`WHAT_STANDS.md` — whose §3 currently says this question is unresolvable *on the old corpus*, and
will need amending to say what the new one showed.

---

## Falsification Map

| ladder outcome | reading |
|---|---|
| gradient ratio rises clearly with N | the failure was corpus size — the headline becomes "TDA needs more data than robot corpora carry", with the N where it crosses |
| gradient ratio flat from 50 → 500 | **the strong result**: gradient attribution does not work on this data at any reachable scale |
| rises but never clears the bar | quantify the gap and extrapolate the N that would — a concrete design number for the field |
| datamodel advantage shrinks with N | the datamodel's edge was a small-corpus artifact; would qualify the campaign's main positive |
| agreement (R3) rises with N | per-demo attribution becomes usable at scale — the most actionable thing here |
| agreement stays ~0.5 | per-demo TDA is not reliable on robot data regardless of size, which is a strong negative for the field's main use case |

| R5 outcome | reading |
|---|---|
| dropping low-influence beats random | attribution is *useful*, not merely measurable — the result the field wants |
| indistinguishable from random | the ~0.5 agreement has teeth: the scores do not support selection |
| dropping top-*k* hurts much more than random | attribution identifies harmful-to-lose data even if it cannot rank the middle |

---

## Risks

| risk | consequence | mitigation |
|---|---|---|
| retrain cost scales with N | the top rung dominates and ~9 h is wrong | **U0 measures it first**; re-size before committing |
| `h100-1` lacks repo and data | cannot start | stage in U0; `runs/` must come from `h200-1` **while it is up** |
| 8-GPU worker placement wrong | silent CPU fallback (#40) | verify non-zero memory on each GPU after launch |
| composition drift along the ladder | trend confounded with task mix | KTD1 — all 10 tasks, N/10 each, at every rung |
| reusing the 25 already-scored demos | selected-upon corpus (#28, #31) | excluded in U1, asserted in tests |
| the ladder returns "flat" and reads as a null | undersold | it is the **strong** outcome; the prereg says so in advance so it cannot be reframed after the fact |

---

## What this does not do

- No real-robot data. The claim stays "on LIBERO-style simulated manipulation corpora".
- No new estimator. The contribution is evaluation, not method.
- Does not revisit anything `WHAT_STANDS.md` marks closed — the grain question stays combinatorially
  shut on the old corpus, and partition sensitivity stays unresolvable there.

## Open question for Jiachen

**Venue and framing.** Two papers are available from this and they want different emphasis: an
evaluation/pitfalls paper (the measurement lessons, publishable without any of the above) or an
empirical paper (this ladder as the centrepiece). The plan above serves the second. If the first is
the target, U3 shrinks and U5 grows.
