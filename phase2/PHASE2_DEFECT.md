# PHASE2_DEFECT.md — instrument defect found before Stage P4 ran

**Found:** 2026-07-11, during P4 implementation, **before any P4 episode was executed.**
**Status:** P4 not yet run. Nothing is contaminated. P1 was running (unaffected — see Scope).
**Affects:** the Phase-2 brief's §6 design ("90 rollouts per probe task"). Nothing in Phase 1.

---

## The defect

The Phase-2 brief specifies re-evaluating checkpoints "at 90 rollouts per probe task" to measure
how closed-loop success reliability improves with episode count. **A LIBERO task cannot supply 90
distinct episodes.** The rollout worker selects an episode's initial condition with

```python
init_states = np.asarray(B.get_task_init_states(ti))     # src/rollout.py:_rollout_task
obs = env.set_init_state(init_states[ep % init_states.shape[0]])
```

and every probe task of C1 and C2 has **exactly 50** init states:

| cluster | suite | probe task | init states |
|---|---|---|---|
| C1 | libero_goal | open_the_middle_drawer_of_the_cabinet | **50** |
| C1 | libero_goal | open_the_top_drawer_and_put_the_bowl_inside | **50** |
| C1 | libero_goal | push_the_plate_to_the_front_of_the_stove | **50** |
| C2 | libero_spatial | pick_up_the_black_bowl_between_the_plate_and_the_ramekin… | **50** |
| C2 | libero_spatial | pick_up_the_black_bowl_from_table_center… | **50** |
| C2 | libero_spatial | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet… | **50** |

So `ep % 50` makes **episodes 50…89 reuse initial states 0…39** (verified: all 40 rows identical;
episodes 0…89 span **50 distinct init states, not 90**).

Every other source of variation is deterministic, so the reused init state produces a *bit-identical
episode*, not merely a similar one:

* **Policy is deterministic.** `policy.BCTransformer.act` = "mean of the highest-weight mode":
  `k = logits.argmax(-1)`, return `mu[:, k].clamp(-1,1)`. No sampling from the GMM.
* **Env is deterministic.** `make_env(..., seed=0)`, then `env.reset()` → `set_init_state(...)` →
  a fixed number of zero-action settle steps.

**Consequence:** a nominal "90-rollout" success estimate is really **50 distinct episodes with
episodes 0–39 counted twice** (weights 2/90 vs 1/90). It is *not* a 90-independent-sample estimate.
Reporting a reliability curve whose x-axis reached 90 would have overstated the achievable episode
budget and mislabelled a 50-episode measurement as 90.

## Scope — what is and is not affected

| stage | rollouts/task | ≤ 50 distinct? | affected |
|---|---|---|---|
| Phase 1 Stage B / C / I | 20 | yes | **no** |
| Phase 1 Stage E / F / G (probe battery) | 10 | yes | **no** |
| Phase 2 **P1** (probe battery) | 10 | yes | **no** |
| Phase 2 **P3** (full Goal suite) | 20 | yes | **no** |
| Phase 2 **P4** as briefed | **90** | **NO — only 50** | **YES** |

**No Phase-1 result is affected**: every Phase-1 rollout count (10, 20, 30 per task) is ≤ 50, so
Phase-1 episodes are all distinct. The Gate-1 "success ceiling = −0.93" finding stands, and P1/P3
are unaffected.

## Amendment to P4 (recorded BEFORE P4 ran)

P4 is the one stage with **no preregistered pass/fail criterion** — the preregistration declares it
"purely descriptive instrumentation science". This amendment therefore changes **no criterion and no
verdict**; it corrects the measurement ladder to what the instrument can actually deliver.

* **Episode ladder per probe task:** `10 / 30 / 90` → **`10 / 30 / 50`** (50 = every distinct init
  state the task has). At the cluster level (3 probe tasks pooled) that is **30 / 90 / 150 distinct
  episodes** per success estimate — still a 5× range, and still nested (the 10-episode subsample is
  exactly Phase-1's Stage-G episode set).
* **Rollouts actually executed:** 50/task, and 10 and 30 are formed by subsampling those 50.
* **Episodes:** arm a = 48 models × 6 tasks × 50 = **14,400** (was 25,920 — *under* ledger).
* **Extrapolation caveat (now a finding, not a footnote):** the Spearman–Brown extrapolation to the
  episode budget that would reach a ceiling of 0.5 / 0.8 must be reported as an extrapolation
  **beyond what a single task can supply**. Past 50 episodes/task the only ways to buy more
  independent closed-loop samples are *more tasks*, *more training seeds*, or *new initial states* —
  you cannot buy them by rolling out the same task again. This is a real constraint on the
  CUPID-style closed-loop-return curation objective and is reported as such.

## Verification still owed

The init-state duplication is proven at the array level (above). An **end-to-end** confirmation —
run episodes 0–4 and 50–54 of one probe task and check the success/step vectors are bit-identical —
requires a GPU; all four allowed GPUs are currently running P1. It will be run **before P4
executes**, and its result recorded here. If (contrary to the code) the pipeline turns out to be
non-deterministic, this amendment will be revisited and the finding corrected.

---

### END-TO-END CONFIRMATION — **RAN, DEFECT CONFIRMED**

Artifact: `phase2/results/p4_determinism_check.json`. GPU 4, task
`libero_goal/open_the_middle_drawer_of_the_cabinet`.

A near-floor checkpoint would fail *every* episode at the full 600-step horizon, so its vectors
would match **trivially** and prove nothing. The check therefore uses a **discriminating**
checkpoint — `runs/stage_C/Q50_cotrain_s101/final.pt` (~50% success) — whose per-episode outcomes
genuinely vary:

| episodes | success | steps |
|---|---|---|
| 0–9 | `F F F F F F T F F T` | `600 600 600 600 600 600 **154** 600 600 **158**` |
| 50–59 | `F F F F F F T F F T` | `600 600 600 600 600 600 **154** 600 600 **158**` |
| 0–9 (repeat) | `F F F F F F T F F T` | identical |

Episodes 50–59 reproduce episodes 0–9 **exactly, including the step counts at which the two
successes occur (154 and 158)**. Re-running 0–9 reproduces them bit-for-bit, so the pipeline is
deterministic. `vectors_are_discriminating = true`, `pipeline_is_deterministic = true`,
`eps 50–59 duplicate 0–9 = true` → **DEFECT_CONFIRMED = true.**

The amendment above stands: P4 runs 50 rollouts/task and subsamples 10/30/50.
