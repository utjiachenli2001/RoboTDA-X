# HANDOFF — the strong version (corpus-size ladder), to run on h200-1

**Written 2026-08-04. Repo at `3c92509`, clean, pushed.** Everything below is self-contained; you do
not need the previous session.

---

## 0. Read these first, in this order

| file | why |
|---|---|
| `if_repair/WHAT_STANDS.md` | **START HERE.** The single authoritative statement of what the campaign established, what was retracted, and what is unresolved. It wins over any older entry it contradicts. |
| this file | the plan for the one remaining question |
| `docs/plans/2026-08-04-001-feat-strong-version-plan.md` | the long-form plan (copy it in from the attachment if not present) |
| `if_repair/BLOCKERS.md` #41–56 | only when you intend to quote a specific claim |

**Do not read FINDINGS/BLOCKERS front-to-back to orient.** They are append-only and carry one
retraction, one withdrawal, one downgrade, four qualifications and two factual corrections layered in
place. `WHAT_STANDS.md` exists precisely because they are no longer safely readable as a whole.

---

## 1. The question, and why it is the only one left

Gradient-based attribution clears a usefulness bar at **no** unit size on the 135-demo corpus
(12–45% of attainable). A design-based datamodel **does** clear it and genuinely attributes — fit on
one partition it predicts an independent partition's outcomes at 4.0–4.8× the gradient estimator.
Its per-demo attributions, however, agree across partitions at only **~0.5**.

Every other question is answered or shown combinatorially closed. **The one unresolved question is
whether the gradient failure reflects the corpus size or the approach.**

## 2. The experiment

**A corpus-size ladder at a fixed task distribution**, on data already on disk.

`libero_goal` is **10 tasks × exactly 50 demos = 500**, of which the old campaign touched only 25.
So build rungs at **N ∈ {50, 100, 200, 500}** with **N/10 demos per task at every rung** — all ten
tasks always present. Size becomes the only moving part. (Growing the *existing* 135-demo corpus
instead would confound size with composition: the unused pool is 71% `libero_goal`.)

Because it is also a different corpus from the 135, the same runs buy **generalisation** and
**size-scaling** together.

**Measure at each rung:** gradient estimator and datamodel — LDS, ceiling, `ρ/r`, `ρ/√r`, bootstrap
CI on the ratio with the ceiling recomputed per resample. **Two independent partitions per rung**, so
cross-partition per-demo agreement becomes a curve in N.

**Then the downstream test** (nothing in this project has done it): at the largest rung, drop the
bottom-*k* demos by influence, retrain, and compare against dropping *k* at **random** and against
dropping the **top-*k***. Both controls are required or the result is uninterpretable.

### Both outcomes are publishable — say so before running

| ladder result | reading |
|---|---|
| gradient ratio **flat** 50 → 500 | **the strong result**: gradient TDA does not work on this data at any reachable scale |
| gradient ratio **rises** | the failure was corpus size — report the N where it would cross |
| rises but never clears | quantify the gap; extrapolate the N that would |
| datamodel edge shrinks with N | its advantage was a small-corpus artifact — would qualify the campaign's main positive |
| agreement rises with N | per-demo attribution becomes usable at scale — the most actionable outcome |
| agreement stays ~0.5 | per-demo TDA is unreliable on robot data regardless of size |

Fix these readings **in the prereg, before any run**, so a null cannot be reframed afterwards.

---

## 3. THE BOX — read before launching

**h200-1 is `ssh h200-1`, Nebius `jli-gold-lemming-instance-8`, ONE NVIDIA H200 (140 GB, index 0).**
Repo and data are already there at `~/code/RoboTDA-X` — no staging needed.

**⚠ It is currently shared.** An **EgoVerse** training run (`egomimic/trainHydra.py`, `G3p_TP1`) was
using it as of this handoff. **Check before launching and coordinate — do not evict it.**

```bash
ssh h200-1 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv; pgrep -af trainHydra | head -3'
```

### Cost on ONE GPU — this is the number that changed

The campaign measured **~88 retrains/hour with 3 workers** on this card. The ladder is **~6,000
retrains ≈ 68 hours**. (On the idle 8-GPU `h100-1` it would be ~9 h, but that box has **no repo and
no data**, and `if_repair/runs/` is 6.7 GB, gitignored and **not on GitHub** — it would have to rsync
off h200-1 first. Your call; the plan works either way.)

### VERIFY THIS BEFORE COMMITTING THE BUDGET

`retrain.py` uses `total_steps` (8000), which suggests **retrain cost is independent of corpus
size**. The whole 68-hour estimate rests on that. **Two timed retrains settle it** — one at a
275-demo training set, one at 75. If cost scales with N instead, the top rung dominates and the
ladder must be re-sized. Do this first.

### Non-negotiable box rules

- **`export CUDA_VISIBLE_DEVICES=0` on every command.** `src/bootstrap.py` pins
  `ALLOWED_GPUS=(4,5,6,7)` from the original 8-GPU machine; on this box that silently disables CUDA
  and falls back to CPU — a run that looks healthy but is ~50× too slow. **Never edit
  `bootstrap.py`.** After launching, verify non-zero GPU memory before walking away.
- **Python is `.venv/bin/python`** in `~/code/RoboTDA-X`.
- **The shell is zsh.** Unquoted `$var` does **not** word-split — this silently broke a watchdog into
  a false "campaign died" alarm. Use `awk` or explicit field extraction in any monitor.
- **Commit messages containing double quotes need `git commit -F file`, not `-m`.**
- **Push needs `ssh -A h200-1`** — a detached job has no agent, so pushes must be a foreground step.

---

## 4. Design rules the campaign paid for — the ladder must respect all of them

| rule | requirement here |
|---|---|
| **#41** a nuisance axis that moves outcomes *and* predictions credits both sides | every mask at a rung keeps **exactly the same number of demos**; never pool over training-set size |
| **#42** the ceiling is a reliability, so `ρ/r` is inflated and depth-dependent | report `ρ/√r` beside `ρ/r`; **identical seed depth at every rung** or the ladder measures depth, not size |
| **#39** the split-half ceiling returns NaN at odd depth | even depth everywhere (use 2) |
| **#47** two point estimates are not a variance | every rung-to-rung comparison quoted with a CI, **never as a percentage** |
| **#54** per-demo agreement is only ~0.5 | two partitions at every rung |
| **#56** a ratio inherits its denominator's luck | hold the fit fixed, vary only the target; never divide by a fit from another draw |
| **#28/#31** discovery draws are selected-upon | **exclude the 25 `libero_goal` demos the old campaign used** |
| **#29** masks beat seeds ~5:1 | depth 2, spend on masks |

**Preregister, score once.** `confirm_*.py` refuses to overwrite, and — added after it nearly bit —
refuses to write at all unless *every* preregistered arm has usable data, so a partial campaign
cannot consume the one-shot score.

---

## 5. Work order

1. **U0 — verify the cost model** (2 retrains, ~10 min). Time one retrain at 275 demos vs one at 75.
   Replace the ~88/hour assumption with a measured figure.
2. **U1 — corpus and ladder** (zero GPU). `p18_corpus.py`, `p18_masks.py`: build the `libero_goal`
   pool excluding the used 25; rungs with N/10 per task; two partitions per rung; fixed retained
   fraction; exact group balance; exact signature disjointness across rungs and partitions. Tests.
3. **U2 — prereg** (`p18_prereg.md`), frozen at zero runs, with §2's decision table.
4. **U3 — the ladder** (campaign `T`, ~6,000 retrains). Seed-major so every prefix is balanced.
5. **U4 — the pruning test** (~100–200 retrains) with random and top-*k* controls.
6. **U5 — analysis and write-up.** Ratio-vs-N and agreement-vs-N with CIs; FINDINGS/BLOCKERS/HANDOFF;
   **amend `WHAT_STANDS.md` §3**, which currently says this question is unresolvable *on the old
   corpus*.

**Use the adversarial-judge step.** Send the plan and the prereg to Fable before spending GPU. It has
killed three plans in this project before they cost anything — roughly 60 GPU-hours saved — and each
time the flaw was one written into the plan and not seen. It also caught, in a claims audit, four
committed results that did not survive their own error bars.

---

## 6. Reusable machinery already in the repo

| file | what it gives you |
|---|---|
| `if_repair/p9_grain.py` | partitions a corpus into equal groups, seeded and committed |
| `if_repair/p9_masks.py` | complementary-pair mask construction → exact balance, signature disjointness |
| `if_repair/p10_masks2.py` | a second independent partition (zero shared groups) |
| `if_repair/confirm_oseries.py` | scored-once scorer with the stopping rule and partial-score guard |
| `if_repair/p9_stratum_control.py` | `boot_ratio` (ceiling recomputed per resample), `perm_pooled` |
| `if_repair/p11_transfer.py` | fit-on-A / score-on-B, coefficient→per-demo mapping |
| `if_repair/p15_overdet.py` | the transfer÷within design that divides out fit quality |
| `if_repair/p17_demo_scores.py` | per-demo influence under both partitions |
| `retrain.py` | `jobs(campaign)` — add campaign `T` as a new branch **and to the argparse choices** |

`pytest if_repair/tests -q` → **151 passed, 3 skipped, ~30 s, all CPU.** Keep it green.

---

## 7. The paper this is for

**One paper, empirical.** Methodology as the spine, the ladder as the result, pruning as the payoff.
Not a method paper — the datamodel is Ilyas et al.; the contribution is evaluation.

The measurement lessons are **not** a separate contribution — they are the validity argument.
Reviewers attack an empirical paper's measurement first, and "we found four ways this metric misleads,
including four of our own claims we retracted with the numbers that killed them" is the strongest
available answer.

Claim, roughly:

> On simulated robot imitation corpora, gradient-based training-data attribution reaches only a
> fraction of the achievable ceiling and clears a usefulness bar at no unit size from 3 to 15
> demonstrations and no corpus size from 50 to 500. A design-based datamodel does clear it and
> genuinely attributes — it transfers across an independent re-partition — but its per-demonstration
> attributions agree across partitions at only ~0.5, which bounds what data selection can expect.

**One open item:** a pitfalls-only paper would need a **synthetic demonstration with known
ground-truth influence** to generalise beyond four self-inflicted cases. Cheap, no GPU. Worth doing
regardless — it strengthens the merged paper's methodology section too.

---

## 8. Current state, for the record

- Repo `3c92509`, clean, pushed. 30 commits across passes 9–16 plus the gallery pipeline.
- **No experiment running for this project.** The GPU work is finished; h200-1 is otherwise occupied.
- The demo-gallery website is **outside the repo** at `~/Desktop/first try/robotda-web`
  (self-contained; `open index.html`). Rebuild with `python3 build_site.py`.
- The advisor report is `docs/PASS9_10_REPORT.md`, current through all corrections.
- `if_repair/runs/` is **6.7 GB, gitignored, not on GitHub**. Stopping the instance preserves it;
  deleting the instance destroys it. Everything derived from it is committed.
