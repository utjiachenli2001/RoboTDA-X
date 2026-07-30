# HANDOFF — GPU / retrain follow-ups (all out of scope for this pass)

This pass was pure re-scoring of cached Gram matrices against cached outcomes: no training,
no GPU, no repo file edited. Everything below needs resources this pass did not have.
Ordered by what would actually change a conclusion.

## 1. Confirm the Hodges–Lehmann effect with fresh masks (highest value)

**What was seen.** Within the champion's unit-L2 normalization, Hodges–Lehmann aggregation
beat the mean on **all four** hold-out targets (Δratio +0.032…+0.067) and on C1
(0.5130 → 0.5678). Uniform sign on 4/4 unseen targets is the only consistent effect this
project found. It did **not** survive Bonferroni-8 (best cell C4, p=0.0076 vs α=0.003125).

**Why it cannot be settled with what is cached.** The 24 demo-grain masks are fixed and
already used. Re-testing the same masks cannot raise the evidence — only a fresh mask draw
can. At n=24 and Bonferroni-8 the critical ρ is ≈ 0.60, which no estimator here approaches;
either more masks or a smaller preregistered family is required.

**What to run.** Draw a fresh set of demo-grain masks with the same generator
(`results/demo_mask_manifest.json` was produced at seed/K recorded in its header), retrain
S=10 seeds per mask on the targets where HL led (C4 first, then C2/C9), and test the single
preregistered hypothesis "HL > mean within unit-L2". A family of 1 needs p < 0.025, which
C4's 0.0076 already clears — so this is a genuinely decidable experiment, not a fishing trip.

## 2. Common-random-seed ground truth

Every outcome here is a seed *mean* (or median) over independently seeded retrains, so the
mask-to-mask contrast carries the full between-seed variance. Re-running the mask sweep with
**common random seeds** across masks (same init and data order, only the mask differing)
removes that variance from the comparison and should raise the ceiling materially. The
ceilings are currently 0.75–0.96 (BC) and 0.76–0.90 (diffusion); every ratio in `RESULTS.md`
and `FINDINGS.md` is divided by one of those. This is the cheapest way to make all existing
numbers sharper without inventing a new estimator.

## 3. Task 4 — per-layer / subspace Φ (blocked, see BLOCKERS.md #3)

Not attempted: the repo contains **no checkpoints** (`find . -name '*.pt' | wc -l` → 0).
`runs/stage_E/ens_s201..210/` hold only `demos.json`, `outcomes.json`, `train_meta.json` and
markers. The cached `G` has already collapsed layers, so per-layer Φ cannot be recovered
from it — it must be recomputed from weights.

**Why this is now the most interesting blocked experiment.** The k-sweep found k\* ≈ 1:
the 135×135 Gram carries about one eigendirection distinguishable from random demo pairing.
That is a statement about the *dimensionality of Φ relative to the number of demos*, and the
direct remedy is to shrink Φ rather than to regularize the solve. Restricting Φ to the action
head or last block is exactly that experiment, and the k-sweep predicts it should move the
needle where truncation and damping did not.

**What to run.** With `runs/stage_E/ens_s*/final.pt` present:
`src/attribution.py:build_targets` + `demo_gradient`, restricted per layer group; rebuild
`G`, `K` per group; then reuse `if_repair/eval.py:evaluate` unchanged — the tier plumbing,
LDS, ceilings and CLI guards all work on any `(G, K)` of the same shape. Layer-ablation
table: which layer groups carry LDS signal.

## 4. Port to robomimic / Can

Everything here is LIBERO-goal, 135 demos, 9 clusters, one policy class plus a 5-member
diffusion arm. Whether "the identity is the best preconditioner" is a property of influence
functions at this demo count or a property of *this* dataset is not answerable from inside
this repo. A second task suite with a different demo count is the cleanest external check —
and the k\*≈1 diagnostic is cheap to compute there before committing to a full LDS sweep.

## 5. Re-run the diffusion arm with more members

The diffusion Gram has E=5. At that depth Hodges–Lehmann and the mean induce the same
ranking over the 24 masks (`FINDINGS.md`), so the diffusion comparison carries only one
effective estimator and cannot separate the two frozen candidates. E=10 would make the
cross-policy-class check informative rather than merely directional.

---

## Reproducing this pass

```bash
ln -s /path/to/RoboTDA-X /mnt/sdb/ljc/RoboTDA-X   # bootstrap.py hardcodes this ROOT
python -m venv .venv && . .venv/bin/activate
pip install numpy==2.2.6 scipy==1.18.0 pandas==2.3.3 pyarrow==24.0.0 \
            matplotlib==3.11.0 scikit-learn==1.9.0 pytest==9.1.1 PyYAML==6.0.3
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu  # import-only

python -m pytest if_repair/tests -q
python -m if_repair.run anchors
python -m if_repair.run dev     --config if_repair/configs/dev.yaml
python -m if_repair.aggregate            # Task 1 + tail diagnostic
python -m if_repair.spectral             # Task 2 k-sweep, spectrum null, damping
python -m if_repair.shrinkage            # Task 3
python -m if_repair.figs                 # figs/k_sweep.png, figs/spectrum.png
# hold-out: already computed once; re-running overwrites results/holdout_table.csv
python -m if_repair.run holdout   --config if_repair/configs/frozen.yaml
python -m if_repair.run diffusion --config if_repair/configs/frozen.yaml
```

Runtime is seconds to ~1 minute per step on CPU (135×135 × ≤20 members × 9 targets); the
k-sweep is the slowest at ~21 s. No GPU is used anywhere.

---

# Pass 2 follow-ups — B2/B3/B4 are now UNBLOCKED and cheap

Pass 1 listed Task 4 as blocked for want of checkpoints. That is resolved: checkpoints are
**regenerable** in 94 s/member via `if_repair/regen_ckpt.sh <member>`, which also writes the
5 intermediate `ckpt_0..4.pt` that TracIn needs. Only B1 was run (session limit, not budget:
**0.15 of 12 GPU-h used**). The remaining Phase-B items now cost well under an hour each.

**Read first:** regenerated weights are NOT the originals (BLOCKERS #6). Recompute every
baseline on whichever ensemble you use; never mix regenerated numbers with the E=20 cached
Gram rows.

## B2 — target-functional redesign (highest upside, ~2-3 h)

`if_repair/gradients.py:build_gram` already isolates the test-side term (`AT.build_targets`
→ `TG`). Swap the uniform held-out-L2 target for (a) ensemble-action-variance weighting, (b)
failure-divergence-state weighting, (c) rollout-visited states. **Gate each new functional
through its own split-half ceiling ≥0.4 BEFORE scoring any estimator against it** — a
misaligned proxy scored −0.48 in the sibling study, and B1 already shows this corpus will
happily produce large negative LDS (block_01/C7 = −0.142).

## B3 — KFAC / EK-FAC (~2-3 h)

The one genuinely different `H^-1`. Given B1's mechanism result, the specific prediction is
sharp: KFAC's *structure* is what should help, because it estimates curvature in a factored
space far smaller than 19.2M, i.e. it raises the effective `k*` the way layer restriction
did. Test it per-block as well as globally — if KFAC beats GradDot only where `k*` is already
> 1, that confirms the p/N account rather than the "better solve" account.

## B4 — TracIn density & grain (~1-2 h)

`ckpt_0..4.pt` per regenerated member are already on disk. TracIn is the diffusion winner
(0.479 vs GradDot 0.414), so this is the direct probe of the policy-class flip, and the
cheapest remaining item.

## B5 — common-random-seed estimand (~4-5 h, unchanged from pass 1)

Still the highest-value retrain: init is ~72 % of outcome variance, and every ratio in
RESULTS.md is divided by a ceiling inflated by it.

## The two results most worth defending

1. **Datamodel on C2** — the only hold-out pass in either pass (ratio 0.639, p = 0.00115).
   Confirm on fresh masks; note it needs outcome data, so it is a different estimator class.
2. **`k*` rises when Φ is restricted** (1 → 6-9). This is the mechanism claim. Strengthen it
   by sweeping Φ width directly (random parameter subsets of increasing size) and showing
   `k*` and the best-k LDS move together — that turns a 5-group observation into a curve.

---

# Pass 3 follow-ups — read FINDINGS.md §1 first

## What changed about the project's story (updated after B8 + the depth re-run)

Pass 2 ended "generality NO, unification NO, **mechanism YES**". Pass 3 **retracts the mechanism
claim as stated** (B6: k\* is a property of which subspace, not of p/N — two groups of identical
dimension differ 6 vs 1, and random subspaces never leave k\* ≈ 1) and **re-confirms it at the
right level** (H3: KFAC on `embed`, C5, 0.563 on fresh masks p=0.0032 — curvature from 92k frames,
not from 135 demos).

The bigger correction is methodological. The absolute LDS ratio is unresolvable at n=24 (B8:
sampling sd ~0.15 against a 0.5 bar). But the mask draw is a *shared* nuisance, so the **paired**
comparison "does X beat GradDot on these masks?" is decisive. Two estimators beat GradDot on C5 in
~100% of mask draws: **KFAC-on-embed (+0.48)** and the **datamodel (+0.56)**. The confirmatory
family reads 2 of 5 at matched 10-seed depth (H1 datamodel/C2, H3 KFAC/C5).

The most robust single result is the **datamodel**: C2 confirmed on two independent draws (0.639,
0.626) and C5 in 99.7% of B8 subsets.

## Do NOT re-run these

- **B5 (common random seeds).** Retired, not deferred. Stage G is already common-init (read
  `src/train.py`: the model is seeded before it is built), the batch-order main effect is
  0.1–0.5% of variance, and the binding noise is the mask×init interaction, which no seeding
  protocol removes. The three designs differ by ≤0.05 in reliability. See FINDINGS §6.
- **All THREE mask sets are now consumed dev data.** The archived 24 (G), the fresh 24 (H), and
  their pool (B8 read it). Both confirmatory families are spent (BLOCKERS #18/#19). A genuinely
  new confirmation needs a THIRD draw: `retrain.fresh_demo_masks(seed=<new>)`, ~5.8 solo-h at 10
  seeds.
- **Comparing seed depths.** Ratio-to-ceiling is not invariant to depth (BLOCKERS #17). Always
  match the depth of anything you compare; the archived protocol is 10.
- **Six concurrent trainers.** Slower than three (BLOCKERS #16).

## Open threads, ordered by what would change a conclusion

1. **Report paired differences, not absolute ratios.** B8 shows this is the only way to resolve an
   estimator comparison at n=24, and the machinery is in `b8_maskdraw.py`. Any new estimator
   should be run through it against GradDot on shared masks. A K=48 fresh draw at 10 seeds
   (~5.8 solo-h) would tighten the paired sd further, but B8 already turns the existing 48 masks
   into a decisive test — that is the cheaper first move. This
   is the one place where spending GPU on *more data* is not a violation of the prime directive
   but its precondition — the directive was about not buying stability with seeds, and this buys
   adjudicability with masks.
2. **Why `embed` and `block_00` and not the others?** B6 establishes that k\* tracks the subspace
   rather than its size and leaves the obvious question open. The cheap probe is *structured*
   subspaces of matched size — one attention head at a time, input-side vs output-side
   projections within a block — to see whether "input-side" or "early" is the predictor. All of
   it reuses the cached Φ; zero retrains.
3. **The target functional is the largest untried lever** (+0.161 on C5 dev, vs +0.130 for the
   whole curvature sweep) and per-frame held-out losses are now on disk, so a **new weighting
   costs zero GPU**: `functionals.weights()` plus `campaign_outcomes()` give it an outcome and a
   ceiling immediately. The obvious untried ones are behavioural rather than model-derived, since
   the two model-derived ones (`ens_var`, `fail_div`) did not transfer across targets.
4. **The intermediate-checkpoint effect.** Step 6400/8000 attributes better than the final
   weights on C1 (0.598 vs 0.506) while adjacent checkpoints swing by 0.42. Either there is a
   real "attribute from the middle of training" effect or the LDS is far noisier than n = 24
   suggests — and #1 says the second. A denser checkpoint grid on one member separates them for
   ~0.2 solo-h.
5. **Rollout-based functionals** remain blocked on LIBERO/robosuite (BLOCKERS #15). Success rate
   is still the only behaviourally meaningful outcome the study never predicts.

## What is cheap now that was not

- **Per-frame held-out losses for 456 retrains** (`runs/campaigns/{A,B,C}/*.npz`). Any frame
  weighting `w` yields an outcome `-Σ w_f l2_f / Σ w_f` and its own split-half ceiling for zero
  GPU. **Machine-local and gitignored (~40 MB); ~11 solo-h to rebuild.** The derived outcome
  table is committed at `results/campaign_outcomes.parquet` (153 KB) — enough for everything
  except defining a *new* weighting.
- **A split init/order trainer** (`retrain.train_one`) the repo's single `--seed` cannot express,
  with bit-for-bit legacy equivalence pinned by a test.
- **A regenerated diffusion ensemble with checkpoints** (`runs/regen_dp/`), which p17 never had.
  Machine-local; 0.46 solo-h to rebuild via `if_repair/regen_dp.sh`.
- **A ceiling recipe** (`functionals.split_half_ceiling`) that reproduces all nine archived p12
  ceilings to <1e−12, so any new outcome can be gated the same way the archived ones were.

## Invariants for the next session

- Run from `/mnt/sdb/ljc/RoboTDA-X`; `export CUDA_VISIBLE_DEVICES=0` before anything GPU.
- Three concurrent trainers.
- Never mix regenerated Gram numbers with the E=20 cached-Gram rows. **Outcomes are the
  exception** and the reason is measured (BLOCKERS #12): mask-level outcomes regenerate with
  Spearman 0.61–0.93.
- Any estimator fitted on mask outcomes goes through the leave-one-mask-out path.
- Report LDS, ceiling, ratio, p, pass. Never bare ρ. And after pass 3, never a single mask draw.

---

# HANDOFF -- passes 4-6 (for pass 7)

## The state pass 6 inherits
Pass 4 attacked demo attribution from four untried directions and found the FIRST gradient-side
estimators to beat GradDot out of sample (RelatIF/C5, TRAK-head/C2, exact-surrogate-LOO/C5). Pass 5
unified them into a two-parameter leverage family that beats the canonical GradDot_dmean OOS on 5
targets (per-target configs) and 2 with one config. Two fresh-draw confirmations were run: campaign
J (pass 4) and campaign K (pass 5). [J/K verdicts: TBD]

## Do NOT re-run
- W1 unlearning-LOO (ascent/finetune-forget/scrub-lite): screened negative, killed. A converged
  model barely moves in 200 SGD steps; the held-out effect is noise.
- W5 mid-training GradDot: single-checkpoint LDS swings +-1.0 between adjacent 400-step ckpts. There
  is no "attribute from the middle of training" effect; it is checkpoint noise.
- W6 win-condition-2 (does a gradient prior let the datamodel hit 24-mask LDS with 12 masks): no.
- Benchmarking any unit-L2-aggregated estimator against GradDot_unitL2 (BLOCKERS #23).

## What pass 7 should do (updated after J/K/L)
1. **Fix the measurement, not the estimator.** The binding constraint is n=24: even the confirmed
   C5 effect (+0.34 over GradDot on J) clears only the absolute bar, missing paired-p<0.05, and
   dev wins on 4 targets evaporated on fresh draws. The single highest-value spend is a
   48- or 72-mask confirmation draw (or pooling several fresh 24-draws preregistered together), which
   halves the paired sd and would actually adjudicate C5 (and tell whether C7 is real). More masks
   beats more estimators on this corpus.
2. **The C5 self-influence result is the one to defend.** RelatIF (K/G_dd) confirmed on J; [campaign
   L: TBD]. It plus its near-orthogonal exact-LOO partner (ensemble dev +0.298) is the project's
   strongest gradient attributor. Any pass-7 estimator work should build here, not chase new targets.
3. Wider / second-order exact LOO (P3, not run): the exact frozen-trunk surrogate beats GradDot on
   C5; extend the exact counterfactual to wider Phi (early blocks) or a Gauss-Newton influence on
   the true GMM head. Closed-form, ~0.5 GPU-h. Only worth it if #1 gives it the power to be tested.
4. Do NOT re-chase C2/C8/C7 as single-draw wins -- K showed C2/C8 are dev overfitting and C7 is
   underpowered. They return only with the higher-power protocol in #1.

## Machinery added in pass 4/5 (all under if_repair/)
b11_unlearn.py, b12_headloo.py (exact frozen-trunk LOO + feature extraction), b13_trak.py,
b14_rescoring.py (RelatIF), b15_ckptgrid.py, b16_hybrid.py, p1_leverage_family.py (the family +
generality scan), p2_ensemble.py, p4_why.py, confirm_jseries.py, confirm_kseries.py, campaigns J/K
wired in retrain.py, tests/test_jseries.py.

---

# HANDOFF -- pass 7 (for pass 8)

## The state pass 8 inherits

The question passes 4-6 opened is **closed to the resolution this corpus can support**. The C5
self-influence effect (RelatIF, K/G_dd) is **Delta rho = +0.06, 95% CI [-0.02, +0.14]** over 192
out-of-sample masks across 8 independent draws. Real in direction, small in size, about a quarter
of the +0.34 the discovery draw suggested, clearing the paired bar on the largest virgin draw
(p=0.031) but not when pooled (p=0.057), and clearing the absolute half-ceiling bar nowhere.

The write-up sentence this pass earned:

> A self-influence correction of GradDot improves demo-grain attribution on this corpus by
> Delta rho = +0.06, 95% CI [-0.02, +0.14], measured on 192 out-of-sample masks across 8
> independent draws -- an effect roughly a quarter of what its discovery draw suggested and not
> separable from zero at the sample sizes any comparable study uses. The binding constraint on
> demo-grain TDA benchmarks is the number of counterfactual retrains, not the estimator; we
> quantify the exchange rate at about five seeds per mask.

## Do NOT re-run

- **Everything in the passes 4-6 "do not re-run" list**, unchanged (W1 unlearning-LOO, W5
  mid-training GradDot, W6 win-condition-2, benchmarking against `GradDot_unitL2`, six concurrent
  trainers, robust aggregation / spectral sweeps / EK-FAC / KFAC subspaces / TracIn density, the
  redesigned target functionals).
- **The W2 duel design** (BLOCKERS #35). Killed by its own pilot: one demo out of 68 moves the
  outcome by 0.44 seed-noise sd, and resolving a duel sign would cost ~22 solo-h for a single n=18
  binomial test. Not viable at any budget this project can reach. The pairing premise DID hold
  (2.39x noise reduction) -- that part is settled and needs no re-measurement.
- **The archived 24 masks x 10 seeds allocation** (BLOCKERS #29). It is close to the worst
  allocation available. Any new campaign buys masks at depth 2-3.
- **W3 (wider / second-order exact LOO).** Dropped in `p7_prereg.md` §7: its gate was "+0.10 above
  the current C5 champion" and the champion is now measured at +0.06 with an interval touching
  zero, so the gate no longer identifies anything worth confirming.
- **Extending campaign M to chase significance.** `p7_prereg.md` §7 fixed score-once with no
  optional stopping, and p = 0.057 is exactly the situation that rule exists to protect. Adding
  masks until p < 0.05 would invalidate the result. A larger campaign must be preregistered fresh,
  as its own confirmation, with the sizing below.

## If pass 8 wants significance rather than an interval

It is cheap now, and the sizing is measured rather than guessed. Pooled paired sd at 192 masks is
~0.038 (kendall). For a one-sided alpha=0.05 test of the observed +0.060 at 80% power you need
sd ~0.029, i.e. **~330 masks**; from a standing start at depth 2 that is ~660 retrains ~ 17 solo-h.
Preregister it as a fresh single-hypothesis confirmation, score once. Note honestly that a result
which needs 330 masks to separate from zero is, for most purposes, the same as no effect.

## The three things worth carrying to another corpus

1. **Score every frozen config on every draw it is out-of-sample on, before building anything.**
   This project carried an unscored draw for a whole pass, and scoring it reversed the headline
   (BLOCKERS #28). Zero GPU.
2. **Measure the mask/seed exchange rate first.** On LIBERO-goal at 135 demos it is ~5:1 for
   masks, and the conventional depth-10 protocol wastes most of its GPU (BLOCKERS #29). The
   measurement costs nothing once one campaign exists.
3. **Choose the LDS statistic on reliability and noise, never on the contrast.** Worth ~33% of CI
   width here, and the choice is defensible precisely because it cannot see the hypothesis
   (BLOCKERS #30).

## Open threads, ordered by what would change a conclusion

1. **Port to a corpus with more demos.** BLOCKERS #33/#35 say the demo grain is signal-starved at
   135 demos from both the estimator side (the two estimators rank-correlate 0.547) and the outcome
   side (one demo = 0.44 seed-noise sd). Whether "self-influence correction helps a little" is a
   property of influence functions or of 135 demos is not answerable from inside this repo. A
   corpus with 500+ demos would make the per-demo signal measurable and is the single highest-value
   move left.
2. **The datamodel remains the only estimator with a large replicable OOS advantage**, and pass 7
   did not touch it. It is outcome-consuming, so it plays by different input rules, but it is the
   result to defend if the paper needs a positive.
3. **Cluster grain instead of demo grain.** Everything above says individual demos are too small a
   unit on this corpus. The cluster-grain masks (Stage F, 72 masks x 5 of 9 clusters) already exist
   in `masks.py` and were never used for attribution. A grain where the unit is 15 demos rather
   than 1 should sit far above the noise floor measured in #35.
4. **The absolute half-ceiling bar may be the wrong bar.** Nothing in seven passes has cleared it
   out of sample, while the paired bar has now been cleared once. If the bar is unreachable for
   every estimator anyone would try, it is not discriminating between hypotheses -- it is only
   measuring the ceiling. Worth an explicit argument in the paper rather than another attempt.

## Reproducing pass 7

```bash
python -m if_repair.p7_pooled_oos --stage validity    # the pooling rule, zero GPU
python -m if_repair.p7_pooled_oos --stage contrast    # the correction to passes 4-6
python -m if_repair.p7_design --stage allocation      # the exchange rate
python -m if_repair.p7_design --stage statistic       # primary statistic, hypothesis-blind
python -m if_repair.p7_design --stage bystat          # robustness across statistics
python -m if_repair.p7_duels --stage select           # duel manifest (frozen)
# GPU:
python -m if_repair.retrain --campaign M --worker {0,1,2} --nworkers 3   # 288 retrains
python -m if_repair.confirm_mseries                                      # score ONCE
python -m if_repair.retrain --campaign D --limit 48 --worker {0,1,2} --nworkers 3
python -m if_repair.p7_duels --stage pilot                               # kill rule
```

`P7_SLOW=1` enables the GPU-dependent surrogate reproduction tests. Everything else in
`pytest if_repair/tests -q` is CPU and runs in ~25 s.

---

# HANDOFF -- pass 8 (for pass 9)

## The state pass 9 inherits

Pass 8 changed the unit of attribution from one demonstration to a cluster of fifteen, and the
answer is unambiguous in both directions:

- **Plain GradDot_dmean, no correction at all, clears the absolute half-ceiling bar**: ratio
  **0.707** Kendall (0.834 Spearman), 149 out-of-sample masks, preregistered as a family of one
  and scored once. No estimator cleared that bar out of sample in seven passes at demo grain.
- **Every self-influence and leverage correction from passes 4-7 reverses**, to between -0.47 and
  -0.75 paired, with intervals nowhere near zero -- replicating across two independent draws and
  two different outcome pipelines.

The bar was never the problem and the estimator was never the problem. **The unit was too small.**

**The caveat that must travel with the headline:** a coarser grain is partly an easier prediction
problem. Normalising by the cluster-grain ceiling controls outcome noise but not degrees of
freedom. Pass 8 shows the measurement works at cluster grain and that demo-grain corrections do
not survive a grain with signal. It does **not** rescue per-demo attribution.

## Do NOT re-run

- **Everything on the passes 4-7 do-not-re-run lists**, unchanged.
- **The self-influence / leverage corrections, anywhere, without re-testing them at the grain the
  new corpus supports.** BLOCKERS #37. They are not a general improvement; on present evidence
  they are a demo-grain noise artifact.
- **A fresh cluster draw at |S| in {4,5,6}.** Campaign N consumed all 278 that Stage F left. The
  strata are exhausted. What remains unused is |S| in {2,3,7,8} -- 36 + 84 + 36 + 9 masks -- at
  training-set sizes (30/45/105/120 demos) far enough from the others that |S| stops being a
  controllable covariate and becomes the experiment.
- **An ODD seed depth in any prereg.** BLOCKERS #39: the split-half ceiling silently returns NaN.

## Open threads, ordered by what would change a conclusion

1. **Find the crossover. Pass 8 jumped from 1 demo to 15 and learned only that the floor is
   somewhere in between.** Intermediate grains -- groups of 3 and 5 demos, within or across
   clusters -- would locate the unit size at which attribution becomes measurable on this corpus.
   That number is the transferable quantity: it is what tells another project what unit to design
   its benchmark around, and nothing in the literature reports it. Mask space at sub-cluster grain
   is large again, so the demo-grain allocation rule (#29) applies rather than #38.

2. **The datamodel at cluster grain -- pass 8 did not touch it.** It remains the only estimator
   with a large replicable OOS advantage at demo grain (C5 ratio 0.882, Delta rho +0.358). Does
   that advantage survive the grain change, or does it evaporate the way every gradient-side
   correction just did? It is outcome-consuming, so it must be scored leave-one-mask-out, and
   campaign N's 1390 retrains make this **zero additional GPU**. This is the cheapest remaining
   experiment with a real chance of changing a conclusion.

3. **Why do the corrections reverse?** RelatIF divides by self-influence; a cluster prediction sums
   75 such scores, so a handful of demos with tiny `G_dd` could dominate the sum. If the reversal
   is a scaling pathology it is fixable and the correction may be salvageable; if the ranking
   itself is wrong the correction is simply false. Diagnosing it costs no GPU -- the scores and
   the outcomes are all on disk -- and it decides whether BLOCKERS #37 is a caveat or an epitaph.

4. **Port to a corpus with 500+ demos** (pass 7's thread #1, still open and still the biggest
   move). Pass 8 sharpens the question from "does attribution work?" to "at what unit size does
   attribution become measurable, as a function of corpus size?" -- which is answerable, and a
   better paper.

## Reproducing pass 8

```bash
export CUDA_VISIBLE_DEVICES=0          # REQUIRED, see BLOCKERS #40
python -m if_repair.p8_cluster_grain               # Stage F OOS scan, zero GPU
python -m if_repair.p8_design --stage all          # ceiling / allocation / statistic, zero GPU
python -m if_repair.p8_masks                       # 278 fresh masks, zero GPU
# GPU (15.75 h wall, 33.59 solo-GPU-h):
python -m if_repair.retrain --campaign N --worker {0,1,2} --nworkers 3    # 1390 retrains
python -m if_repair.confirm_nseries --i_understand_this_scores_once       # score ONCE
python -m if_repair.p8_figs
python -m if_repair.gpu_ledger_pass8
```

`pytest if_repair/tests -q` -> 116 passed, 3 skipped, ~25 s, all CPU.

## Where this project now lives (moved 2026-07-28)

**The box is `h200-1`, not h200-3.** `~/code/RoboTDA-X`, migrated after pass 8 and verified by
re-deriving the Stage F scan and diffing it bit-for-bit against the committed
`results/p8_stageF_oos.csv`. RoboTDA-X was then deleted from h200-3.

Three things that will otherwise cost an hour to rediscover:

1. **`export CUDA_VISIBLE_DEVICES=0` on every command** (BLOCKERS #40). Both h200-1 and h200-3 are
   1-GPU boxes and `src/bootstrap.py` pins `ALLOWED_GPUS = (4,5,6,7)` from the original 8-GPU
   machine. It respects an already-set value, so never edit bootstrap.py for one box.
2. **`if_repair/runs/` is gitignored and is NOT recoverable from GitHub.** 6.1G: the `regen`
   ensemble checkpoints every estimator score reads, the cached Grams, and the campaign A-N npz.
   Moving this project again means an rsync, not a clone. Between Nebius boxes it runs ~300 MB/s
   (6.5G in 21 s) with agent forwarding:
   `ssh -A <src> 'rsync -a if_repair/runs/ jli@<dst-ip>:~/code/RoboTDA-X/if_repair/runs/'`
   Everything else -- code, `runs/` stage outcomes, `data/proc`, `results/` including
   `campaign_outcomes.parquet` -- is in git and clones fine.
3. **`ssh -A h200-1` when pushing.** This host has no `ForwardAgent` in the Mac's ssh config, and a
   detached/nohup job has no agent at all, so pushes must be a foreground step.

---

# HANDOFF -- pass 9 (for pass 10)

## The state pass 10 inherits

Pass 9 set out to locate the grain crossover and instead corrected the result it was building on.

- **Campaign N's committed primary is substantially a training-set-SIZE effect.** GradDot is a fixed
  estimator, yet on outcomes shuffled within stratum it still scores Kendall 0.353 pooled against a
  real 0.475 -- which cannot be leakage. Within stratum the null collapses to ~0.000 and the
  half-ceiling bar is cleared in NO stratum (ratios 0.411 / 0.496 / 0.317). BLOCKERS #41.
- **At a fixed 75-demo training set, no grain clears the bar.** Campaign O, 800 fresh masks scored
  once: k=3 ratio 0.356 [0.180, 0.550], k=5 0.365 [0.204, 0.559], k=15 0.666 [0.194, 2.349]. All
  three beat their permutation nulls, so the attribution is real -- it is just weak.
- **The corrections are dead.** Neither rank/scale swap recovers them, and within stratum their
  ordering is actively anti-predictive. BLOCKERS #37 is an epitaph, #43 has the 2x2.
- **The datamodel is the last estimator standing.** It beats GradDot within every stratum by
  +0.31..+0.38 Kendall at cluster grain -- but at that grain it faces 149 masks against 9
  coefficients, against the demo-grain 24-vs-135 it earned its reputation on. Not like-for-like, and
  it has not been scored at sub-cluster grain on campaign O's masks (see thread 2 below).
- **Every ratio in this repo is on an inflated scale.** The ceiling is a Spearman-Brown reliability
  r; the attainable maximum is ~sqrt(r). BLOCKERS #42. Two ratios at different seed depths are not
  comparable, which is why campaign O is depth-matched.

## Do NOT re-run

- Everything on the passes 4-8 do-not-re-run lists, unchanged.
- **The self-influence / leverage corrections, anywhere.** #37 + #43. Not a caveat any more.
- **A fresh cluster draw at |S| in {4,5,6}.** Exhausted since pass 8.
- **Campaign O's masks.** Scored once, `results/confirm_oseries.csv` is frozen.
- **Any pooled-over-|S| absolute bar.** #41. It flatters whatever it is applied to; the same effect
  silently changed a verdict in #43.

## Open threads, ordered by what would change a conclusion

1. **Finish the k=15 rung for 66 retrains (~45 min). The cheapest experiment left in the project.**
   Conditional on C5 the 5of9 stratum has C(8,4) = 70 masks; campaign N holds 37 and Stage F holds
   the other 33. Re-running Stage F's 33 through `retrain.heldout_frame_losses` at depth 2 gives the
   complete 70-mask enumeration in ONE pipeline and roughly doubles n on the only rung whose interval
   is currently useless ([0.19, 2.35]). Until that is done the grain trend -- the pass's whole
   subject -- stays suggestive and unestablished. Do it first; it is 45 minutes.

2. **Score the datamodel at sub-cluster grain on campaign O -- zero additional GPU.** It is the only
   estimator with a surviving advantage, campaign O's 1600 retrains already exist, and the fixed
   training-set size means the result cannot be an |S| artifact. It must be leave-one-mask-out with
   alpha refit inside each fold, and note that at sub-cluster grain the design matrix regains real
   width (45 or 27 columns against 400 masks), so unlike cluster grain this is closer to the
   estimation problem the datamodel was actually celebrated for.

3. **Raise depth before adding masks.** Campaign O's ceilings are 0.376 / 0.386 at depth 2 -- the
   noisiest allocation available. #42 means low depth inflates the ratio, so the failure is robust,
   but a depth-4 re-read of the SAME 800 masks (another 1600 retrains, ~18 h) would tighten the
   ceiling and give a non-inflated read. BLOCKERS #29's masks-beat-seeds rule was measured for a
   PAIRED statistic; campaign O's primary is absolute, and the two do not have the same allocation
   optimum. That mismatch was noted in the pass-9 prereg as a stated proxy and is still unresolved.

4. **Ask whether the half-ceiling bar is the right standard at all.** Eight passes have now failed
   it, and pass 9 showed the one apparent success was a confound. Either the bar is the wrong
   standard for this corpus, or per-demo/per-small-group attribution genuinely does not work at 135
   demos. Distinguishing those is a better paper than another estimator. The `rho/sqrt(r)` scale
   (#42) is the honest place to argue it.

5. **Port to a corpus with 500+ demos** (pass 7 thread #1, pass 8 thread #4, still open and still the
   biggest move). Pass 9 sharpens the question again: not "does attribution work?" nor "at what unit
   size?", but "is the signal weak because the unit is small, or because the corpus is?" A 500-demo
   corpus separates those; nothing on this one can.

## Reproducing pass 9

```bash
export CUDA_VISIBLE_DEVICES=0          # REQUIRED, see BLOCKERS #40
python -m if_repair.p9_grain                       # the grain ladder, zero GPU
python -m if_repair.p9_stratum_control             # the |S| correction, zero GPU
python -m if_repair.p9_datamodel_cluster           # thread 2 at cluster grain, zero GPU
python -m if_repair.p9_why_reverse                 # the rank/scale 2x2, zero GPU
python -m if_repair.p9_masks                       # 800 masks at a fixed 75 demos, zero GPU
# GPU (18.2 h wall, 1600 retrains, 3 workers):
python -m if_repair.retrain --campaign O --worker {0,1,2} --nworkers 3
python -m if_repair.confirm_oseries --i_understand_this_scores_once     # score ONCE
```

`pytest if_repair/tests -q` -> 151 passed, 3 skipped, ~36 s, all CPU.

## The write-up sentence this pass earned

> Holding the training set fixed at 75 demonstrations, leave-one-out attribution on this corpus
> reaches 22-45% of the achievable ceiling at every unit size from 3 demos to 15, and clears a
> half-ceiling bar at none of them. The previously reported cluster-grain success -- 71% of ceiling
> -- was substantially an artifact of pooling over training-set size: a fixed estimator reproduces
> three quarters of that correlation on shuffled outcomes. Attribution at this corpus size is real,
> weak, and not obviously improved by coarsening the unit.

---

# HANDOFF -- pass 10 (for pass 11)

## The state pass 11 inherits

- **The grain question is closed on this corpus.** The k=15 conditional population is capped at 70
  and campaign P exhausted it. No design tightens that rung below a CI width of ~0.6 (#45).
- **The bar is reachable and discriminates** -- the datamodel clears it at both sub-cluster grains
  (0.640 / 0.674 attainable) while 10 gradient attempts across 5 designs clear it nowhere once size
  and depth are controlled (12-45%) (#48). Pass 7's HANDOFF #4 is answered.
- **Partition sensitivity is UNRESOLVED** at both grains: the two partitions differ by z = 1.23 at k=3 and z = 0.11 at k=5, so the originally-reported 42% movement is noise-consistent (#47, corrected).
- **The ceiling is noisy enough at these n to drive a ratio comparison on its own** (#46b).
- **No alpha remains at k=15, ever.** The unselected-upon masks are spent (#45 corollary).

## Do NOT re-run

- Everything on the passes 4-9 lists, unchanged.
- The self-influence / leverage corrections, anywhere (#37, #43).
- Any pooled-over-|S| absolute bar (#41).
- Any alpha-bearing test at k=15, or on the 33 Stage F masks (#28, #31, #45).
- Campaign O's or campaign R's masks -- both scored once, both result files frozen.

## Open threads, ordered by what would change a conclusion

1. **Port to a corpus of 500+ demonstrations.** No longer merely the biggest move -- for the grain
   question it is now the ONLY move (#45), and it is simultaneously the only clean discriminator for
   whether the gradient failure in #48 is a corpus-size limit or a limit of the approach. Everything
   else on this list is secondary to it.

2. ~~**Audit the historical ratio comparisons for the #46(b) ceiling effect.**~~ **DONE** --
   BLOCKERS #49. 5,535 comparisons across 89% of eligible files; 32 (1%) denominator-driven, all in
   pass-10's cross-design overview table or in pass-9 rows already known to be confounded. **Zero
   flagged anywhere in passes 1-8, and both load-bearing pass-9/10 conclusions are estimator-driven.**
   No committed conclusion needs amending. #46(b) is NOT repealed: future cross-subset ratio
   comparisons at n ~ 40-150 still need decomposing before they are believed.

3. ~~**A third partition at k=3.**~~ **DROPPED, and do not revive it at this scale.** The two draws
   differ by 1.23 sigma (#47 corrected), the data-consistent between-partition SD (~0.021) sits below
   a single partition's own SE (~0.032), and a six-partition test of sigma_b = 0 has ~10-15% power at
   that effect size. Resolving it needs ~20 partitions at 800 masks (~50,000 retrains). A third
   partition would return a near-certain null that says nothing.

4. ~~**Settle whether the datamodel attributes or fits the outcome surface.**~~ **DONE, zero GPU** --
   BLOCKERS #50. It ATTRIBUTES: fit on campaign O and scored on campaign R's independent partition it
   reaches ratio 0.781 (k=3) and 0.754 (k=5), still clearing the bar, at 4.0x and 2.4x GradDot on the
   same masks. Two qualifications: at k=5 the within-campaign figure overstates it (transfer loses 32%
   of the LDS, z=3.7; at k=3 the loss is undetectable), and coefficient stability across disjoint
   halves is 0.69 (k=3) / 0.90 (k=5) Pearson. The design the earlier HANDOFF said this needed -- one
   where the datamodel is not over-determined -- turned out to be unnecessary; two independent
   partitions answered it directly.

5. **Depth-4 re-read of campaign O / R.** Deferred twice on the argument that its direction is already
   measured (#42). Still true, still a footnote, still the right thing to run only if the box would
   otherwise idle.

## Reproducing pass 10

```bash
export CUDA_VISIBLE_DEVICES=0          # REQUIRED, see BLOCKERS #40
python -m if_repair.p10_k15_census --manifest-only   # the 70/37/33 gate, zero GPU
python -m if_repair.retrain --campaign P --worker {0,1,2} --nworkers 3   # 132 retrains, 1.5 h
python -m if_repair.p10_k15_census                   # the census read
python -m if_repair.p10_masks2                       # second partition, zero GPU
python -m if_repair.retrain --campaign R --worker {0,1,2} --nworkers 3   # 1600 retrains, 18.2 h
python -m if_repair.confirm_rseries --i_understand_this_scores_once      # score ONCE
python -m if_repair.p10_datamodel_subcluster         # zero GPU
python -m if_repair.p10_bar                          # every bar attempt on one scale
```

`pytest if_repair/tests -q` -> 151 passed, 3 skipped.

## The write-up sentence this pass earned

> On a 135-demonstration corpus, gradient-based training-data attribution reaches 12-45% of the
> achievable ceiling at every unit size from 3 demonstrations to 15, and clears a half-ceiling
> usefulness bar at none of them -- while a design-based datamodel that reads retraining outcomes
> directly clears the same bar comfortably. The bar is therefore reachable and does discriminate; what
> it discriminates against is the gradient approach at this corpus size. Whether that is a property of
> the corpus or of the approach is the question the next corpus has to answer.
