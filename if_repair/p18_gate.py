"""PASS 18 -- the campaign-U convergence gate (prereg §3.4). Outcome-blind.

WHY THIS EXISTS. `total_steps` is fixed at 8000 for every run, so a retrain can finish
under-converged; the diagnostic pilot saw roughly one run in sixteen land ~6 nats off its
neighbours, and a single such run inflated a pool's seed-noise estimate 8x and drove its measured
ceiling to zero. The ceiling is the denominator of everything campaign U reports, so an
unguarded outlier is not a nuisance, it is the result.

THE RULE OF RECORD, and why it is at PAIR level. Revision 3 flagged runs by deviation from their
own mask's median. At depth 2 the median of two runs is their midpoint and both deviate from it
identically, so the rule could not name a culprit -- it was written for depth 4 and the depth
reversal broke it. Revision 4 restated it on the seed PAIR:

    flag mask m  <=>  |Δ_m| > median(|Δ|) + 3 * MAD(|Δ|),  both computed WITHIN m's pool

where Δ_m is the difference of the mask's two runs' final TRAINING losses. Three properties matter:

  outcome-blind    training loss is not the outcome, so the rule cannot select on the thing it
                   protects.
  content-invariant |Δ| is a seed property at fixed mask content. A pool-median rule on the loss
                   LEVEL would instead flag masks that are merely hard to fit, and the resulting
                   censoring would correlate with the outcome -- biasing the across-mask signal,
                   which is the measured quantity.
  self-budgeting   under Gaussian seed noise |Δ| is HALF-normal (median 0.674σ, MAD 0.399σ), so
                   the threshold sits at 1.871σ and flags 6.1% -- matching the preregistered ~6%
                   re-run budget. Round 4 caught that dropping the median term puts it at 1.197σ
                   and flags 23.1%, which would both blow the budget and select a quarter of masks
                   for small |Δ|, truncating the noise distribution and inflating the ceiling.

DISPOSITION. A flagged mask has BOTH seeds re-run at the next reserve pair -- (4403,4404), then
(4405,4406) -- never one seed, which would select the survivor. A replacement pair is re-tested
under the same rule. If both reserve pairs are exhausted the mask is dropped whole and reported.
The per-pool threshold is computed ONCE from the first complete pass and frozen, so the rule
cannot chase its own tail as replacements arrive.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p18_campaign_u as U  # noqa: E402

D.add_repo_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CAMP = os.path.join(HERE, "runs", "campaigns", "U")
THRESH_PATH = os.path.join(RESULTS, "p18_gate_thresholds.json")


def load_runs(outdir=CAMP):
    """-> {sig: {seed: (plain_loss, final_train_loss)}}, and {sig: pool}."""
    runs, pool_of = {}, {}
    for f in sorted(glob.glob(os.path.join(outdir, "*.npz"))):
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        head, sig, si, _ = meta["run_id"].split("_")
        runs.setdefault(sig, {})[int(si[1:])] = (meta["outcomes"]["plain_loss"],
                                                 meta["final_loss"])
        pool_of[sig] = int(head[1:])
    return runs, pool_of


def _pairs(runs, pool_of, seeds):
    """-> {pool: {sig: |delta train loss|}} for masks complete at this seed pair."""
    out = {}
    for sig, byseed in runs.items():
        if seeds[0] in byseed and seeds[1] in byseed:
            d = abs(byseed[seeds[0]][1] - byseed[seeds[1]][1])
            out.setdefault(pool_of[sig], {})[sig] = d
    return out


def pass_is_complete(runs, seeds=U.SEEDS):
    """Every campaign-U mask has BOTH primary seeds on disk."""
    want = {m["sig"] for pool in U.N_MASKS for p in ("A", "B") for m in U.build(pool, p)[0]}
    have = {s for s, byseed in runs.items() if all(x in byseed for x in seeds)}
    return want <= have, len(want - have)


def thresholds(runs, pool_of, seeds=U.SEEDS, path=THRESH_PATH, force=False):
    """Per-pool flag threshold, computed ONCE from the first complete pass and frozen.

    GUARD: refuses to compute from a partial pass. The threshold is a per-pool location+scale of
    |Δtrain|; deriving it from whichever masks happened to finish first would freeze a statistic
    of the scheduler, and because it is frozen it would then never be corrected. A partial-pass
    freeze is exactly the kind of silent, unstated analytic choice this campaign has already had
    to withdraw one table over.
    """
    if os.path.exists(path) and not force:
        return {int(k): v for k, v in json.load(open(path)).items()}
    ok, missing = pass_is_complete(runs, seeds)
    if not ok and not force:
        raise RuntimeError(
            f"refusing to freeze gate thresholds from a PARTIAL pass: {missing} masks still "
            f"lack one or both primary seeds. Wait for the campaign, or pass force=True and "
            f"record why.")
    per = _pairs(runs, pool_of, seeds)
    out = {}
    for pool, d in per.items():
        v = np.array(list(d.values()))
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        out[pool] = {"median_abs_delta": med, "mad_abs_delta": mad,
                     "threshold": med + 3 * mad, "n_masks": int(v.size)}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump({str(k): v for k, v in out.items()}, open(path, "w"), indent=1)
    return out


def flagged(runs, pool_of, seeds=U.SEEDS, thr=None):
    thr = thr or thresholds(runs, pool_of, seeds)
    per, out = _pairs(runs, pool_of, seeds), []
    for pool, d in per.items():
        t = thr[pool]["threshold"]
        for sig, v in d.items():
            if v > t:
                out.append({"sig": sig, "pool": pool, "abs_delta": v, "threshold": t})
    return sorted(out, key=lambda r: (-r["abs_delta"], r["sig"]))


def rerun_jobs(runs, pool_of):
    """Job list for flagged masks at the next unused reserve pair. Both seeds, never one."""
    thr = thresholds(runs, pool_of)
    by_sig = {m["sig"]: m for pool in U.N_MASKS for p in ("A", "B")
              for m in U.build(pool, p)[0]}
    jobs, exhausted = [], []
    # a mask is re-tested at each reserve pair in turn; the pair in force is the first whose
    # runs are absent, and a mask whose every reserve pair is present and still flagged is done
    for f in flagged(runs, pool_of, thr=thr):
        sig = f["sig"]
        m = by_sig.get(sig)
        if m is None:
            continue
        placed = False
        for pair in U.RESERVE_PAIRS:
            if all(s in runs.get(sig, {}) for s in pair):
                continue                      # this reserve pair already ran; test the next
            for s in pair:
                if s in runs.get(sig, {}):
                    continue
                jobs.append({"run_id": f"U{f['pool']}_{sig}_i{s}_o{s}", "mask_id": m["mask_id"],
                             "pool": f["pool"], "partition": m["partition"], "sig": sig,
                             "demos": m["demos"], "seed_init": s, "seed_order": s})
            placed = True
            break
        if not placed:
            exhausted.append(sig)
    return jobs, exhausted


def report(runs=None, pool_of=None):
    if runs is None:
        runs, pool_of = load_runs()
    thr = thresholds(runs, pool_of)
    fl = flagged(runs, pool_of, thr=thr)
    per = _pairs(runs, pool_of, U.SEEDS)
    rows = []
    for pool in sorted(per):
        n = len(per[pool])
        k = sum(1 for f in fl if f["pool"] == pool)
        rows.append({"pool": pool, "n_masks": n, "flagged": k,
                     "flag_rate": round(k / n, 4) if n else 0.0,
                     "threshold": round(thr[pool]["threshold"], 4),
                     "median_abs_delta": round(thr[pool]["median_abs_delta"], 4),
                     "mad_abs_delta": round(thr[pool]["mad_abs_delta"], 4)})
    return rows, fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--emit-reruns", action="store_true")
    ap.add_argument("--force-thresholds", action="store_true")
    a = ap.parse_args()
    runs, pool_of = load_runs()
    if a.force_thresholds:
        thresholds(runs, pool_of, force=True)
    rows, fl = report(runs, pool_of)
    print(f"[p18/gate] {len(runs)} distinct training sets on disk; rule of record: "
          f"|Δtrain| > median + 3*MAD, within pool, frozen from the first complete pass")
    for r in rows:
        print(f"  pool {r['pool']:3d}: {r['n_masks']:5d} masks  flagged {r['flagged']:4d} "
              f"({r['flag_rate']:.2%})  thr={r['threshold']:.4f} "
              f"(med {r['median_abs_delta']:.4f} + 3x MAD {r['mad_abs_delta']:.4f})")
    print(f"[p18/gate] total flagged: {len(fl)} (preregistered budget ~6%)")
    if a.emit_reruns:
        jobs, exhausted = rerun_jobs(runs, pool_of)
        p = os.path.join(RESULTS, "p18_gate_reruns.json")
        json.dump({"jobs": jobs, "exhausted": exhausted}, open(p, "w"), indent=1)
        print(f"[p18/gate] {len(jobs)} re-run jobs -> {p}"
              f"{f'; {len(exhausted)} masks exhausted their reserves' if exhausted else ''}")


if __name__ == "__main__":
    main()
