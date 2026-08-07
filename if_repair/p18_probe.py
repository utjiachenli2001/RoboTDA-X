"""PASS 18 -- the sensitivity probe. Diagnostic only; carries no alpha and scores no estimator.

THE QUESTION IT SETTLES. The variance pilot found that at rung 370, removing 50% of the corpus
moves held-out loss less than a random seed does (ceiling r = 0), while at rung 50 the same 50%
removal is clearly measurable (r = 0.73). Two different mechanisms could produce that, and they
imply completely different experiments:

  (a) RETAINED COUNT governs sensitivity. A model trained on 185 demos is near its data ceiling,
      so swapping which 185 barely matters. Then the ladder's fixed-FRACTION design is the
      problem, and an experiment holding the retained count fixed while growing the POOL would
      stay measurable at every rung.

  (b) POOL SIZE governs sensitivity through redundancy. A 370-demo pool is internally redundant
      in a way a 50-demo pool is not, so any subset resembles any other. Then no removal scheme
      rescues the top rung, and the size ladder is dead above some N regardless of design.

These are distinguishable with one measurement: at rung 370, retain only 25 demos -- the SAME
absolute training-set size as a rung-50 mask -- and see whether the outcome disperses like rung 50
(supporting (a)) or stays flat (supporting (b)).

WHY THIS IS NOT PART OF CAMPAIGN T. It draws unbalanced, non-complementary subsets at retention
levels the campaign does not use, purely to map where the outcome is sensitive. It writes to its
own directory, computes no influence score, and is reported as descriptive.
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
from if_repair import p18_corpus as C  # noqa: E402
from if_repair import p18_masks as M  # noqa: E402

D.add_repo_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "runs", "campaigns", "T_probe")

PROBE_SEED = 20260811
PROBE_MASKS = 8

# (rung, retained demos). The rung-50 cell reproduces the pilot as a positive control; the
# rung-370 cells sweep retention down to the rung-50 training-set size.
CELLS = ((370, 25), (370, 50), (370, 100))
CELLS_COND = ((370, 25),)   # conditioned mini-gate: campaign U's ACTUAL mask distribution


def probe_jobs(cells=CELLS, n_masks=PROBE_MASKS, conditioned=False):
    """Random GROUP subsets of a pool at a given retained COUNT.

    Retention is expressed in groups, not in demos-per-task, because that is what a campaign
    mask is: `keep` demos means keep/GROUP_SIZE whole groups, each spanning 5 distinct tasks.
    A rung-50 mask retains 5 groups = 25 demos spread over all 10 tasks but NOT 2.5 per task,
    so a task-balanced draw would not be the same object and the comparison would not be like
    for like -- which is the entire point of the probe.
    """
    out = []
    for sd in M.SEEDS:
        for rung, keep in cells:
            gs = C.groups(rung, "A")
            n_keep_groups = keep // C.GROUP_SIZE
            if n_keep_groups * C.GROUP_SIZE != keep:
                raise ValueError(f"retained {keep} is not a whole number of "
                                 f"{C.GROUP_SIZE}-demo groups")
            if n_keep_groups > len(gs):
                raise ValueError(f"pool {rung} has {len(gs)} groups, cannot keep "
                                 f"{n_keep_groups}")
            for k in range(n_masks):
                rng = np.random.default_rng([PROBE_SEED + (1 if conditioned else 0),
                                             rung, keep, k])
                # §2.2 coverage conditioning: 31.2% of unconditioned 5-group masks miss a task
                # entirely, and a missing task moves 10% of the eval bank hard. The campaign
                # rejects those, so a gate that does not is certifying a different distribution.
                for _try in range(10000):
                    sel = rng.permutation(len(gs))[:n_keep_groups]
                    demos = sorted(d for i in sel for d in gs[i]["demos"])
                    if not conditioned:
                        break
                    if len({d.split("/")[1] for d in demos}) == C.N_TASKS:
                        break
                else:
                    raise RuntimeError("coverage rejection failed to converge")
                sig = M.signature(demos)
                tag = "C" if conditioned else ""
                out.append({"run_id": f"TQ{tag}{rung}k{keep}_{sig}_i{sd}_o{sd}",
                            "mask_id": f"Q{tag}{rung}k{keep}_{k:02d}", "rung": rung, "keep": keep,
                            "sig": sig, "demos": demos, "seed_init": sd, "seed_order": sd,
                            "conditioned": bool(conditioned)})
    return out


def run(worker=0, nworkers=1, steps=None, conditioned=False):
    import time

    from if_repair import retrain as R
    from if_repair import p18_eval as EVT
    import train as TR

    cfg = TR.load_cfg()
    if steps:
        cfg["total_steps"] = steps
    os.makedirs(OUTDIR, exist_ok=True)
    J = probe_jobs(CELLS_COND if conditioned else CELLS, conditioned=conditioned)
    mine = [j for k, j in enumerate(J) if k % nworkers == worker]
    fidx = EVT.frame_index()
    print(f"[p18/probe] worker {worker}/{nworkers}: {len(mine)} of {len(J)} jobs, "
          f"steps={cfg['total_steps']}", flush=True)
    t0 = time.time()
    for k, j in enumerate(mine):
        r = R.run_job(j, cfg, fidx, OUTDIR, ev=EVT)
        if r == "skip":
            continue
        print(f"[p18/probe] {k + 1}/{len(mine)} {j['run_id']} keep={j['keep']} "
              f"loss={r['final_loss']:.4f} elapsed={(time.time() - t0) / 60:.1f}m", flush=True)
    print(f"[p18/probe] worker {worker} DONE ({(time.time() - t0) / 60:.1f} m)", flush=True)


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(OUTDIR, "*.npz"))):
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        head, sig, si, _ = meta["run_id"].split("_")
        cond = head[2:3] == "C"
        rung, keep = head[2:].lstrip("C").split("k")
        out.setdefault((int(rung), int(keep), cond), {}).setdefault(sig, {})[int(si[1:])] = \
            meta["outcomes"]["plain_loss"]
    return out


def analyse(drop_nonconverged=True):
    """Same variance decomposition as the pilot, per (rung, retained-count) cell.

    `drop_nonconverged` removes runs whose final_loss is a gross outlier -- ~1 run in 16 at large
    N fails to converge and single-handedly inflates the seed-noise term.
    """
    data, rows = load(), []
    for (rung, keep, cond), v in sorted(data.items()):
        per = {s: x for s, x in v.items() if len(x) == len(M.SEEDS)}
        if len(per) < 3:
            rows.append({"rung": rung, "keep": keep, "conditioned": cond,
                         "n_masks": len(per), "status": "insufficient"})
            continue
        means = np.array([np.mean(list(x.values())) for x in per.values()])
        diffs = np.array([x[M.SEEDS[0]] - x[M.SEEDS[1]] for x in per.values()])
        if drop_nonconverged and len(diffs) >= 5:
            # a single run 10x off the typical seed difference is a training failure, not noise
            med = np.median(np.abs(diffs - np.median(diffs)))
            good = np.abs(diffs - np.median(diffs)) <= max(8 * med, 1e-9)
            means, diffs = means[good], diffs[good]
        s2_seed = float(diffs.var(ddof=1) / 2.0)
        s2_obs = float(means.var(ddof=1))
        s2_mask = max(s2_obs - s2_seed / len(M.SEEDS), 0.0)
        rows.append({
            "rung": rung, "keep": keep, "conditioned": cond,
            "removed_frac": round(1 - keep / rung, 3),
            "n_masks": int(len(diffs)), "status": "ok",
            "outcome_mean": float(means.mean()),
            "sd_across_masks": float(np.sqrt(s2_obs)),
            "sd_across_seeds": float(np.sqrt(s2_seed)),
            "sd_signal": float(np.sqrt(s2_mask)),
            "implied_ceiling_r_depth2": float(s2_mask / (s2_mask + s2_seed / len(M.SEEDS)))
            if s2_obs > 0 else float("nan"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--conditioned", action="store_true")
    a = ap.parse_args()
    if a.run:
        run(a.worker, a.nworkers, a.steps, a.conditioned)
    if a.analyse:
        rows = analyse()
        print("[p18/probe] outcome sensitivity vs RETAINED COUNT at fixed pool size")
        for r in rows:
            if r["status"] != "ok":
                print(f"  rung {r['rung']} keep {r['keep']}: {r['status']}")
                continue
            print(f"  pool={r['rung']:3d} keep={r['keep']:3d} "
                  f"{'COND' if r.get('conditioned') else 'uncond'} "
                  f"(removed {r['removed_frac']:.0%})  "
                  f"outcome={r['outcome_mean']:.5f}  signal={r['sd_signal']:.5f}  "
                  f"noise={r['sd_across_seeds']:.5f}  r@d2={r['implied_ceiling_r_depth2']:.3f}")
        os.makedirs(RESULTS, exist_ok=True)
        p = os.path.join(RESULTS, "p18_probe.json")
        json.dump({"cells": [list(c) for c in CELLS], "n_masks": PROBE_MASKS,
                   "seeds": list(M.SEEDS), "rows": rows}, open(p, "w"), indent=1)
        print(f"[p18/probe] wrote {p}")


if __name__ == "__main__":
    main()
