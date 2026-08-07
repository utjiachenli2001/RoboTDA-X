"""PASS 18 -- the variance pilot. The cheapest thing that can invalidate the top rung.

WHY THIS EXISTS. Stage C measured rollout success on this suite saturating hard with corpus size:
21% at 15 demos, 48% at 50, 94% at 490 (`results/stage_C_quantity.json`). Campaign T's outcome is
held-out l2 rather than success, so that does not transfer directly -- but if the outcome stops
MOVING when demos are removed at the top rung, then across-mask outcome variance collapses toward
the across-seed noise, the split-half ceiling `r` goes toward zero, and every ratio built on it
becomes unstable. A ladder that fell for that would report "attribution degrades with N" when the
truth was "the outcome stopped moving with N". That is a 2,400-retrain mistake, and 32 retrains
can rule it out.

WHY BOTH ENDS OF THE LADDER. A single rung's outcome SD in isolation means nothing -- there is no
scale to read it against. The decisive quantity is the RATIO of across-mask signal to across-seed
noise at rung 370 compared with the same thing at rung 50. So the pilot runs both ends.

WHAT IT DELIBERATELY DOES NOT DO. It computes no estimator, no influence score and no correlation.
It looks only at the dispersion of the OUTCOME. The preregistered hypothesis is about the ratio
trend, and nothing here touches it -- but the prereg records that this pilot ran, on which rungs,
and exactly what was inspected, so the top rung's retention is an auditable decision rather than a
silent one.

ISOLATION. Results go to `runs/campaigns/T_pilot`, NOT to `T`. Campaign T's job list is seed-major
so that every prefix is a complete balanced design; dropping 16 rung-370 runs into it early would
make a truncated analysis quietly unbalanced. The 32 retrains are duplicated later at a cost of
about 11 minutes, which is the cheaper side of that trade.
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
from if_repair import p18_masks as M  # noqa: E402

D.add_repo_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "runs", "campaigns", "T_pilot")

PILOT_RUNGS = (50, 370)
PILOT_MASKS = 8
PILOT_PARTITION = "A"


def pilot_jobs(rungs=None):
    """8 masks per rung, both seeds. Seed-major, so a half-finished pilot is still balanced."""
    out = []
    for sd in M.SEEDS:
        for n in (rungs or PILOT_RUNGS):
            for m in M.build(n, PILOT_PARTITION)[0][:PILOT_MASKS]:
                out.append({"run_id": f"TP{n}_{m['sig']}_i{sd}_o{sd}", "mask_id": m["mask_id"],
                            "rung": n, "partition": PILOT_PARTITION, "sig": m["sig"],
                            "demos": m["demos"], "seed_init": sd, "seed_order": sd})
    return out


def run(worker=0, nworkers=1, steps=None, rungs=None):
    import time

    import torch  # noqa: F401
    from if_repair import retrain as R
    from if_repair import p18_eval as EVT
    import train as TR

    cfg = TR.load_cfg()
    if steps:
        cfg["total_steps"] = steps
    os.makedirs(OUTDIR, exist_ok=True)
    J = pilot_jobs(rungs)
    mine = [j for k, j in enumerate(J) if k % nworkers == worker]
    fidx = EVT.frame_index()
    print(f"[p18/pilot] worker {worker}/{nworkers}: {len(mine)} of {len(J)} jobs, "
          f"steps={cfg['total_steps']}, bank={fidx['n']} frames", flush=True)
    t0 = time.time()
    for k, j in enumerate(mine):
        r = R.run_job(j, cfg, fidx, OUTDIR, ev=EVT)
        if r == "skip":
            continue
        print(f"[p18/pilot] {k + 1}/{len(mine)} {j['run_id']} rung={j['rung']} "
              f"loss={r['final_loss']:.4f} wall={r['wall_s']:.0f}s "
              f"elapsed={(time.time() - t0) / 60:.1f}m", flush=True)
    print(f"[p18/pilot] worker {worker} DONE ({(time.time() - t0) / 60:.1f} m)", flush=True)


def load():
    """-> {rung: {sig: {seed: plain_loss}}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(OUTDIR, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        rid = meta["run_id"]                      # TP<rung>_<sig>_i<seed>_o<seed>
        head, sig, si, _ = rid.split("_")
        rung = int(head[2:])
        out.setdefault(rung, {}).setdefault(sig, {})[int(si[1:])] = \
            meta["outcomes"]["plain_loss"]
    return out


def analyse():
    """Across-mask signal vs across-seed noise, per rung.

    The estimator of interest is the reliability of a DEPTH-2 mask mean:

        r_hat = s2_mask / (s2_mask + s2_seed / depth)

    which is what the split-half ceiling measures on the full campaign. `s2_seed` is estimated
    from the within-mask seed pairs (each pair contributes one difference), and `s2_mask` from
    the between-mask spread of the depth-2 means with the seed component removed.
    """
    data, rows = load(), []
    for rung in sorted(data):
        per = {s: v for s, v in data[rung].items() if len(v) == len(M.SEEDS)}
        if len(per) < 3:
            rows.append({"rung": rung, "n_masks": len(per), "status": "insufficient"})
            continue
        means = np.array([np.mean(list(v.values())) for v in per.values()])
        diffs = np.array([v[M.SEEDS[0]] - v[M.SEEDS[1]] for v in per.values()])
        s2_seed = float(diffs.var(ddof=1) / 2.0)          # var of ONE observation
        s2_obs = float(means.var(ddof=1))                 # var of the depth-2 mean
        s2_mask = max(s2_obs - s2_seed / len(M.SEEDS), 0.0)
        r_hat = s2_mask / (s2_mask + s2_seed / len(M.SEEDS)) if s2_obs > 0 else float("nan")
        rows.append({
            "rung": rung, "n_masks": len(per), "status": "ok",
            "outcome_mean": float(means.mean()),
            "sd_across_masks_depth2": float(np.sqrt(s2_obs)),
            "sd_across_seeds_single": float(np.sqrt(s2_seed)),
            "sd_mask_signal": float(np.sqrt(s2_mask)),
            "signal_to_noise": float(np.sqrt(s2_mask) / np.sqrt(s2_seed)) if s2_seed > 0
            else float("inf"),
            "implied_ceiling_r_depth2": float(r_hat),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--rungs", type=int, nargs="+", default=None,
                    help="override the piloted rungs (default 50 and 370)")
    a = ap.parse_args()
    if a.run:
        run(a.worker, a.nworkers, a.steps, a.rungs)
    if a.analyse:
        rows = analyse()
        print(f"[p18/pilot] {len(rows)} rungs, partition {PILOT_PARTITION}, "
              f"{PILOT_MASKS} masks x depth {len(M.SEEDS)}")
        for r in rows:
            if r["status"] != "ok":
                print(f"  rung {r['rung']:3d}: {r['status']} ({r['n_masks']} complete masks)")
                continue
            print(f"  rung {r['rung']:3d}: outcome={r['outcome_mean']:.5f}  "
                  f"sd_mask(signal)={r['sd_mask_signal']:.5f}  "
                  f"sd_seed(noise)={r['sd_across_seeds_single']:.5f}  "
                  f"S/N={r['signal_to_noise']:.2f}  "
                  f"implied r@depth2={r['implied_ceiling_r_depth2']:.3f}")
        os.makedirs(RESULTS, exist_ok=True)
        p = os.path.join(RESULTS, "p18_pilot.json")
        json.dump({"rungs": PILOT_RUNGS, "n_masks": PILOT_MASKS,
                   "partition": PILOT_PARTITION, "seeds": list(M.SEEDS), "rows": rows},
                  open(p, "w"), indent=1)
        print(f"[p18/pilot] wrote {p}")


if __name__ == "__main__":
    main()
