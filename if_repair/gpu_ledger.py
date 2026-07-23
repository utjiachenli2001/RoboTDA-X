"""The GPU ledger, computed from run metadata rather than tallied by hand.

Concurrency makes "GPU hours" ambiguous, and the three defensible readings differ by 3x here, so
all three are reported rather than whichever one flatters the budget.

  job_h        sum of each job's own wall time. This is the pass-1/2 convention -- but those jobs
               ran one at a time, so it coincided with everything else there. Under 3 concurrent
               workers a retrain's own wall time inflates from 87 s to ~137-160 s, so this figure
               DOUBLE-COUNTS contention and is the largest of the three.
  solo_h       n_jobs x the measured single-job time (87 s for a BC retrain, 331 s for a
               diffusion member). This is the honest measure of how much WORK was done: what the
               same jobs would have cost run one at a time.
  occupancy_h  wall-clock span during which that stage's jobs were being produced. This is how
               long the GPU was actually busy, and stages that overlapped (B3/B4/B6/B7 ran
               alongside campaign A) are marked, since their occupancy is not additive.

solo_h is the figure to compare against the 12 h budget: it is neither inflated by contention
nor deflated by running things in parallel.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")


# measured single-job times on an otherwise idle GPU
SOLO_S = {"retrain": 87.0, "diffusion_member": 331.0, "phi_pass": 9.2, "dp_phi_pass": 146.0}


def _span(paths):
    """Wall-clock span over which a set of outputs was written."""
    ts = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    return (max(ts) - min(ts)) if len(ts) > 1 else 0.0


def campaign_seconds(campaign):
    tot, n = 0.0, 0
    files = glob.glob(os.path.join(RUNS, "campaigns", campaign, "*.npz"))
    for f in files:
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        tot += float(m["wall_s"])
        n += 1
    return tot, n, _span(files)


def regen_seconds(subdir, wall_file="wall.txt"):
    tot, n = 0.0, 0
    for d in sorted(glob.glob(os.path.join(RUNS, subdir, "*"))):
        mp = os.path.join(d, "train_meta.json")
        wp = os.path.join(d, wall_file)
        if os.path.exists(mp):
            tot += float(json.load(open(mp)).get("wall_s", 0.0))
            n += 1
        elif os.path.exists(wp):
            tot += float(open(wp).read().strip())
            n += 1
    return tot, n


def log_seconds(logname, pattern):
    """Last elapsed-seconds stamp of a build log, e.g. '(3648s)'."""
    import re
    p = os.path.join(RUNS, "logs", logname)
    if not os.path.exists(p):
        return 0.0
    last = 0.0
    for line in open(p):
        for m in re.finditer(pattern, line):
            last = max(last, float(m.group(1)))
    return last


def main():
    rows = []

    def add(item, sec, n, solo_s, span=0.0, note=""):
        rows.append({"item": item, "n_jobs": n, "job_h": sec / 3600.0,
                     "solo_h": (n * solo_s) / 3600.0 if n else 0.0,
                     "occupancy_h": span / 3600.0, "note": note})

    for c, label in (("A", "campaign A -- 24 archived masks x 10 seeds, per-frame losses"),
                     ("C", "campaign C -- 8 masks x 3 inits x 3 orders (B5)"),
                     ("B", "campaign B -- 24 FRESH H-series x 10 seeds (confirmatory)"),
                     ("I", "campaign I -- 24 FRESH I-series x 10 seeds (out-of-sample)")):
        s, n, span = campaign_seconds(c)
        if n:
            add(label, s, n, SOLO_S["retrain"], span, "3 workers")

    s, n = regen_seconds("regen_dp")
    if n:
        add("diffusion ensemble regeneration (B7)", s, n, SOLO_S["diffusion_member"],
            0.0, "overlapped campaign A")

    for label, logname, n, solo in (
            ("B7 diffusion TracIn cache (25 Phi passes)", "b7.log", 25,
             SOLO_S["dp_phi_pass"]),
            ("B4 TracIn cache (25 Phi passes)", "b4.log", 25, SOLO_S["phi_pass"]),
            ("B3 KFAC factors + Phi (5 members)", "b3.log", 5, SOLO_S["phi_pass"] * 3),
            ("B6 width cache (5 members x 28 subsets)", "b6.log", 5, SOLO_S["phi_pass"]),
            ("B2 Gram, 7 functionals x 9 clusters (5 members)", "b2_archived.log", 5,
             SOLO_S["phi_pass"] * 3)):
        sec = log_seconds(logname, r"\((\d+)s\)")
        add(label, sec, n, solo, sec, "overlapped campaign A")

    df = pd.DataFrame(rows)
    for c in ("job_h", "solo_h", "occupancy_h"):
        df[c] = df[c].round(3)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "gpu_ledger_pass3.csv"), index=False)
    print("=" * 110)
    print("GPU LEDGER -- pass 3.  job_h double-counts contention; solo_h is the work done; "
          "occupancy_h is GPU-busy time")
    print("=" * 110)
    print(df.to_string(index=False))
    seq = df[~df.note.str.contains("overlapped")].occupancy_h.sum()
    print(f"\npass-3  job_h {df.job_h.sum():.2f}   solo_h {df.solo_h.sum():.2f}   "
          f"occupancy (non-overlapping stages) {seq:.2f} h")
    print(f"pass-1 + pass-2 archived ledger: 0.15 job-h (those ran serially)")
    print(f"\nPROJECT TOTAL vs the 12 h budget, on the comparable measure: "
          f"{df.solo_h.sum() + 0.15:.2f} solo-h")


if __name__ == "__main__":
    main()
