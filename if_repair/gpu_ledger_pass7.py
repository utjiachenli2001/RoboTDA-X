"""GPU ledger for pass 7. Same three readings as passes 3-6 (gpu_ledger.py): job_h double-counts
3-worker contention, solo_h is the honest single-job total, occupancy_h is wall-clock GPU-busy
time.

Pass 7 is unusual in this project: its central result (W0.1, the pooled out-of-sample analysis
that overturned the passes 4-6 headline) cost ZERO GPU. It was a re-scoring of frozen estimator
configs against outcomes already on disk. The GPU line items below are the two campaigns that
followed the W0 branch decision, and they are deliberately small -- W0.2 measured that the
archived 24-masks x 10-seeds allocation is close to the worst available, so campaign M buys 144
masks at depth 2 and spends FEWER retrains than any single prior campaign while returning a
95% CI roughly half as wide.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair.gpu_ledger import campaign_seconds, SOLO_S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main():
    rows = []

    def add(item, sec, n, solo_s, span=0.0, note=""):
        rows.append({"item": item, "n_jobs": n, "job_h": round(sec / 3600.0, 3),
                     "solo_h": round((n * solo_s) / 3600.0 if n else 0.0, 3),
                     "occupancy_h": round(span / 3600.0, 3), "note": note})

    add("W0.1 pooled out-of-sample analysis (the pass's main result)", 0.0, 0, 0.0, 0.0,
        "ZERO GPU -- re-scored frozen configs against outcomes already on disk")
    add("W0.2 allocation / statistic / baseline-instability study", 0.0, 0, 0.0, 0.0,
        "ZERO GPU -- resampled the existing 144 masks x 10 seeds")

    for c, label in (
            ("M", "campaign M -- 144 FRESH masks (6 sub-draws) x depth 2, the resolving campaign"),
            ("D", "campaign D -- W2 duels, one-demo-different mask pairs at matched seeds")):
        s, n, span = campaign_seconds(c)
        if n:
            add(label, s, n, SOLO_S["retrain"], span, "3 workers")

    # Estimator computes. The surrogate-LOO view needs a forward pass + head downdate per member
    # and has no disk cache, so W0.1 and every confirm run that touches it pays this again.
    add("W0.1 surrogate-LOO feature passes (b12, ~4 scoring runs x 5 members)", 0.0, 20,
        SOLO_S["phi_pass"], 0.0, "no disk cache; overlapped")
    add("regenE5 head-Gram builds (b1 build_ensemble, ~4 calls x 5 members)", 0.0, 20,
        SOLO_S["phi_pass"], 0.0, "overlapped")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "gpu_ledger_pass7.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\npass 7 total: job_h {df.job_h.sum():.2f}  solo_h {df.solo_h.sum():.2f}  "
          f"occupancy_h {df.occupancy_h.sum():.2f}")
    print("budget: 22 solo-h soft cap, 28 hard (p7 brief §0)")


if __name__ == "__main__":
    main()
