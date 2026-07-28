"""GPU ledger for pass 8. Same three readings as passes 3-7 (gpu_ledger.py): job_h double-counts
3-worker contention, solo_h is the honest single-job total, occupancy_h is wall-clock GPU-busy.

Pass 8 inherits pass 7's shape -- the finding that reframed the pass cost ZERO GPU -- but for a
different reason. Pass 7's free result came from re-scoring frozen configs against campaign
outcomes already on disk. Pass 8's came from `runs/stage_F`: 168 retrains the ORIGINAL project
paid for and never used for attribution, on which every frozen config is out-of-sample by
construction. Those 168 retrains are listed below as inherited, at zero cost to this pass, because
attributing them to pass 8 would overstate what pass 8 spent and understate what the repo already
had lying unused.

The GPU line item is campaign N, whose sizing follows directly from a cluster-grain fact rather
than the demo-grain rule: the cluster mask space is only C(9,4)+C(9,5)+C(9,6) = 336 subsets, of
which Stage F consumed 58 distinct. Campaign N takes all 278 that remain. With the mask axis
exhausted, depth is the only axis left to buy -- the inverse of BLOCKERS #29, and a consequence of
the combinatorics rather than a contradiction of it.
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

    add("W1 cluster-grain OOS scan of every frozen config on Stage F", 0.0, 0, 0.0, 0.0,
        "ZERO GPU beyond re-deriving cached estimator scores -- Stage F's 168 retrains "
        "were paid for by the original project and never used for attribution")
    add("W2 design: ceiling / allocation / statistic", 0.0, 0, 0.0, 0.0,
        "ZERO GPU -- resampled Stage F")
    add("p8_masks: complete enumeration of |S| in {4,5,6} minus Stage F", 0.0, 0, 0.0, 0.0,
        "ZERO GPU -- combinatorial construction, no sampling")

    s, n, span = campaign_seconds("N")
    if n:
        add("campaign N -- 278 fresh CLUSTER masks x depth 5 (seed-major)", s, n,
            SOLO_S["retrain"], span, "3 workers; the pass's entire GPU spend")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "gpu_ledger_pass8.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nPASS 8 TOTAL: {df.job_h.sum():.2f} job-h, {df.solo_h.sum():.2f} solo-GPU-h, "
          f"{df.occupancy_h.sum():.2f} occupancy-h  -> {out}")
    print("\nInherited at zero cost: Stage F, 168 retrains (72 cluster masks x seeds 301/302 "
          "+ 12 noise-ceiling masks x 303/304), built by the original project.")


if __name__ == "__main__":
    main()
