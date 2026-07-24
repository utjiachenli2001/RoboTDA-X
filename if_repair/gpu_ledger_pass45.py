"""GPU ledger for passes 4-6. The retrain campaigns J/K/L dominate; the estimator computes
(unlearning pilot, surrogate feature passes, Gram builds) are minor and overlapped. Same three
readings as pass 3 (gpu_ledger.py): job_h double-counts 3-worker contention, solo_h is the honest
single-job total, occupancy_h is wall-clock GPU-busy time."""
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

    for c, label in (("J", "campaign J -- 24 FRESH J-series x 10 seeds (pass-4 confirmation)"),
                     ("K", "campaign K -- 24 FRESH K-series x 10 seeds (pass-5 confirmation)"),
                     ("L", "campaign L -- 24 FRESH L-series x 10 seeds (pass-6 C5 capstone)")):
        s, n, span = campaign_seconds(c)
        if n:
            add(label, s, n, SOLO_S["retrain"], span, "3 workers")

    # estimator computes (measured roughly; all overlapped with a campaign or each other)
    add("W1 unlearning pilot (1 member, 2 variants, k<=200)", 600, 1, 600, 0.0,
        "one-member screen, killed")
    add("W2/P2/P6/L surrogate feature passes (~6 x 5-member forward+LOO)", 0, 30,
        SOLO_S["phi_pass"], 0.0, "overlapped campaigns")
    add("regenE5 head-Gram builds (b1 build_ensemble, ~8 calls x 5 members)", 0, 40,
        SOLO_S["phi_pass"], 0.0, "overlapped campaigns")
    add("W5 mid-training ckpt grid (1 retrain + 20 head-Gram builds)", 0, 1,
        SOLO_S["retrain"], 0.0, "one member")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "gpu_ledger_pass456.csv"), index=False)
    print("=" * 110)
    print("GPU LEDGER -- passes 4-6")
    print("=" * 110)
    print(df.to_string(index=False))
    campaigns = df[df.item.str.startswith("campaign")]
    print(f"\nthree confirmation campaigns (J,K,L): solo {campaigns.solo_h.sum():.2f} h, "
          f"occupancy {campaigns.occupancy_h.sum():.2f} h (3 workers)")
    print(f"passes 4-6 total solo_h (incl. estimator computes): {df.solo_h.sum():.2f} h")


if __name__ == "__main__":
    main()
