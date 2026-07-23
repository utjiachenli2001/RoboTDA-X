"""Export the campaign outcome tables as a committable artifact.

The raw per-frame losses (runs/campaigns/*/*.npz, ~40 MB) are gitignored with the rest of
runs/, and regenerating them costs ~9.5 GPU-h. Everything downstream of them except the
ability to define a NEW frame weighting is captured by the per-(weighting, target, mask, seed)
outcome, which is a few hundred KB. That is what this writes.

To define a new weighting later you still need the per-frame losses, so keep them if the box
survives; if it does not, `retrain.py --campaign A` rebuilds them.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def export(campaign, weightings=F.WEIGHTINGS, targets=D.ALL_TARGETS):
    rows = []
    for w in weightings:
        raw = F.campaign_outcomes(campaign, w, targets=targets)
        for t, per_mask in raw.items():
            for m, per_seed in per_mask.items():
                for (si, so), v in per_seed.items():
                    rows.append({"campaign": campaign, "weighting": w, "target": t,
                                 "mask_id": m, "seed_init": si, "seed_order": so,
                                 "neg_weighted_loss": v})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaigns", default="A,B,C")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    frames, ceilings = [], []
    for c in a.campaigns.split(","):
        try:
            df = export(c)
        except Exception as e:                                  # noqa: BLE001
            print(f"[export] campaign {c}: {e}")
            continue
        frames.append(df)
        ct = F.ceiling_table("campaign", campaign=c)
        ct.insert(0, "campaign", c)
        ceilings.append(ct)
        print(f"[export] campaign {c}: {len(df)} outcome rows, "
              f"{df.mask_id.nunique()} masks, {len(df.groupby(['seed_init','seed_order']))} "
              f"seed cells, {df.weighting.nunique()} weightings")
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(os.path.join(RESULTS, "campaign_outcomes.parquet"), index=False)
    cc = pd.concat(ceilings, ignore_index=True)
    cc.to_csv(os.path.join(RESULTS, "campaign_ceilings.csv"), index=False)
    print(f"[export] wrote results/campaign_outcomes.parquet ({len(out)} rows) "
          f"and results/campaign_ceilings.csv ({len(cc)} rows)")


if __name__ == "__main__":
    main()
