"""P2 supplement: how seed-unstable is the ATTRIBUTION itself?

The Stage-E ensemble (seeds 201-210) and the Stage-B co-train ensemble (seeds 101-105 x 3
targets) are trained on the IDENTICAL 135-demo pool -- verified: their demos.json sets are equal.
They differ ONLY in training seed. Yet the preregistered P2 statistic (pooled Spearman of
predicted-benefit vs measured per-task margin) reads +0.163 on one and +0.345 on the other.

This quantifies that instability directly:
  (a) Spearman between the two ensembles' per-task predicted benefit (do they even agree?)
  (b) the sampling distribution of the pooled P2 statistic across DISJOINT 5-member sub-ensembles
      drawn from Stage-E's 10 members -- i.e. how much the headline number moves on seeds alone.

DESCRIPTIVE supplement. It changes no preregistered verdict (the primary remains Stage-E/IF).
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT  # noqa: E402
from lds import spearman  # noqa: E402

P2 = os.path.join(ROOT, "phase2")
ATTRS = ["IF", "TRAK", "TracIn"]


def pred_from(pm, members, attr):
    """Per-task predicted benefit = sum of OUTSIDER attributions, from a subset of members."""
    s = pm[(pm.attributor == attr) & (pm.member.isin(members))]
    agg = s.groupby(["task", "demo_id"], as_index=False).score.mean()
    meta = pm[["demo_id", "task"]].drop_duplicates()
    return agg, meta


def main():
    meas = pd.read_csv(f"{P2}/results/p2_measured_margins.csv")[["task_key", "margin_pts"]]
    full = {e: pd.read_parquet(f"{P2}/results/per_task_influence_{e}.parquet")
            for e in ("stageE", "stageB")}
    out = {"note": "Stage-E and Stage-B co-train are trained on the IDENTICAL 135-demo pool; "
                   "they differ only in training seed.",
           "agreement_between_ensembles": {}, "subensemble_spread": {}}

    # ---------------------------------------------------- (a) do the two ensembles agree at all?
    for attr in ATTRS:
        a = (full["stageE"][full["stageE"].attributor == attr]
             .groupby("task").apply(lambda g: g.loc[~g.is_insider, "score"].sum(),
                                    include_groups=False).rename("E"))
        b = (full["stageB"][full["stageB"].attributor == attr]
             .groupby("task").apply(lambda g: g.loc[~g.is_insider, "score"].sum(),
                                    include_groups=False).rename("B"))
        j = pd.concat([a, b], axis=1).dropna()
        out["agreement_between_ensembles"][attr] = {
            "spearman_predicted_benefit_E_vs_B": spearman(j.E.values, j.B.values),
            "n_tasks": int(len(j)),
        }

    # ------------------------- (b) spread of the headline statistic over disjoint 5-member halves
    pm = pd.read_parquet(f"{P2}/results/per_task_influence_stageE_per_member.parquet")
    members = sorted(pm.member.unique())
    assert len(members) == 10, members
    ins = full["stageE"][["task", "demo_id", "is_insider"]].drop_duplicates()

    for attr in ATTRS:
        vals, pairs = [], []
        seen = set()
        for half in itertools.combinations(members, 5):
            other = tuple(m for m in members if m not in half)
            key = frozenset([half, other])
            if key in seen:
                continue
            seen.add(key)
            row = []
            for sub in (half, other):
                s = pm[(pm.attributor == attr) & (pm.member.isin(sub))]
                agg = s.groupby(["task", "demo_id"], as_index=False).score.mean().merge(ins,
                                                                                        on=["task", "demo_id"])
                pred = (agg[~agg.is_insider].groupby("task").score.sum().rename("pred")
                        .reset_index().rename(columns={"task": "task_key"}))
                m = pred.merge(meas, on="task_key")
                row.append(spearman(m.pred.values, m.margin_pts.values))
            vals.extend(row)
            pairs.append(row)
        vals = [v for v in vals if np.isfinite(v)]
        out["subensemble_spread"][attr] = {
            "n_subensembles": len(vals),
            "pooled_rho_mean": float(np.mean(vals)), "pooled_rho_sd": float(np.std(vals)),
            "pooled_rho_min": float(np.min(vals)), "pooled_rho_max": float(np.max(vals)),
            "range": float(np.max(vals) - np.min(vals)),
            "frac_that_would_PASS_alpha05_uncorrected": float(np.mean([v > 0.323 for v in vals])),
        }

    json.dump(out, open(f"{P2}/results/p2_attribution_stability.json", "w"), indent=1, default=float)

    print("=" * 92)
    print("P2 SUPPLEMENT -- ATTRIBUTION SEED-STABILITY (descriptive; changes no verdict)")
    print("  Stage-E and Stage-B co-train use the IDENTICAL 135 demos; only the training seed differs.")
    print("=" * 92)
    print("  (a) do the two ensembles' per-task predictions agree?")
    for attr in ATTRS:
        e = out["agreement_between_ensembles"][attr]
        print(f"      {attr:7s} Spearman(E, B) over 27 tasks = "
              f"{e['spearman_predicted_benefit_E_vs_B']:+.3f}")
    print("\n  (b) spread of the POOLED P2 statistic over disjoint 5-member sub-ensembles of Stage-E")
    print(f"      {'attr':7s} {'mean':>7s} {'sd':>6s} {'min':>7s} {'max':>7s} {'range':>7s} "
          f"{'%>crit(.32)':>12s}")
    for attr in ATTRS:
        e = out["subensemble_spread"][attr]
        print(f"      {attr:7s} {e['pooled_rho_mean']:+7.3f} {e['pooled_rho_sd']:6.3f} "
              f"{e['pooled_rho_min']:+7.3f} {e['pooled_rho_max']:+7.3f} {e['range']:7.3f} "
              f"{100*e['frac_that_would_PASS_alpha05_uncorrected']:11.0f}%")
    print("=" * 92)
    print("READ: if a 5-seed swap moves the headline statistic by more than the effect being")
    print("      claimed, then a single ensemble's P2 'significance' is a seed lottery.")
    print("=" * 92)


if __name__ == "__main__":
    main()
