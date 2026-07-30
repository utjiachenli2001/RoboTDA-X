"""PASS 10 U4 -- every absolute-bar attempt on ONE honest scale. Zero GPU.

Nine passes have now been judged against the "half-ceiling bar": LDS >= 0.5 x noise ceiling. One
attempt appeared to clear it (pass 8's cluster-grain 0.707) and pass 9 showed that was substantially
a training-set-size artifact. That pattern is itself worth looking at directly, which is what this
module assembles.

THE SCALE MATTERS AND THE HISTORICAL ONE IS INFLATED. `confirm_mseries.ceiling` estimates a
Spearman-Brown RELIABILITY r. The largest correlation any predictor can have with an observation of
reliability r is about sqrt(r), not r. So the project's historical `ratio = rho/r` is inflated by
~1/sqrt(r) and tops out near 1/sqrt(r) rather than at 1 -- and the inflation GROWS as r falls, i.e.
at lower seed depth. Two ratios measured at different depths are therefore not comparable, which is
why this table carries depth and reports `rho/sqrt(r)` beside `rho/r`.

WHAT THIS MODULE DOES NOT DO. It does not decide between the two readings of the pattern:

  (a) the bar is mis-specified for a 135-demo corpus, or
  (b) small-unit attribution genuinely does not work at this corpus size.

Both predict exactly what is observed -- weak-positive signal everywhere, bar cleared nowhere -- so
the evidence in hand does not discriminate them, and pass 10 does not pretend otherwise. The honest
discriminator is corpus-size scaling, which is off-box. See the FINDINGS pass-10 section, where the
position is labelled a judgment rather than a finding.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
STAT = "kendall_tau_b"
BAR = 0.5


def _csv(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def assemble():
    """Every absolute-bar attempt this project has a committed number for."""
    rows = []

    def add(pas, label, n, depth, lds, ceil, note, size_controlled, outcome_consuming=False):
        ratio = lds / ceil if ceil else np.nan
        rows.append({
            "pass": pas, "attempt": label, "n_masks": n, "depth": depth,
            "lds": lds, "ceiling": ceil, "ratio_r": ratio,
            "ratio_sqrt_r": lds / np.sqrt(ceil) if ceil and ceil > 0 else np.nan,
            "clears_bar_on_ratio_r": bool(np.isfinite(ratio) and ratio >= BAR),
            "training_set_size_controlled": size_controlled,
            "outcome_consuming": outcome_consuming, "note": note,
        })

    n = _csv("confirm_nseries.csv")
    if n is not None:
        r = n[n.statistic == STAT].iloc[0]
        add(8, "cluster grain, POOLED over |S| (the published headline)", int(r["n_masks"]),
            int(r["depth"]), float(r["lds"]), float(r["ceiling"]),
            "pass 9: ~0.353 of the 0.475 survives an outcome shuffle -- a size effect", False)

    sc = _csv("p9_stratum_control.csv")
    if sc is not None:
        for _, r in sc[(sc.statistic == STAT) & (sc.scope.str.startswith("within"))].iterrows():
            add(9, f"cluster grain, {r['scope']}", int(r["n_masks"]), int(r["depth"]),
                float(r["lds"]), float(r["ceiling"]),
                "the same computation with |S| held constant", True)

    o = _csv("confirm_oseries.csv")
    if o is not None:
        for _, r in o[o.statistic == STAT].iterrows():
            add(9, f"sub-cluster {r['rung']}, fixed 75 demos (preregistered)", int(r["n_masks"]),
                int(r["depth"]), float(r["lds"]), float(r["ceiling"]),
                "campaign O; size channel removed by construction", True)

    c = _csv("p10_k15_census.csv")
    if c is not None:
        for _, r in c[(c.statistic == STAT) & (c.subset.str.startswith("70"))].iterrows():
            add(10, "k=15 CENSUS, complete conditional population", int(r["n_masks"]),
                int(r["depth"]), float(r["lds"]), float(r["ceiling"]),
                "all 70 masks; the population is exhausted", True)

    rr = _csv("confirm_rseries.csv")
    if rr is not None:
        for _, r in rr[rr.statistic == STAT].iterrows():
            add(10, f"sub-cluster {r['rung']}, SECOND partition (preregistered)",
                int(r["n_masks"]), int(r["depth"]), float(r["lds"]), float(r["ceiling"]),
                "campaign R; k=3 moved 44% from the first partition", True)

    dm = _csv("p10_datamodel_subcluster.csv")
    if dm is not None:
        for _, r in dm[dm.statistic == STAT].iterrows():
            add(10, f"DATAMODEL, sub-cluster {r['grain']}, fixed 75 demos", int(r["n_masks"]),
                int(r["depth"]), float(r["lds"]), float(r["ceiling"]),
                "reads outcomes directly; over-determined; descriptive, not preregistered",
                True, outcome_consuming=True)

    return pd.DataFrame(rows)


def summarise(df):
    grad = df[~df.outcome_consuming]
    dm = df[df.outcome_consuming]
    controlled = grad[grad.training_set_size_controlled]
    return {
        "attempts_total": len(df),
        "gradient_attempts": len(grad),
        "gradient_clearing_bar": int(grad.clears_bar_on_ratio_r.sum()),
        "gradient_clearing_bar_WITH_size_controlled": int(
            controlled.clears_bar_on_ratio_r.sum()),
        "gradient_ratio_sqrt_r_range": (round(float(controlled.ratio_sqrt_r.min()), 3),
                                        round(float(controlled.ratio_sqrt_r.max()), 3)),
        "datamodel_attempts": len(dm),
        "datamodel_clearing_bar": int(dm.clears_bar_on_ratio_r.sum()),
        "datamodel_ratio_sqrt_r_range": (round(float(dm.ratio_sqrt_r.min()), 3),
                                         round(float(dm.ratio_sqrt_r.max()), 3)) if len(dm) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p10_bar_attempts.csv"))
    a = ap.parse_args()
    df = assemble()
    df.to_csv(a.out, index=False)
    cols = ["pass", "attempt", "n_masks", "depth", "lds", "ceiling", "ratio_r", "ratio_sqrt_r",
            "clears_bar_on_ratio_r", "training_set_size_controlled", "outcome_consuming"]
    with pd.option_context("display.width", 260, "display.max_columns", 40,
                           "display.max_colwidth", 52):
        print(df[cols].to_string(index=False))
    print()
    for k, v in summarise(df).items():
        print(f"  {k}: {v}")
    print(f"\n[p10/bar] -> {a.out}")


if __name__ == "__main__":
    main()
