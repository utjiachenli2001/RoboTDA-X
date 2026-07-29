"""PASS 10 -- the datamodel at SUB-CLUSTER grain, on campaign O's existing retrains. Zero GPU.

WHAT THIS TESTS, AND WHAT THE PLAN'S FIRST VERSION WRONGLY CLAIMED IT TESTS.

v1 of the pass-10 plan argued that sub-cluster grain puts the datamodel "much closer to the
demo-grain 24-vs-135 regime it was celebrated for". That is backwards, and the error came from
HANDOFF thread 2 and was repeated without checking. As masks/coefficients:

    demo grain     24 / 135 = 0.18    UNDER-determined -- the ridge prior does the work
    cluster grain  149 /  9 = 16.6    over-determined
    sub-cluster    400 /  45 = 8.9    (k=3)   over-determined
    sub-cluster    400 /  27 = 14.8   (k=5)   over-determined

k=5 is close in conditioning to the cluster grain pass 9 declined to call like-for-like, and k=3 is
still over-determined by ~9x. These are the OPPOSITE regime from 24-vs-135, not nearer to it.

So the honest framing: this tests whether the datamodel's advantage survives non-trivial DESIGN WIDTH
(45 and 27 coefficients rather than 9) at a fixed training-set size where the #41 size channel cannot
operate. **A win here says nothing about the p >> n regime the demo-grain result lived in.** That
question needs a design with more coefficients than masks, which this corpus cannot supply at any
grain -- 400 masks is simply more than 45 groups.

The one trap that does carry over unchanged: the datamodel is OUTCOME-CONSUMING, so every prediction
is leave-one-mask-out with alpha refit INSIDE each fold. A single global alpha leaks the held-out mask
into its own prediction through the regularisation path.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import functionals as F  # noqa: E402
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import p9_grain as G  # noqa: E402
from if_repair import p9_masks as P9M  # noqa: E402
from if_repair.confirm_mseries import ceiling, stratified_bootstrap, STATS  # noqa: E402
from if_repair.confirm_nseries import achieved_depth, analysis_depth  # noqa: E402
from if_repair.p9_datamodel_cluster import loo_predict, permutation_control  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TARGET = P9M.TARGET
PRIMARY_STAT = "kendall_tau_b"


def group_design(masks, group_ids):
    """X (n_masks, n_groups) binary inclusion over the partition's groups."""
    idx = {g: i for i, g in enumerate(group_ids)}
    X = np.zeros((len(masks), len(group_ids)))
    for r, m in enumerate(masks):
        for g in m["groups"]:
            X[r, idx[g]] = 1.0
    return X


def evaluate(campaign="O", target=TARGET, models=("ridge",), n_perm=10):
    man = P9M.manifest()
    rows = []
    for k in man["grains"]:
        k = int(k)
        masks = man["masks"][str(k)]
        raw_all = F.campaign_outcomes(campaign, "plain", targets=(target,))[target]
        raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in masks if m["mask_id"] in raw_all}
        if not raw:
            continue
        depth, seeds = achieved_depth(raw, len(masks))
        d = analysis_depth(depth)
        if d == 0:
            continue
        seeds = list(seeds)[:d]
        raw = {m: {s: v[s] for s in seeds} for m, v in raw.items() if all(s in v for s in seeds)}
        obs = F.seed_mean(raw)
        use = [m for m in masks if m["mask_id"] in obs]
        y = np.array([obs[m["mask_id"]] for m in use], float)
        pg = P7.mask_pred(P7._graddot("cached")[target], use)
        st = np.array([m["stratum"] for m in use])
        ok = np.isfinite(y) & np.isfinite(pg)
        y, pg, st, use = y[ok], pg[ok], st[ok], [m for m, q in zip(use, ok) if q]
        raw = {m["mask_id"]: raw[m["mask_id"]] for m in use}

        gids = [g["group_id"] for g in G.groups(k)]
        X = group_design(use, gids)

        for model in models:
            pred, alphas = loo_predict(X, y, model=model)
            good = np.isfinite(pred)
            perm = permutation_control(X, y, st, model=model, n_perm=n_perm)
            for sname, fn in STATS.items():
                c = ceiling(raw, fn)
                lds = fn(pred[good], y[good])
                lds_g = fn(pg[good], y[good])
                d0, pp, lo, hi = stratified_bootstrap(pred[good], pg[good], y[good], st[good], fn)
                rows.append({
                    "estimator": f"datamodel_{model}", "grain": f"k={k}", "campaign": campaign,
                    "statistic": sname, "primary": sname == PRIMARY_STAT,
                    "n_masks": int(good.sum()), "n_coef": X.shape[1],
                    "masks_per_coef": round(X.shape[1] and int(good.sum()) / X.shape[1], 2),
                    "regime": "over-determined" if int(good.sum()) > X.shape[1]
                              else "under-determined",
                    "depth": d, "retained_demos": P9M.RETAINED_DEMOS,
                    "lds": lds, "graddot_lds": lds_g, "ceiling": c,
                    "ratio": lds / c if np.isfinite(c) and c else np.nan,
                    "graddot_ratio": lds_g / c if np.isfinite(c) and c else np.nan,
                    "ratio_sqrt": lds / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan,
                    "paired_delta": d0, "paired_p": pp, "ci_lo": lo, "ci_hi": hi,
                    "alpha_n_distinct": int(len({float(x) for x in alphas if np.isfinite(x)})),
                    "perm_control_mean": float(np.mean(perm)),
                    "perm_control_max_abs": float(np.max(np.abs(perm))),
                    "note": "LOO over masks, alpha refit inside every fold; fixed training-set size "
                            "so the #41 size channel cannot operate. Over-determined -- says nothing "
                            "about the p >> n regime the demo-grain result lived in.",
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="O")
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(RESULTS, "p10_datamodel_subcluster.csv"))
    a = ap.parse_args()
    df = evaluate(campaign=a.campaign, n_perm=a.n_perm)
    if df.empty:
        raise SystemExit("no outcomes at a complete even depth")
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["grain", "statistic", "n_masks", "n_coef", "masks_per_coef", "regime", "lds",
            "graddot_lds", "ratio", "graddot_ratio", "ratio_sqrt", "paired_delta", "ci_lo",
            "ci_hi", "perm_control_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df[df.primary][cols].to_string(index=False))
    print(f"\n[p10/datamodel-subcluster] -> {a.out}")


if __name__ == "__main__":
    main()
