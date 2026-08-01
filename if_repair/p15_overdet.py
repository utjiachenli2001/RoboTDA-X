"""PASS 15 -- test the over-determination hypothesis AT FIXED GRAIN. Zero GPU.

WHY THIS EXISTS. BLOCKERS #52 was downgraded from result to hypothesis by the 2026-08-01 audit on an
identification argument: with exactly TWO grains, "coefficient count" is perfectly confounded with
every other property distinguishing k=5 fits from k=3 fits. The cross-grain design showed the transfer
loss follows the FIT rather than the target, which is real, but it cannot say WHICH property of the
fit is responsible. #54's attempt to corroborate via cross-partition agreement was retracted outright
(Pearson-only, reversing under Spearman).

So the hypothesis -- **over-determination buys within-sample performance and pays for it in
transfer** -- currently rests on one ~2-sigma observation (#50's within-vs-transfer asymmetry) and has
never been tested with coefficient count varied independently.

**IT CAN BE, at fixed grain, on data already on disk.** Hold the grain constant and vary the number of
masks the model is FIT on. Masks-per-coefficient then moves from ~1 to ~7 without changing the
grouping, the estimand, or anything else.

WHY THIS IS NOT THE LADDER FABLE KILLED IN PASS 11. That one compared the datamodel's LOO prediction
against GradDot's cached score as n fell, and the paired delta shrank MECHANICALLY because one arm
refits and the other does not. Here **both arms use the SAME fit** -- one fit on n masks of campaign O,
scored two ways:

    within    -> scored on campaign O's HELD-OUT masks (a different sample of the SAME partition)
    transfer  -> scored on campaign R's masks (a DIFFERENT partition)

The ratio `transfer / within` therefore divides out fit quality: a worse fit at small n hurts both
arms together. What survives in the ratio is how much of the fit is specific to its own mask ensemble.

THE PREDICTION, stated before looking. If over-determination causes ensemble-specific absorption, then
as n grows the fit has more capacity to absorb structure peculiar to campaign O's masks, so
**transfer/within should DECREASE with n**. Mechanical fit-quality degradation at small n pushes both
arms the same direction and cannot produce that. A flat ratio refutes the hypothesis; a rising ratio
refutes it more strongly.

REPORTED WITH ERROR BARS, BECAUSE THAT IS THE FAILURE THIS PROJECT KEEPS REPEATING. Every point is
many independent subsamples, and the trend is quoted with a slope and its CI rather than as a
percentage change between two endpoints.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import p9_grain as G  # noqa: E402
from if_repair import p9_masks as P9M  # noqa: E402
from if_repair import p10_masks2 as P10M  # noqa: E402
from if_repair.confirm_mseries import STATS  # noqa: E402
from if_repair.p11_transfer import _load, coefficients_to_demo_scores  # noqa: E402
from if_repair.p9_datamodel_cluster import _inner_alpha, MODELS  # noqa: E402
from if_repair.p10_datamodel_subcluster import group_design  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FN = STATS["kendall_tau_b"]
LADDER = (50, 100, 200, 300)      # fit-set sizes; 100+ O masks always held out
N_DRAWS = 12


def one_draw(o_use, o_y, r_use, r_y, k, n_fit, rng, model="ridge"):
    idx = rng.permutation(len(o_use))
    fit_i, held_i = idx[:n_fit], idx[n_fit:]
    gids = [g["group_id"] for g in G.groups(k)]
    X = group_design([o_use[i] for i in fit_i], gids)
    y = o_y[fit_i]
    alpha = _inner_alpha(X, y, model)
    coef = np.asarray(MODELS[model](alpha).fit(X, y).coef_, float)
    scores = coefficients_to_demo_scores(coef, gids, k, G.GROUP_SEED)

    held = [o_use[i] for i in held_i]
    p_within = P7.mask_pred(scores, held)
    p_transfer = P7.mask_pred(scores, r_use)
    w = FN(p_within, o_y[held_i])
    t = FN(p_transfer, r_y)
    return w, t


def evaluate(model="ridge"):
    rows = []
    for k in (3, 5):
        o_use, _, o_y, _ = _load("O", P9M.manifest()["masks"][str(k)])
        r_use, _, r_y, _ = _load("R", P10M.manifest()["masks"][str(k)])
        n_coef = len(G.groups(k))
        for n_fit in LADDER:
            rng = np.random.default_rng([20260801, k, n_fit])
            ws, ts = [], []
            for _ in range(N_DRAWS):
                w, t = one_draw(o_use, o_y, r_use, r_y, k, n_fit, rng, model=model)
                ws.append(w)
                ts.append(t)
            ws, ts = np.array(ws), np.array(ts)
            ratios = ts / ws
            rows.append({
                "grain": f"k={k}", "n_fit": n_fit, "n_coef": n_coef,
                "masks_per_coef": round(n_fit / n_coef, 2),
                "n_held_out": len(o_use) - n_fit, "n_draws": N_DRAWS,
                "within_mean": float(ws.mean()), "within_se": float(ws.std(ddof=1) / np.sqrt(len(ws))),
                "transfer_mean": float(ts.mean()),
                "transfer_se": float(ts.std(ddof=1) / np.sqrt(len(ts))),
                "ratio_mean": float(ratios.mean()),
                "ratio_se": float(ratios.std(ddof=1) / np.sqrt(len(ratios))),
            })
    return pd.DataFrame(rows)


def trend(df):
    """Slope of transfer/within against masks-per-coefficient, with a bootstrap CI.

    The hypothesis predicts a NEGATIVE slope. A CI containing zero refutes it as an explanation of
    anything, which is the outcome the entry must be prepared to report.
    """
    out = []
    rng = np.random.default_rng(0)
    for g, sub in df.groupby("grain"):
        x = sub["masks_per_coef"].to_numpy()
        y = sub["ratio_mean"].to_numpy()
        se = sub["ratio_se"].to_numpy()
        slope = np.polyfit(x, y, 1)[0]
        bs = np.empty(4000)
        for i in range(4000):
            bs[i] = np.polyfit(x, y + rng.normal(0, se), 1)[0]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append({"grain": g, "slope_per_mask_per_coef": slope, "ci_lo": lo, "ci_hi": hi,
                    "negative_as_predicted": bool(hi < 0),
                    "verdict": "supports over-determination" if hi < 0
                               else ("REFUTES (positive)" if lo > 0
                                     else "no trend -- hypothesis unsupported")})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p15_overdet.csv"))
    a = ap.parse_args()
    df = evaluate()
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["grain", "n_fit", "masks_per_coef", "n_held_out", "within_mean", "transfer_mean",
            "ratio_mean", "ratio_se"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[cols].to_string(index=False))
        print("\n  trend of transfer/within against masks-per-coefficient "
              "(hypothesis predicts NEGATIVE):")
        t = trend(df)
        t.to_csv(os.path.join(RESULTS, "p15_overdet_trend.csv"), index=False)
        print(t.to_string(index=False))
    print(f"\n[p15/overdet] -> {a.out}")


if __name__ == "__main__":
    main()
