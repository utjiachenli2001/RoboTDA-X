"""PASS 9 -- WHY do the passes-4-to-7 corrections reverse at cluster grain? Zero GPU.

BLOCKERS #37 records that every self-influence / leverage correction flips sign at cluster grain,
paired against GradDot, to between -0.47 and -0.75 with intervals nowhere near zero. It does not
say why, and the difference matters: if the reversal is a SCALING pathology the correction is
salvageable and #37 is a caveat; if the RANKING itself is wrong the correction is false and #37 is
an epitaph.

The mechanism under suspicion. RelatIF divides by self-influence, `K / G_dd`. A cluster-grain
prediction is `mask_pred` = the SUM of per-demo scores over the 75 demos a mask keeps. A handful of
demos with tiny `G_dd` therefore produce enormous scores that can dominate the sum, so the cluster
prediction could be driven by a few denominator outliers rather than by the ordering the estimator
believes in. At demo grain this never surfaced: one demo is one score, and a heavy tail in the
marginal distribution does not matter to a rank correlation over single demos. Summation is what
exposes it.

THE DECOMPOSITION. An estimator contributes two separable things to a summed prediction: the ORDER
it puts demos in, and the MARGINAL DISTRIBUTION of values it spreads over that order. Rank
transforms let those be swapped independently, which turns "scaling or ranking?" into a 2x2 anyone
can read:

    as_is                 est order,   est scale     -- the reversal as reported
    est_order_base_scale  est order,   GradDot scale -- keeps the ranking, kills the heavy tail
    base_order_est_scale  GradDot order, est scale   -- keeps the tail, uses a ranking known to work
    base                  GradDot order, GradDot scale

Reading it:

  * If `est_order_base_scale` recovers to roughly `base`, the correction's ORDER is fine and its
    heavy-tailed marginal is what destroys the sum. Scaling pathology -- fixable, and #37 is a
    caveat about aggregation rather than about the correction.
  * If `est_order_base_scale` still reverses, the ordering itself is wrong at this grain and no
    rescaling saves it. #37 is an epitaph.
  * If `base_order_est_scale` also collapses, that independently confirms the marginal is toxic,
    because it breaks an ordering that is known to work on these very masks.

Both arms are reported. They are not mutually exclusive and the informative outcome is that one of
them is clean.

NOTHING HERE IS INFERENTIAL. No alpha, no bar, no preregistered hypothesis. This is a diagnosis of
an already-established result, run on campaign N's existing retrains, and it is reported as such.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import p8_masks as P8M  # noqa: E402
from if_repair.confirm_mseries import stratified_bootstrap, STATS  # noqa: E402
from if_repair.confirm_nseries import (achieved_depth, analysis_depth,  # noqa: E402
                                       conditional_masks)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PRIMARY_STAT = "kendall_tau_b"


def _vec(sc, ids):
    return np.array([sc.get(d, 0.0) for d in ids], float)


def swap_marginal(order_from, scale_from, ids):
    """Scores with `order_from`'s ranking and `scale_from`'s marginal distribution.

    Ties in the order source are broken by the source's own value order, which is stable, so the
    result is a pure re-expression of one estimator's ranking on another's value scale.
    """
    a, b = _vec(order_from, ids), _vec(scale_from, ids)
    rank = np.argsort(np.argsort(a, kind="stable"), kind="stable")
    vals = np.sort(b)
    return {d: float(vals[rank[i]]) for i, d in enumerate(ids)}


def concentration(sc, masks, top=5):
    """Median share of a mask's summed |contribution| coming from its `top` largest demos.

    A summed predictor whose value is set by 5 of 75 demos is not aggregating, it is sampling an
    outlier. GradDot supplies the reference for what a well-behaved share looks like on the same
    masks, so this is read as a contrast and never as an absolute threshold.
    """
    shares = []
    for m in masks:
        v = np.abs(np.array([sc.get(d, 0.0) for d in m["demos"]], float))
        tot = v.sum()
        if not np.isfinite(tot) or tot <= 0:
            continue
        shares.append(np.sort(v)[-top:].sum() / tot)
    return float(np.median(shares)) if shares else np.nan


def evaluate(campaign="N"):
    ids = list(D.cache_for("bc_s10")["train_ids"])
    masks_all = P8M.manifest()["masks"]
    rows = []

    for cname, cfg in P7.CONFIGS.items():
        t = cfg["target"]
        est = cfg["scores"]()
        base = P7._graddot(cfg["baseline"])[t]

        raw_all = F.campaign_outcomes(campaign, "plain", targets=(t,))[t]
        ms = conditional_masks(t, masks_all)
        raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in ms if m["mask_id"] in raw_all}
        depth, seeds = achieved_depth(raw, len(ms))
        d_even = analysis_depth(depth)
        if d_even == 0:
            continue
        seeds = list(seeds)[:d_even]
        raw = {m: {s: v[s] for s in seeds} for m, v in raw.items() if all(s in v for s in seeds)}
        obs = F.seed_mean(raw)
        use = [m for m in ms if m["mask_id"] in obs]
        o = np.array([obs[m["mask_id"]] for m in use], float)
        st = np.array([m["stratum"] for m in use])

        arms = {
            "as_is": est,
            "est_order_base_scale": swap_marginal(est, base, ids),
            "base_order_est_scale": swap_marginal(base, est, ids),
            "base": base,
        }
        pg = P7.mask_pred(base, use)
        for arm, sc in arms.items():
            px = P7.mask_pred(sc, use)
            ok = np.isfinite(o) & np.isfinite(px) & np.isfinite(pg)
            if ok.sum() < 8 or np.std(px[ok]) == 0 or np.std(pg[ok]) == 0:
                continue
            for sname, fn in STATS.items():
                d0, pp, lo, hi = stratified_bootstrap(px[ok], pg[ok], o[ok], st[ok], fn)
                rows.append({
                    "config": cname, "label": cfg["label"], "target": t, "arm": arm,
                    "statistic": sname, "primary": sname == PRIMARY_STAT,
                    "n_masks": int(ok.sum()), "depth": d_even,
                    "lds": fn(px[ok], o[ok]), "graddot_lds": fn(pg[ok], o[ok]),
                    "paired_delta": d0, "paired_p": pp, "ci_lo": lo, "ci_hi": hi,
                    "top5_share": concentration(sc, use),
                })
    return pd.DataFrame(rows)


def verdict(df):
    """Read the 2x2 per config, on the primary statistic only."""
    out = []
    d = df[df.primary]
    for cname, g in d.groupby("config"):
        a = g.set_index("arm")
        if not {"as_is", "est_order_base_scale", "base_order_est_scale"} <= set(a.index):
            continue
        as_is = a.loc["as_is", "paired_delta"]
        keep_order = a.loc["est_order_base_scale", "paired_delta"]
        keep_scale = a.loc["base_order_est_scale", "paired_delta"]
        ci_lo = a.loc["est_order_base_scale", "ci_lo"]
        recovered = ci_lo > -0.05          # order-preserving arm is no longer clearly negative
        if recovered:
            v = "SCALING PATHOLOGY -- ranking survives, heavy tail destroys the sum"
        elif keep_order < -0.2:
            v = "RANKING ERROR -- order still reverses on a well-behaved scale"
        else:
            v = "MIXED / UNDETERMINED"
        out.append({"config": cname, "as_is_delta": as_is,
                    "est_order_base_scale_delta": keep_order,
                    "base_order_est_scale_delta": keep_scale,
                    "verdict": v,
                    "top5_share_as_is": a.loc["as_is", "top5_share"],
                    "top5_share_base": a.loc["base", "top5_share"]
                    if "base" in a.index else np.nan})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="N")
    ap.add_argument("--out", default=os.path.join(RESULTS, "p9_why_reverse.csv"))
    a = ap.parse_args()
    df = evaluate(campaign=a.campaign)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df[df.primary][["config", "arm", "n_masks", "lds", "graddot_lds", "paired_delta",
                              "ci_lo", "ci_hi", "top5_share"]].to_string(index=False))
        print()
        print(verdict(df).to_string(index=False))
    print(f"\n[p9/why-reverse] -> {a.out}")


if __name__ == "__main__":
    main()
