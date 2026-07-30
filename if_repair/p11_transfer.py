"""PASS 11 -- does the datamodel ATTRIBUTE, or does it fit its own mask ensemble? Zero GPU.

Pass 10 found the design-based datamodel is the only estimator clearing the half-ceiling bar
(k=3 ratio 1.044, k=5 1.084, against GradDot's 0.356 / 0.365) and left the obvious question open:
with 400 masks against 45 or 27 coefficients it is a well-posed regression of outcomes on an
inclusion matrix, so is it attributing influence to demonstrations, or fitting the outcome surface of
the particular mask ensemble it was trained on?

THE TEST. Campaigns O and R are two INDEPENDENT partitions of the same 135 demos, sharing zero
groups, each with 400 frozen masks per grain. So:

    fit the datamodel on campaign O's masks  ->  45 group coefficients
    map each group coefficient to its demos  ->  135 per-demo scores
    aggregate over campaign R's masks        ->  400 predictions in R's group vocabulary
    correlate with campaign R's outcomes     ->  out-of-sample, out-of-PARTITION

**A method that attributes to demonstrations transfers across an arbitrary re-partition. A method
that fits its own mask ensemble does not.** The demo vocabulary is shared between campaigns; only the
grouping differs. Nothing about R was seen during the fit.

WHY THIS BEATS THE SUBSAMPLING LADDER IT REPLACES. The first version of pass 11 proposed walking
masks-per-coefficient down through 1.0 by subsampling, to see whether the datamodel's advantage
survives a narrow design. That cannot work: the datamodel's LOO prediction is refit on n-1 masks
while GradDot's per-mask prediction is a fixed cached score that does not depend on n at all. The
paired delta therefore shrinks **mechanically** as n falls, for any regression, informative or not.
A decaying curve would have been guaranteed by estimation theory rather than evidence of an artifact.

WHAT THE READS MEAN.

    transfer ~ within-campaign LOO   -> it attributes; the coefficients are about demos
    transfer ~ 0, or below GradDot   -> it fits its own mask ensemble; pass 10's headline is about
                                        a well-posed regression, not about attribution
    in between                       -> partially; report the gap, claim neither

GradDot is the reference and transfers perfectly by construction: its scores are per-demo and cached,
computed from gradients with no campaign involved, so its number on R is the same object either way.
That makes it the natural yardstick for what "transfers" looks like.

TWO THINGS THE MAPPING GETS RIGHT ON PURPOSE. Within a group the demos are perfectly collinear -- no
procedure can identify them separately -- so a group's coefficient is spread evenly over its k demos,
which is the only defensible split and is exactly what `P7.mask_pred` then re-sums. And the
intercept is dropped: every mask in both campaigns retains exactly 75 demos, so it is a constant shift
that cannot affect a rank correlation.
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
from if_repair import p10_masks2 as P10M  # noqa: E402
from if_repair.confirm_mseries import ceiling, STATS  # noqa: E402
from if_repair.confirm_nseries import achieved_depth, analysis_depth  # noqa: E402
from if_repair.p9_datamodel_cluster import _inner_alpha, MODELS, loo_predict  # noqa: E402
from if_repair.p10_datamodel_subcluster import group_design  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TARGET = "C5"
PRIMARY_STAT = "kendall_tau_b"
N_BOOT = 2000


def _load(campaign, masks, target=TARGET):
    raw_all = F.campaign_outcomes(campaign, "plain", targets=(target,))[target]
    raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in masks if m["mask_id"] in raw_all}
    d, seeds = achieved_depth(raw, len(masks))
    d = analysis_depth(d)
    seeds = list(seeds)[:d]
    raw = {m: {s: v[s] for s in seeds} for m, v in raw.items() if all(s in v for s in seeds)}
    obs = F.seed_mean(raw)
    use = [m for m in masks if m["mask_id"] in obs]
    y = np.array([obs[m["mask_id"]] for m in use], float)
    ok = np.isfinite(y)
    use = [m for m, q in zip(use, ok) if q]
    return use, {m["mask_id"]: raw[m["mask_id"]] for m in use}, y[ok], d


def coefficients_to_demo_scores(coef, group_ids, k, seed):
    """A group's coefficient spread evenly over its k demos -- the only identifiable split."""
    idx = {g["group_id"]: g for g in G.groups(k, seed=seed)}
    out = {}
    for c, gid in zip(coef, group_ids):
        for dd in idx[gid]["demos"]:
            out[dd] = out.get(dd, 0.0) + float(c) / k
    return out


def fit_on(campaign_masks, y, k, seed, model="ridge"):
    """Fit the datamodel on a whole campaign (no LOO -- this fit is never scored on its own data)."""
    gids = [g["group_id"] for g in G.groups(k, seed=seed)]
    X = group_design(campaign_masks, gids)
    alpha = _inner_alpha(X, y, model)
    fit = MODELS[model](alpha).fit(X, y)
    return np.asarray(fit.coef_, float), gids, alpha


def evaluate(k, model="ridge"):
    rows = []
    o_masks_all = P9M.manifest()["masks"][str(k)]
    r_masks_all = P10M.manifest()["masks"][str(k)]
    o_use, o_raw, o_y, o_d = _load("O", o_masks_all)
    r_use, r_raw, r_y, r_d = _load("R", r_masks_all)

    # ---- fit on O, transfer to R
    coef, gids, alpha = fit_on(o_use, o_y, k, P9M.GROUP_SEED if hasattr(P9M, "GROUP_SEED")
                               else G.GROUP_SEED, model=model)
    demo_scores = coefficients_to_demo_scores(coef, gids, k, G.GROUP_SEED)
    pred_transfer = P7.mask_pred(demo_scores, r_use)

    # ---- references on the SAME campaign-R masks
    pred_graddot = P7.mask_pred(P7._graddot("cached")[TARGET], r_use)
    r_gids = [g["group_id"] for g in G.groups(k, seed=P10M.GROUP_SEED2)]
    Xr = group_design(r_use, r_gids)
    pred_within, _ = loo_predict(Xr, r_y, model=model)

    rng = np.random.default_rng(0)
    for sname, fn in STATS.items():
        c = ceiling(r_raw, fn)
        for label, p, note in (
            ("datamodel fit on O -> scored on R (TRANSFER)", pred_transfer,
             "out-of-sample AND out-of-partition; nothing about R was seen"),
            ("datamodel fit on R -> scored on R (LOO, within)", pred_within,
             "pass 10's number; same mask ensemble it was fit on"),
            ("GradDot (cached, campaign-independent)", pred_graddot,
             "transfers by construction -- the yardstick"),
        ):
            g = np.isfinite(p) & np.isfinite(r_y)
            lds = fn(p[g], r_y[g])
            bs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                j = rng.integers(0, g.sum(), g.sum())
                bs[i] = fn(p[g][j], r_y[g][j])
            rows.append({
                "grain": f"k={k}", "arm": label, "statistic": sname,
                "primary": sname == PRIMARY_STAT, "n_masks": int(g.sum()), "depth": r_d,
                "lds": lds, "lds_se": float(bs.std(ddof=1)),
                "ceiling": c, "ratio": lds / c if np.isfinite(c) and c else np.nan,
                "ratio_sqrt": lds / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan,
                "alpha": alpha, "note": note,
            })
    return pd.DataFrame(rows)


def coefficient_stability(k, model="ridge", seed=0):
    """Split campaign O's masks in disjoint halves, fit each, correlate the coefficients.

    Predicting outcomes well with unstable coefficients is surface fitting: the coefficients are what
    the method claims are per-group influences, so if they do not replicate across halves of the same
    campaign they are not measuring a property of the groups.
    """
    o_masks_all = P9M.manifest()["masks"][str(k)]
    use, _, y, _ = _load("O", o_masks_all)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(use))
    h = len(use) // 2
    out = []
    for name, sel in (("half A", idx[:h]), ("half B", idx[h:2 * h])):
        ms = [use[i] for i in sel]
        coef, gids, _ = fit_on(ms, y[sel], k, G.GROUP_SEED, model=model)
        out.append(coef)
    a, b = out
    return {"grain": f"k={k}", "n_per_half": h, "n_coef": len(a),
            "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(STATS["spearman"](a, b)),
            "kendall": float(STATS["kendall_tau_b"](a, b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p11_transfer.csv"))
    a = ap.parse_args()
    df = pd.concat([evaluate(k) for k in (3, 5)], ignore_index=True)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["grain", "arm", "n_masks", "lds", "lds_se", "ceiling", "ratio", "ratio_sqrt"]
    with pd.option_context("display.width", 240, "display.max_columns", 30,
                           "display.max_colwidth", 46):
        print(df[df.primary][cols].to_string(index=False))
    print("\n  coefficient stability (disjoint halves of campaign O):")
    st = pd.DataFrame([coefficient_stability(k) for k in (3, 5)])
    st.to_csv(os.path.join(RESULTS, "p11_coef_stability.csv"), index=False)
    print(st.to_string(index=False))
    print(f"\n[p11/transfer] -> {a.out}")


if __name__ == "__main__":
    main()
