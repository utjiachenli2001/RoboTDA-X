"""PASS 9 -- the datamodel at CLUSTER grain. Zero additional GPU.

Pass 8 reversed every gradient-side correction and left exactly one estimator untouched: the
design-based datamodel, which ignores gradients entirely and regresses observed mask outcomes on
the mask's inclusion vector. At demo grain it is still the only estimator with a large replicable
out-of-sample advantage (pass 3: C5 ratio 0.882, Delta rho +0.358). Pass 8's HANDOFF thread 2 asks
whether that advantage survives the grain change or evaporates the way the corrections did. The
retrains it needs already exist -- campaign N's 1390 -- so the answer costs no GPU.

WHY THE DESIGN MATRIX IS 9 COLUMNS, NOT 135. At cluster grain the 15 demos of a cluster are
perfectly collinear: they enter and leave every mask together, so no procedure can identify their
coefficients separately. A 135-column ridge does not fail on this, it just splits each cluster's
coefficient evenly across its 15 demos -- and `P7.mask_pred`, which sums per-demo scores over the
demos a mask keeps, then sums those fifteenths straight back into the cluster coefficient. The
9-column fit is therefore the SAME estimator with the unidentifiable direction removed rather than
silently regularised, and it is the honest way to write it down.

THE THING THAT MAKES THIS COMPARISON INTERESTING. At demo grain the datamodel is badly
underdetermined -- 24 observations against 135 coefficients -- and that underdetermination is the
whole difficulty the estimator has to survive. At cluster grain it inverts: ~149 conditional masks
against 9 coefficients. Whatever this number turns out to be, it is not measuring the same
statistical problem, and the write-up must say so rather than reporting a ratio that looks like a
like-for-like improvement over 0.882.

THE TRAP THIS FILE EXISTS TO AVOID. The datamodel is OUTCOME-CONSUMING: it is fit on mask outcomes
and evaluated against mask outcomes. Scoring it in sample is not a weak result, it is a meaningless
one. Every prediction here is leave-one-mask-out, and alpha is selected INSIDE each fold -- a
single global alpha chosen on all masks leaks the held-out mask into its own prediction through the
regularisation path, which is the quiet version of the same error.

CONDITIONING. Following campaign N, masks are conditioned on target-in-mask. Inside that set the
target cluster's own column is constant, so the datamodel predicts C5's outcome from which OTHER
clusters are present. That is exactly the structure GradDot faces on the same masks (its prediction
also varies only through the kept non-target demos), so the comparison is like-for-like.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import functionals as F  # noqa: E402
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import p8_masks as P8M  # noqa: E402
from if_repair.confirm_mseries import ceiling, stratified_bootstrap, STATS  # noqa: E402
from if_repair.confirm_nseries import (achieved_depth, analysis_depth,  # noqa: E402
                                       conditional_masks)
from if_repair.datamodel import ALPHA_GRID  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

PRIMARY_STAT = "kendall_tau_b"
TARGET = "C5"
MODELS = {
    "ridge": lambda a: Ridge(alpha=a, fit_intercept=True),
    "lasso": lambda a: Lasso(alpha=a, fit_intercept=True, max_iter=50000),
}


def cluster_design(masks, clusters):
    """X (n_masks, n_clusters) binary inclusion, in the given mask order."""
    idx = {c: i for i, c in enumerate(clusters)}
    X = np.zeros((len(masks), len(clusters)))
    for r, m in enumerate(masks):
        for c in m["clusters"]:
            X[r, idx[c]] = 1.0
    return X


def _inner_alpha(X, y, model, folds=5, seed=0):
    """Pick alpha by out-of-fold MSE on the TRAINING masks only."""
    from sklearn.model_selection import KFold
    best, best_s = ALPHA_GRID[0], -np.inf
    kf = KFold(n_splits=min(folds, len(y)), shuffle=True, random_state=seed)
    for a in ALPHA_GRID:
        pred = np.full(len(y), np.nan)
        for tr, te in kf.split(X):
            if len(tr) < 3:
                continue
            pred[te] = MODELS[model](a).fit(X[tr], y[tr]).predict(X[te])
        ok = np.isfinite(pred)
        if ok.sum() < 4:
            continue
        s = -np.mean((pred[ok] - y[ok]) ** 2)
        if s > best_s:
            best, best_s = a, s
    return best


def loo_predict(X, y, model="ridge", nested_alpha=True, alpha=None, seed=0):
    """Leave-one-MASK-out predictions. alpha is refit inside every fold by default.

    Returns (pred, alphas). `nested_alpha=False` is provided only so a test can demonstrate that
    the global-alpha variant leaks; it is never the reported path.
    """
    n = len(y)
    pred = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    glob = alpha if alpha is not None else (
        None if nested_alpha else _inner_alpha(X, y, model, seed=seed))
    for i in range(n):
        tr = np.arange(n) != i
        a = glob if glob is not None else _inner_alpha(X[tr], y[tr], model, seed=seed)
        alphas[i] = a
        pred[i] = MODELS[model](a).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return pred, alphas


def permutation_control(X, y, st, model="ridge", n_perm=20, seed=0):
    """Shuffle the outcomes and re-run the WHOLE LOO pipeline. Must collapse to ~0.

    This is the decisive leakage test and the reason it is in the module rather than a notebook.
    If any part of the fold construction let a mask see its own outcome, the shuffled run would
    still predict, because the leak travels with the label. Shuffling WITHIN stratum keeps the
    training-set-size structure intact so the control isolates leakage rather than also destroying
    the design.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        yp = y.copy()
        for s in set(st):
            j = np.where(st == s)[0]
            yp[j] = y[j][rng.permutation(len(j))]
        pred, _ = loo_predict(X, yp, model=model)
        g = np.isfinite(pred)
        out.append(STATS[PRIMARY_STAT](pred[g], yp[g]))
    return np.array(out)


def evaluate(campaign="N", target=TARGET, models=("ridge", "lasso")):
    masks_all = P8M.manifest()["masks"]
    ms = conditional_masks(target, masks_all)
    raw_all = F.campaign_outcomes(campaign, "plain", targets=(target,))[target]
    raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in ms if m["mask_id"] in raw_all}

    depth, seeds = achieved_depth(raw, len(ms))
    if depth == 0:
        raise SystemExit(f"campaign {campaign}: no complete depth across {len(ms)} masks")
    d_even = analysis_depth(depth)
    if d_even == 0:
        raise SystemExit(f"campaign {campaign}: achieved depth {depth} has no even prefix")
    seeds = list(seeds)[:d_even]
    raw = {m: {s: v[s] for s in seeds} for m, v in raw.items() if all(s in v for s in seeds)}
    obs = F.seed_mean(raw)

    use = [m for m in ms if m["mask_id"] in obs]
    y = np.array([obs[m["mask_id"]] for m in use], float)
    pg = P7.mask_pred(P7._graddot("cached")[target], use)
    st = np.array([m["stratum"] for m in use])
    ok = np.isfinite(y) & np.isfinite(pg)
    y, pg, st, use = y[ok], pg[ok], st[ok], [m for m, k in zip(use, ok) if k]
    raw = {m["mask_id"]: raw[m["mask_id"]] for m in use}

    clusters = sorted({c for m in masks_all for c in m["clusters"]})
    X = cluster_design(use, clusters)

    rows = []
    for model in models:
        pred, alphas = loo_predict(X, y, model=model)
        good = np.isfinite(pred)
        perm = permutation_control(X, y, st, model=model)

        # ---- the |S| control. POOLED IS NOT THE HONEST NUMBER.
        #
        # |S| (how many clusters the mask keeps) sets the training-set size -- 60/75/90 demos --
        # and pass 8's own prereg says that "moves the outcome directly", which is why campaign N
        # treats |S| as a STRATUM and not a covariate to pool over. A pooled datamodel is free to
        # spend its 9 coefficients on reproducing that size effect, which is a property of the
        # design rather than an attribution of anything. GradDot cannot do this: its response to
        # |S| is whatever summing its fixed per-demo scores gives.
        #
        # Within a stratum |S| is constant, so the size effect is differenced out and what is left
        # is ordering within a fixed training-set size -- the quantity the LDS is supposed to be
        # about. If the pooled advantage is a size effect it collapses here; if it survives, the
        # datamodel really is attributing.
        for s in sorted(set(st[good])):
            j = good & (st == s)
            if j.sum() < 8:
                continue
            raw_s = {m["mask_id"]: raw[m["mask_id"]]
                     for m, k in zip(use, j) if k and m["mask_id"] in raw}
            for sname, fn in STATS.items():
                if np.std(pred[j]) == 0 or np.std(pg[j]) == 0:
                    continue
                c_s = ceiling(raw_s, fn)
                l_s, lg_s = fn(pred[j], y[j]), fn(pg[j], y[j])
                rows.append({
                    "estimator": f"datamodel_{model}", "grain": "cluster", "target": target,
                    "campaign": campaign, "statistic": sname, "primary": sname == PRIMARY_STAT,
                    "stratum": s, "n_masks": int(j.sum()), "depth": d_even,
                    "n_coef": X.shape[1], "lds": l_s, "graddot_lds": lg_s, "ceiling": c_s,
                    "ratio": l_s / c_s if np.isfinite(c_s) and c_s else np.nan,
                    "graddot_ratio": lg_s / c_s if np.isfinite(c_s) and c_s else np.nan,
                    "paired_delta": l_s - lg_s,
                    "note": "WITHIN-STRATUM: |S| constant, so the training-set-size effect is "
                            "differenced out. This is the honest attribution number.",
                })

        for sname, fn in STATS.items():
            c = ceiling(raw, fn)
            lds = fn(pred[good], y[good])
            lds_g = fn(pg[good], y[good])
            d0, pp, lo, hi = stratified_bootstrap(pred[good], pg[good], y[good], st[good], fn)
            rows.append({
                "estimator": f"datamodel_{model}", "grain": "cluster", "target": target,
                "campaign": campaign, "statistic": sname, "primary": sname == PRIMARY_STAT,
                "stratum": "POOLED (confounded by |S|, see note)",
                "n_masks": int(good.sum()), "depth": d_even, "depth_achieved": depth,
                "n_coef": X.shape[1], "alpha_median": float(np.nanmedian(alphas)),
                "alpha_n_distinct": int(len({float(a) for a in alphas if np.isfinite(a)})),
                "lds": lds, "graddot_lds": lds_g, "ceiling": c,
                "ratio": lds / c if np.isfinite(c) and c else np.nan,
                "graddot_ratio": lds_g / c if np.isfinite(c) and c else np.nan,
                # A ratio above 1 is NOT prima facie evidence of leakage. This repo's `ceiling`
                # is a Spearman-Brown RELIABILITY, and the largest correlation an oracle
                # predictor can have with a noisy observation of reliability r is sqrt(r), not r.
                # So the attainable maximum on this scale is 1/sqrt(ceiling) ~ 1.22 here, and a
                # ratio between 1 and 1.22 is a saturating estimator rather than a broken one.
                # `ratio_sqrt` is the normalisation that tops out at 1; both are reported because
                # every historical number in this repo uses the first one.
                "ratio_sqrt": lds / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan,
                "graddot_ratio_sqrt": lds_g / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan,
                "perm_control_mean": float(np.mean(perm)),
                "perm_control_max_abs": float(np.max(np.abs(perm))),
                "paired_delta": d0, "paired_p": pp, "ci_lo": lo, "ci_hi": hi,
                "note": "leave-one-mask-out, alpha refit inside every fold; outcome-consuming "
                        "estimator so the in-sample number is not reported at all",
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="N")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--out", default=os.path.join(RESULTS, "p9_datamodel_cluster.csv"))
    a = ap.parse_args()
    df = evaluate(campaign=a.campaign, target=a.target)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["estimator", "statistic", "stratum", "n_masks", "lds", "graddot_lds", "ceiling",
            "ratio", "graddot_ratio", "ratio_sqrt", "graddot_ratio_sqrt", "paired_delta",
            "ci_lo", "ci_hi", "perm_control_mean", "perm_control_max_abs"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[cols].to_string(index=False))
    print(f"\n[p9/datamodel] -> {a.out}")


if __name__ == "__main__":
    main()
