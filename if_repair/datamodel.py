"""A4 -- design-based datamodel baseline. Zero gradients, zero GPU.

Every other estimator here scores a demo by what its GRADIENT does. This one ignores
gradients entirely and reads the outcomes directly: regress the observed mask outcome on
the mask's inclusion vector and take each demo's coefficient as its influence. That is the
RoboTDA-P style estimator applied to X's corpus -- a genuinely different attributor, and
the honest baseline for "is any of the gradient machinery earning its keep?"

    X : (24 masks x 135 demos) binary inclusion design
    y : (24,) observed neg_plain_loss for that mask
    beta = argmin ||y - X beta||^2 + alpha * pen(beta)      -> per-demo influence

The design is badly underdetermined (24 observations, 135 coefficients), which is the whole
difficulty and why regularization choice matters. alpha is selected by cross-validation
OVER MASKS -- never on the LDS being reported -- and the estimator is then scored by the
same demo-grain LDS as everything else.

HONEST CAVEAT, stated up front: the LDS evaluates prediction of held-out MASK outcomes, and
this estimator is fit on mask outcomes. Fitting and evaluation therefore share a data
source, which the gradient estimators do not. To keep the comparison meaningful, the headline
number is the LOO/K-fold CROSS-VALIDATED prediction (each mask predicted by a model that
never saw it); the in-sample fit is reported alongside only to show the gap.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, LeaveOneOut

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds, ALPHA  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def design_matrix(tier="bc_s10"):
    """-> X (n_masks, n_demos) binary, demo_ids, mask_ids."""
    Z = D.cache_for(tier)
    demo_ids = list(Z["train_ids"])
    idx = {d: i for i, d in enumerate(demo_ids)}
    masks = D.demo_masks()
    X = np.zeros((len(masks), len(demo_ids)))
    for r, m in enumerate(masks):
        for d in m["demos"]:
            if d in idx:
                X[r, idx[d]] = 1.0
    return X, demo_ids, [m["mask_id"] for m in masks]


def outcome_vector(tier, target, mask_ids):
    obs = D.outcomes(tier)[target]
    return np.array([obs.get(m, np.nan) for m in mask_ids], float)


MODELS = {
    "ridge": lambda a: Ridge(alpha=a, fit_intercept=True),
    "lasso": lambda a: Lasso(alpha=a, fit_intercept=True, max_iter=50000),
    "elasticnet": lambda a: ElasticNet(alpha=a, l1_ratio=0.5, fit_intercept=True,
                                       max_iter=50000),
}
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]


def cv_predict(X, y, model_name, alpha, folds=None, seed=0):
    """Out-of-fold predictions: every mask predicted by a fit that excluded it."""
    n = len(y)
    pred = np.full(n, np.nan)
    splitter = (LeaveOneOut() if folds in (None, "loo", n)
                else KFold(n_splits=int(folds), shuffle=True, random_state=seed))
    for tr, te in splitter.split(X):
        ok = np.isfinite(y[tr])
        if ok.sum() < 3:
            continue
        m = MODELS[model_name](alpha).fit(X[tr][ok], y[tr][ok])
        pred[te] = m.predict(X[te])
    return pred


def select_alpha(X, y, model_name, folds=5, seed=0):
    """Choose alpha by out-of-fold predictive R-like score over MASKS. Never uses LDS."""
    ok = np.isfinite(y)
    best, best_s = None, -np.inf
    for a in ALPHA_GRID:
        p = cv_predict(X[ok], y[ok], model_name, a, folds=folds, seed=seed)
        m = np.isfinite(p)
        if m.sum() < 4:
            continue
        s = -np.mean((p[m] - y[ok][m]) ** 2)          # negative CV MSE
        if s > best_s:
            best, best_s = a, s
    return best, best_s


def fit_datamodel(tier, target, model_name="ridge", alpha=None, folds=5, seed=0):
    """-> dict with per-demo coefficients (the influence scores) and CV diagnostics."""
    X, demo_ids, mask_ids = design_matrix(tier)
    y = outcome_vector(tier, target, mask_ids)
    ok = np.isfinite(y)
    if alpha is None:
        alpha, cvscore = select_alpha(X[ok], y[ok], model_name, folds=folds, seed=seed)
    else:
        cvscore = np.nan
    fit = MODELS[model_name](alpha).fit(X[ok], y[ok])
    coef = np.asarray(fit.coef_, float)
    # honesty diagnostics: out-of-fold mask-level prediction quality
    oof = cv_predict(X[ok], y[ok], model_name, alpha, folds=folds, seed=seed)
    m = np.isfinite(oof)
    return {
        "scores": {demo_ids[i]: float(coef[i]) for i in range(len(demo_ids))},
        "alpha": float(alpha), "model": model_name,
        "cv_neg_mse": float(cvscore),
        "oof_spearman_masks": float(spearman(oof[m], y[ok][m])),
        "insample_spearman_masks": float(spearman(fit.predict(X[ok]), y[ok])),
        "n_nonzero_coef": int(np.sum(np.abs(coef) > 1e-12)),
    }


def evaluate_datamodel(tier="bc_s10", targets=("C1", "C5"), models=("ridge", "lasso",
                                                                   "elasticnet"),
                       folds=5, seed=0) -> pd.DataFrame:
    """Headline `lds` is the OUT-OF-FOLD mask-prediction Spearman.

    CRITICAL -- why not the in-sample coefficients. demo_grain_lds forms its mask
    prediction as sum_{d in mask} score_d = X @ beta, which for a linear datamodel IS
    the model's own fitted prediction. Scoring the in-sample coefficients therefore
    measures how well a 135-parameter fit reproduces the 24 points it was fit on, and
    it duly returns rho up to 0.986 with ratio-to-ceiling ABOVE 1.0 -- impossible for
    any real estimator, since the ceiling is the reliability of the outcome itself.
    That number is a fit statistic, not a faithfulness measure.

    The honest analogue of what every gradient estimator is asked to do -- predict the
    outcome of a mask it has never seen -- is the out-of-fold prediction: each mask
    predicted by a model fit on the other masks only. That is `lds` here.
    `lds_insample_INVALID` is retained purely so the gap is visible.
    """
    gm, ceil = D.demo_masks(), D.ceilings(tier)
    X, demo_ids, mask_ids = design_matrix(tier)
    rows = []
    for mn in models:
        for t in targets:
            r = fit_datamodel(tier, t, mn, folds=folds, seed=seed)
            y = outcome_vector(tier, t, mask_ids)
            ok = np.isfinite(y)
            oof = cv_predict(X[ok], y[ok], mn, r["alpha"], folds=folds, seed=seed)
            m = np.isfinite(oof)
            rho = spearman(oof[m], y[ok][m])
            n = int(m.sum())
            p = spearman_p_onesided(rho, n)
            insample, _, _, _, _ = demo_grain_lds(r["scores"], gm,
                                                  D.outcomes(tier)[t])
            c = float(ceil[t])
            rows.append({"tier": tier, "estimator": f"datamodel_{mn}", "target": t,
                         "alpha": r["alpha"], "lds": float(rho), "ceiling": c,
                         "ratio": float(rho) / c, "bar": 0.5 * c, "p": float(p),
                         "n": n,
                         "passed": bool(np.isfinite(rho) and rho >= 0.5 * c
                                        and p < ALPHA),
                         "lds_insample_INVALID": float(insample),
                         "n_nonzero_coef": r["n_nonzero_coef"]})
    return pd.DataFrame(rows)


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df = evaluate_datamodel()
    print("=" * 108)
    print("A4 -- DESIGN-BASED DATAMODEL (no gradients, no GPU), tier=bc_s10, demo grain n=24")
    print("=" * 108)
    print(df[["estimator", "target", "alpha", "lds", "ceiling", "ratio", "p", "passed",
              "lds_insample_INVALID", "n_nonzero_coef"]].to_string(index=False))
    df.to_csv(os.path.join(HERE, "results", "a4_datamodel.csv"), index=False)
    print("\nNOTE: oof_spearman_masks is the datamodel's OUT-OF-FOLD ability to predict the")
    print("mask outcomes it was fit on. If it is near 0, the datamodel has not learned the")
    print("design at all and its per-demo coefficients are noise -- read the LDS with that")
    print("in mind. in-sample vs oof gap quantifies the overfit of a 135-coef fit to 24 obs.")


if __name__ == "__main__":
    main()
