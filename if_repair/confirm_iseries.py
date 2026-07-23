"""I-series confirmation -- the THIRD mask draw, preregistered, computed once.

By the time B8 ran, the G-series (dev) and H-series (pass-3 confirmatory) were BOTH consumed, and
B8 pooled them. So B8's headline -- KFAC-on-embed and the datamodel beat GradDot on C5 in ~100%
of mask draws -- is measured on masks the estimators have effectively been selected against. That
is strong evidence but not an out-of-sample confirmation.

Campaign I is a genuinely fresh 24-mask draw (Stage-G generator, seed 9973, disjoint from both
prior draws; `tests/test_iseries.py` pins it), retrained at 10 seeds. This file freezes the
hypotheses BEFORE campaign I exists and evaluates each exactly once.

TWO TESTS, because B8 taught us the absolute ratio is unresolvable at n=24 and the paired
difference is the right statistic:

  ABSOLUTE  (the project's historical bar): ratio-to-ceiling >= 0.5 and p < Bonferroni-alpha.
            Kept so this family is comparable to confirm3, and because a paired win with a
            negative absolute ratio would be a hollow victory.
  PAIRED    (the B8 statistic): does the estimator beat GradDot_dmean(ALL) on the SAME 24 masks?
            Reported as the sign of the difference and its one-sided significance via a
            bootstrap over masks. This is the claim B8 actually made, now on fresh masks.

Bonferroni is over the 3 preregistered hypotheses: alpha = 0.025 / 3 = 0.00833.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402
from if_repair import b8_maskdraw as B8  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import ALPHA  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Frozen before campaign I. Each names an estimator (built by b8_maskdraw.estimators() or the
# datamodel LOO path), a target, and the direction the B8/pooled analysis predicted.
PREREG_I = {
    "J1_KFAC_embed_C5": {
        "estimator": "KFAC_embed_1e-4", "target": "C5",
        "why": "B8 pooled-48 ratio 0.631; beats GradDot on C5 in 100.0% of bootstrap draws. "
               "This is the mechanism result (curvature from 92k frames on the one subspace "
               "with real Gram structure). H3 confirmed it on the H-series at 0.563.",
    },
    "J2_datamodel_C5": {
        "estimator": "datamodel_LOO", "target": "C5",
        "why": "B8 pooled-48 ratio 0.684; beats GradDot on C5 in 99.7% of draws. The datamodel "
               "is the only estimator that consumes outcomes.",
    },
    "J3_datamodel_C2": {
        "estimator": "datamodel_LOO", "target": "C2",
        "why": "H1 passed on the H-series twice (0.639, 0.626). The only hypothesis confirmed in "
               "the pass-3 family; a third draw makes it three-for-three.",
    },
}
BONF = ALPHA / len(PREREG_I)


def paired_bootstrap(pred_x, pred_g, out, n_boot=5000, seed=0):
    """One-sided bootstrap over masks: is rho(X) - rho(GradDot) > 0 on this mask set?

    Both estimators are scored on the SAME masks, so resampling masks jointly preserves the
    pairing and the shared mask-draw noise cancels -- exactly the B8 logic, made inferential.
    """
    rng = np.random.default_rng(seed)
    pred_x, pred_g, out = map(lambda a: np.asarray(a, float), (pred_x, pred_g, out))
    n = len(out)
    d0 = spearman(pred_x, out) - spearman(pred_g, out)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rx, rg = spearman(pred_x[i], out[i]), spearman(pred_g[i], out[i])
        if np.isfinite(rx) and np.isfinite(rg):
            diffs.append(rx - rg)
    diffs = np.array(diffs)
    p = float((diffs <= 0).mean()) if len(diffs) else np.nan   # H1: difference > 0
    return d0, p, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def mask_pred(scores, masks):
    return np.array([sum(scores.get(d, 0.0) for d in m["demos"]) for m in masks])


def main():
    campaign = "I"
    # Build the I-series masks directly (do NOT pool with G/H here).
    from if_repair import retrain as RT
    imasks = [{"mask_id": m["mask_id"], "demos": m["demos"]}
              for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0]]

    ests = B8.estimators()
    graddot = ests["GradDot_ALL"]
    rows = []
    for name, spec in PREREG_I.items():
        t = spec["target"]
        weighting = spec.get("weighting", "plain")
        raw = F.campaign_outcomes(campaign, weighting, targets=(t,))[t]
        c = F.split_half_ceiling(raw)["ceiling"]
        obs = F.seed_mean(raw)
        out = np.array([obs.get(m["mask_id"], np.nan) for m in imasks])
        ok = np.isfinite(out)

        if spec["estimator"] == "datamodel_LOO":
            rho, p_abs, n = B8.datamodel_loo(imasks, obs)
            # paired: datamodel prediction vs GradDot on the same masks
            from if_repair.datamodel import MODELS, select_alpha
            Z = D.cache_for("bc_s10"); demo_ids = list(Z["train_ids"])
            idx = {d: i for i, d in enumerate(demo_ids)}
            X = np.zeros((len(imasks), len(demo_ids)))
            for r, m in enumerate(imasks):
                for d in m["demos"]:
                    if d in idx:
                        X[r, idx[d]] = 1.0
            y = out.copy(); okm = np.isfinite(y)
            alpha, _ = select_alpha(X[okm], y[okm], "lasso", folds=5)
            pr = np.full(len(imasks), np.nan)
            for r in np.nonzero(okm)[0]:
                keep = okm.copy(); keep[r] = False
                fit = MODELS["lasso"](alpha).fit(X[keep], y[keep])
                pr[r] = float(np.dot(X[r], fit.coef_))
            px = pr[ok]
        else:
            sc = ests[spec["estimator"]][t]
            px = mask_pred(sc, imasks)[ok]
            rho = spearman(px, out[ok]); p_abs = spearman_p_onesided(rho, int(ok.sum()))
            n = int(ok.sum())

        pg = mask_pred(graddot[t], imasks)[ok]
        rho_g = spearman(pg, out[ok])
        d0, p_paired, ci = paired_bootstrap(px, pg, out[ok])

        rows.append({
            "name": name, "target": t, "estimator": spec["estimator"],
            "lds": float(rho), "lds_graddot": float(rho_g), "ceiling": float(c),
            "ratio": float(rho) / c, "bar": 0.5 * c, "p_abs": float(p_abs),
            "paired_delta_rho": float(d0), "paired_p": float(p_paired),
            "paired_ci_lo": ci[0], "paired_ci_hi": ci[1], "alpha": BONF,
            "PASS_abs": bool(np.isfinite(rho) and rho / c >= 0.5 and p_abs < BONF),
            "PASS_paired": bool(np.isfinite(d0) and d0 > 0 and p_paired < BONF)})

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "confirm_iseries.csv"), index=False)
    print("=" * 108)
    print(f"I-SERIES CONFIRMATION (3 preregistered hypotheses, Bonferroni alpha = {BONF:.5f}), "
          f"24 fresh masks seed 9973, 10 seeds")
    print("=" * 108)
    print(df[["name", "target", "estimator", "lds", "lds_graddot", "ceiling", "ratio",
              "p_abs", "PASS_abs", "paired_delta_rho", "paired_p",
              "PASS_paired"]].to_string(index=False))
    print(f"\nABSOLUTE passes:  {int(df.PASS_abs.sum())}/{len(df)}")
    print(f"PAIRED  passes:   {int(df.PASS_paired.sum())}/{len(df)}  "
          f"(beats GradDot on the fresh masks, Bonferroni-significant)")


if __name__ == "__main__":
    main()
