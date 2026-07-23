"""Task 9 -- confirmatory hold-out for the Phase A/B candidates, computed ONCE.

Each estimator family is evaluated the way that is HONEST for that family:

  gradient estimators  -- direct. Their scores never touch mask outcomes.
  datamodel            -- leave-one-mask-out. Its coefficients are fit ON outcomes, so
                          in-sample scoring is circular (it returns ratio > 1; see
                          datamodel.py). Each mask is predicted by a refit that excluded it.
  fusion w/ datamodel  -- leave-one-mask-out on the datamodel component only.
  B1 layer groups      -- direct, but on the REGENERATED E=5 ensemble (BLOCKERS #6), so
                          its own baseline is recomputed on that same ensemble.

Bonferroni is applied over the whole confirmatory family actually tested.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import spectral as SP  # noqa: E402
from if_repair.eval import build_scores  # noqa: E402
from if_repair.datamodel import (design_matrix, outcome_vector, select_alpha,  # noqa: E402
                                 MODELS)
from if_repair.fusion import COMPONENTS, zscore_average  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds, ALPHA  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _rec(name, target, rho, n, ceil, extra=None):
    c = float(ceil)
    r = {"estimator": name, "target": target, "lds": float(rho), "ceiling": c,
         "ratio": float(rho) / c, "bar": 0.5 * c,
         "p": float(spearman_p_onesided(rho, n)), "n": int(n)}
    r.update(extra or {})
    return r


def eval_direct(scores, target, tier):
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    rho, p, n, _, _ = demo_grain_lds(scores, gm, obs[target])
    return _rec("", target, rho, n, ceil[target])


def eval_loo(target, tier, model="lasso", with_fusion=None):
    """Leave-one-mask-out datamodel (optionally fused with gradient scores)."""
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    X, demo_ids, mask_ids = design_matrix(tier)
    y = outcome_vector(tier, target, mask_ids)
    ok = np.isfinite(y)
    alpha, _ = select_alpha(X[ok], y[ok], model, folds=5)
    grad = None
    if with_fusion:
        grad = [build_scores(COMPONENTS[c], tier)[target] for c in with_fusion]
    pred, out = [], []
    for r in range(len(mask_ids)):
        if not np.isfinite(y[r]):
            continue
        keep = np.array([q != r and np.isfinite(y[q]) for q in range(len(mask_ids))])
        fit = MODELS[model](alpha).fit(X[keep], y[keep])
        dm = {d: float(v) for d, v in zip(demo_ids, fit.coef_)}
        sc = zscore_average(grad + [dm], demo_ids) if grad else dm
        pred.append(sum(sc.get(d, 0.0) for d in gm[r]["demos"]))
        out.append(y[r])
    rho = spearman(pred, out)
    return _rec("", target, rho, len(pred), ceil[target], {"alpha": alpha})


def regen_ensemble(group):
    from if_repair import b1_layerwise as B1
    from if_repair import gradients as GR
    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    ens = B1.build_ensemble(members)
    return ens[group], members


def main():
    tier = "bc_s10"
    HOLD = list(D.HOLDOUT_TARGETS)
    rows = []

    # ---- family 1: cached-Gram estimators (no GPU, E=20)
    for name, spec in [("GradDot_dmean_BASELINE",
                        {"kind": "GradDot", "normalize": "dmean", "aggregator": "mean"})]:
        sc = build_scores(spec, tier)
        for t in HOLD:
            r = eval_direct(sc[t], t, tier); r["estimator"] = name
            r["family"] = "reference"; rows.append(r)
    for t in HOLD:
        sc = {c: build_scores(COMPONENTS[c], tier)[t] for c in COMPONENTS}
        Z = D.cache_for(tier); ids = list(Z["train_ids"])
        fused = zscore_average([sc[c] for c in COMPONENTS], ids)
        r = eval_direct(fused, t, tier)
        r["estimator"] = "A5_fusion_gradients_zscore"; r["family"] = "tested"
        rows.append(r)
    for t in HOLD:
        r = eval_loo(t, tier, "lasso")
        r["estimator"] = "A4_datamodel_lasso_LOO"; r["family"] = "tested"; rows.append(r)

    # ---- family 2: B1 layer-restricted, regenerated E=5 ensemble
    for group, est, k in [("head", "GradDot_dmean", None),
                          ("block_01", "trunc_k10", 10)]:
        Z, members = regen_ensemble(group)
        tids = list(Z["train_ids"])
        if k is None:
            sc_all = scores_graddot(Z, normalize_per_member=True)
        else:
            S = SP.truncated_if(Z, k, normalize="dmean")
            sc_all = {tg: {tids[i]: float(S[i, j]) for i in range(len(tids))}
                      for j, tg in enumerate(list(Z["targets"]))}
        for t in HOLD:
            r = eval_direct(sc_all[t], t, tier)
            r["estimator"] = f"B1_{group}_{est}_regenE{len(members)}"
            r["family"] = "tested"; rows.append(r)

    df = pd.DataFrame(rows)
    tested = df[df.family == "tested"]
    nfam = len(tested)
    df["alpha_bonf"] = np.where(df.family == "tested", ALPHA / nfam, ALPHA)
    df["PASS"] = (df.ratio >= 0.5) & (df.p < df.alpha_bonf)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "holdout_phase2.csv"), index=False)

    print("=" * 104)
    print(f"TASK 9 -- CONFIRMATORY HOLD-OUT (C2,C4,C7,C9), computed once. "
          f"Bonferroni family = {nfam}, alpha = {ALPHA/nfam:.5f}")
    print("=" * 104)
    print(df[["estimator", "family", "target", "lds", "ceiling", "ratio", "bar", "p",
              "alpha_bonf", "PASS"]].to_string(index=False))
    print("\n--- generality count (win condition 1: >=3 of C1,C2,C4,C7,C9) ---")
    for e, sub in df[df.family == "tested"].groupby("estimator"):
        print(f"  {e:42s} hold-out passes: {int(sub.PASS.sum())}/4  "
              f"(ratios {[round(x,3) for x in sub.ratio]})")


if __name__ == "__main__":
    main()
