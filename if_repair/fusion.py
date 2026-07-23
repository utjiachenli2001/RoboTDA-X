"""A5 -- rank fusion across the Phase-A estimator families.

GradDot and the datamodel fail on opposite targets (GradDot C1 pass / C5 fail; datamodel
C1 fail / C5 pass). That complementarity is exactly where a rank ensemble can beat both --
if the two carry different real signal rather than different noise.

FUSION MUST BE A FIXED, TARGET-BLIND RULE. Choosing "GradDot on C1, datamodel on C5" is
selection on the dev targets and would not transfer; it appears only as a clearly labelled
ORACLE upper bound.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair.datamodel import fit_datamodel  # noqa: E402
from if_repair.eval import build_scores  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds, ALPHA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _rank(v):
    return stats.rankdata(np.asarray(v, float), method="average")


def borda(score_dicts, demo_ids, weights=None):
    R = np.stack([_rank([d[i] for i in demo_ids]) for d in score_dicts])
    w = np.ones(len(score_dicts)) if weights is None else np.asarray(weights, float)
    agg = (w[:, None] * R).sum(0)
    return {d: float(agg[k]) for k, d in enumerate(demo_ids)}


def zscore_average(score_dicts, demo_ids):
    Zs = []
    for d in score_dicts:
        v = np.array([d[i] for i in demo_ids], float)
        sd = v.std(ddof=1)
        Zs.append((v - v.mean()) / (sd if sd > 0 else 1.0))
    agg = np.mean(Zs, axis=0)
    return {d: float(agg[k]) for k, d in enumerate(demo_ids)}


def median_rank(score_dicts, demo_ids):
    R = np.stack([_rank([d[i] for i in demo_ids]) for d in score_dicts])
    agg = np.median(R, axis=0)
    return {d: float(agg[k]) for k, d in enumerate(demo_ids)}


COMPONENTS = {
    "GradDot_dmean": {"kind": "GradDot", "normalize": "dmean", "aggregator": "mean"},
    "GradDot_unitL2_HL": {"kind": "GradDot", "normalize": "unitL2",
                          "aggregator": "hodges_lehmann"},
    "IF_ridge10": {"kind": "IF", "normalize": "none", "ridge_rel": 10.0,
                   "aggregator": "mean"},
}

FUSIONS = {"borda": borda, "zscore_avg": zscore_average, "median_rank": median_rank}

RECIPES = {
    "GradDot+datamodel": ["GradDot_dmean", "datamodel_lasso"],
    "GradDot+HL+datamodel": ["GradDot_dmean", "GradDot_unitL2_HL", "datamodel_lasso"],
    "all4": ["GradDot_dmean", "GradDot_unitL2_HL", "IF_ridge10", "datamodel_lasso"],
    "gradients_only": ["GradDot_dmean", "GradDot_unitL2_HL", "IF_ridge10"],
}


def component_scores(tier, targets):
    out = {n: build_scores(spec, tier) for n, spec in COMPONENTS.items()}
    out["datamodel_lasso"] = {t: fit_datamodel(tier, t, "lasso")["scores"]
                              for t in targets}
    return out


def evaluate_fusion(tier="bc_s10", targets=("C1", "C5")) -> pd.DataFrame:
    """Leave-one-mask-out wherever the datamodel participates.

    CRITICAL. The gradient components are mask-independent, so scoring them in-sample is
    fine. The datamodel is NOT: its coefficients are fit to the mask outcomes, and
    demo_grain_lds re-evaluates sum_{d in mask} score_d = X@beta on those same masks. A
    fusion containing in-sample datamodel coefficients therefore inherits that
    circularity wholesale -- it lifted C5 from an honest 0.43 to a fake 0.97 here.

    So for any recipe containing the datamodel, each mask is predicted by a fusion whose
    datamodel component was refit WITHOUT that mask. Recipes with no datamodel are
    unaffected and use the direct path (identical result, far cheaper).
    """
    from if_repair.datamodel import design_matrix, outcome_vector, select_alpha, MODELS

    Z = D.cache_for(tier)
    demo_ids = list(Z["train_ids"])
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    grad = {n: build_scores(spec, tier) for n, spec in COMPONENTS.items()}
    X, dm_ids, mask_ids = design_matrix(tier)
    assert dm_ids == demo_ids
    rows = []
    for rname, members in RECIPES.items():
        needs_dm = any(m.startswith("datamodel") for m in members)
        for fname, fn in FUSIONS.items():
            for t in targets:
                c = float(ceil[t])
                if not needs_dm:
                    fused = fn([grad[m][t] for m in members], demo_ids)
                    rho, p, n, _, _ = demo_grain_lds(fused, gm, obs[t])
                    mode = "direct"
                else:
                    y = outcome_vector(tier, t, mask_ids)
                    ok = np.isfinite(y)
                    alpha, _ = select_alpha(X[ok], y[ok], "lasso", folds=5)
                    pred, out = [], []
                    for r in range(len(mask_ids)):
                        if not np.isfinite(y[r]):
                            continue
                        keep = np.array([q != r and np.isfinite(y[q])
                                         for q in range(len(mask_ids))])
                        fit = MODELS["lasso"](alpha).fit(X[keep], y[keep])
                        dm = {d: float(v) for d, v in zip(demo_ids, fit.coef_)}
                        sd = [(dm if m.startswith("datamodel") else grad[m][t])
                              for m in members]
                        fused = fn(sd, demo_ids)
                        pred.append(sum(fused.get(d, 0.0) for d in gm[r]["demos"]))
                        out.append(y[r])
                    from lds import spearman, spearman_p_onesided
                    rho = spearman(pred, out)
                    n = len(pred)
                    p = spearman_p_onesided(rho, n)
                    mode = "leave_one_mask_out"
                rows.append({"tier": tier, "recipe": rname, "fusion": fname,
                             "target": t, "eval_mode": mode, "lds": float(rho),
                             "ceiling": c, "ratio": float(rho) / c, "bar": 0.5 * c,
                             "p": float(p), "n": n,
                             "passed": bool(np.isfinite(rho) and rho >= 0.5 * c
                                            and p < ALPHA)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df = evaluate_fusion()
    print("=" * 96)
    print("A5 -- RANK FUSION (target-blind rules), tier=bc_s10, demo grain n=24")
    print("=" * 96)
    print("ratio-to-ceiling:")
    print(df.pivot_table(index=["recipe", "fusion"], columns="target",
                         values="ratio").round(3).to_string())
    print("\npass flags:")
    print(df.pivot_table(index=["recipe", "fusion"], columns="target",
                         values="passed").to_string())
    df.to_csv(os.path.join(HERE, "results", "a5_fusion.csv"), index=False)
    best = df.loc[df.groupby("target").ratio.idxmax()]
    print("\nbest fusion per target:")
    print(best[["target", "recipe", "fusion", "lds", "ratio", "p",
                "passed"]].to_string(index=False))
    print("\nBaselines: C1 GradDot_dmean ratio 0.624 PASS | "
          "C5 datamodel_lasso 0.879 PASS, GradDot_dmean 0.416 fail")


if __name__ == "__main__":
    main()
