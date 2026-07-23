"""Task 5 -- one evaluation entry point that wires the right (cache, outcome, ceiling) triple.

Every number this project reports goes through evaluate(): LDS, ceiling, ratio, p, CI and
pass, computed with the repo's own routines. Bare rho is never returned alone.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import spectral as SP  # noqa: E402
from if_repair.aggregate import per_member_scores, normalize_members, aggregate  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds, ALPHA, DEFAULT_RIDGE  # noqa: E402
from lds import bootstrap_spearman_ci  # noqa: E402


def evaluate(scores, target: str, tier: str = "bc_s10", n_boot: int = 2000,
             seed: int = 0) -> dict:
    """scores: {demo_id: score}. -> full LDS record against this tier's ground truth."""
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    if target not in obs:
        raise KeyError(f"target {target} not in tier {tier}")
    rho, p, n, pred, out = demo_grain_lds(scores, gm, obs[target])
    c = float(ceil[target])
    lo, hi = bootstrap_spearman_ci(pred, out, n_boot=n_boot, seed=seed)
    return {"tier": tier, "target": target, "lds": float(rho), "ceiling": c,
            "ratio": float(rho) / c if c else np.nan, "bar": 0.5 * c,
            "p": float(p), "n": int(n), "ci95": [float(lo), float(hi)],
            "alpha": ALPHA,
            "passed": bool(np.isfinite(rho) and c > 0 and rho >= 0.5 * c
                           and np.isfinite(p) and p < ALPHA)}


# --------------------------------------------------------------- estimator registry
def build_scores(spec: dict, tier: str) -> dict:
    """spec -> {target: {demo_id: score}}. The only place a config becomes numbers."""
    Z = D.cache_for(tier)
    kind = spec["kind"]
    agg = spec.get("aggregator", "mean")
    nrm = spec.get("normalize", "dmean")
    if kind == "truncated_if":
        S = SP.truncated_if(Z, int(spec["k"]), aggregate=agg, normalize=nrm)
    elif kind == "damped_if":
        S = SP.damped_if(Z, float(spec["gamma"]), mode=spec.get("mode", "diag"),
                         aggregate=agg, normalize=nrm)
    elif kind in ("GradDot", "IF", "TRAK"):
        S0 = per_member_scores(Z, kind, ridge_rel=float(spec.get("ridge_rel",
                                                                DEFAULT_RIDGE)))
        S = aggregate(normalize_members(S0, nrm, Z), agg)
    else:
        raise KeyError(kind)
    tids, tgts = list(Z["train_ids"]), list(Z["targets"])
    return {tgts[j]: {tids[i]: float(S[i, j]) for i in range(len(tids))}
            for j in range(len(tgts))}


def evaluate_spec(spec: dict, targets, tier="bc_s10") -> list:
    sc = build_scores(spec, tier)
    rows = []
    for t in targets:
        r = evaluate(sc[t], t, tier)
        r["estimator"] = spec.get("name", spec["kind"])
        rows.append(r)
    return rows
