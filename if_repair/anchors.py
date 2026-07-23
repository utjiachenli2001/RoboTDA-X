"""Task 0 -- reproduce the paper's anchors from the cached Grams. No new estimator.

Four anchors, all computed with the repo's own routines:

  A1  GradDot champion, C1, E=20, bc_s10                 -> rho = 0.5130434782608695
  A2  GradDot dmean    (scores_graddot), C1, E=20        -> rho = 0.593  (a DIFFERENT estimator)
  A3  cross-validated IF (tuned on C1, frozen, eval C5)  -> rho ~ 0.40   (fails the bar)
  A4  lambda -> inf collapse: IF and TRAK -> GradDot_dmean, gap -> 0
  A5  dev triple (E=10 + stage_G6 + 6-seed ceiling)      -> GradDot_dmean C1 = 0.504

IMPORTANT -- two distinct "GradDot" estimators exist in this repo, and conflating them
is the single biggest trap here (see BLOCKERS.md #1):

  GradDot_unitL2 : per-member score column scaled to unit L2 over demos, then mean.
                   This is p16's PRIMARY "GradDot_E20_normalized" and IS the paper's
                   headline 0.513 at E=20. Implemented here (p5lib.normalized_ensemble_scores
                   cannot be called directly without building the per-member long frame).
  GradDot_dmean  : K_m / mean(diag G_m), then mean over members. This is exactly
                   phase3/src/p6_lambda_extend.py:scores_graddot(normalize_per_member=True),
                   and it is the true lambda -> inf limit of the ensemble-mean IF/TRAK.
                   At E=20 on C1 it scores 0.593, NOT 0.513.

Both are reproduced bit-for-bit against the archived p16 table by tests/test_anchors.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import scores_at_ridge, demo_grain_lds, RIDGE_GRID, ALPHA  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman_p_onesided, bootstrap_spearman_ci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# archived constants (phase5/src/p16_analyze.py, phase3/results/p6_lambda_sweep.json)
C1_GRADDOT_ARCHIVED = 0.5130434782608695
CV_IF_ARCHIVED_E10 = 0.40347826086956523


# --------------------------------------------------------------------- estimators
def graddot_unit_l2(Z) -> dict:
    """The paper's champion. Per-member unit-L2 over the demo axis, then mean over members.

    Mirrors p5lib.normalized_ensemble_scores(normalize=True) applied to raw K.
    """
    K = np.asarray(Z["K"], float)                       # (M,N,T)
    nrm = np.linalg.norm(K, axis=1, keepdims=True)      # per (member,target) over demos
    if np.any(nrm == 0):
        raise RuntimeError("zero-norm member score vector")
    S = (K / nrm).mean(0)                               # (N,T)
    tids, tgts = list(Z["train_ids"]), list(Z["targets"])
    return {tgts[j]: {tids[i]: float(S[i, j]) for i in range(len(tids))}
            for j in range(len(tgts))}


def graddot_dmean(Z) -> dict:
    """The true lambda->inf limit. Repo function, called verbatim."""
    return scores_graddot(Z, normalize_per_member=True)


# --------------------------------------------------------------------- helpers
def _lds(scores, gmasks, obs_t):
    rho, p, n, pred, out = demo_grain_lds(scores, gmasks, obs_t)
    return {"lds": float(rho), "p": float(p), "n": int(n)}


def _report(scores, gmasks, obs, ceil, t):
    r = _lds(scores[t], gmasks, obs[t])
    c = float(ceil[t])
    r.update(target=t, ceiling=c, ratio=r["lds"] / c,
             bar=0.5 * c,
             passed=bool(np.isfinite(r["lds"]) and r["lds"] >= 0.5 * c and r["p"] < ALPHA))
    return r


# --------------------------------------------------------------------- anchors
def cv_if(Z, obs, ceil, gmasks, tune_on="C1", eval_on="C5", grid=None):
    """Repo's MANDATORY cross-validation: pick (attributor, ridge) maximising demo LDS on
    `tune_on`, freeze it, report on `eval_on`. Never tuned on the evaluated target."""
    grid = list(grid or RIDGE_GRID)
    best = None
    cache = {}
    for rr in grid:
        S = cache[rr] = scores_at_ridge(Z, rr)
        for a in ("IF", "TRAK"):
            rho = demo_grain_lds(S[a][tune_on], gmasks, obs[tune_on])[0]
            if np.isfinite(rho) and (best is None or rho > best[0]):
                best = (float(rho), a, rr)
    tuned_rho, attr, rr = best
    S = cache[rr]
    rho, p, n, _, _ = demo_grain_lds(S[attr][eval_on], gmasks, obs[eval_on])
    c = float(ceil[eval_on])
    return {"frozen_attributor": attr, "frozen_ridge_rel": rr,
            "tuned_on": tune_on, "tuned_lds_on_tuning_target": tuned_rho,
            "evaluated_on": eval_on, "lds": float(rho), "p": float(p), "n": int(n),
            "ceiling": c, "ratio": float(rho) / c, "bar": 0.5 * c,
            "passed": bool(rho >= 0.5 * c and p < ALPHA)}


def lambda_collapse(Z, obs, gmasks, targets=("C1", "C5"),
                    ridges=(1e1, 1e2, 1e3, 1e5, 1e7)):
    """IF and TRAK must both converge to GradDot_dmean as ridge_rel -> inf."""
    gd = graddot_dmean(Z)
    ref = {t: demo_grain_lds(gd[t], gmasks, obs[t])[0] for t in targets}
    rows = []
    for rr in ridges:
        S = scores_at_ridge(Z, rr)
        row = {"ridge_rel": rr}
        for a in ("IF", "TRAK"):
            for t in targets:
                rho = demo_grain_lds(S[a][t], gmasks, obs[t])[0]
                row[f"{a}_{t}"] = float(rho)
                row[f"gap_{a}_{t}"] = abs(float(rho) - float(ref[t]))
        rows.append(row)
    return {"graddot_dmean_reference": {t: float(v) for t, v in ref.items()}, "sweep": rows,
            "max_gap_at_largest_ridge": max(
                rows[-1][k] for k in rows[-1] if k.startswith("gap_"))}


def run():
    gmasks = D.demo_masks()
    Z20, Z10 = D.gram_e20(), D.gram_e10()
    obs_bc, ceil_bc = D.outcomes("bc_s10"), D.ceilings("bc_s10")
    obs_dev, ceil_dev = D.outcomes("dev_s6"), D.ceilings("dev_s6")

    gl2_20, gdm_20 = graddot_unit_l2(Z20), graddot_dmean(Z20)
    gdm_10 = graddot_dmean(Z10)

    out = {
        "A1_graddot_unitL2_C1_E20_bc_s10": _report(gl2_20, gmasks, obs_bc, ceil_bc, "C1"),
        "A1b_graddot_unitL2_C5_E20_bc_s10": _report(gl2_20, gmasks, obs_bc, ceil_bc, "C5"),
        "A2_graddot_dmean_C1_E20_bc_s10": _report(gdm_20, gmasks, obs_bc, ceil_bc, "C1"),
        "A2b_graddot_dmean_C5_E20_bc_s10": _report(gdm_20, gmasks, obs_bc, ceil_bc, "C5"),
        "A3_cv_if_tuneC1_evalC5_E20_bc_s10": cv_if(Z20, obs_bc, ceil_bc, gmasks, "C1", "C5"),
        "A3b_cv_if_tuneC1_evalC5_E10_dev_s6": cv_if(Z10, obs_dev, ceil_dev, gmasks, "C1", "C5"),
        "A3c_cv_if_tuneC5_evalC1_E10_dev_s6": cv_if(Z10, obs_dev, ceil_dev, gmasks, "C5", "C1"),
        "A4_lambda_collapse_E20_bc_s10": lambda_collapse(Z20, obs_bc, gmasks),
        "A5_graddot_dmean_C1_E10_dev_s6": _report(gdm_10, gmasks, obs_dev, ceil_dev, "C1"),
        "archived_constants": {
            "C1_GRADDOT_unitL2_E20": C1_GRADDOT_ARCHIVED,
            "CV_IF_tuneC1_evalC5_E10": CV_IF_ARCHIVED_E10,
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "anchors.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def _fmt(r):
    return (f"lds={r['lds']:+.4f}  ceiling={r['ceiling']:.4f}  ratio={r['ratio']:.3f}  "
            f"p={r['p']:.4f}  n={r['n']}  pass={r['passed']}")


def main():
    o = run()
    print("=" * 94)
    print("TASK 0 -- ANCHORS (demo grain; every number as LDS / ceiling / ratio / p)")
    print("=" * 94)
    a1 = o["A1_graddot_unitL2_C1_E20_bc_s10"]
    print(f"A1 GradDot_unitL2  C1 E=20 bc_s10 : {_fmt(a1)}")
    print(f"   archived target {C1_GRADDOT_ARCHIVED!r}  |diff| = "
          f"{abs(a1['lds'] - C1_GRADDOT_ARCHIVED):.2e}   <-- the paper's 0.513 champion")
    a2 = o["A2_graddot_dmean_C1_E20_bc_s10"]
    print(f"A2 GradDot_dmean   C1 E=20 bc_s10 : {_fmt(a2)}   <-- scores_graddot(); NOT 0.513")
    a3 = o["A3_cv_if_tuneC1_evalC5_E20_bc_s10"]
    print(f"A3 CV-IF tune C1 -> eval C5, E=20 : {_fmt(a3)}  "
          f"[frozen {a3['frozen_attributor']} @ ridge_rel={a3['frozen_ridge_rel']:.0e}]")
    a3b = o["A3b_cv_if_tuneC1_evalC5_E10_dev_s6"]
    print(f"A3b CV-IF tune C1 -> eval C5, E=10: {_fmt(a3b)}")
    print(f"    archived target {CV_IF_ARCHIVED_E10!r}  |diff| = "
          f"{abs(a3b['lds'] - CV_IF_ARCHIVED_E10):.2e}")
    a5 = o["A5_graddot_dmean_C1_E10_dev_s6"]
    print(f"A5 GradDot_dmean   C1 E=10 dev_s6 : {_fmt(a5)}   <-- the 0.504 dev anchor")
    c = o["A4_lambda_collapse_E20_bc_s10"]
    print(f"A4 lambda->inf collapse: GradDot_dmean ref "
          f"{ {k: round(v, 4) for k, v in c['graddot_dmean_reference'].items()} }")
    for r in c["sweep"]:
        gaps = {k[4:]: round(v, 6) for k, v in r.items() if k.startswith("gap_")}
        print(f"   ridge_rel={r['ridge_rel']:8.0e}  gaps {gaps}")
    print(f"   max gap at largest ridge = {c['max_gap_at_largest_ridge']:.2e}  (-> 0 expected)")
    print("=" * 94)


if __name__ == "__main__":
    main()
