"""P6.5-EXT -- close the right edge of the ridge sweep. Zero GPU (reuses the cached Gram).

WHY THIS EXISTS (a disclosed extension, not a post-hoc criterion change):

The preregistered 8-point grid (ridge_rel 1e-6 .. 1e+1) returned its MAXIMUM AT THE RIGHT EDGE
for C1 (LDS +0.498 at ridge_rel = 1e+1, still climbing). A maximum at a grid boundary is not a
maximum -- the preregistered read-out "the MAX over the lambda grid" would be reporting a
truncation artifact. So we close the boundary ANALYTICALLY.

THE LIMIT IS KNOWN IN CLOSED FORM. Both estimators are one-parameter families in lambda:

    TRAK  = (G + lam I)^{-1} K              -> (1/lam) K   as lam -> inf
    IF    = [K - G (lam N I + G)^{-1} K]/lam -> (1/lam) K   as lam -> inf

Spearman is invariant to a positive scale, so BOTH estimators converge, as lambda -> inf, to the
SAME ranking: that of K itself, where K_ij = <g_i, g_test_j> -- the RAW GRADIENT DOT PRODUCT
(no preconditioning at all). We therefore add:

  * four more grid points (1e2 .. 1e5), and
  * the analytic limit estimator "GradDot" (score = K exactly),

which makes the sweep COMPLETE: the curve now runs from the unregularized inverse (lam -> 0) to
the unpreconditioned dot product (lam -> inf), and the maximum can no longer hide outside it.

The preregistered 8-point read-out is preserved VERBATIM in p6_lambda_sweep.json. This file
reports the extended sweep alongside it, and re-runs the SAME preregistered decision rule
(oracle-tuned max + mandatory cross-validation) on the extended grid.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, RESULTS, P2_RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402
from p6_lambda_sweep import (GK_CACHE, RIDGE_GRID, DEFAULT_RIDGE, FOCAL, ALPHA,  # noqa: E402
                             demo_grain_lds, cluster_grain_lds, scores_at_ridge)

EXT_GRID = RIDGE_GRID + [1e2, 1e3, 1e4, 1e5]


def scores_graddot(Z, normalize_per_member=True):
    """The lam -> inf limit of the ENSEMBLE-MEAN IF/TRAK score.

    CAREFUL -- the naive limit is WRONG, and the convergence check below caught it.

    Per member m, lam_m = ridge_rel * mean(diag(G_m)) -- an ADAPTIVE, per-member scale (that is
    how attribution.py sets it). So as ridge_rel -> inf,

        IF_m, TRAK_m  ->  K_m / lam_m  =  K_m / (ridge_rel * d_m),    d_m := mean(diag(G_m))

    and the ENSEMBLE MEAN of the scores tends to

        (1/ridge_rel) * (1/M) * sum_m  K_m / d_m                       <-- weights 1/d_m !

    Spearman kills the leading 1/ridge_rel, but NOT the per-member 1/d_m weights. The true limit
    is therefore the PER-MEMBER SCALE-NORMALIZED dot product, not the plain mean of K_m. Members
    have different gradient norms, so averaging K_m unnormalized lets the largest-gradient member
    dominate -- a different estimator entirely (it scores 0.397 on C1 vs 0.504 for the true limit).

    normalize_per_member=True  -> "GradDot_limit"     : the actual lam -> inf limit
    normalize_per_member=False -> "GradDot_unnorm"    : plain mean of K_m (reported for contrast)
    """
    K = Z["K"]                                            # (M,N,T)
    G = Z["G"]                                            # (M,N,N)
    train_ids, tgts = list(Z["train_ids"]), list(Z["targets"])
    if normalize_per_member:
        d = np.array([np.mean(np.diag(G[m])) for m in range(K.shape[0])])   # (M,)
        S = (K / d[:, None, None]).mean(0)
    else:
        S = K.mean(0)
    return {tgts[j]: {train_ids[i]: float(S[i, j]) for i in range(len(train_ids))}
            for j in range(len(tgts))}


def main():
    Z = np.load(GK_CACHE, allow_pickle=True)
    clusters = dataset.clusters()

    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    fman = json.load(open(os.path.join(RESULTS, "mask_manifest.json")))
    fmasks = [{"mask_id": m["mask_id"], "demos": m["demos"], "clusters": m["clusters"]}
              for m in fman["masks"]]

    G6 = pd.read_parquet(os.path.join(P2_RESULTS, "stage_G6_outcomes.parquet"))
    dfF = pd.read_parquet(os.path.join(RESULTS, "stage_F_outcomes.parquet"))
    dfF["neg_plain_loss"] = -dfF.plain_loss
    F2 = dfF[dfF.seed.isin([301, 302])]

    p1 = json.load(open(os.path.join(P2_RESULTS, "p1_demo_grain.json")))
    ceil_demo = {t: p1["all_targets"][t]["neg_plain_loss"]["ceiling_6seed_SB"] for t in clusters}
    nc = json.load(open(os.path.join(RESULTS, "noise_ceilings.json")))
    ceil_clu = {t: nc[t]["neg_plain_loss"]["ceiling"] for t in clusters}

    obs_demo = {t: G6[G6.target == t].groupby("mask_id")["neg_plain_loss"].mean().to_dict()
                for t in clusters}
    obs_clu = {t: F2[F2.target == t].groupby("mask_id")["neg_plain_loss"].mean().to_dict()
               for t in clusters}

    rows = []

    def add(attr, rr, t, sc):
        rho_d, p_d, n_d, _, _ = demo_grain_lds(sc, gman, obs_demo[t])
        rho_c, p_c, n_c = cluster_grain_lds(sc, fmasks, obs_clu[t], t)
        rows.append({"ridge_rel": rr, "attributor": attr, "target": t, "focal": t in FOCAL,
                     "demo_lds": rho_d, "demo_p1": p_d, "demo_n": n_d,
                     "demo_ceiling": ceil_demo[t],
                     "demo_ratio": rho_d / ceil_demo[t],
                     "demo_bar_half_ceiling": 0.5 * ceil_demo[t],
                     "demo_PASS": bool(np.isfinite(rho_d) and rho_d >= 0.5 * ceil_demo[t]
                                       and np.isfinite(p_d) and p_d < ALPHA),
                     "cluster_lds": rho_c, "cluster_p1": p_c, "cluster_n": n_c,
                     "cluster_ceiling": ceil_clu[t],
                     "cluster_ratio": rho_c / ceil_clu[t] if ceil_clu[t] > 0 else np.nan})

    for rr in EXT_GRID:
        S = scores_at_ridge(Z, rr)
        for a in ("IF", "TRAK"):
            for t in clusters:
                add(a, rr, t, S[a][t])
        print(f"[6.5x] ridge_rel={rr:.0e} done", flush=True)

    SG = scores_graddot(Z, normalize_per_member=True)
    for t in clusters:
        add("GradDot_limit", np.inf, t, SG[t])
    SGu = scores_graddot(Z, normalize_per_member=False)
    for t in clusters:
        add("GradDot_unnorm", np.inf, t, SGu[t])
    print("[6.5x] analytic lam->inf limit (GradDot_limit) + unnormalized contrast done")

    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(P3_RESULTS, "p6_lambda_sweep_extended.csv"), index=False)

    # ---- CONVERGENCE CHECK (this is what caught the naive-limit error; it must be ~0 now)
    conv, conv_ok = {}, True
    for t in clusters:
        lim = float(T[(T.attributor == "GradDot_limit") & (T.target == t)].demo_lds.iloc[0])
        unn = float(T[(T.attributor == "GradDot_unnorm") & (T.target == t)].demo_lds.iloc[0])
        big = {a: float(T[(T.attributor == a) & (T.target == t)
                          & (T.ridge_rel == 1e5)].demo_lds.iloc[0]) for a in ("IF", "TRAK")}
        gap = max(abs(lim - v) for v in big.values())
        conv_ok &= (gap < 1e-6)
        conv[t] = {"analytic_limit_lds": lim, "unnormalized_variant_lds": unn,
                   "lds_at_ridge_1e5": big, "max_abs_gap_vs_limit": gap,
                   "converged": bool(gap < 1e-6)}
        if t in FOCAL:
            print(f"[6.5x] convergence {t}: limit={lim:+.4f} ridge1e5 IF={big['IF']:+.4f} "
                  f"TRAK={big['TRAK']:+.4f} gap={gap:.2e} "
                  f"| unnormalized variant={unn:+.4f}")
    if not conv_ok:
        raise RuntimeError(
            "P6.5-EXT convergence FAILED: IF/TRAK at ridge_rel=1e5 do not agree with the analytic "
            "lambda->inf limit. Either the limit derivation or the sweep is wrong. INSTRUMENT "
            "DEFECT -- stop and write PHASE3_DEFECT.md.")
    print("[6.5x] CONVERGENCE VERIFIED: IF and TRAK at ridge 1e5 == the analytic limit to <1e-6")

    # ---- the SAME preregistered decision rule, on the extended grid
    R = T[T.attributor.isin(["IF", "TRAK", "GradDot_limit"])]   # unnorm variant is a contrast, not a candidate
    tuned, cv = {}, {}
    for t in FOCAL:
        s = R[R.target == t]
        b = s.loc[s.demo_lds.idxmax()]
        tuned[t] = {"best_ridge_rel": float(b.ridge_rel), "best_attributor": str(b.attributor),
                    "best_demo_lds": float(b.demo_lds), "ceiling": float(b.demo_ceiling),
                    "bar": float(b.demo_bar_half_ceiling), "ratio": float(b.demo_ratio),
                    "p_onesided": float(b.demo_p1),
                    "CROSSES_HALF_CEILING": bool(b.demo_ratio >= 0.5),
                    "PASS_ratio_AND_p": bool(b.demo_PASS),
                    "LABEL": "ORACLE-TUNED UPPER BOUND on the EXTENDED grid. NOT held-out."}
    for tune_on, eval_on in [("C1", "C5"), ("C5", "C1")]:
        s = R[R.target == tune_on]
        i = s.demo_lds.idxmax()
        rr, at = float(s.loc[i].ridge_rel), str(s.loc[i].attributor)
        e = R[(R.target == eval_on) & (R.attributor == at)
              & ((R.ridge_rel == rr) | (np.isinf(R.ridge_rel) & np.isinf(rr)))].iloc[0]
        cv[f"tune_on_{tune_on}__evaluate_on_{eval_on}"] = {
            "frozen_ridge_rel": rr, "frozen_attributor": at,
            "heldout_demo_lds": float(e.demo_lds), "heldout_ceiling": float(e.demo_ceiling),
            "heldout_ratio": float(e.demo_ratio),
            "heldout_bar": float(e.demo_bar_half_ceiling),
            "heldout_p_onesided": float(e.demo_p1),
            "heldout_CROSSES_HALF_CEILING": bool(e.demo_ratio >= 0.5),
            "heldout_PASS_ratio_AND_p": bool(e.demo_PASS),
            "LABEL": "CROSS-VALIDATED on the EXTENDED grid -- lambda + attributor frozen on the "
                     "OTHER focal target. This is the number that licenses a claim."}

    any_cv = any(v["heldout_PASS_ratio_AND_p"] for v in cv.values())
    default_lds = {t: float(R[(R.target == t) & (R.ridge_rel == DEFAULT_RIDGE)].demo_lds.max())
                   for t in FOCAL}
    lim_lds = {t: float(T[(T.attributor == "GradDot_limit") & (T.target == t)].demo_lds.iloc[0])
               for t in clusters}
    lim_ratio = {t: lim_lds[t] / ceil_demo[t] for t in clusters}

    out = {
        "stage": "P6.5-EXT -- extended ridge sweep with the analytic lambda -> inf limit",
        "n_retrains": 0, "n_gpu": 0,
        "WHY": ("The preregistered 8-point grid returned its maximum AT ITS RIGHT EDGE for C1 "
                "(+0.498 at ridge_rel=1e+1, still climbing). A boundary maximum is a truncation "
                "artifact, so the boundary is closed analytically. Disclosed as an EXTENSION; "
                "the preregistered 8-point read-out is preserved verbatim in p6_lambda_sweep.json."),
        "extended_grid": [float(x) for x in EXT_GRID],
        "analytic_limit": ("Per member, lam_m = ridge_rel * mean(diag(G_m)). As ridge_rel -> inf, "
                           "IF_m and TRAK_m -> K_m / lam_m, so the ENSEMBLE-MEAN score tends to "
                           "(1/ridge_rel) * mean_m [ K_m / mean(diag(G_m)) ]. Spearman kills the "
                           "leading constant but NOT the per-member 1/mean(diag(G_m)) weights. The "
                           "limit is therefore the PER-MEMBER SCALE-NORMALIZED gradient dot product "
                           "('GradDot_limit'), NOT the plain mean of K_m ('GradDot_unnorm', reported "
                           "as a contrast: it scores materially worse, so per-member scale "
                           "normalization matters more than the preconditioner does)."),
        "convergence_check": conv,
        "ORACLE_TUNED_MAX_extended": tuned,
        "MANDATORY_CROSS_VALIDATION_extended": cv,
        "demo_lds_at_default_ridge_1e-2": default_lds,
        "GradDot_limit_demo_lds_all_targets": lim_lds,
        "GradDot_limit_ratio_to_ceiling_all_targets": lim_ratio,
        "THE_SUBSTANTIVE_FINDING": (
            "Demo-grain LDS increases MONOTONICALLY with the ridge across ~7 orders of magnitude "
            "and is MAXIMIZED at the degenerate limit where the preconditioner is switched off "
            "entirely. The exact empirical-Fisher inverse (IF) and the exact Gram inverse (TRAK) "
            "-- the machinery the field considers principled, and which Phase 1 deliberately "
            "computed EXACTLY rather than approximately -- ACTIVELY HURT the ranking here. A raw "
            "gradient dot product beats both at their default lambda."),
        "CONSEQUENCE_FOR_PHASE_2": (
            "Phase 2's claim 'attribution was given its best shot' was, at the default lambda, "
            "FALSE: C1's demo-grain LDS nearly doubles (+0.256 -> +0.498) when lambda is tuned. "
            "The VERDICT (FAIL) nevertheless survives, because the cross-validated read-out -- the "
            "only one that licenses a claim -- still does not clear the half-ceiling bar. But the "
            "margin is much thinner than Phase 2 reported, and this must be stated plainly."),
        "VERDICT": ("MATERIAL_UPDATE_TO_PHASE_2" if any_cv
                    else "PHASE_2_VERDICT_STANDS__BUT_MARGIN_IS_THINNER_THAN_REPORTED"),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p6_lambda_sweep_extended.json"), out)

    print("\n" + "=" * 96)
    print("P6.5-EXT -- EXTENDED RIDGE SWEEP (demo grain, held-out L2, 6-seed ground truth)")
    print("=" * 96)
    print(f"{'target':7s} {'ridge':>9s} {'attr':>14s} {'LDS':>8s} {'bar':>7s} {'ratio':>6s} {'p1':>7s}")
    for t in FOCAL:
        for rr in EXT_GRID:
            s = R[(R.target == t) & (R.ridge_rel == rr)]
            b = s.loc[s.demo_lds.idxmax()]
            mark = "  <-- DEFAULT" if rr == DEFAULT_RIDGE else ""
            print(f"{t:7s} {rr:9.0e} {b.attributor:>14s} {b.demo_lds:+8.3f} "
                  f"{b.demo_bar_half_ceiling:7.3f} {b.demo_ratio:6.2f} {b.demo_p1:7.4f}{mark}")
        g = R[(R.attributor == "GradDot_limit") & (R.target == t)].iloc[0]
        print(f"{t:7s} {'inf':>9s} {'GradDot_limit':>14s} {g.demo_lds:+8.3f} "
              f"{g.demo_bar_half_ceiling:7.3f} {g.demo_ratio:6.2f} {g.demo_p1:7.4f}  <-- LIMIT")
        print("-" * 96)
    print("\nORACLE-TUNED MAX (extended; upper bound, NOT held-out):")
    for t, v in tuned.items():
        print(f"  {t}: ridge={v['best_ridge_rel']:.0e} {v['best_attributor']:>14s} "
              f"LDS={v['best_demo_lds']:+.3f} ratio={v['ratio']:.2f} p={v['p_onesided']:.4f} "
              f"crosses_half={v['CROSSES_HALF_CEILING']} PASS={v['PASS_ratio_AND_p']}")
    print("\nCROSS-VALIDATED (the number that licenses a claim):")
    for k, v in cv.items():
        print(f"  {k}: ridge={v['frozen_ridge_rel']:.0e} {v['frozen_attributor']:>14s} "
              f"LDS={v['heldout_demo_lds']:+.3f} ratio={v['heldout_ratio']:.2f} "
              f"p={v['heldout_p_onesided']:.4f} PASS={v['heldout_PASS_ratio_AND_p']}")
    print("\nGradDot (no preconditioning) vs ceiling, ALL 9 targets:")
    for t in clusters:
        print(f"  {t}: LDS={lim_lds[t]:+.3f} ceiling={ceil_demo[t]:.3f} ratio={lim_ratio[t]:.2f}")
    print("=" * 96)
    print(f"VERDICT: {out['VERDICT']}")
    print("=" * 96)


if __name__ == "__main__":
    main()
