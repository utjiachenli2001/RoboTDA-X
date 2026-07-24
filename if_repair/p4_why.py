"""PASS 5, P4 -- why does leverage correction help C2/C5/C7/C8 and not C1/C4/C6/C9?

Mechanistic hypothesis. The leverage correction diag(G)^-beta down-weights demos with large
self-influence G[d,d]. It therefore HELPS a target t exactly when GradDot's raw kernel K[:,t] is
contaminated by self-influence -- i.e. when high-|K[d,t]| demos are high-G[d,d] demos that are not
actually the most useful for t. The clean diagnostic is the correlation, over demos, between the
target's raw scores and the self-influence:

    contamination(t) = Spearman_d( |K[d,t]| , diag(G)[d] )     (member-averaged)

Prediction: contamination(t) correlates POSITIVELY with the per-target leverage responsiveness
(the best pooled Delta_rho vs GradDot_dmean the family achieves on t, from P1). Also tabulated:
ceiling, mask-outcome variance, datamodel per-target LDS, and cluster index. n=9 targets, so all
correlations are DESCRIPTIVE, not inferential.

Deliverable: the diagnostic with the strongest |rho| against responsiveness, as a candidate
outcome-free rule for when to apply leverage correction. Zero GPU.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402
from if_repair import gradients as GR  # noqa: E402
from if_repair import b8_maskdraw as B8  # noqa: E402

D.add_repo_paths()
from lds import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TARGETS = tuple(f"C{i}" for i in range(1, 10))


def contamination(Z):
    """-> {target: member-averaged Spearman(|K[:,t]|, diag(G))}."""
    G, K = np.asarray(Z["G"], float), np.asarray(Z["K"], float)
    M, N, T = K.shape
    tgts = list(Z["targets"])
    out = {}
    for j, t in enumerate(tgts):
        rs = [spearman(np.abs(K[m, :, j]), np.diag(G[m])) for m in range(M)]
        out[t] = float(np.nanmean(rs))
    return out


def diagG_dispersion(Z):
    """CV of the self-influence diag(G), member-averaged -- how spread the leverage is."""
    G = np.asarray(Z["G"], float)
    cvs = [np.std(np.diag(G[m])) / (np.mean(np.diag(G[m])) + 1e-30) for m in range(G.shape[0])]
    return float(np.mean(cvs))


def per_target_responsiveness():
    """-> {target: best pooled Delta_rho vs GradDot_dmean over the P1 family (sign-consistent
    flag too)}. Read from the committed P1 csv."""
    df = pd.read_csv(os.path.join(RESULTS, "p1_leverage_family.csv"))
    out = {}
    for t in TARGETS:
        sub = df[df.target == t]
        best = sub.sort_values("d_pooled", ascending=False).iloc[0]
        sc = bool(best.dG > 0 and best.dH > 0 and best.dI > 0)
        out[t] = {"best_pooled_delta": float(best.d_pooled), "sign_consistent": sc}
    return out


def main():
    members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(x, "final.pt")))
    from if_repair import b1_layerwise as B1
    ens = B1.build_ensemble(members)
    Zh = ens["head"]
    Zc = D.cache_for("bc_s10")

    cont_h = contamination(Zh)
    cont_c = contamination(Zc)
    resp = per_target_responsiveness()

    # datamodel per-target LDS (pooled G/H/I) and outcome variance + ceiling from campaign A
    rows = []
    for t in TARGETS:
        rawA = F.campaign_outcomes("A", "plain", targets=(t,))[t]
        obsA = F.seed_mean(rawA)
        ceil = F.split_half_ceiling(rawA)["ceiling"]
        ovar = float(np.var(list(obsA.values())))
        # datamodel LOO on the archived G masks
        gm = [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in D.demo_masks()]
        dm_rho, _, _ = B8.datamodel_loo(gm, obsA)
        rows.append({"target": t, "contam_head": cont_h[t], "contam_cached": cont_c[t],
                     "ceiling": ceil, "outcome_var": ovar,
                     "datamodel_lds": dm_rho,
                     "best_leverage_delta": resp[t]["best_pooled_delta"],
                     "sign_consistent": resp[t]["sign_consistent"]})
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "p4_why.csv"), index=False)

    print("=" * 96)
    print("P4 -- per-target leverage responsiveness and its predictors (n=9, DESCRIPTIVE)")
    print("=" * 96)
    print(df.round(3).to_string(index=False))
    print(f"\nhead-Phi diag(G) dispersion (CV): {diagG_dispersion(Zh):.3f}   "
          f"cached-E20 diag(G) dispersion: {diagG_dispersion(Zc):.3f}")

    y = df.best_leverage_delta.values
    print("\nSpearman(predictor, best_leverage_delta) over the 9 targets:")
    for col in ("contam_head", "contam_cached", "ceiling", "outcome_var", "datamodel_lds"):
        r = spearman(df[col].values, y)
        print(f"  {col:16s}: rho = {r:+.3f}")
    # binary: does contamination separate sign-consistent winners from the rest?
    win = df[df.sign_consistent]
    lose = df[~df.sign_consistent]
    print(f"\nsign-consistent winners ({','.join(win.target)}): "
          f"mean contam_head {win.contam_head.mean():+.3f}")
    print(f"non-winners ({','.join(lose.target)}): "
          f"mean contam_head {lose.contam_head.mean():+.3f}")


if __name__ == "__main__":
    main()
