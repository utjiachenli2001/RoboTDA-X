"""Campaign K -- the pass-5 out-of-sample confirmation of the unified-leverage-family wins (P1),
preregistered, computed once.

P1 established that the leverage family
    S_m(lam_rel, beta) = diag(G_m)^-beta . (G_m + lam I)^-1 . K_m
beats the CANONICAL GradDot_dmean bar (BLOCKERS #1) out of sample on several targets, with the
single best UNIFIED config (regen E=5 head-Phi, dmean agg, lam_rel=0.3, beta=1) generalizing
cleanly to C2 and C8 across all three dev draws (pooled Delta_rho +0.28 / +0.39, p<0.01). The
strongest single-target win is C7 on the cached E=20 full-model Gram (dmean, lam_rel=1, beta=1,
+0.316). All dev draws are consumed; campaign K (seed 20260724, disjoint from G/H/I/J;
tests/test_jseries.py) is the fresh confirmation.

Three preregistered hypotheses (Bonferroni over the family, alpha_abs = 0.0167). Each estimator is
paired against GradDot_dmean on ITS OWN ensemble (BLOCKERS #6). Two bars: ABSOLUTE (ratio>=0.5,
p<alpha_abs) and PAIRED (beats GradDot_dmean on the same masks, one-sided bootstrap p<0.05).

Frozen and committed while campaign K has ZERO runs.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402
from if_repair import retrain as RT  # noqa: E402
from if_repair import gradients as GR  # noqa: E402
from if_repair.p1_leverage_family import family_scores  # noqa: E402

D.add_repo_paths()
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ---------------------------------------------------------------- FROZEN prereg (edit ONLY before K)
PREREG_K = {
    "K1_leverage_head_l0p3_b1_C2": {
        "source": "regenE5_head", "agg": "dmean", "lam_rel": 0.3, "beta": 1.0, "target": "C2",
        "why": "unified leverage config, dev pooled G/H/I Delta_rho +0.276 (p=0.006), "
               "sign-consistent vs GradDot_dmean.",
    },
    "K2_leverage_head_l0p3_b1_C8": {
        "source": "regenE5_head", "agg": "dmean", "lam_rel": 0.3, "beta": 1.0, "target": "C8",
        "why": "SAME unified config as K1 (this is the generality test: one config, two targets), "
               "dev pooled +0.387 (p=0.001), sign-consistent.",
    },
    "K3_leverage_cachedE20_l1_b1_C7": {
        "source": "cachedE20_full", "agg": "dmean", "lam_rel": 1.0, "beta": 1.0, "target": "C7",
        "why": "strongest single-target leverage win on the cached E=20 full-model Gram, dev "
               "pooled +0.316, sign-consistent vs GradDot_dmean.",
    },
}
ALPHA_ABS = 0.05 / len(PREREG_K)


def _sources():
    if "src" not in _CACHE:
        members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                         if os.path.exists(os.path.join(x, "final.pt")))
        from if_repair import b1_layerwise as B1
        _CACHE["src"] = {"cachedE20_full": D.cache_for("bc_s10"),
                         "regenE5_head": B1.build_ensemble(members)["head"]}
    return _CACHE["src"]


_CACHE = {}
_SCORECACHE = {}


def scores_for(spec):
    """-> (estimator_scores_target, graddot_dmean_scores_target) for one hypothesis."""
    Z = _sources()[spec["source"]]
    key = (spec["source"], spec["lam_rel"], spec["beta"], spec["agg"])
    if key not in _SCORECACHE:
        _SCORECACHE[key] = family_scores(Z, spec["lam_rel"], spec["beta"], spec["agg"])
    gkey = ("gd", spec["source"])
    if gkey not in _SCORECACHE:
        _SCORECACHE[gkey] = scores_graddot(Z, normalize_per_member=True)
    t = spec["target"]
    return _SCORECACHE[key][t], _SCORECACHE[gkey][t]


def paired_bootstrap(px, pg, o, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(o)
    d0 = spearman(px, o) - spearman(pg, o)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rx, rg = spearman(px[i], o[i]), spearman(pg[i], o[i])
        if np.isfinite(rx) and np.isfinite(rg):
            diffs.append(rx - rg)
    diffs = np.array(diffs)
    return d0, (float((diffs <= 0).mean()) if len(diffs) else np.nan)


def mask_pred(sc, masks):
    return np.array([sum(sc.get(dd, 0.0) for dd in m["demos"]) for m in masks])


def evaluate_family(campaign, masks):
    rows = []
    for name, spec in PREREG_K.items():
        t = spec["target"]
        xs, gs = scores_for(spec)
        raw = F.campaign_outcomes(campaign, "plain", targets=(t,))[t]
        ceil = F.split_half_ceiling(raw)["ceiling"]
        obs = F.seed_mean(raw)
        out = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
        ok = np.isfinite(out)
        px, pg, o = mask_pred(xs, masks)[ok], mask_pred(gs, masks)[ok], out[ok]
        n = int(ok.sum())
        rho = spearman(px, o)
        p_abs = spearman_p_onesided(rho, n)
        d0, p_paired = paired_bootstrap(px, pg, o)
        rows.append({"name": name, "target": t, "source": spec["source"],
                     "lam_rel": spec["lam_rel"], "beta": spec["beta"], "n": n,
                     "lds": rho, "graddot_lds": spearman(pg, o), "ceiling": ceil,
                     "ratio": rho / ceil, "p_abs": p_abs, "paired_delta_rho": d0,
                     "paired_p": p_paired,
                     "PASS_abs": bool(np.isfinite(rho) and rho / ceil >= 0.5 and p_abs < ALPHA_ABS),
                     "PASS_paired": bool(np.isfinite(d0) and d0 > 0 and p_paired < 0.05)})
    return pd.DataFrame(rows)


def dev_pooled():
    """Pooled G/H/I paired vs GradDot_dmean -- the dev evidence behind PREREG_K (NOT a confirmation)."""
    draws = [(D.demo_masks(), "A"),
             (RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0], "B"),
             (RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0], "I")]
    rows = []
    for name, spec in PREREG_K.items():
        t = spec["target"]
        xs, gs = scores_for(spec)
        pool = {"px": [], "pg": [], "o": []}
        for ms, camp in draws:
            masks = [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in ms]
            raw = F.campaign_outcomes(camp, "plain", targets=(t,))[t]
            obs = F.seed_mean(raw)
            out = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
            ok = np.isfinite(out)
            pool["px"] += list(mask_pred(xs, masks)[ok])
            pool["pg"] += list(mask_pred(gs, masks)[ok])
            pool["o"] += list(out[ok])
        px, pg, o = (np.array(pool[k]) for k in ("px", "pg", "o"))
        d0, pp = paired_bootstrap(px, pg, o)
        rows.append({"name": name, "target": t, "pooled_delta_rho": d0, "pooled_p": pp})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev", choices=["dev", "K"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.mode == "dev":
        df = dev_pooled()
        df.to_csv(os.path.join(RESULTS, "confirm_kseries_DEVCHECK.csv"), index=False)
        print("=" * 84)
        print("K DEV EVIDENCE: PREREG_K estimators vs GradDot_dmean, pooled G/H/I (not a confirmation)")
        print("=" * 84)
        print(df.round(4).to_string(index=False))
    else:
        masks = [{"mask_id": m["mask_id"], "demos": m["demos"]}
                 for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_K, prefix="K")[0]]
        df = evaluate_family("K", masks)
        df.to_csv(os.path.join(RESULTS, "confirm_kseries.csv"), index=False)
        print("=" * 104)
        print(f"CAMPAIGN K CONFIRMATION (leverage family), Bonferroni abs alpha = {ALPHA_ABS:.4f}")
        print("=" * 104)
        print(df[["name", "target", "lds", "graddot_lds", "ceiling", "ratio", "p_abs", "PASS_abs",
                  "paired_delta_rho", "paired_p", "PASS_paired"]].round(4).to_string(index=False))
        both = int((df.PASS_abs & df.PASS_paired).sum())
        print(f"\nABSOLUTE passes: {int(df.PASS_abs.sum())}/3   PAIRED passes: "
              f"{int(df.PASS_paired.sum())}/3   BOTH bars: {both}/3")
        print("GENERALITY (>=3 paired) CONFIRMED" if int(df.PASS_paired.sum()) >= 3
              else f"leverage family confirmed OOS on {int(df.PASS_paired.sum())} target(s) (paired bar)")


if __name__ == "__main__":
    main()
