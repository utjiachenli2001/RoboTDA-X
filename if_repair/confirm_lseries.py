"""Campaign L -- the pass-6 capstone: a SECOND independent fresh draw for the C5 leverage win.

Campaign J confirmed RelatIF/C5 on the absolute half-ceiling bar out of sample (ratio 0.533,
p=0.005) with a large paired advantage over GradDot (+0.34), but the paired one-sided bootstrap p
(0.073) narrowly missed 0.05 at n=24. Pass-6 P2 showed the two C5 views (RelatIF, the exact
frozen-trunk surrogate-LOO) are near-orthogonal (rank-corr 0.20) and their z-score ensemble is the
strongest single C5 attributor found (+0.298 pooled dev). Campaign L (seed 20260725, disjoint from
G/H/I/J/K) is a second fresh draw so the C5 result is replicated out of sample, and the ensemble --
which was never preregistered -- gets its own confirmation.

PREREG_L (3 hypotheses, Bonferroni abs alpha=0.0167), each vs GradDot_dmean on its own ensemble.
Frozen and committed with campaign L at ZERO runs. Two bars per protocol.
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

D.add_repo_paths()
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ---------------------------------------------------------------- FROZEN prereg (edit ONLY before L)
# L1 the C5 ensemble (z-avg of RelatIF + exact surrogate-LOO), paired vs GradDot_dmean-cached (the
#    stronger of the two same-ensemble GradDots on C5); L2/L3 the components, replicating campaign J.
PREREG_L = {
    "L1_C5_ensemble_relatif+surrogate": {
        "kind": "ensemble", "target": "C5", "baseline": "cached",
        "why": "z-avg of RelatIF (cached E=20) and exact surrogate-LOO (regen E=5 head); dev pooled "
               "+0.298 (p=0.0018), the strongest C5 attributor; never preregistered until now.",
    },
    "L2_relatif_lin_unitL2_C5": {
        "kind": "relatif", "target": "C5", "baseline": "cached",
        "why": "RelatIF K/G_dd unit-L2 (cached E=20); campaign J ratio 0.533 ABS PASS -- L is the "
               "second fresh draw.",
    },
    "L3_surrogate_loo_C5": {
        "kind": "surrogate", "target": "C5", "baseline": "head",
        "why": "exact frozen-trunk surrogate-LOO lam0.001 (regen E=5); campaign J ratio 0.479.",
    },
}
ALPHA_ABS = 0.05 / len(PREREG_L)
_C = {}


def _ens_head():
    if "h" not in _C:
        members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                         if os.path.exists(os.path.join(x, "final.pt")))
        from if_repair import b1_layerwise as B1
        _C["members"] = members
        _C["h"] = B1.build_ensemble(members)["head"]
    return _C["h"]


def _relatif():
    from if_repair.b14_rescoring import relatif_scores
    return relatif_scores(D.cache_for("bc_s10"), 1.0, aggregate="unitl2_then_mean")


def _surrogate():
    if "surr" not in _C:
        from if_repair import b12_headloo as B12
        _ens_head()
        agg = None
        for m in _C["members"]:
            sc, _ = B12.member_scores(m)
            if agg is None:
                agg = {t: {d: 0.0 for d in sc[0.001][t]} for t in B12.DEV_TARGETS}
            for t in B12.DEV_TARGETS:
                for d, v in sc[0.001][t].items():
                    agg[t][d] += v / len(_C["members"])
        _C["surr"] = agg
    return _C["surr"]


def _zavg(dicts, demo_ids):
    mats = []
    for sc in dicts:
        v = np.array([sc.get(d, 0.0) for d in demo_ids], float)
        mats.append((v - v.mean()) / (v.std() + 1e-30))
    m = np.mean(mats, axis=0)
    return {d: float(m[i]) for i, d in enumerate(demo_ids)}


def scores_for(spec):
    demo_ids = list(D.cache_for("bc_s10")["train_ids"])
    t = spec["target"]
    if spec["kind"] == "ensemble":
        est = _zavg([_relatif()[t], _surrogate()[t]], demo_ids)
    elif spec["kind"] == "relatif":
        est = _relatif()[t]
    else:
        est = _surrogate()[t]
    if spec["baseline"] == "cached":
        base = scores_graddot(D.cache_for("bc_s10"), normalize_per_member=True)[t]
    else:
        base = scores_graddot(_ens_head(), normalize_per_member=True)[t]
    return est, base


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


def evaluate(campaign, masks):
    rows = []
    for name, spec in PREREG_L.items():
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
        rows.append({"name": name, "target": t, "n": n, "lds": rho, "graddot_lds": spearman(pg, o),
                     "ceiling": ceil, "ratio": rho / ceil, "p_abs": p_abs,
                     "paired_delta_rho": d0, "paired_p": p_paired,
                     "PASS_abs": bool(np.isfinite(rho) and rho / ceil >= 0.5 and p_abs < ALPHA_ABS),
                     "PASS_paired": bool(np.isfinite(d0) and d0 > 0 and p_paired < 0.05)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev", choices=["dev", "L"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.mode == "dev":
        # pooled G/H/I evidence behind PREREG_L (not a confirmation)
        draws = [(D.demo_masks(), "A"),
                 (RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0], "B"),
                 (RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0], "I")]
        rows = []
        for name, spec in PREREG_L.items():
            t = spec["target"]
            xs, gs = scores_for(spec)
            pool = {"px": [], "pg": [], "o": []}
            for ms, camp in draws:
                masks = [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in ms]
                raw = F.campaign_outcomes(camp, "plain", targets=(t,))[t]
                obs = F.seed_mean(raw)
                out = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
                ok = np.isfinite(out)
                pool["px"] += list(mask_pred(xs, masks)[ok]); pool["pg"] += list(mask_pred(gs, masks)[ok])
                pool["o"] += list(out[ok])
            px, pg, o = (np.array(pool[k]) for k in ("px", "pg", "o"))
            d0, pp = paired_bootstrap(px, pg, o)
            rows.append({"name": name, "pooled_delta_rho": d0, "pooled_p": pp})
        df = pd.DataFrame(rows)
        print("K DEV EVIDENCE behind PREREG_L (pooled G/H/I vs GradDot_dmean):")
        print(df.round(4).to_string(index=False))
    else:
        masks = [{"mask_id": m["mask_id"], "demos": m["demos"]}
                 for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_L, prefix="L")[0]]
        df = evaluate("L", masks)
        df.to_csv(os.path.join(RESULTS, "confirm_lseries.csv"), index=False)
        print("=" * 104)
        print(f"CAMPAIGN L CONFIRMATION (C5 capstone), Bonferroni abs alpha = {ALPHA_ABS:.4f}")
        print("=" * 104)
        print(df[["name", "target", "lds", "graddot_lds", "ceiling", "ratio", "p_abs", "PASS_abs",
                  "paired_delta_rho", "paired_p", "PASS_paired"]].round(4).to_string(index=False))
        print(f"\nABSOLUTE passes: {int(df.PASS_abs.sum())}/3   PAIRED passes: "
              f"{int(df.PASS_paired.sum())}/3")


if __name__ == "__main__":
    main()
