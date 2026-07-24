"""Campaign J -- the pass-4 out-of-sample confirmation, preregistered, computed once.

Passes 1-3 taught the lesson the hard way (BLOCKERS #20): an estimator selected on G/H/I and then
"confirmed" on a bootstrap of those same masks is not out of sample -- KFAC-on-embed won 100% of B8
draws and then LOST on the clean I-series. Pass 4's dev candidates were all selected on the pooled
G/H/I draws, so they need a genuinely fresh draw. Campaign J (seed 20260723, disjoint from G/H/I;
tests/test_jseries.py) is that draw. This file freezes the hypotheses BEFORE campaign J exists and
scores each exactly once.

TWO BARS per hypothesis (BLOCKERS #20):
  ABSOLUTE  ratio-to-ceiling >= 0.5 and p < Bonferroni alpha (0.05 / |family|).
  PAIRED    beats GradDot_dmean on the SAME masks, one-sided mask-bootstrap p < 0.05.

Each estimator's per-demo scores are FIXED by the dev ensembles (cached E=20 or regen E=5); only the
outcome (campaign J masks) is new -- exactly the confirm_iseries pattern. Each estimator is paired
against the GradDot_dmean computed on ITS OWN ensemble/space (self-consistent; BLOCKERS #6).

PREREG_J is frozen below and this file is committed while campaign J has ZERO runs (verifiable in
git history, as confirm3.py and confirm_iseries.py were).
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
from if_repair.eval import build_scores  # noqa: E402

D.add_repo_paths()
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ---------------------------------------------------------------- FROZEN prereg (edit ONLY before J)
# Each hypothesis: the dev finding it confirms, the estimator+config, the target, and which GradDot
# it is paired against. All three cleared the pass-4 dev screen (pooled G/H/I paired Delta >= +0.15).
PREREG_J = {
    "J1_relatif_lin_unitL2_C5": {
        "estimator": "relatif_lin_unitL2", "target": "C5", "baseline": "graddot_cachedE20_full",
        "why": "W4: RelatIF K[d,t]/G[d,d], per-member unit-L2 then mean, on the cached E=20 Gram. "
               "Pooled G/H/I paired Delta_rho +0.210 (p=0.016); positive on all three draws.",
    },
    "J2_trak_head_lam0p1_C2": {
        "estimator": "trak_head_lam0.1", "target": "C2", "baseline": "graddot_regenE5_head",
        "why": "W3: exact-dual TRAK on the regen E=5 head-Phi Gram, lambda_rel=0.1. Pooled G/H/I "
               "paired Delta_rho +0.222 (p=0.016). Inverting the head Gram helps C2.",
    },
    "J3_surrogate_loo_lam0p001_C5": {
        "estimator": "surrogate_loo_lam0.001", "target": "C5", "baseline": "graddot_regenE5_head",
        "why": "W2: exact frozen-trunk head surrogate-LOO, lambda_rel=0.001. Pooled G/H/I paired "
               "Delta_rho +0.222 (p=0.044). Independent (exact-LOO) replication of the C5 win.",
    },
}
ALPHA_ABS = 0.05 / len(PREREG_J)     # Bonferroni over the preregistered family


# ---------------------------------------------------------------- estimator builders (dev ensembles)
def regen_members():
    return sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                  if os.path.exists(os.path.join(x, "final.pt")))


_CACHE = {}


def _ens_head():
    if "ens_head" not in _CACHE:
        from if_repair import b1_layerwise as B1
        _CACHE["ens_head"] = B1.build_ensemble(regen_members())["head"]
    return _CACHE["ens_head"]


def estimator_scores(estimator):
    """-> {demo_id: score} for the estimator's target-agnostic full score dict {target:{demo:score}}."""
    if estimator == "relatif_lin_unitL2":
        from if_repair.b14_rescoring import relatif_scores
        return relatif_scores(D.cache_for("bc_s10"), 1.0, aggregate="unitl2_then_mean")
    if estimator == "trak_head_lam0.1":
        from if_repair.b13_trak import scores_from
        return scores_from(_ens_head(), "TRAK", 0.1, "dmean")
    if estimator == "surrogate_loo_lam0.001":
        from if_repair import b12_headloo as B12
        agg = None
        members = regen_members()
        for m in members:
            sc, _ = B12.member_scores(m)
            if agg is None:
                agg = {t: {d: 0.0 for d in sc[0.001][t]} for t in B12.DEV_TARGETS}
            for t in B12.DEV_TARGETS:
                for d, v in sc[0.001][t].items():
                    agg[t][d] += v / len(members)
        return agg
    raise KeyError(estimator)


def baseline_scores(baseline):
    if baseline == "graddot_cachedE20_full":
        return build_scores({"kind": "GradDot", "normalize": "dmean", "aggregator": "mean"},
                            "bc_s10")
    if baseline == "graddot_regenE5_head":
        return scores_graddot(_ens_head(), normalize_per_member=True)
    raise KeyError(baseline)


# ---------------------------------------------------------------- scoring
def paired_bootstrap(px, pg, out, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    px, pg, out = map(lambda a: np.asarray(a, float), (px, pg, out))
    n = len(out)
    d0 = spearman(px, out) - spearman(pg, out)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rx, rg = spearman(px[i], out[i]), spearman(pg[i], out[i])
        if np.isfinite(rx) and np.isfinite(rg):
            diffs.append(rx - rg)
    diffs = np.array(diffs)
    return d0, (float((diffs <= 0).mean()) if len(diffs) else np.nan)


def mask_pred(sc, masks):
    return np.array([sum(sc.get(dd, 0.0) for dd in m["demos"]) for m in masks])


def confirm(campaign, prefix, seed):
    masks = [{"mask_id": m["mask_id"], "demos": m["demos"]}
             for m in RT.fresh_demo_masks(seed=seed, prefix=prefix)[0]]
    rows = []
    for name, spec in PREREG_J.items():
        t = spec["target"]
        xs = estimator_scores(spec["estimator"])[t]
        gs = baseline_scores(spec["baseline"])[t]
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
        rows.append({
            "name": name, "target": t, "estimator": spec["estimator"], "n": n,
            "lds": rho, "graddot_lds": spearman(pg, o), "ceiling": ceil,
            "ratio": rho / ceil, "bar": 0.5 * ceil, "p_abs": p_abs, "alpha_abs": ALPHA_ABS,
            "paired_delta_rho": d0, "paired_p": p_paired,
            "PASS_abs": bool(np.isfinite(rho) and rho / ceil >= 0.5 and p_abs < ALPHA_ABS),
            "PASS_paired": bool(np.isfinite(d0) and d0 > 0 and p_paired < 0.05),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="J", choices=["I", "J"],
                    help="J = the real confirmation; I = wiring smoke test on the existing draw")
    a = ap.parse_args()
    if a.campaign == "J":
        df = confirm("J", "J", RT.FRESH_MASK_SEED_J)
        outname = "confirm_jseries.csv"
        title = f"CAMPAIGN J CONFIRMATION (Bonferroni abs alpha = {ALPHA_ABS:.4f}), 24 fresh masks"
    else:
        df = confirm("I", "I", RT.FRESH_MASK_SEED_I)
        outname = "confirm_jseries_SMOKE_iseries.csv"
        title = "WIRING SMOKE TEST on the I-series (NOT a confirmation; checks the scorer plumbing)"
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, outname), index=False)
    print("=" * 112)
    print(title)
    print("=" * 112)
    print(df[["name", "target", "estimator", "lds", "graddot_lds", "ceiling", "ratio", "p_abs",
              "PASS_abs", "paired_delta_rho", "paired_p", "PASS_paired"]].round(4).to_string(index=False))
    print(f"\nABSOLUTE passes: {int(df.PASS_abs.sum())}/{len(df)}   "
          f"PAIRED passes: {int(df.PASS_paired.sum())}/{len(df)}   "
          f"BOTH bars: {int((df.PASS_abs & df.PASS_paired).sum())}/{len(df)}")


if __name__ == "__main__":
    main()
