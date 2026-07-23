"""B8 -- the sampling distribution of the demo-grain LDS over MASK DRAWS.

Pass 3's most consequential finding was a two-point observation: GradDot_dmean on C1 scores
0.509 on the archived 24 masks and 0.337 on a fresh 24, with C5 flipping 0.401 -> -0.106
(BLOCKERS #17). Two points cannot say whether that gap is typical or a coincidence, and every
number this project reports is a single draw.

It can be turned into a distribution for ZERO GPU. Campaigns A and B produced outcomes for the
24 archived (G-series) and 24 fresh (H-series) masks. Both draws come from the SAME Stage-G
generator (`masks.build_demo_masks`, seeds 11 and 4711), both are 68-demo within-cluster
stratified masks over the same 135 demos, and both were scored on the same fixed 14,461-frame
held-out bank by the same trainer. They are exchangeable, so:

  * POOLED-48: evaluate on all 48 at once -- ~sqrt(2) more power than anything reported so far,
    and the best point estimate available for any estimator.
  * BOOTSTRAP-24: resample 24-mask subsets from the 48 and recompute LDS, ceiling and ratio on
    each. That IS the sampling distribution of a 24-mask result, and the two published points
    can be located inside it.

Seed depth is matched at 6 (campaign A is truncated to its first 6 seeds) so the two halves
contribute outcomes of the same reliability; the ceiling is recomputed per subset, because the
ceiling is a mask-draw quantity too.

WHAT THIS IS AND IS NOT. It is descriptive: it measures the variance of a statistic. It is NOT a
hypothesis test, and no claim selected on the G-series may be "confirmed" here -- the G-series is
consumed dev data and the H-series is a consumed confirmatory family. Pooling them is legitimate
for characterising spread and illegitimate for adjudicating a selected hypothesis.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402

D.add_repo_paths()
import masks as MK  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
N_SEEDS = 6            # campaign B's depth; campaign A is truncated to match
N_BOOT = 2000


# --------------------------------------------------------------------------- the pool
def pooled(weighting="plain", targets=D.ALL_TARGETS, n_seeds=None):
    """-> (masks, {target: {mask: {seed_slot: v}}}) over the 48 G+H masks at matched depth."""
    from if_repair import retrain as RT
    gm = [{"mask_id": m["mask_id"], "demos": m["demos"]}
          for m in MK.demo_mask_manifest()["masks"]]
    hm, _ = RT.fresh_demo_masks()
    hm = [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in hm]

    ra = F.campaign_outcomes("A", weighting, targets=targets)
    rb = F.campaign_outcomes("B", weighting, targets=targets)
    out = {}
    for t in targets:
        d = {}
        for src in (ra, rb):
            if t not in src:
                continue
            for m, per_seed in src[t].items():
                keys = sorted(per_seed)[:(n_seeds or N_SEEDS)]   # matched depth, fixed slots
                d[str(m)] = {i: per_seed[k] for i, k in enumerate(keys)}
        out[t] = d
    return gm + hm, out


def lds_on(scores, masks, obs_t):
    pred = np.array([sum(scores.get(d, 0.0) for d in m["demos"]) for m in masks])
    out = np.array([obs_t.get(m["mask_id"], np.nan) for m in masks])
    ok = np.isfinite(out)
    rho = spearman(pred[ok], out[ok])
    return rho, spearman_p_onesided(rho, int(ok.sum())), int(ok.sum())


def datamodel_loo(masks, obs_t, model="lasso"):
    """Leave-one-mask-out, the only honest path for an outcome-fitted estimator (BLOCKERS #7)."""
    from if_repair.datamodel import MODELS, select_alpha
    Z = D.cache_for("bc_s10")
    demo_ids = list(Z["train_ids"])
    idx = {d: i for i, d in enumerate(demo_ids)}
    X = np.zeros((len(masks), len(demo_ids)))
    for r, m in enumerate(masks):
        for d in m["demos"]:
            if d in idx:
                X[r, idx[d]] = 1.0
    y = np.array([obs_t.get(m["mask_id"], np.nan) for m in masks], float)
    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan, int(ok.sum())
    alpha, _ = select_alpha(X[ok], y[ok], model, folds=5)
    pred, out = [], []
    for r in np.nonzero(ok)[0]:
        keep = ok.copy()
        keep[r] = False
        fit = MODELS[model](alpha).fit(X[keep], y[keep])
        pred.append(float(np.dot(X[r], fit.coef_)))
        out.append(y[r])
    rho = spearman(pred, out)
    return rho, spearman_p_onesided(rho, len(pred)), len(pred)


# --------------------------------------------------------------------------- estimators
def estimators():
    """{name: {target: {demo: score}}} -- the ones worth locating in the distribution."""
    import glob
    from if_repair import b1_layerwise as B1
    from if_repair import gradients as GR
    from if_repair import b4_tracin as B4
    from if_repair import b3_kfac as B3
    import json

    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    ens = B1.build_ensemble(members)
    out = {"GradDot_ALL": scores_graddot(ens["ALL"], normalize_per_member=True),
           "GradDot_head": scores_graddot(ens["head"], normalize_per_member=True)}
    if os.path.exists(B4.CACHE):
        z = np.load(B4.CACHE, allow_pickle=True)
        ix = json.loads(str(z["index"]))
        Z = B4.tracin_Z(ix, z, "head", list(range(len(ix["ckpts"]))), lr_weight=True)
        out["TracIn_head_last5_lr"] = scores_graddot(Z, normalize_per_member=True)
    if os.path.exists(B3.CACHE):
        e = B3.build_ensemble()
        out["KFAC_embed_1e-4"] = scores_graddot(e[("embed", 1e-4, "kfac")],
                                                normalize_per_member=True)
    return out


# --------------------------------------------------------------------------- driver
def run(target="C1", weighting="plain", n_boot=N_BOOT, seed=0, n_seeds=None):
    masks, obs_raw = pooled(weighting, targets=(target,), n_seeds=n_seeds)
    raw = obs_raw[target]
    ests = estimators()
    rng = np.random.default_rng(seed)
    gids = {m["mask_id"] for m in masks if m["mask_id"].startswith("G")}

    def eval_subset(sub_masks):
        sub_raw = {m["mask_id"]: raw[m["mask_id"]] for m in sub_masks
                   if m["mask_id"] in raw}
        c = F.split_half_ceiling(sub_raw)["ceiling"]
        obs = F.seed_mean(sub_raw)
        row = {"ceiling": c}
        for name, sc in ests.items():
            rho, p, n = lds_on(sc[target], sub_masks, obs)
            row[name] = rho / c if (c and np.isfinite(rho)) else np.nan
        rho, p, n = datamodel_loo(sub_masks, obs)
        row["datamodel_LOO"] = rho / c if (c and np.isfinite(rho)) else np.nan
        return row

    rows = []
    full = eval_subset(masks); full["draw"] = "POOLED_48"; rows.append(full)
    g = eval_subset([m for m in masks if m["mask_id"] in gids]); g["draw"] = "G_only_24"
    rows.append(g)
    h = eval_subset([m for m in masks if m["mask_id"] not in gids]); h["draw"] = "H_only_24"
    rows.append(h)

    boot = []
    for b in range(n_boot):
        sel = rng.choice(len(masks), 24, replace=False)
        r = eval_subset([masks[i] for i in sel])
        r["draw"] = f"boot{b}"
        boot.append(r)
    return pd.DataFrame(rows), pd.DataFrame(boot)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="C1,C5,C2")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS,
                    help="seed depth, matched across both mask sets")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    allboot, allfix = [], []
    for t in a.targets.split(","):
        fixed, boot = run(t, n_boot=a.n_boot, n_seeds=a.n_seeds)
        fixed.insert(0, "target", t); boot.insert(0, "target", t)
        allfix.append(fixed); allboot.append(boot)

        cols = [c for c in fixed.columns if c not in ("target", "draw", "ceiling")]
        print("=" * 104)
        print(f"B8 -- {t}: ratio-to-ceiling over mask draws (pooled 48 = G-series + H-series, "
              f"matched at {a.n_seeds} seeds)")
        print("=" * 104)
        print(fixed[["draw", "ceiling"] + cols].round(3).to_string(index=False))
        print(f"\n  bootstrap over {len(boot)} random 24-mask subsets of the 48:")
        print(f"  {'estimator':22s} {'mean':>7s} {'sd':>7s} {'p05':>7s} {'p50':>7s} "
              f"{'p95':>7s} {'range':>7s}")
        for c in cols:
            v = boot[c].dropna().values
            if not len(v):
                continue
            print(f"  {c:22s} {v.mean():7.3f} {v.std():7.3f} "
                  f"{np.percentile(v,5):7.3f} {np.percentile(v,50):7.3f} "
                  f"{np.percentile(v,95):7.3f} "
                  f"{np.percentile(v,95)-np.percentile(v,5):7.3f}")
        print(f"\n  ceiling itself over subsets: mean {boot.ceiling.mean():.3f} "
              f"sd {boot.ceiling.std():.3f} "
              f"[{np.percentile(boot.ceiling,5):.3f}, {np.percentile(boot.ceiling,95):.3f}]")

        # PAIRED differences. The mask draw is a shared nuisance: every estimator is scored on
        # the SAME subset, so the draw-to-draw noise largely cancels in a difference. Comparing
        # levels at n=24 is hopeless (sd ~0.15 against a 0.5 bar); comparing two estimators on
        # the same masks need not be.
        base = "GradDot_ALL"
        print(f"\n  PAIRED vs {base} on the same subsets "
              f"(sd of the difference, and how often it is positive):")
        print(f"  {'estimator':22s} {'mean_d':>7s} {'sd_d':>7s} {'p05':>7s} {'p95':>7s} "
              f"{'win%':>6s}")
        for c in cols:
            if c == base:
                continue
            d = (boot[c] - boot[base]).dropna().values
            if not len(d):
                continue
            print(f"  {c:22s} {d.mean():+7.3f} {d.std():7.3f} "
                  f"{np.percentile(d,5):+7.3f} {np.percentile(d,95):+7.3f} "
                  f"{100*(d > 0).mean():5.1f}%")

    pd.concat(allfix, ignore_index=True).to_csv(
        os.path.join(RESULTS, "b8_maskdraw_fixed.csv"), index=False)
    pd.concat(allboot, ignore_index=True).to_csv(
        os.path.join(RESULTS, "b8_maskdraw_bootstrap.csv"), index=False)
    print("\nwrote results/b8_maskdraw_{fixed,bootstrap}.csv")


if __name__ == "__main__":
    main()
