"""PASS 6, P6.1 -- the multi-view fixed estimator (MV): the generality shot.

Pass 5 proved the leverage family wins as a FAMILY but not as a single estimator (best unified
config: 2 targets vs the strict bar). The winning Phi is target-specific -- C5/C7 on the cached
E=20 full-model Gram, C2/C8 on the regen E=5 head Gram -- and P2 showed the two C5 views (RelatIF
and exact surrogate-LOO) rank-correlate only 0.20, i.e. they are near-orthogonal. So a FIXED
per-demo combination across both Phis plus the exact LOO should inherit the UNION of the win sets
{C2,C8} u {C5,C7} without any per-target selection.

Three pre-declared combination rules (frozen before scoring; no post-hoc variants):
  MV-A : equal-weight z-score average of 3 views
  MV-B : equal-weight rank average of 3 views
  MV-C : z-score average of views (i)+(ii) only (drop the surrogate)
Views:
  (i)   leverage S_m(lam_rel=0.3, beta=1) on the regen E=5 head Gram
  (ii)  leverage S_m(lam_rel=0.3, beta=1) on the cached E=20 full-model Gram
  (iii) exact frozen-trunk surrogate-LOO (b12_headloo, lam_rel=0.001), regen E=5

THE BAR (preregistered form): MV spans two ensembles, so "same-ensemble GradDot" is ambiguous.
A target counts only if MV beats the MAX of GradDot_dmean(cached E=20) and GradDot_dmean(regen
head) on that target, paired on shared masks, on EVERY draw (sign-consistent) with pooled Delta
>= +0.10. Strictly harder than either single bar; preempts the BLOCKERS #1/#23 objection.

Win: one MV variant clears the bar on >=3 targets -> the generality claim, to be confirmed on a
fresh draw (campaign L). Zero GPU beyond the surrogate feature recompute.
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
from if_repair import retrain as RT  # noqa: E402
from if_repair import gradients as GR  # noqa: E402
from if_repair.p1_leverage_family import family_scores  # noqa: E402

D.add_repo_paths()
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TARGETS = tuple(f"C{i}" for i in range(1, 10))


def zscore(vec):
    v = np.asarray(vec, float)
    return (v - v.mean()) / (v.std() + 1e-30)


def rankscore(vec):
    v = np.asarray(vec, float)
    order = v.argsort().argsort().astype(float)
    return (order - order.mean()) / (order.std() + 1e-30)


def combine(views, demo_ids, target, how):
    """views: list of {target:{demo:score}} -> {demo: combined score}."""
    mats = []
    for v in views:
        if target not in v:
            continue
        vec = np.array([v[target].get(d, 0.0) for d in demo_ids])
        mats.append(zscore(vec) if how == "z" else rankscore(vec))
    if not mats:
        return None
    m = np.mean(mats, axis=0)
    return {d: float(m[i]) for i, d in enumerate(demo_ids)}


def mask_pred(sc, masks):
    return np.array([sum(sc.get(dd, 0.0) for dd in m["demos"]) for m in masks])


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


def main():
    members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(x, "final.pt")))
    from if_repair import b1_layerwise as B1
    ens = B1.build_ensemble(members)
    Zh, Zc = ens["head"], D.cache_for("bc_s10")
    demo_ids = list(Zc["train_ids"])

    view_i = family_scores(Zh, 0.3, 1.0, "dmean")
    view_ii = family_scores(Zc, 0.3, 1.0, "dmean")
    # surrogate-LOO across ALL 9 targets, averaged over members
    from if_repair import b12_headloo as B12
    agg = None
    for m in members:
        sc, _ = B12.member_scores(m, targets=TARGETS)
        if agg is None:
            agg = {t: {d: 0.0 for d in sc[0.001][t]} for t in TARGETS}
        for t in TARGETS:
            for d, v in sc[0.001][t].items():
                agg[t][d] += v / len(members)
    view_iii = agg

    gd_c = scores_graddot(Zc, normalize_per_member=True)
    gd_h = scores_graddot(Zh, normalize_per_member=True)

    draws = [(D.demo_masks(), "A"),
             (RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0], "B"),
             (RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0], "I")]
    # precompute outcomes per (draw,target)
    OBS = {}
    for k, (ms, camp) in enumerate(draws):
        masks = [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in ms]
        OBS[k] = {"masks": masks}
        for t in TARGETS:
            raw = F.campaign_outcomes(camp, "plain", targets=(t,))[t]
            obs = F.seed_mean(raw)
            OBS[k][t] = np.array([obs.get(m["mask_id"], np.nan) for m in masks])

    MVS = {"MV-A": (["i", "ii", "iii"], "z"), "MV-B": (["i", "ii", "iii"], "rank"),
           "MV-C": (["i", "ii"], "z")}
    viewmap = {"i": view_i, "ii": view_ii, "iii": view_iii}

    rows = []
    for mvname, (vs, how) in MVS.items():
        for t in TARGETS:
            mv = combine([viewmap[v] for v in vs], demo_ids, t, how)
            if mv is None:
                continue
            # per draw: MV lds vs max(gd_c, gd_h); pooled too
            perdraw_win = []
            pooled = {"px": [], "gc": [], "gh": [], "o": []}
            for k in range(3):
                masks, out = OBS[k]["masks"], OBS[k][t]
                ok = np.isfinite(out)
                pmv = mask_pred(mv, masks)[ok]
                pgc = mask_pred(gd_c[t], masks)[ok]
                pgh = mask_pred(gd_h[t], masks)[ok]
                o = out[ok]
                lds_mv, lds_gc, lds_gh = spearman(pmv, o), spearman(pgc, o), spearman(pgh, o)
                perdraw_win.append(lds_mv > max(lds_gc, lds_gh))
                pooled["px"] += list(pmv); pooled["gc"] += list(pgc)
                pooled["gh"] += list(pgh); pooled["o"] += list(o)
            px = np.array(pooled["px"]); o = np.array(pooled["o"])
            gc = np.array(pooled["gc"]); gh = np.array(pooled["gh"])
            lds_mv = spearman(px, o)
            ref = max(spearman(gc, o), spearman(gh, o))
            # paired p vs the STRONGER baseline
            stronger = gc if spearman(gc, o) >= spearman(gh, o) else gh
            d0, pp = paired_bootstrap(px, stronger, o)
            rows.append({"mv": mvname, "target": t, "mv_lds": lds_mv, "max_graddot_lds": ref,
                         "pooled_delta_vs_max": lds_mv - ref, "paired_p": pp,
                         "sign_consistent": bool(all(perdraw_win)),
                         "qualifies": bool(all(perdraw_win) and (lds_mv - ref) >= 0.10)})
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "p6_multiview.csv"), index=False)

    print("=" * 100)
    print("P6.1 -- multi-view estimator vs MAX(GradDot_dmean cached, GradDot_dmean head), pooled G/H/I")
    print("=" * 100)
    for mvname in MVS:
        sub = df[df.mv == mvname]
        q = sub[sub.qualifies]
        print(f"\n{mvname}: qualifying targets (sign-consistent + pooled Delta>=+0.10 vs max-bar): "
              f"{list(q.target)}  ({len(q)})")
        print(sub[["target", "mv_lds", "max_graddot_lds", "pooled_delta_vs_max", "paired_p",
                   "sign_consistent", "qualifies"]].round(3).to_string(index=False))
    best = max(MVS, key=lambda mv: int(df[(df.mv == mv) & df.qualifies].shape[0]))
    nb = int(df[(df.mv == best) & df.qualifies].shape[0])
    print(f"\n>>> BEST: {best} qualifies on {nb} targets "
          f"({list(df[(df.mv == best) & df.qualifies].target)})")
    print(">>> GENERALITY (>=3, single estimator, strict max-bar, OOS-dev)" if nb >= 3
          else f">>> {nb} targets -- generality not reached at dev")


if __name__ == "__main__":
    main()
