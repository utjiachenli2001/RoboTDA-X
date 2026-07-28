"""PASS 8 W2 -- cluster-grain DESIGN measurement. Hypothesis-blind, zero GPU.

Everything here is computed and committed BEFORE p8_prereg.md names a hypothesis, because two of
the three stages would be indefensible otherwise:

  --stage ceiling     the absolute bar at cluster grain, from the 12 noise-ceiling masks x seeds
                      301-304 that Stage F already paid for.
  --stage allocation  paired sd as a function of (n_masks, seed_depth) AT CLUSTER GRAIN.
  --stage statistic   Kendall tau_b vs Spearman on split-half reliability and noise ONLY, never
                      on the contrast (BLOCKERS #30). Worth ~33% of CI width at demo grain.

WHY THE PASS-7 ALLOCATION RULE CANNOT SIMPLY BE INHERITED. BLOCKERS #29 measured masks beating
seeds ~5:1 and concluded "buy masks at depth 2". That measurement was taken where the mask supply
was effectively unlimited: demo masks are 68-of-135 subsets, so the pool is astronomically large.
At cluster grain the pool is C(9,5) = 126 masks TOTAL, and Stage F has already consumed 72 of
them. Masks are a depletable resource here and seeds are not, so the exchange rate has to be
re-measured against a hard cap rather than assumed. This stage reports the sd reachable at the
cap, which is what actually sizes campaign N.

The subsampling below draws from Stage F`s 72 masks (40 conditional per target) and from the 12
noise-ceiling masks, which carry 4 seeds each and are therefore the only place in the corpus where
cluster-grain seed depth beyond 2 can be observed at all.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import p8_cluster_grain as CG  # noqa: E402

P7.D.add_repo_paths()
import lds  # noqa: E402
import masks as MK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

F_SEEDS = [301, 302]
F_NC_SEEDS = [303, 304]
ALL_F_SEEDS = F_SEEDS + F_NC_SEEDS
CLUSTERS_PER_MASK = 5
N_CLUSTERS = 9
MASK_SPACE = math.comb(N_CLUSTERS, CLUSTERS_PER_MASK)   # 126
STAGE_F_USED = 72
SUB_SEED = 11
N_SUB = 400


def _targets():
    return sorted({c["target"] for c in P7.CONFIGS.values()})


# ------------------------------------------------------------------ stage: ceiling
def ceiling_rows():
    """Split-half noise ceiling at cluster grain, from the 12 replicate masks x 4 seeds.

    Conditional on target-in-mask, mirroring src/analysis.py: a mask that drops the target has an
    outcome dominated by that removal, so including it inflates the ceiling exactly as it inflates
    the LDS.
    """
    df = CG.stage_f_table()
    man = MK.cluster_mask_manifest()
    nc = set(man["noise_ceiling_masks"])
    incl = {m["mask_id"]: set(m["clusters"]) for m in man["masks"]}
    rows = []
    for t in P7.D.dataset.clusters() if hasattr(P7.D, "dataset") else _all_clusters(man):
        for key in (CG.PRIMARY_OUTCOME,):
            sub = df[(df.target == t) & (df.mask_id.isin(nc)) & (df.seed.isin(ALL_F_SEEDS))]
            obms = {}
            for mid, g in sub.groupby("mask_id"):
                obms[mid] = {int(s): float(v) for s, v in zip(g.seed, g[key])}
            cond = {m: v for m, v in obms.items() if t in incl.get(m, set())}
            if len(cond) < 3:
                continue
            c_all = lds.noise_ceiling(obms, seeds=tuple(ALL_F_SEEDS))
            c_cond = lds.noise_ceiling(cond, seeds=tuple(ALL_F_SEEDS))
            # A depth-2-only ceiling is NOT computable with this recipe: noise_ceiling builds
            # two DISJOINT seed pairs and so needs all 4 replicate seeds. Reporting one anyway
            # would mean inventing a different estimator and calling it the same bar.
            c_d2 = {"ceiling": float("nan")}
            rows.append({
                "stage": "ceiling", "grain": "cluster", "target": t, "outcome": key,
                "ceiling_conditional": c_cond["ceiling"],
                "ceiling_conditional_sb": c_cond.get("ceiling_sb"),
                "ci_lo": c_cond["ci95"][0], "ci_hi": c_cond["ci95"][1],
                "n_masks": c_cond["n_masks"],
                "ceiling_depth2_only_NOT_COMPUTABLE": c_d2["ceiling"],
                "ceiling_allmasks_INFLATED_BY_EXCLUSION": c_all["ceiling"],
                "n_masks_all": c_all["n_masks"],
            })
    return pd.DataFrame(rows)


def _all_clusters(man):
    return sorted({c for m in man["masks"] for c in m["clusters"]})


# ------------------------------------------------------------------ stage: allocation
def _pair_series(target, statname, masks, obs):
    """Per-mask prediction/outcome/baseline arrays for the target`s own frozen config."""
    cfg = next(c for c in P7.CONFIGS.values() if c["target"] == target)
    sc = cfg["scores"]()
    base = P7._graddot(cfg["baseline"])[target]
    use = [m for m in masks if target in m["clusters"] and m["mask_id"] in obs]
    o = np.array([obs[m["mask_id"]] for m in use], float)
    p = P7.mask_pred(sc, use)
    b = P7.mask_pred(base, use)
    ok = np.isfinite(o) & np.isfinite(p) & np.isfinite(b)
    return p[ok], b[ok], o[ok]


def allocation_rows():
    """sd of the paired delta vs n_masks (and vs seed depth) at cluster grain.

    n_masks is subsampled from the 40 conditional Stage F masks. seed depth is observable only on
    the 12 replicate masks, which is the honest limit of what this corpus can say about depth.
    """
    df = CG.stage_f_table()
    man = MK.cluster_mask_manifest()
    nc = set(man["noise_ceiling_masks"])
    masks = CG.cluster_masks()
    rng = np.random.default_rng(SUB_SEED)
    rows = []
    for t in _targets():
        obs = CG.seedmean(df, t, CG.PRIMARY_OUTCOME)
        for statname, fn in CG.STATS.items():
            p, b, o = _pair_series(t, statname, masks, obs)
            n_avail = len(o)
            for n in [k for k in (10, 15, 20, 25, 30, 40) if k <= n_avail]:
                ds = []
                for _ in range(N_SUB):
                    j = rng.choice(n_avail, n, replace=False)
                    if np.std(p[j]) == 0 or np.std(b[j]) == 0 or np.std(o[j]) == 0:
                        continue
                    ds.append(fn(p[j], o[j]) - fn(b[j], o[j]))
                if len(ds) < 20:
                    continue
                rows.append({"stage": "allocation", "axis": "masks", "target": t,
                             "statistic": statname, "n_masks": n, "seed_depth": 2,
                             "paired_sd": float(np.std(ds, ddof=1)),
                             "n_subsamples": len(ds), "n_available": n_avail})
            # seed depth, on the replicate masks only
            sub = df[(df.target == t) & (df.mask_id.isin(nc))]
            cond_ids = [m["mask_id"] for m in masks
                        if t in m["clusters"] and m["mask_id"] in set(sub.mask_id)]
            if len(cond_ids) >= 5:
                for depth in (1, 2, 3, 4):
                    ds = []
                    for _ in range(N_SUB):
                        seeds = list(rng.choice(ALL_F_SEEDS, depth, replace=False))
                        sm = {}
                        for mid, g in sub[sub.seed.isin(seeds)].groupby("mask_id"):
                            sm[mid] = float(g[CG.PRIMARY_OUTCOME].mean())
                        use = [m for m in masks if m["mask_id"] in sm and t in m["clusters"]]
                        if len(use) < 5:
                            continue
                        cfg = next(c for c in P7.CONFIGS.values() if c["target"] == t)
                        pp = P7.mask_pred(cfg["scores"](), use)
                        bb = P7.mask_pred(P7._graddot(cfg["baseline"])[t], use)
                        oo = np.array([sm[m["mask_id"]] for m in use], float)
                        if np.std(pp) == 0 or np.std(bb) == 0 or np.std(oo) == 0:
                            continue
                        ds.append(fn(pp, oo) - fn(bb, oo))
                    if len(ds) < 20:
                        continue
                    rows.append({"stage": "allocation", "axis": "seeds", "target": t,
                                 "statistic": statname, "n_masks": len(cond_ids),
                                 "seed_depth": depth, "paired_sd": float(np.std(ds, ddof=1)),
                                 "n_subsamples": len(ds), "n_available": len(cond_ids),
                                 # Only 4 replicate seeds exist, so depth d is a draw WITHOUT
                                 # replacement from a 4-seed pool: sd shrinks toward 0 at d=4 by
                                 # construction, not because seeds stopped mattering. Depths 3-4
                                 # are contaminated; only the 1->2 step is interpretable.
                                 "finite_pop_contaminated": bool(depth >= 3),
                                 "seed_pool": 4})
    out = pd.DataFrame(rows)
    out.attrs["mask_space"] = MASK_SPACE
    return out


def allocation_advice(df_alloc):
    """What the curve implies for campaign N, given the hard cap on 5-of-9 masks."""
    rows = []
    fresh_5of9 = MASK_SPACE - STAGE_F_USED
    for (t, st), g in df_alloc[df_alloc.axis == "masks"].groupby(["target", "statistic"]):
        # DROP the n == n_available point. Subsampling all 40 of 40 masks without replacement is
        # deterministic, so its sd is identically 0 -- not a measurement. Including it in the fit
        # drags c down by a factor of (k-1)/k and would undersize campaign N.
        g = g[g.n_masks < g.n_available].sort_values("n_masks")
        if len(g) < 2:
            continue
        # sd ~ c / sqrt(n): fit c on the observed curve, then read off the cap
        c = float(np.mean(g.paired_sd.values * np.sqrt(g.n_masks.values)))
        rows.append({
            "stage": "allocation_advice", "target": t, "statistic": st,
            "sd_model_c_over_sqrt_n": c,
            "fresh_5of9_masks_available": fresh_5of9,
            "sd_at_fresh_5of9_cap": c / math.sqrt(max(fresh_5of9, 1)),
            "sd_at_100_masks_multistratum": c / math.sqrt(100),
            "sd_at_200_masks_multistratum": c / math.sqrt(200),
            "note": "conditional masks are ~40/72 of a draw, so a stratified multi-|S| design is "
                    "the only way past the 5-of-9 cap of "
                    f"{fresh_5of9} fresh masks",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stage: statistic
def statistic_rows():
    """Choose the primary statistic on RELIABILITY and NOISE only. No contrast column is emitted.

    Reliability: split-half correlation of the seed-mean outcome vector across disjoint halves of
    the replicate seeds, Spearman-Brown corrected. Noise: the paired sd already measured above.
    A structural test asserts this frame carries no contrast, so a later edit cannot leak the
    hypothesis into the choice.
    """
    df = CG.stage_f_table()
    man = MK.cluster_mask_manifest()
    nc = set(man["noise_ceiling_masks"])
    incl = {m["mask_id"]: set(m["clusters"]) for m in man["masks"]}
    rows = []
    halves = [((301, 302), (303, 304)), ((301, 303), (302, 304)), ((301, 304), (302, 303))]
    for t in _all_clusters(man):
        sub = df[(df.target == t) & (df.mask_id.isin(nc))]
        ids = [m for m in sub.mask_id.unique() if t in incl.get(m, set())]
        if len(ids) < 5:
            continue
        for statname, fn in CG.STATS.items():
            rs = []
            for h1, h2 in halves:
                a, b = [], []
                for mid in ids:
                    g = sub[sub.mask_id == mid]
                    v1 = g[g.seed.isin(h1)][CG.PRIMARY_OUTCOME].mean()
                    v2 = g[g.seed.isin(h2)][CG.PRIMARY_OUTCOME].mean()
                    if np.isfinite(v1) and np.isfinite(v2):
                        a.append(v1); b.append(v2)
                if len(a) >= 5 and np.std(a) > 0 and np.std(b) > 0:
                    rs.append(fn(np.array(a), np.array(b)))
            if not rs:
                continue
            r = float(np.mean(rs))
            sb = 2 * r / (1 + r) if r > -1 else np.nan
            rows.append({"stage": "statistic", "target": t, "statistic": statname,
                         "split_half_r": r, "spearman_brown": sb,
                         "n_masks": len(ids), "n_splits": len(rs)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["ceiling", "allocation", "statistic", "all"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    stages = ["ceiling", "allocation", "statistic"] if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "ceiling":
            d = ceiling_rows()
            d.to_csv(os.path.join(RESULTS, "p8_design_ceiling.csv"), index=False)
            print(f"[p8/W2 ceiling] {len(d)} rows")
            if len(d):
                print(d[["target", "ceiling_conditional", "ci_lo", "ci_hi",
                         "n_masks"]].to_string(index=False))
        elif s == "allocation":
            d = allocation_rows()
            d.to_csv(os.path.join(RESULTS, "p8_design_allocation.csv"), index=False)
            adv = allocation_advice(d)
            adv.to_csv(os.path.join(RESULTS, "p8_design_allocation_advice.csv"), index=False)
            print(f"[p8/W2 allocation] {len(d)} rows; 5-of-9 mask space={MASK_SPACE}, "
                  f"Stage F used {STAGE_F_USED}, fresh available={MASK_SPACE - STAGE_F_USED}")
            if len(d):
                print(d.to_string(index=False))
            if len(adv):
                print(adv.to_string(index=False))
        else:
            d = statistic_rows()
            d.to_csv(os.path.join(RESULTS, "p8_design_statistic.csv"), index=False)
            print(f"[p8/W2 statistic] {len(d)} rows (reliability only -- no contrast computed)")
            if len(d):
                print(d.groupby("statistic")[["split_half_r", "spearman_brown"]]
                      .mean().to_string())


if __name__ == "__main__":
    main()
