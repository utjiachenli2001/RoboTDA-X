"""PASS 5, P1 -- the unified leverage family, and the generality question.

Pass 4's three out-of-sample winners are all corners of ONE estimator family: a leverage/Gram
correction of GradDot's raw kernel K_m[d,t] = <z_d, g_t>.

    S_m(lam_rel, beta)[:, t] = diag(G_m)^(-beta) . INV . K_m[:, t]
        INV = (G_m + lam I)^(-1)   if lam_rel is finite   (lam = lam_rel * mean(diag G_m))
        INV = I                    if lam_rel = inf        ("skip the inversion")

Corners (exact):
    (inf, 0)  = GradDot          (S_m = K_m)
    (inf, 1)  = RelatIF          (S_m = diag(G_m)^-1 K_m)          -- W4 winner on C5
    (inf, .5) = RelatIF-sqrt
    (0.1, 0)  = TRAK, lam_rel .1 (S_m = (G+lam I)^-1 K_m)          -- W3 winner on C2

Two aggregations carried: dmean (the baseline's) and unitL2 (the one RelatIF won with). The
(inf, 0, dmean) cell IS GradDot_dmean, so it appears at Delta_rho = 0 -- a built-in sanity check.

Selection is LEAVE-ONE-DRAW-OUT over {G,H,I}: a configuration is a GENERALITY candidate only if,
on the held-out draw, its paired Delta_rho vs GradDot is > 0 on >= 3 targets in ALL three
rotations, AND pooled Delta_rho >= +0.10 on those targets. This is the discipline that separated
the real pass-4 winners from the KFAC-style mirages. Zero GPU (135x135 solves on cached Grams).
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

D.add_repo_paths()
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TARGETS = tuple(f"C{i}" for i in range(1, 10))
LAM_RELS = (None, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)   # None = inf = skip inversion
BETAS = (0.0, 0.5, 1.0)
AGGS = ("dmean", "unitL2")
EPS = 1e-30


def family_scores(Z, lam_rel, beta, agg):
    """-> {target: {demo: score}} for the (lam_rel, beta, agg) cell on ensemble Z."""
    G = np.asarray(Z["G"], float)
    K = np.asarray(Z["K"], float)
    M, N, T = K.shape
    tids, tgts = list(Z["train_ids"]), list(Z["targets"])
    S = np.empty((M, N, T))
    for m in range(M):
        Gm, Km = G[m], K[m]
        mu = float(np.mean(np.diag(Gm)))
        if lam_rel is None:
            inv_K = Km
        else:
            inv_K = np.linalg.solve(Gm + lam_rel * mu * np.eye(N), Km)
        if beta != 0.0:
            d = np.clip(np.diag(Gm), EPS, None) ** (-beta)
            inv_K = d[:, None] * inv_K
        S[m] = inv_K
    # aggregation across members
    if agg == "unitL2":
        S = S / np.clip(np.linalg.norm(S, axis=1, keepdims=True), EPS, None)
    elif agg == "dmean":
        d = np.array([np.mean(np.diag(G[m])) for m in range(M)])
        S = S / d[:, None, None]
    Sag = S.mean(axis=0)
    return {tgts[j]: {tids[i]: float(Sag[i, j]) for i in range(N)} for j in range(T)}


def draws():
    return [("G", [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in D.demo_masks()], "A"),
            ("H", [{"mask_id": m["mask_id"], "demos": m["demos"]}
                   for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0]], "B"),
            ("I", [{"mask_id": m["mask_id"], "demos": m["demos"]}
                   for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0]], "I")]


def mask_pred(sc, masks):
    return np.array([sum(sc.get(dd, 0.0) for dd in m["demos"]) for m in masks])


def outcomes_by_draw():
    """-> {draw: {target: (masks_ok, pred-ready obs vector aligned to masks)}} cache."""
    out = {}
    for name, masks, camp in draws():
        out[name] = {"masks": masks}
        for t in TARGETS:
            raw = F.campaign_outcomes(camp, "plain", targets=(t,))[t]
            obs = F.seed_mean(raw)
            out[name][t] = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
    return out


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
    OBS = outcomes_by_draw()
    dl = draws()

    members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(x, "final.pt")))
    from if_repair import b1_layerwise as B1
    ens = B1.build_ensemble(members)
    sources = {"cachedE20_full": D.cache_for("bc_s10"), "regenE5_head": ens["head"]}

    rows = []      # per (source, agg, lam_rel, beta, target): per-draw + pooled point Delta_rho
    for sname, Z in sources.items():
        for agg in AGGS:
            gbase = (scores_graddot(Z, normalize_per_member=True) if agg == "dmean"
                     else _graddot_unitl2(Z))
            for lam_rel in LAM_RELS:
                for beta in BETAS:
                    sc = family_scores(Z, lam_rel, beta, agg)
                    for t in TARGETS:
                        # per-draw point Delta_rho + pooled
                        pooled_px, pooled_pg, pooled_o = [], [], []
                        perdraw = {}
                        for name, masks, camp in dl:
                            out = OBS[name][t]
                            ok = np.isfinite(out)
                            px = mask_pred(sc[t], masks)[ok]
                            pg = mask_pred(gbase[t], masks)[ok]
                            o = out[ok]
                            perdraw[name] = spearman(px, o) - spearman(pg, o)
                            pooled_px += list(px); pooled_pg += list(pg); pooled_o += list(o)
                        px, pg, o = map(np.array, (pooled_px, pooled_pg, pooled_o))
                        rows.append({"source": sname, "agg": agg,
                                     "lam_rel": (np.inf if lam_rel is None else lam_rel),
                                     "beta": beta, "target": t,
                                     "dG": perdraw["G"], "dH": perdraw["H"], "dI": perdraw["I"],
                                     "d_pooled": spearman(px, o) - spearman(pg, o)})
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "p1_leverage_family.csv"), index=False)

    # ---- generality analysis: per (source, agg, lam_rel, beta) config, count targets with
    #      sign-consistent Delta_rho across all 3 draws AND pooled >= +0.10
    print("=" * 100)
    print("P1 -- generality scan: configs beating GradDot OOS on >=3 targets "
          "(sign-consistent on G,H,I + pooled >= +0.10)")
    print("=" * 100)
    best = []
    for (s, agg, lr, b), g in df.groupby(["source", "agg", "lam_rel", "beta"]):
        wins = g[(np.sign(g.dG) > 0) & (np.sign(g.dH) > 0) & (np.sign(g.dI) > 0)
                 & (g.d_pooled >= 0.10)]
        if len(wins) >= 1:
            best.append({"source": s, "agg": agg, "lam_rel": lr, "beta": b,
                         "n_targets_generalizing": len(wins),
                         "targets": ",".join(sorted(wins.target)),
                         "mean_pooled_delta": float(wins.d_pooled.mean())})
    bdf = pd.DataFrame(best).sort_values("n_targets_generalizing", ascending=False)
    if len(bdf):
        print(bdf.to_string(index=False))
        gen = bdf[bdf.n_targets_generalizing >= 3]
        print(f"\n>>> GENERALITY (>=3 targets) configs: {len(gen)}")
        if len(gen):
            top = gen.iloc[0]
            top_agg = top["agg"]
            print(f">>> BEST: source={top.source} agg={top_agg} lam_rel={top.lam_rel} "
                  f"beta={top.beta} -> {top.n_targets_generalizing} targets "
                  f"({top.targets}), mean pooled Delta {top.mean_pooled_delta:+.3f}")
    else:
        print("No config beats GradDot on any target with full sign-consistency + pooled>=0.10.")

    # corner sanity + per-target champion table
    print("\n--- per-target best pooled Delta_rho over the whole family (any config) ---")
    for t in TARGETS:
        sub = df[df.target == t].sort_values("d_pooled", ascending=False).iloc[0]
        sub_agg = sub["agg"]
        tag = "SIGN-CONSISTENT" if (sub.dG > 0 and sub.dH > 0 and sub.dI > 0) else "mixed-sign"
        print(f"  {t}: pooled {sub.d_pooled:+.3f}  ({sub.source}/{sub_agg}/lam{sub.lam_rel}/"
              f"b{sub.beta})  draws G{sub.dG:+.2f} H{sub.dH:+.2f} I{sub.dI:+.2f}  [{tag}]")


def _graddot_unitl2(Z):
    from if_repair.aggregate import per_member_scores, normalize_members, aggregate
    S0 = per_member_scores(Z, "GradDot")
    S = aggregate(normalize_members(S0, "unitL2", Z), "mean")
    tids, tgts = list(Z["train_ids"]), list(Z["targets"])
    return {tgts[j]: {tids[i]: float(S[i, j]) for i in range(len(tids))}
            for j in range(len(tgts))}


if __name__ == "__main__":
    main()
