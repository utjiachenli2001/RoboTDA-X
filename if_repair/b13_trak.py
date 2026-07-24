"""W3 -- TRAK, the field-standard estimator, run here to close the "did you try TRAK?" question.

TRAK score = (G + lambda I)^-1 K, per member, ensemble-averaged. attribution.py already argues the
EXACT dual is strictly better than the paper's random-projection sketch at N=135 << p (the k x k
projected Gram is singular for any useful k), so we compute the exact dual and sweep lambda. Two
Phi choices, both justified by the head-identity finding:

  cached E=20 full-model Gram   -- the real ensemble, evaluated vs the true 0.593 (BLOCKERS #1)
  regen E=5 head-Phi Gram       -- head space, baseline GradDot recomputed on the same ensemble

Prior is honestly low: N=135 caps the kernel rank exactly as Phase A found (k* = 1), so inverting
the Gram amplifies noise past the first eigendirection. Run it, report the number, move on.
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
from if_repair import retrain as RT  # noqa: E402
from if_repair.aggregate import per_member_scores, normalize_members, aggregate  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LAMS = (1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e2)
TARGETS = ("C1", "C5", "C2")


def scores_from(Z, kind, ridge_rel, norm):
    S0 = per_member_scores(Z, kind, ridge_rel=ridge_rel)
    S = aggregate(normalize_members(S0, norm, Z), "mean")
    tids, tgts = list(Z["train_ids"]), list(Z["targets"])
    return {tgts[j]: {tids[i]: float(S[i, j]) for i in range(len(tids))}
            for j in range(len(tgts))}


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


def eval_gseries(sc, tag, ridge):
    gm, obs, ceil = D.demo_masks(), D.outcomes("bc_s10"), D.ceilings("bc_s10")
    rows = []
    for t in TARGETS:
        rho, p, n, _, _ = demo_grain_lds(sc[t], gm, obs[t])
        rows.append({"phi": tag, "lambda_rel": ridge, "target": t, "lds": rho,
                     "ceiling": ceil[t], "ratio": rho / ceil[t], "p": p})
    return rows


def main():
    members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(x, "final.pt")))
    from if_repair import b1_layerwise as B1
    ens = B1.build_ensemble(members)

    Zc = D.cache_for("bc_s10")             # cached E=20 full-model
    Zh = ens["head"]                        # regen E=5 head-Phi

    rows = []
    # baselines (GradDot) on each ensemble
    for tag, Z in [("cachedE20_full", Zc), ("regenE5_head", Zh)]:
        g = scores_graddot(Z, normalize_per_member=True)
        for t in TARGETS:
            rho, p, n, _, _ = demo_grain_lds(g[t], D.demo_masks(), D.outcomes("bc_s10")[t])
            c = D.ceilings("bc_s10")[t]
            rows.append({"phi": tag, "lambda_rel": np.nan, "target": t, "estimator": "GradDot",
                         "lds": rho, "ceiling": c, "ratio": rho / c, "p": p})
    # TRAK sweep; track best lambda PER (phi, target) by G-series LDS
    best = {}     # (tag, target) -> (lds, lr, scores_at_that_lr)
    scache = {}   # (tag, lr) -> scores dict
    for tag, Z in [("cachedE20_full", Zc), ("regenE5_head", Zh)]:
        for lr in LAMS:
            sc = scores_from(Z, "TRAK", lr, "dmean")
            scache[(tag, lr)] = sc
            for r in eval_gseries(sc, tag, lr):
                r["estimator"] = "TRAK"
                rows.append(r)
                key = (tag, r["target"])
                if r["lds"] > best.get(key, (-9, None))[0]:
                    best[key] = (r["lds"], lr)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b13_trak_gseries.csv"), index=False)
    print("=" * 96)
    print("W3 -- TRAK (exact dual) G-series LDS. Bar = GradDot on the SAME ensemble.")
    print("=" * 96)
    for tag in ("cachedE20_full", "regenE5_head"):
        sub = df[df.phi == tag]
        print(f"\n--- Phi = {tag} ---")
        print(sub[["estimator", "lambda_rel", "target", "lds", "ratio", "p"]].round(4)
              .to_string(index=False))

    # paired G/H/I for the best TRAK cell per phi vs GradDot on that ensemble
    print("\n" + "=" * 96)
    print("W3 -- best TRAK cell PAIRED vs GradDot on G/H/I")
    print("=" * 96)
    draws = [("G", [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in D.demo_masks()], "A"),
             ("H", [{"mask_id": m["mask_id"], "demos": m["demos"]}
                    for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0]], "B"),
             ("I", [{"mask_id": m["mask_id"], "demos": m["demos"]}
                    for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0]], "I")]
    prows = []
    for tag, Z in [("cachedE20_full", Zc), ("regenE5_head", Zh)]:
        g = scores_graddot(Z, normalize_per_member=True)
        pool = {t: {"px": [], "pg": [], "o": []} for t in TARGETS}
        lr_of = {t: best[(tag, t)][1] for t in TARGETS}   # per-target best lambda
        for name, masks, camp in draws:
            for t in TARGETS:
                sc = scache[(tag, lr_of[t])]
                raw = F.campaign_outcomes(camp, "plain", targets=(t,))[t]
                obs_t = F.seed_mean(raw)
                out = np.array([obs_t.get(m["mask_id"], np.nan) for m in masks])
                ok = np.isfinite(out)
                px, pg = mask_pred(sc[t], masks)[ok], mask_pred(g[t], masks)[ok]
                pool[t]["px"] += list(px); pool[t]["pg"] += list(pg); pool[t]["o"] += list(out[ok])
        for t in TARGETS:
            px, pg, o = (np.array(pool[t][k]) for k in ("px", "pg", "o"))
            d0, pp = paired_bootstrap(px, pg, o)
            prows.append({"phi": tag, "lambda_rel": lr_of[t], "target": t,
                          "trak_lds": spearman(px, o), "graddot_lds": spearman(pg, o),
                          "paired_delta_rho": d0, "paired_p": pp})
    pdf = pd.DataFrame(prows)
    pdf.to_csv(os.path.join(RESULTS, "b13_trak_paired_ghi.csv"), index=False)
    print(pdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
