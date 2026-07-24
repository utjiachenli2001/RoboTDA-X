"""PASS 5, P2 -- are the C5 winners the same demos, or complementary? And does an ensemble win?

Pass 4 got ~+0.22 on C5 out of sample from two DIFFERENT mechanisms: RelatIF (self-influence
normalization on the cached E=20 Gram) and the exact frozen-trunk head surrogate-LOO (regen E=5).
If their per-demo score vectors are only loosely correlated, a rank/z-score ensemble may beat
either alone. Same question for C2 (TRAK-head vs the leverage family). Zero GPU except the
surrogate feature recompute.
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


def zavg(score_dicts, demo_ids):
    """z-score each score vector over demos, then average -> {demo: score}."""
    mats = []
    for sc in score_dicts:
        v = np.array([sc.get(d, 0.0) for d in demo_ids], float)
        v = (v - v.mean()) / (v.std() + 1e-30)
        mats.append(v)
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


def pooled_delta(sc, base, t):
    draws = [(D.demo_masks(), "A"),
             (RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0], "B"),
             (RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0], "I")]
    px_all, pg_all, o_all = [], [], []
    for ms, camp in draws:
        masks = [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in ms]
        raw = F.campaign_outcomes(camp, "plain", targets=(t,))[t]
        obs = F.seed_mean(raw)
        out = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
        ok = np.isfinite(out)
        px_all += list(mask_pred(sc, masks)[ok])
        pg_all += list(mask_pred(base, masks)[ok])
        o_all += list(out[ok])
    return paired_bootstrap(np.array(px_all), np.array(pg_all), np.array(o_all))


def main():
    members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(x, "final.pt")))
    from if_repair import b1_layerwise as B1
    ens = B1.build_ensemble(members)
    Zh, Zc = ens["head"], D.cache_for("bc_s10")
    demo_ids = list(Zc["train_ids"])

    from if_repair.b14_rescoring import relatif_scores
    relatif = relatif_scores(Zc, 1.0, aggregate="unitl2_then_mean")   # C5 winner
    lev_head = family_scores(Zh, 0.3, 1.0, "dmean")                    # unified leverage config
    trak_head = family_scores(Zh, 0.1, 0.0, "dmean")                   # TRAK-head (C2)
    gd_cached = scores_graddot(Zc, normalize_per_member=True)
    gd_head = scores_graddot(Zh, normalize_per_member=True)

    # surrogate-LOO (regen E=5), C5, lambda 0.001
    from if_repair import b12_headloo as B12
    agg = None
    for m in members:
        sc, _ = B12.member_scores(m)
        if agg is None:
            agg = {t: {d: 0.0 for d in sc[0.001][t]} for t in B12.DEV_TARGETS}
        for t in B12.DEV_TARGETS:
            for d, v in sc[0.001][t].items():
                agg[t][d] += v / len(members)
    surrogate = agg

    print("=" * 84)
    print("P2 -- complementarity of the C5 winners and ensembles")
    print("=" * 84)
    # rank correlation over demos
    def rc(a, b):
        va = np.array([a.get(d, 0.0) for d in demo_ids])
        vb = np.array([b.get(d, 0.0) for d in demo_ids])
        return spearman(va, vb)
    print(f"C5  Spearman(RelatIF, surrogate-LOO) over 135 demos = {rc(relatif['C5'], surrogate['C5']):+.3f}")
    print(f"C5  Spearman(RelatIF, leverage-head)              = {rc(relatif['C5'], lev_head['C5']):+.3f}")
    print(f"C2  Spearman(TRAK-head, leverage-head)            = {rc(trak_head['C2'], lev_head['C2']):+.3f}")

    rows = []
    # C5 components + ensemble (paired vs GradDot_dmean cached, the canonical bar)
    ens_c5 = zavg([relatif["C5"], surrogate["C5"]], demo_ids)
    for label, sc, base in [("RelatIF_C5", relatif["C5"], gd_cached["C5"]),
                            ("surrogate_C5", surrogate["C5"], gd_head["C5"]),
                            ("leverage_head_C5", lev_head["C5"], gd_head["C5"]),
                            ("ENSEMBLE_relatif+surrogate_C5", ens_c5, gd_cached["C5"])]:
        d0, pp = pooled_delta(sc, base, "C5")
        rows.append({"target": "C5", "estimator": label, "pooled_delta_rho": d0, "pooled_p": pp})
    # C2 ensemble
    ens_c2 = zavg([trak_head["C2"], lev_head["C2"]], demo_ids)
    for label, sc in [("TRAK_head_C2", trak_head["C2"]), ("leverage_head_C2", lev_head["C2"]),
                      ("ENSEMBLE_trak+leverage_C2", ens_c2)]:
        d0, pp = pooled_delta(sc, gd_head["C2"], "C2")
        rows.append({"target": "C2", "estimator": label, "pooled_delta_rho": d0, "pooled_p": pp})
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "p2_ensemble.csv"), index=False)
    print("\npooled G/H/I paired Delta_rho vs GradDot_dmean:")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
