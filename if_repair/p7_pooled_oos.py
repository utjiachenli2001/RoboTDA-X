"""W0.1 (pass 7) -- pooled OUT-OF-SAMPLE analysis of the frozen pass-4/5/6 estimators.

Zero GPU. The observation this rests on: several estimator configs were frozen in a PREREG_*
dict and committed while their campaign had zero runs, and were never retuned afterwards. Each
is therefore legitimately out-of-sample on every draw that came after it was frozen -- including
draws whose own prereg family named a DIFFERENT hypothesis. Nobody has ever scored RelatIF/C5 on
campaign K.  The "48-72-mask confirmation draw" HANDOFF.md asks for already exists on disk.

Provenance, verified in code rather than from commit timestamps alone:

  RelatIF/C5            frozen in PREREG_J (c7659cf, campaign J at zero runs), never retuned.
                        Selected on pooled G/H/I dev (pass-4 W4).           OOS: J, K, L  (72)
  surrogate-LOO/C5      frozen in PREREG_J likewise.                        OOS: J, K, L  (72)
  C5 ensemble           defined in pass-6 P2; `p2_ensemble.pooled_delta` reads draws A/B/I ONLY,
                        so no J/K/L outcome entered its selection. PREREG_L (2a1f4e0) was
                        committed while K was already mid-flight, but K's outcomes were never
                        used to select or tune it.                          OOS: K, L     (48)
  leverage-E20/C7       from pass-5 P1; `p1_leverage_family.draws()` reads G/H/I ONLY, so no J
                        data entered its selection.                         OOS: J, K, L  (72)

WINNER'S CURSE. Being frozen before a draw makes a config out-of-sample on it, but the draw on
which an effect was first REPORTED as a success is still selected-upon at the level of "which
hypothesis do we follow up". So every contrast is reported twice: the full-information estimate
over all OOS draws, and the honest estimate with that config's discovery draw removed. The
without-discovery number is the one to believe.

POOLING VALIDITY, prespecified before any contrast is computed (--stage validity, run and
committed first). Pooling raw outcomes across draws is valid only if mask-level outcomes are
exchangeable across draws. A one-way random-effects ANOVA of the seed-mean mask outcome by draw
gives the between-draw variance component as a fraction of total (ICC). Rule, frozen here:

    ICC > 0.10 on the J/K/L outcome set for a target  ->  that target's primary statistic pools
    WITHIN-DRAW RANKS of both the prediction and the outcome, instead of raw values.

The rule is decided per TARGET on the J/K/L set and then applied to every pooled analysis for
that target, so the with-J and without-J analyses always use the same statistic.
"""
from __future__ import annotations

import argparse
import glob
import json
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

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
FIGS = os.path.join(HERE, "figs")

N_BOOT = 5000
BOOT_SEED = 7
ICC_GATE = 0.10
RULE_DRAWS = ("J", "K", "L")          # the set the raw/rank rule is decided on
CAMPAIGN_OF = {"G": "A", "H": "B", "I": "I", "J": "J", "K": "K", "L": "L"}
DEV_DRAWS = ("G", "H", "I")

_C = {}


# ------------------------------------------------------------------ masks / outcomes
def masks_for(draw):
    if draw == "G":
        ms = D.demo_masks()
    elif draw == "H":
        ms = RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0]
    elif draw == "I":
        ms = RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0]
    elif draw == "J":
        ms = RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_J, prefix="J")[0]
    elif draw == "K":
        ms = RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_K, prefix="K")[0]
    elif draw == "L":
        ms = RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_L, prefix="L")[0]
    else:
        raise KeyError(draw)
    return [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in ms]


def raw_outcomes(draw, target):
    """{mask_id: {seed_key: v}} for one draw/target, at the archived depth 10."""
    key = ("raw", draw, target)
    if key not in _C:
        _C[key] = F.campaign_outcomes(CAMPAIGN_OF[draw], "plain", targets=(target,))[target]
    return _C[key]


# ------------------------------------------------------------------ estimators (frozen configs)
def _members():
    if "members" not in _C:
        _C["members"] = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                               if os.path.exists(os.path.join(x, "final.pt")))
    return _C["members"]


def _ens_head():
    if "head" not in _C:
        from if_repair import b1_layerwise as B1
        _C["head"] = B1.build_ensemble(_members())["head"]
    return _C["head"]


def _relatif():
    """PREREG_J/L `L2`: RelatIF K/G_dd, per-member unit-L2 then mean, cached E=20 Gram."""
    if "relatif" not in _C:
        from if_repair.b14_rescoring import relatif_scores
        _C["relatif"] = relatif_scores(D.cache_for("bc_s10"), 1.0, aggregate="unitl2_then_mean")
    return _C["relatif"]


def _surrogate():
    """PREREG_J/L `L3`: exact frozen-trunk head surrogate-LOO, lam=0.001, regen E=5, member-mean."""
    if "surr" not in _C:
        from if_repair import b12_headloo as B12
        agg = None
        for m in _members():
            sc, _ = B12.member_scores(m)
            if agg is None:
                agg = {t: {d: 0.0 for d in sc[0.001][t]} for t in B12.DEV_TARGETS}
            for t in B12.DEV_TARGETS:
                for d, v in sc[0.001][t].items():
                    agg[t][d] += v / len(_members())
        _C["surr"] = agg
    return _C["surr"]


def _zavg(dicts, demo_ids):
    mats = []
    for sc in dicts:
        v = np.array([sc.get(d, 0.0) for d in demo_ids], float)
        mats.append((v - v.mean()) / (v.std() + 1e-30))
    m = np.mean(mats, axis=0)
    return {d: float(m[i]) for i, d in enumerate(demo_ids)}


def _leverage_c7():
    """PREREG_K `K3`: leverage family on the cached E=20 full-model Gram, lam_rel=1, beta=1, dmean."""
    if "lev7" not in _C:
        _C["lev7"] = family_scores(D.cache_for("bc_s10"), 1.0, 1.0, "dmean")
    return _C["lev7"]


def _graddot(which):
    key = ("gd", which)
    if key not in _C:
        Z = D.cache_for("bc_s10") if which == "cached" else _ens_head()
        _C[key] = scores_graddot(Z, normalize_per_member=True)
    return _C[key]


# name -> (target, baseline ensemble, oos draws, discovery draw, scorer)
CONFIGS = {
    "relatif_C5": {
        "target": "C5", "baseline": "cached", "oos": ("J", "K", "L"), "discovery": "J",
        "label": "RelatIF K/G_dd (cached E=20, unitL2)",
        "frozen": "PREREG_J c7659cf",
        "scores": lambda: _relatif()["C5"],
    },
    "surrogate_C5": {
        "target": "C5", "baseline": "head", "oos": ("J", "K", "L"), "discovery": "J",
        "label": "exact frozen-trunk surrogate-LOO (regen E=5 head, lam=1e-3)",
        "frozen": "PREREG_J c7659cf",
        "scores": lambda: _surrogate()["C5"],
    },
    "ensemble_C5": {
        "target": "C5", "baseline": "cached", "oos": ("K", "L"), "discovery": None,
        "label": "z-avg ensemble of RelatIF + surrogate-LOO",
        "frozen": "PREREG_L 2a1f4e0 (selected on G/H/I dev only)",
        "scores": lambda: _zavg([_relatif()["C5"], _surrogate()["C5"]],
                                list(D.cache_for("bc_s10")["train_ids"])),
    },
    "leverage_C7": {
        "target": "C7", "baseline": "cached", "oos": ("J", "K", "L"), "discovery": "K",
        "label": "leverage family lam_rel=1 beta=1 (cached E=20, dmean)",
        "frozen": "PREREG_K 5a5d554 (selected on G/H/I dev only)",
        "scores": lambda: _leverage_c7()["C7"],
    },
}


def mask_pred(sc, masks):
    return np.array([sum(sc.get(dd, 0.0) for dd in m["demos"]) for m in masks])


# ------------------------------------------------------------------ pooling validity (W0.1, stage 1)
def icc_by_draw(target, draws=RULE_DRAWS):
    """One-way random-effects ANOVA of the seed-mean mask outcome by draw.

    Returns the between-draw variance component as a fraction of total (ICC), plus the F test.
    """
    groups = []
    for dr in draws:
        obs = F.seed_mean(raw_outcomes(dr, target))
        v = np.array([obs[m["mask_id"]] for m in masks_for(dr) if m["mask_id"] in obs], float)
        groups.append(v[np.isfinite(v)])
    k = len(groups)
    ns = np.array([len(g) for g in groups], float)
    N = ns.sum()
    gm = np.concatenate(groups).mean()
    ssb = float(sum(len(g) * (g.mean() - gm) ** 2 for g in groups))
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    var_b = max(0.0, (msb - msw) / n0)
    icc = var_b / (var_b + msw) if (var_b + msw) > 0 else np.nan
    fstat = msb / msw if msw > 0 else np.inf
    return {"target": target, "draws": "+".join(draws), "k": k, "n_total": int(N),
            "MSB": msb, "MSW": msw, "var_between": var_b, "var_within": msw,
            "ICC_between_draw": icc, "F": fstat,
            "p_anova": float(1.0 - stats.f.cdf(fstat, k - 1, N - k)),
            "rule": "rank_within_draw" if icc > ICC_GATE else "raw"}


def validity_table(targets=("C5", "C7")):
    rows = []
    for t in targets:
        rows.append(icc_by_draw(t, RULE_DRAWS))
        rows.append(icc_by_draw(t, DEV_DRAWS + RULE_DRAWS))   # context only, not the rule
    return pd.DataFrame(rows)


def rule_for(target):
    """The frozen raw/rank decision for a target, from the J/K/L ANOVA."""
    key = ("rule", target)
    if key not in _C:
        _C[key] = icc_by_draw(target, RULE_DRAWS)["rule"]
    return _C[key]


# ------------------------------------------------------------------ pooled statistic
def _rank_within(v, draw_idx):
    out = np.empty(len(v), float)
    for d in np.unique(draw_idx):
        s = draw_idx == d
        out[s] = stats.rankdata(v[s])
    return out


def assemble(cfg, draws):
    """-> px, pg, o, draw_idx  (already transformed per the frozen rule for this target)."""
    t = cfg["target"]
    est, base = cfg["scores"](), _graddot(cfg["baseline"])[t]
    px, pg, o, di = [], [], [], []
    for j, dr in enumerate(draws):
        masks = masks_for(dr)
        obs = F.seed_mean(raw_outcomes(dr, t))
        out = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
        ok = np.isfinite(out)
        px += list(mask_pred(est, masks)[ok])
        pg += list(mask_pred(base, masks)[ok])
        o += list(out[ok])
        di += [j] * int(ok.sum())
    px, pg, o, di = (np.array(x) for x in (px, pg, o, np.array(di)))
    if rule_for(t) == "rank_within_draw":
        px, pg, o = _rank_within(px, di), _rank_within(pg, di), _rank_within(o, di)
    return px, pg, o, di


def stratified_bootstrap(px, pg, o, di, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample masks WITHIN each draw, preserving draw sizes. -> (d0, p_onesided, lo, hi, se)."""
    rng = np.random.default_rng(seed)
    d0 = spearman(px, o) - spearman(pg, o)
    idx_by_draw = [np.flatnonzero(di == d) for d in np.unique(di)]
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_draw])
        rx, rg = spearman(px[i], o[i]), spearman(pg[i], o[i])
        if np.isfinite(rx) and np.isfinite(rg):
            diffs.append(rx - rg)
    diffs = np.array(diffs)
    if not len(diffs):
        return d0, np.nan, np.nan, np.nan, np.nan
    return (d0, float((diffs <= 0).mean()),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)),
            float(diffs.std(ddof=1)))


# ------------------------------------------------------------------ pooled ceiling
def pooled_ceiling(draws, target, rule, max_splits=200):
    """The ARCHIVED split-half recipe (functionals.split_half_ceiling) over a UNION of draws.

    rule="raw"  -> identical to F.split_half_ceiling on the merged dict (asserted in tests).
    rule="rank_within_draw" -> each seed-half's per-mask vector is converted to within-draw ranks
    before the Spearman, so the ceiling measures the reliability of the SAME statistic the
    contrast uses.
    """
    merged, owner = {}, {}
    for dr in draws:
        for m, d in raw_outcomes(dr, target).items():
            merged[m] = d
            owner[m] = dr
    masks = sorted(merged)
    seeds = sorted({s for m in masks for s in merged[m]}, key=lambda x: (str(type(x)), str(x)))
    masks = [m for m in masks if all(s in merged[m] for s in seeds)]
    S = len(seeds)
    if S < 2 or len(masks) < 4:
        return {"ceiling": np.nan, "half": np.nan, "n_masks": len(masks), "n_seeds": S}
    di = np.array([draws.index(owner[m]) for m in masks])
    h = S // 2
    seen, splits = set(), []
    for c in itertools.combinations(range(S), h):
        rest = tuple(sorted(set(range(S)) - set(c)))
        if len(rest) != h:
            continue
        key = frozenset([c, rest])
        if key in seen:
            continue
        seen.add(key)
        splits.append((c, rest))
    rs = []
    for a, b in splits[:max_splits]:
        va = np.array([np.mean([merged[m][seeds[i]] for i in a]) for m in masks])
        vb = np.array([np.mean([merged[m][seeds[i]] for i in b]) for m in masks])
        if rule == "rank_within_draw":
            va, vb = _rank_within(va, di), _rank_within(vb, di)
        r = spearman(va, vb)
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return {"ceiling": np.nan, "half": np.nan, "n_masks": len(masks), "n_seeds": S}
    half = float(np.mean(rs))
    k = S / h
    return {"ceiling": float(k * half / (1 + (k - 1) * half)), "half": half,
            "n_masks": len(masks), "n_seeds": S}


# ------------------------------------------------------------------ meta-analysis
def meta(ds, ses):
    """Fixed-effect + DerSimonian-Laird random-effects summary, Cochran's Q, I^2."""
    ds, ses = np.asarray(ds, float), np.asarray(ses, float)
    ok = np.isfinite(ds) & np.isfinite(ses) & (ses > 0)
    ds, ses = ds[ok], ses[ok]
    if len(ds) < 2:
        return {}
    w = 1.0 / ses ** 2
    fe = float((w * ds).sum() / w.sum())
    se_fe = float(np.sqrt(1.0 / w.sum()))
    Q = float((w * (ds - fe) ** 2).sum())
    df = len(ds) - 1
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    denom = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = float(max(0.0, (Q - df) / denom)) if denom > 0 else 0.0
    ws = 1.0 / (ses ** 2 + tau2)
    re = float((ws * ds).sum() / ws.sum())
    se_re = float(np.sqrt(1.0 / ws.sum()))
    return {"k_draws": len(ds), "FE": fe, "FE_se": se_fe, "FE_lo": fe - 1.96 * se_fe,
            "FE_hi": fe + 1.96 * se_fe, "Q": Q, "df": df,
            "p_Q": float(1.0 - stats.chi2.cdf(Q, df)) if df > 0 else np.nan,
            "I2": I2, "tau2": tau2, "RE": re, "RE_se": se_re,
            "RE_lo": re - 1.96 * se_re, "RE_hi": re + 1.96 * se_re}


# ------------------------------------------------------------------ the analysis
def analyse_pooled(name, cfg, draws, tag):
    t = cfg["target"]
    px, pg, o, di = assemble(cfg, draws)
    n = len(o)
    rho, rho_gd = spearman(px, o), spearman(pg, o)
    ceil = pooled_ceiling(tuple(draws), t, rule_for(t))["ceiling"]
    d0, pp, lo, hi, se = stratified_bootstrap(px, pg, o, di)
    p_abs = spearman_p_onesided(rho, n)
    return {"config": name, "target": t, "analysis": tag, "draws": "+".join(draws), "n_masks": n,
            "statistic": rule_for(t), "lds": rho, "graddot_lds": rho_gd, "ceiling": ceil,
            "ratio": rho / ceil if np.isfinite(ceil) and ceil else np.nan, "p_abs": p_abs,
            "paired_delta_rho": d0, "paired_p": pp, "ci_lo": lo, "ci_hi": hi,
            "ci_width": hi - lo, "boot_se": se,
            "PASS_abs": bool(np.isfinite(rho) and np.isfinite(ceil) and rho / ceil >= 0.5
                             and p_abs < 0.05),
            "PASS_paired": bool(np.isfinite(d0) and d0 > 0 and pp < 0.05)}


def analyse_per_draw(name, cfg, draws):
    rows = []
    for dr in draws:
        px, pg, o, di = assemble(cfg, [dr])
        d0, pp, lo, hi, se = stratified_bootstrap(px, pg, o, di)
        c = F.split_half_ceiling(raw_outcomes(dr, cfg["target"]))["ceiling"]
        rho = spearman(px, o)
        rows.append({"config": name, "target": cfg["target"], "draw": dr,
                     "role": "dev" if dr in DEV_DRAWS else
                             ("discovery" if dr == cfg["discovery"] else "oos"),
                     "n_masks": len(o), "lds": rho, "graddot_lds": spearman(pg, o),
                     "ceiling": c, "ratio": rho / c if np.isfinite(c) and c else np.nan,
                     "paired_delta_rho": d0, "paired_p": pp, "ci_lo": lo, "ci_hi": hi,
                     "boot_se": se})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="validity", choices=["validity", "contrast"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    if a.stage == "validity":
        df = validity_table()
        df.to_csv(os.path.join(RESULTS, "p7_pooling_validity.csv"), index=False)
        print("=" * 100)
        print("W0.1 POOLING VALIDITY -- one-way random-effects ANOVA of mask outcomes by draw")
        print(f"RULE (frozen before any contrast): ICC > {ICC_GATE} on {'+'.join(RULE_DRAWS)}"
              " -> pool WITHIN-DRAW RANKS for that target")
        print("=" * 100)
        print(df.round(5).to_string(index=False))
        print()
        for t in ("C5", "C7"):
            print(f"  {t}: primary statistic = {rule_for(t)}")
        return

    # ---- stage: contrast
    pooled, per_draw = [], []
    for name, cfg in CONFIGS.items():
        oos = list(cfg["oos"])
        pooled.append(analyse_pooled(name, cfg, oos, "full_information"))
        if cfg["discovery"] and cfg["discovery"] in oos and len(oos) > 1:
            honest = [d for d in oos if d != cfg["discovery"]]
            pooled.append(analyse_pooled(name, cfg, honest, "without_discovery_draw"))
        per_draw += analyse_per_draw(name, cfg, list(DEV_DRAWS) + oos)

    dfp, dfd = pd.DataFrame(pooled), pd.DataFrame(per_draw)
    dfp.to_csv(os.path.join(RESULTS, "p7_pooled_oos.csv"), index=False)
    dfd.to_csv(os.path.join(RESULTS, "p7_per_draw.csv"), index=False)

    mrows = []
    for name, cfg in CONFIGS.items():
        sub = dfd[(dfd.config == name) & (dfd.role != "dev")]
        m = meta(sub.paired_delta_rho.to_numpy(), sub.boot_se.to_numpy())
        if m:
            mrows.append({"config": name, "target": cfg["target"], **m})
    dfm = pd.DataFrame(mrows)
    dfm.to_csv(os.path.join(RESULTS, "p7_heterogeneity.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=" * 118)
    print("W0.1 POOLED OUT-OF-SAMPLE CONTRAST vs GradDot_dmean  (95% CI = stratified mask bootstrap)")
    print("=" * 118)
    print(dfp[["config", "target", "analysis", "draws", "n_masks", "statistic", "lds",
               "graddot_lds", "ceiling", "ratio", "paired_delta_rho", "ci_lo", "ci_hi",
               "ci_width", "paired_p", "PASS_abs", "PASS_paired"]].round(4).to_string(index=False))
    print("\n" + "=" * 118)
    print("PER-DRAW (dev rows are NOT out-of-sample; 'discovery' is the draw the effect was first reported on)")
    print("=" * 118)
    print(dfd.round(4).to_string(index=False))
    print("\n" + "=" * 118)
    print("BETWEEN-DRAW HETEROGENEITY of the paired contrast (OOS draws only)")
    print("=" * 118)
    print(dfm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
