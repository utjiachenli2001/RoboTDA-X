"""W0.2 (pass 7) -- design and statistic selection, answered empirically from data on disk.

Zero GPU. 144 masks x 10 seeds already exist (draws G/H/I/J/K/L). W0.1's ANOVA measured the
between-draw variance component at ZERO on every set tested, so the six draws are exchangeable
and the 144 masks can be treated as one pool for a design study.

Three questions:

(a) ALLOCATION. For a fixed retrain budget B = n_masks x depth, which split minimises the
    sampling sd of the paired contrast? BLOCKERS #17's depth table suggests seed noise is nearly
    exhausted by depth 4-5 while mask sd falls as 1/sqrt(n), which would make masks the better
    buy -- but #17 also shows LDS is not monotone in depth, so this is measured, not assumed.

(b) STATISTIC. Spearman over 24 masks throws away magnitude. Candidate statistics are scored on
    RELIABILITY and NOISE ONLY -- criteria that never touch the RelatIF-vs-GradDot contrast -- so
    the primary statistic cannot be chosen because it flatters the hypothesis. The contrast under
    every candidate is computed in a SEPARATE stage, run only after the selection is committed.

    Note on the brief: it lists "Pearson on ranks" as a candidate distinct from Spearman. Pearson
    on ranks IS Spearman, by definition. It is kept as an identity check, not a fifth candidate.

(c) BASELINE INSTABILITY. Part of why the C5 contrast swings is that GradDot itself swings
    (0.175 on J, 0.143 on L, -0.032 on K). How much of the paired variance is the baseline's?

Stages, run and committed in order:
    --stage allocation   (a) + (c)
    --stage statistic    (b) selection on reliability/noise only -> names the primary
    --stage bystat       (b) the contrast under every candidate; run AFTER committing selection
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import p7_pooled_oos as P7  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ALL_DRAWS = ("G", "H", "I", "J", "K", "L")
NS = (24, 48, 72, 96, 120, 144)
DEPTHS = (2, 3, 4, 5, 6, 8, 10)
NREP = 500
NREP_CEIL = 100
NSPLIT = 20
BUDGETS = (240, 360, 480, 720)
EFFECTS = (0.10, 0.15, 0.20, 0.30)
Z95 = 1.6448536269514722
SEED = 11


def _sp(a, b):
    r = stats.spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


# ------------------------------------------------------------------ the pooled design matrix
def pool(target, cfg_name="relatif_C5"):
    """-> (px, pg, O, draw_labels).  O is (M, 10) per-seed outcomes at the archived depth."""
    cfg = P7.CONFIGS[cfg_name]
    est, base = cfg["scores"](), P7._graddot(cfg["baseline"])[target]
    px, pg, O, lab = [], [], [], []
    for dr in ALL_DRAWS:
        raw = P7.raw_outcomes(dr, target)
        for m in P7.masks_for(dr):
            mid = m["mask_id"]
            if mid not in raw:
                continue
            d = raw[mid]
            ks = sorted(d, key=lambda x: str(x))
            if len(ks) < 10:
                continue
            O.append([d[k] for k in ks[:10]])
            px.append(sum(est.get(dd, 0.0) for dd in m["demos"]))
            pg.append(sum(base.get(dd, 0.0) for dd in m["demos"]))
            lab.append(dr)
    return np.array(px), np.array(pg), np.array(O), np.array(lab)


def _ceiling(Osub, rng, nsplit=NSPLIT):
    """Monte-Carlo form of the archived split-half recipe: disjoint halves + Spearman-Brown."""
    d = Osub.shape[1]
    h = d // 2
    if h < 1 or Osub.shape[0] < 4:
        return np.nan
    rs = []
    for _ in range(nsplit):
        p = rng.permutation(d)
        r = _sp(Osub[:, p[:h]].mean(1), Osub[:, p[h:2 * h]].mean(1))
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return np.nan
    half, k = float(np.mean(rs)), d / h
    return float(k * half / (1 + (k - 1) * half))


# ------------------------------------------------------------------ (a) allocation
def allocation(target, cfg_name):
    px, pg, O, lab = pool(target, cfg_name)
    M = len(px)
    rng = np.random.default_rng(SEED)
    rows = []
    for n in NS:
        if n > M:
            continue
        for d in DEPTHS:
            dx, lv, lg, ce = [], [], [], []
            for rep in range(NREP):
                # Masks WITH replacement: the estimand is the sd of a campaign whose n masks are
                # fresh iid draws from the generator, so the nonparametric bootstrap is the right
                # model. Subsampling without replacement would carry a finite-population
                # correction (1 - n/144) and collapse to sd = 0 at n = 144, which is an artefact
                # of the pool being finite, not a property of the design.
                mi = rng.choice(M, n, replace=True)
                # Seeds WITHOUT replacement: a real campaign uses d distinct slots B_SEEDS[:d]
                # out of the 10 that exist, so at d = 10 there is genuinely no seed-choice noise.
                si = rng.choice(O.shape[1], d, replace=False)
                o = O[np.ix_(mi, si)].mean(1)
                rx, rg = _sp(px[mi], o), _sp(pg[mi], o)
                if np.isfinite(rx) and np.isfinite(rg):
                    lv.append(rx); lg.append(rg); dx.append(rx - rg)
                if rep < NREP_CEIL:
                    c = _ceiling(O[np.ix_(mi, si)], rng)
                    if np.isfinite(c):
                        ce.append(c)
            if not dx:
                continue
            sd_paired = float(np.std(dx, ddof=1))
            ceil_m = float(np.mean(ce)) if ce else np.nan
            lv_m = float(np.mean(lv))
            row = {"target": target, "config": cfg_name, "n_masks": n, "depth": d,
                   "budget": n * d, "ceiling": ceil_m,
                   "level_sd_est": float(np.std(lv, ddof=1)),
                   "level_sd_graddot": float(np.std(lg, ddof=1)),
                   "lds_est_mean": lv_m, "lds_graddot_mean": float(np.mean(lg)),
                   "ratio_est_mean": lv_m / ceil_m if np.isfinite(ceil_m) and ceil_m else np.nan,
                   "paired_sd": sd_paired, "paired_mean": float(np.mean(dx)),
                   "ci_width_95": 2 * 1.96 * sd_paired}
            for e in EFFECTS:
                row[f"power_{e:.2f}"] = float(stats.norm.cdf(e / sd_paired - Z95))
            rows.append(row)
    return pd.DataFrame(rows)


def iso_budget(df):
    out = []
    for b in BUDGETS:
        sub = df[df.budget == b]
        if not len(sub):
            continue
        best = sub.loc[sub.paired_sd.idxmin()]
        for _, r in sub.sort_values("paired_sd").iterrows():
            out.append({"budget": b, "n_masks": int(r.n_masks), "depth": int(r.depth),
                        "ceiling": r.ceiling, "ratio_est_mean": r.ratio_est_mean,
                        "paired_mean": r.paired_mean, "paired_sd": r.paired_sd,
                        "ci_width_95": r.ci_width_95, "power_0.15": r["power_0.15"],
                        "power_0.20": r["power_0.20"],
                        "BEST": bool(r.n_masks == best.n_masks and r.depth == best.depth)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ (c) baseline instability
def baseline_instability(target, cfg_name):
    px, pg, O, lab = pool(target, cfg_name)
    rows = []
    for dr in ALL_DRAWS:
        s = lab == dr
        o = O[s].mean(1)
        rx, rg = _sp(px[s], o), _sp(pg[s], o)
        rows.append({"target": target, "draw": dr, "n": int(s.sum()),
                     "graddot_lds": rg, "estimator_lds": rx, "delta": rx - rg})
    df = pd.DataFrame(rows)
    v_g = float(df.graddot_lds.var(ddof=1))
    v_x = float(df.estimator_lds.var(ddof=1))
    v_d = float(df.delta.var(ddof=1))
    cov = float(np.cov(df.estimator_lds, df.graddot_lds, ddof=1)[0, 1])
    summ = {"target": target, "config": cfg_name,
            "var_graddot_across_draws": v_g, "var_estimator_across_draws": v_x,
            "var_paired_delta": v_d, "cov_est_graddot": cov,
            "baseline_share_of_paired_var": v_g / v_d if v_d else np.nan,
            "graddot_lds_min": float(df.graddot_lds.min()),
            "graddot_lds_max": float(df.graddot_lds.max()),
            "graddot_lds_mean": float(df.graddot_lds.mean()),
            "graddot_lds_sd": float(df.graddot_lds.std(ddof=1))}
    return df, pd.DataFrame([summ])


# ------------------------------------------------------------------ (b) statistic selection
def _top_overlap(pred, out, k=6):
    a = set(np.argsort(-np.asarray(pred))[:k])
    b = set(np.argsort(-np.asarray(out))[:k])
    return len(a & b) / k


def _quartile_gap(pred, out):
    o = np.asarray(out, float)
    q = max(1, len(o) // 4)
    idx = np.argsort(-np.asarray(pred))
    sd = o.std(ddof=1)
    return float((o[idx[:q]].mean() - o[idx[-q:]].mean()) / sd) if sd > 0 else np.nan


STATS = {
    "spearman": lambda p, o: _sp(p, o),
    "kendall_tau_b": lambda p, o: float(stats.kendalltau(p, o, variant="b").statistic),
    "pearson_raw": lambda p, o: float(stats.pearsonr(p, o).statistic),
    "pearson_on_ranks": lambda p, o: float(stats.pearsonr(stats.rankdata(p),
                                                          stats.rankdata(o)).statistic),
    "top6_overlap": lambda p, o: _top_overlap(p, o, 6),
    "quartile_gap": lambda p, o: _quartile_gap(p, o),
}


def statistic_selection(target, cfg_name, n=24, nrep=NREP):
    """Reliability and noise ONLY. Never sees the estimator-vs-baseline contrast."""
    _, _, O, _ = pool(target, cfg_name)
    M, S = O.shape
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for sname, fn in STATS.items():
        rel, null = [], []
        for _ in range(nrep):
            mi = rng.choice(M, n, replace=False)
            p = rng.permutation(S)
            a = O[np.ix_(mi, p[:S // 2])].mean(1)
            b = O[np.ix_(mi, p[S // 2:])].mean(1)
            v = fn(a, b)                       # reliability: outcome vs outcome, estimator-free
            if np.isfinite(v):
                rel.append(v)
            v0 = fn(rng.normal(size=n), O[mi].mean(1))   # noise floor: random predictor
            if np.isfinite(v0):
                null.append(v0)
        rel, null = np.array(rel), np.array(null)
        nsd = float(null.std(ddof=1))
        rows.append({"target": target, "statistic": sname, "n_masks": n,
                     "reliability_halfsplit": float(rel.mean()),
                     "reliability_sd": float(rel.std(ddof=1)),
                     "null_mean": float(null.mean()), "null_sd": nsd,
                     "resolution": (float(rel.mean()) - float(null.mean())) / nsd if nsd else np.nan})
    return pd.DataFrame(rows)


def contrast_by_statistic(target, cfg_name, draws=("K", "L")):
    """Run ONLY after the selection is committed. Reports the contrast under every candidate."""
    px, pg, O, lab = pool(target, cfg_name)
    s = np.isin(lab, list(draws))
    o = O[s].mean(1)
    rng = np.random.default_rng(SEED + 2)
    idx_by_draw = [np.flatnonzero(lab[s] == d) for d in draws]
    rows = []
    for sname, fn in STATS.items():
        d0 = fn(px[s], o) - fn(pg[s], o)
        boots = []
        for _ in range(2000):
            i = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_draw])
            v = fn(px[s][i], o[i]) - fn(pg[s][i], o[i])
            if np.isfinite(v):
                boots.append(v)
        boots = np.array(boots)
        rows.append({"target": target, "config": cfg_name, "draws": "+".join(draws),
                     "statistic": sname, "delta": d0,
                     "ci_lo": float(np.percentile(boots, 2.5)),
                     "ci_hi": float(np.percentile(boots, 97.5)),
                     "p_onesided": float((boots <= 0).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="allocation",
                    choices=["allocation", "statistic", "bystat"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    pd.set_option("display.width", 220)

    if a.stage == "allocation":
        alloc = pd.concat([allocation("C5", "relatif_C5"), allocation("C7", "leverage_C7")],
                          ignore_index=True)
        alloc.to_csv(os.path.join(RESULTS, "p7_allocation.csv"), index=False)
        iso = pd.concat([iso_budget(alloc[alloc.target == t]).assign(target=t)
                         for t in ("C5", "C7")], ignore_index=True)
        iso.to_csv(os.path.join(RESULTS, "p7_iso_budget.csv"), index=False)
        per, summ = [], []
        for t, c in (("C5", "relatif_C5"), ("C7", "leverage_C7")):
            d, s = baseline_instability(t, c)
            per.append(d); summ.append(s)
        pd.concat(per).to_csv(os.path.join(RESULTS, "p7_baseline_per_draw.csv"), index=False)
        dsum = pd.concat(summ, ignore_index=True)
        dsum.to_csv(os.path.join(RESULTS, "p7_baseline_instability.csv"), index=False)

        print("=" * 130)
        print("W0.2(a) ALLOCATION -- sampling sd of the paired contrast, C5 (RelatIF vs GradDot_dmean)")
        print("=" * 130)
        c5 = alloc[alloc.target == "C5"]
        print(c5.pivot(index="n_masks", columns="depth", values="paired_sd").round(4).to_string())
        print("\nceiling by (n_masks, depth), C5:")
        print(c5.pivot(index="n_masks", columns="depth", values="ceiling").round(4).to_string())
        print("\npower to detect Delta = 0.15 (one-sided a=0.05), C5:")
        print(c5.pivot(index="n_masks", columns="depth", values="power_0.15").round(3).to_string())
        print("\n" + "=" * 130)
        print("W0.2(a) ISO-BUDGET -- same number of retrains, different splits")
        print("=" * 130)
        print(iso.round(4).to_string(index=False))
        print("\n" + "=" * 130)
        print("W0.2(c) BASELINE INSTABILITY -- GradDot_dmean is itself a moving target")
        print("=" * 130)
        print(pd.concat(per).round(4).to_string(index=False))
        print()
        print(dsum.round(4).to_string(index=False))

    elif a.stage == "statistic":
        sel = pd.concat([statistic_selection("C5", "relatif_C5"),
                         statistic_selection("C7", "leverage_C7")], ignore_index=True)
        sel.to_csv(os.path.join(RESULTS, "p7_statistic_selection.csv"), index=False)
        print("=" * 120)
        print("W0.2(b) STATISTIC SELECTION -- reliability and noise ONLY (no estimator contrast)")
        print("=" * 120)
        print(sel.round(4).to_string(index=False))
        for t in ("C5", "C7"):
            s = sel[sel.target == t]
            print(f"\n  {t}: highest resolution = {s.loc[s.resolution.idxmax()].statistic}")

    else:
        by = pd.concat([contrast_by_statistic("C5", "relatif_C5"),
                        contrast_by_statistic("C7", "leverage_C7")], ignore_index=True)
        by.to_csv(os.path.join(RESULTS, "p7_contrast_by_statistic.csv"), index=False)
        print("=" * 120)
        print("W0.2(b) CONTRAST UNDER EVERY CANDIDATE STATISTIC (clean OOS draws K+L)")
        print("=" * 120)
        print(by.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
