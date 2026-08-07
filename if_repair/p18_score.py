"""PASS 18 -- campaign U scoring. Zero GPU. Preregistered in `if_repair/p18_prereg.md` (frozen).

WHAT IS PRIMARY. The pair (tau, r) per pool, NOT a ratio. Campaign T's primary was rho/sqrt(r) and
died because r went to zero at its top rung; more subtly, `sqrt(r)` disattenuation is a PEARSON
identity applied to a KENDALL statistic, so the "normalised" quantity drifts mechanically as r
drifts -- along the very axis being regressed. Campaign U therefore regresses tau directly and
tests the r trend separately (prereg 3.2). rho/sqrt(r) is reported for continuity with campaigns
O/R and carries no alpha.

THE CEILING is a split-half KENDALL between the two seeds' outcome rankings, reported WITHOUT a
Spearman-Brown correction -- S-B is itself a Pearson identity and applying it here would rebuild
the fault in the denominator. It therefore understates the depth-2 reliability by a stable factor
and is compared only against other depth-2 ceilings in this campaign.

FLATNESS IS AN EQUIVALENCE CLAIM. Claiming "no trend" from a failure to reject is how a design
emits its most quotable sentence regardless of truth. Flatness requires TOST: the 95% CI on the
slope must lie inside (-Delta_tau, +Delta_tau) with Delta_tau = 0.04966, frozen from the gate
before any campaign retrain. A CI containing zero but extending past the margin is
"uninformative", not a finding.

BRANCH PRECEDENCE (prereg 5): permutation null -> r-band -> non-monotonicity -> slope/TOST.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p18_corpus as C  # noqa: E402
from if_repair import p18_campaign_u as U  # noqa: E402
from if_repair import p18_gate as G  # noqa: E402
from if_repair import p18_gram as GR  # noqa: E402

D.add_repo_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT_CSV = os.path.join(RESULTS, "confirm_useries.csv")

DELTA_TAU = 0.04966          # frozen from the four-pool conditioned gate; prereg 3.5
R_BAND = (0.65, 0.95)        # prereg 3.3
ALPHA = 0.025                # family of 2 (H1, H2), Bonferroni from 0.05
N_BOOT = 2000


# --------------------------------------------------------------------------- outcomes
def outcome_table():
    """-> {(pool, partition): {sig: (mean_outcome, o_seed_a, o_seed_b)}}.

    The seed pair used for a mask is the FIRST pair that the frozen convergence gate does not
    flag: the primary (4401,4402), else reserve (4403,4404), else (4405,4406). A mask flagged at
    every available pair is DROPPED WHOLE and reported -- never partially, which would select the
    surviving seed and truncate the noise distribution upward, inflating the ceiling.
    """
    runs, pool_of = G.load_runs()
    thr = G.thresholds(runs, pool_of)
    by_sig = {m["sig"]: m for pool in U.N_MASKS for p in C.PARTITIONS
              for m in U.build(pool, p)[0]}
    table, dropped = {}, []
    for sig, m in by_sig.items():
        pool, part = m["pool"], m["partition"]
        chosen = None
        for pair in (U.SEEDS,) + U.RESERVE_PAIRS:
            if not all(s in runs.get(sig, {}) for s in pair):
                continue
            d = abs(runs[sig][pair[0]][1] - runs[sig][pair[1]][1])
            if d <= thr[pool]["threshold"]:
                chosen = pair
                break
        if chosen is None:
            dropped.append(sig)
            continue
        a, b = runs[sig][chosen[0]][0], runs[sig][chosen[1]][0]
        table.setdefault((pool, part), {})[sig] = (0.5 * (a + b), a, b)
    return table, dropped


# --------------------------------------------------------------------------- predictions
def gradient_prediction(pool, partition, sigs, arm="H1"):
    """Predicted outcome for each mask = SUM of per-demo GradDot scores over its RETAINED demos.

    Sign convention: a higher summed score means a training set the estimator expects to reduce
    held-out loss more. Kendall is rank-based, so only the ordering matters.
    """
    sc = GR.load(pool, arm)
    by_sig = {m["sig"]: m for m in U.build(pool, partition)[0]}
    return np.array([sum(sc[d] for d in by_sig[s]["demos"]) for s in sigs])


def datamodel_prediction(pool, fit_partition, score_partition, table, alpha=1.0):
    """Fit group coefficients on `fit_partition`'s masks, score `score_partition`'s masks.

    OUT-OF-PARTITION by construction (prereg 4). WHAT_STANDS #50/#53 established that
    within-partition datamodel figures are inflated and only cross-partition transfer counts, so
    the alpha-carrying arm never scores the masks it was fit on. The coefficient -> per-demo ->
    other-partition-mask path is `p11_transfer`'s, because the two partitions share no groups.

    The fit uses an INFORMATION-EQUALISED subsample (prereg 6): holding masks-per-coefficient
    constant still lets per-coefficient information n*p(1-p) rise 1.87x along the ladder, which
    would bias this arm's slope in a way the permutation null provably cannot detect.
    """
    fit_masks = U.build(pool, fit_partition)[0]
    fit_sigs = [m["sig"] for m in fit_masks if m["sig"] in table[(pool, fit_partition)]]
    n_fit = min(len(fit_sigs), int(round(9.2 * len(C.groups(pool, fit_partition)) ** 2
                                         / (len(C.groups(pool, fit_partition)) - 5))))
    fit_sigs = fit_sigs[:n_fit]
    gids = [g["group_id"] for g in C.groups(pool, fit_partition)]
    gi = {g: k for k, g in enumerate(gids)}
    by_sig = {m["sig"]: m for m in fit_masks}
    X = np.zeros((len(fit_sigs), len(gids)))
    for r, s in enumerate(fit_sigs):
        for g in by_sig[s]["groups"]:
            X[r, gi[g]] = 1.0
    y = np.array([table[(pool, fit_partition)][s][0] for s in fit_sigs])
    Xc, yc = X - X.mean(0), y - y.mean()
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(len(gids)), Xc.T @ yc)
    per_demo = {}
    for g, k in gi.items():
        for d in C.index(pool, fit_partition)[g]["demos"]:
            per_demo[d] = beta[k] / C.GROUP_SIZE
    sc_masks = {m["sig"]: m for m in U.build(pool, score_partition)[0]}
    sigs = [s for s in sc_masks if s in table[(pool, score_partition)]]
    pred = np.array([sum(per_demo[d] for d in sc_masks[s]["demos"]) for s in sigs])
    return sigs, pred


# --------------------------------------------------------------------------- statistics
def r2_from_kendall(tau):
    """Split-half KENDALL (a depth-1 rank quantity) -> the variance-based depth-2 reliability.

        rho1 = sin(tau * pi/2)     Kendall -> Pearson under bivariate normality
        r2   = 2*rho1/(1+rho1)     Spearman-Brown, depth 1 -> depth 2

    Both identities are already used elsewhere in the frozen prereg (§0(a), §3.1). See AMENDMENT 1
    in `p18_prereg.md` for why both scales are reported.
    """
    rho1 = np.sin(np.clip(tau, -1, 1) * np.pi / 2)
    return float(2 * rho1 / (1 + rho1)) if rho1 > -1 else float("nan")


def split_half_kendall(table, pool, partition, sigs=None):
    """Ceiling r: Kendall tau between the two seeds' outcome rankings. No S-B correction."""
    cell = table[(pool, partition)]
    sigs = sigs or sorted(cell)
    a = np.array([cell[s][1] for s in sigs])
    b = np.array([cell[s][2] for s in sigs])
    return float(stats.kendalltau(a, b).statistic)


def cell_tau(table, pool, partition, pred, sigs):
    y = np.array([table[(pool, partition)][s][0] for s in sigs])
    return float(stats.kendalltau(pred, y).statistic)


def wls_slope(x, y, w):
    x, y, w = map(np.asarray, (x, y, w))
    xb = np.sum(w * x) / np.sum(w)
    yb = np.sum(w * y) / np.sum(w)
    return float(np.sum(w * (x - xb) * (y - yb)) / np.sum(w * (x - xb) ** 2))


def tost(ci_lo, ci_hi, margin=DELTA_TAU):
    """-> 'flat' | 'trend' | 'uninformative'."""
    if ci_lo > 0 or ci_hi < 0:
        return "trend"
    if ci_lo >= -margin and ci_hi <= margin:
        return "flat"
    return "uninformative"


# --------------------------------------------------------------------------- the campaign read
def arm_curve(table, arm):
    """-> {pool: {'tau': partition-averaged tau, 'r': partition-averaged ceiling, 'cells': [...]}}"""
    out = {}
    for pool in C.RUNGS:
        taus, rs, cells = [], [], []
        for part in C.PARTITIONS:
            if arm in ("H1", "H1f"):
                sigs = sorted(table[(pool, part)])
                pred = gradient_prediction(pool, part, sigs, arm=arm)
            else:                                   # H2: fit on the OTHER partition
                other = "B" if part == "A" else "A"
                sigs, pred = datamodel_prediction(pool, other, part, table)
            t = cell_tau(table, pool, part, pred, sigs)
            r = split_half_kendall(table, pool, part, sigs)
            taus.append(t)
            rs.append(r)
            cells.append({"partition": part, "n_masks": len(sigs), "tau": t, "r": r,
                          "r2": r2_from_kendall(r),
                          "ratio_rho_over_sqrt_r": t / np.sqrt(r) if r > 0 else float("nan")})
        rbar = float(np.mean(rs))
        out[pool] = {"tau": float(np.mean(taus)), "r": rbar,
                     "r2": r2_from_kendall(rbar), "cells": cells}
    return out


def bootstrap_slope(table, arm, n_boot=N_BOOT, seed=0):
    """Resample MASKS within each cell, recompute the ceiling on every resample, refit the slope.

    The ceiling is recomputed per resample because it is an estimated quantity of the same data;
    holding it fixed would understate the slope's uncertainty (BLOCKERS #42/#56).
    """
    rng = np.random.default_rng(seed)
    x = np.array([np.log2(p / 50) for p in C.RUNGS])
    slopes, r_slopes = [], []
    base = {}
    for pool in C.RUNGS:
        for part in C.PARTITIONS:
            sigs = sorted(table[(pool, part)])
            if arm in ("H1", "H1f"):
                base[(pool, part)] = (sigs, gradient_prediction(pool, part, sigs, arm=arm))
            else:
                other = "B" if part == "A" else "A"
                base[(pool, part)] = datamodel_prediction(pool, other, part, table)
    for _ in range(n_boot):
        taus, rs, ws = [], [], []
        for pool in C.RUNGS:
            tt, rr = [], []
            for part in C.PARTITIONS:
                sigs, pred = base[(pool, part)]
                idx = rng.integers(0, len(sigs), len(sigs))
                ss = [sigs[i] for i in idx]
                y = np.array([table[(pool, part)][s][0] for s in ss])
                a = np.array([table[(pool, part)][s][1] for s in ss])
                b = np.array([table[(pool, part)][s][2] for s in ss])
                tt.append(stats.kendalltau(pred[idx], y).statistic)
                rr.append(stats.kendalltau(a, b).statistic)
            taus.append(np.mean(tt))
            rs.append(np.mean(rr))
            ws.append(len(base[(C.RUNGS[0], "A")][0]))
        n = np.array([len(base[(p, "A")][0]) for p in C.RUNGS], float)
        w = n * (n - 1)                       # inverse-variance up to a constant (var ~ 1/n)
        slopes.append(wls_slope(x, taus, w))
        r_slopes.append(wls_slope(x, rs, w))
    return np.array(slopes), np.array(r_slopes)


def permutation_null(table, arm, n_perm=500, seed=1):
    """Prereg 3.8. A FIXED estimator, outcomes shuffled WITHIN pool, through the whole pipeline
    including the ceiling and the slope. If the null pipeline produces a slope, the pipeline
    manufactures slopes. Named limit: it cannot detect differential attenuation of REAL signal
    (there is none to attenuate under the null), which is why prereg 6 removes that confound by
    construction instead of testing for it."""
    rng = np.random.default_rng(seed)
    x = np.array([np.log2(p / 50) for p in C.RUNGS])
    base = {}
    for pool in C.RUNGS:
        for part in C.PARTITIONS:
            sigs = sorted(table[(pool, part)])
            if arm in ("H1", "H1f"):
                base[(pool, part)] = (sigs, gradient_prediction(pool, part, sigs, arm=arm))
            else:
                other = "B" if part == "A" else "A"
                base[(pool, part)] = datamodel_prediction(pool, other, part, table)
    out = []
    for _ in range(n_perm):
        taus = []
        for pool in C.RUNGS:
            tt = []
            for part in C.PARTITIONS:
                sigs, pred = base[(pool, part)]
                y = np.array([table[(pool, part)][s][0] for s in sigs])
                tt.append(stats.kendalltau(pred, rng.permutation(y)).statistic)
            taus.append(np.mean(tt))
        n = np.array([len(base[(p, "A")][0]) for p in C.RUNGS], float)
        out.append(wls_slope(x, taus, n * (n - 1)))
    return np.array(out)


def score(force=False, n_boot=N_BOOT):
    """The one-shot preregistered read. Refuses to overwrite and refuses partial families."""
    if os.path.exists(OUT_CSV) and not force:
        raise RuntimeError(f"{OUT_CSV} exists -- campaign U is SCORED ONCE. Refusing to overwrite.")
    table, dropped = outcome_table()
    for pool in C.RUNGS:
        for part in C.PARTITIONS:
            if (pool, part) not in table or len(table[(pool, part)]) < 50:
                raise RuntimeError(f"arm incomplete at pool {pool}{part} -- refusing to score "
                                   f"a partial family (prereg 9)")
    res = {"delta_tau": DELTA_TAU, "alpha": ALPHA, "n_boot": n_boot,
           "dropped_masks_all_reserves_exhausted": len(dropped), "arms": {}}
    for arm in ("H1", "H2", "H1f"):
        curve = arm_curve(table, arm)
        sl, rsl = bootstrap_slope(table, arm, n_boot=n_boot)
        lo, hi = np.percentile(sl, [100 * ALPHA, 100 * (1 - ALPHA)])
        rlo, rhi = np.percentile(rsl, [2.5, 97.5])
        nullsl = permutation_null(table, arm) if arm != "H1f" else None
        res["arms"][arm] = {
            "alpha_carrying": arm in ("H1", "H2"),
            "per_pool": {str(p): {"tau": curve[p]["tau"], "r": curve[p]["r"],
                                  "r2": curve[p]["r2"], "cells": curve[p]["cells"]}
                         for p in C.RUNGS},
            "slope_tau": float(np.mean(sl)), "ci": [float(lo), float(hi)],
            "verdict": tost(lo, hi),
            "slope_r": float(np.mean(rsl)), "r_trend_ci": [float(rlo), float(rhi)],
            # AMENDMENT 1: the band is reported on BOTH scales. §3.1's statistic is a depth-1
            # split-half Kendall; §3.3's band, §7's gate and §3.5's C are all on the depth-2
            # variance-based r2. Neither reading is suppressed.
            "r_in_band_kendall_scale": all(R_BAND[0] <= curve[p]["r"] <= R_BAND[1]
                                           for p in C.RUNGS),
            "r_in_band_r2_scale": all(R_BAND[0] <= curve[p]["r2"] <= R_BAND[1]
                                      for p in C.RUNGS),
            "permutation_null_slope_abs_p95": (float(np.percentile(np.abs(nullsl), 95))
                                               if nullsl is not None else None),
        }
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(OUT_CSV.replace(".csv", ".json"), "w"), indent=1)
    with open(OUT_CSV, "w") as fh:
        fh.write("arm,pool,tau,r_kendall,r2,ratio,n_masks_A,n_masks_B\n")
        for arm, a in res["arms"].items():
            for p in C.RUNGS:
                d = a["per_pool"][str(p)]
                fh.write(f"{arm},{p},{d['tau']:.6f},{d['r']:.6f},{d['r2']:.6f},"
                         f"{d['tau'] / np.sqrt(d['r']) if d['r'] > 0 else float('nan'):.6f},"
                         f"{d['cells'][0]['n_masks']},{d['cells'][1]['n_masks']}\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--dry-run", action="store_true", help="outcome table only, no scoring")
    a = ap.parse_args()
    if a.dry_run:
        table, dropped = outcome_table()
        print(f"[p18/score] outcome table: {sum(len(v) for v in table.values())} masks, "
              f"{len(dropped)} dropped (reserves exhausted)")
        for pool in C.RUNGS:
            for part in C.PARTITIONS:
                cell = table.get((pool, part), {})
                r = split_half_kendall(table, pool, part) if cell else float("nan")
                print(f"  pool {pool:3d}{part}: {len(cell):5d} masks  ceiling r={r:.4f}")
        return
    if a.score:
        res = score(force=a.force, n_boot=a.n_boot)
        print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))
        for arm, d in res["arms"].items():
            tag = "ALPHA" if d["alpha_carrying"] else "descriptive"
            print(f"\n[{arm}] ({tag})  slope={d['slope_tau']:+.5f} "
                  f"CI [{d['ci'][0]:+.5f}, {d['ci'][1]:+.5f}]  -> {d['verdict'].upper()}")
            print(f"      r trend CI [{d['r_trend_ci'][0]:+.4f}, {d['r_trend_ci'][1]:+.4f}]  "
                  f"r in band (kendall/r2): {d['r_in_band_kendall_scale']}/"
                  f"{d['r_in_band_r2_scale']}  null |slope| p95="
                  f"{d['permutation_null_slope_abs_p95']}")
            for p in C.RUNGS:
                x = d["per_pool"][str(p)]
                print(f"      pool {p:3d}: tau={x['tau']:+.4f}  "
                      f"r_kendall={x['r']:.4f}  r2={x['r2']:.4f}")


if __name__ == "__main__":
    main()
