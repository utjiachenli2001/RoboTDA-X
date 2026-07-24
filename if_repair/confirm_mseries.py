"""Campaign M -- the pass-7 RESOLVING campaign. Preregistered, computed once.

Passes 4-6 reported a self-influence correction (RelatIF, K/G_dd) as the first gradient-side
estimator to beat GradDot out of sample. W0.1 scored that frozen config on campaign K, which
nobody had ever run it against, and the effect reversed (-0.171). Pooled over its clean
out-of-sample draws (K+L, 48 masks) it is Delta rho = -0.044, 95% CI [-0.264, +0.185] -- a null
with an interval too wide to close the question.

M closes it. W0.2 measured the exchange rate: paired sd is flat in seed depth and falls as
1/sqrt(n_masks), so M buys 144 masks at depth 2 (288 retrains, ~7.5 solo-h) rather than the
archived 24 x 10. Predicted paired sd ~0.045 => 95% CI width ~0.18. That is the resolution the
brief's win condition 1 asks for and which no 72-mask campaign can reach at any depth.

PREREG_M -- FAMILY OF ONE, so alpha_abs = 0.05 (no Bonferroni; stated explicitly per the brief).
Frozen and committed while campaign M has ZERO runs.

  M1  RelatIF/C5, cached E=20 Gram, unitl2_then_mean, paired vs GradDot_dmean on the same
      ensemble and the same masks. THE HYPOTHESIS OF RECORD.

C7 does not enter: on its own honest draw set (J+L, excluding its discovery draw K) it is
+0.032, p=0.406, and the brief readmits it only inside a high-power protocol, which a
single-hypothesis campaign is not.

The estimator and baseline are imported from p7_pooled_oos.CONFIGS rather than rebuilt here, so
this scores the IDENTICAL frozen config that tests/test_p7.py pins bit-for-bit to the committed
campaign J and L confirmation rows. Re-implementing it would open exactly the gap BLOCKERS #14
came through.

Primary statistic: Kendall tau_b (W0.2b, selected on reliability and noise alone, committed in
0666081 before any contrast under it was computed). Mandatory secondary: Spearman, for continuity
with every historical number in the repo. Both bars are reported under both statistics.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import functionals as F  # noqa: E402
from if_repair import retrain as RT  # noqa: E402
from if_repair import p7_pooled_oos as P7  # noqa: E402

P7.D.add_repo_paths()
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ---------------------------------------------------------------- FROZEN prereg (edit ONLY before M)
PREREG_M = {
    "M1_relatif_lin_unitL2_C5": {
        "config": "relatif_C5", "target": "C5",
        "why": "the pass-4/5/6 hypothesis of record. Confirmed on J (ratio 0.533), reversed on "
               "K (-0.171), pooled K+L Delta rho -0.044 CI [-0.264,+0.185]. M is sized to close "
               "the interval, not to give the effect another chance.",
    },
}
ALPHA_ABS = 0.05 / len(PREREG_M)
N_BOOT = 5000
BOOT_SEED = 7

STATS = {
    "kendall_tau_b": lambda p, o: float(stats.kendalltau(p, o, variant="b").statistic),
    "spearman": lambda p, o: spearman(p, o),
}


def _p_onesided(name, px, o):
    if name == "spearman":
        return float(spearman_p_onesided(spearman(px, o), len(o)))
    return float(stats.kendalltau(px, o, alternative="greater").pvalue)


def m_masks():
    return RT.fresh_demo_masks_pooled()


def ceiling(raw, statfn, max_splits=200):
    """The archived split-half recipe with a pluggable statistic.

    For Spearman this IS functionals.split_half_ceiling (asserted in tests/test_p7.py). For
    Kendall the same disjoint-half construction is used and the same Spearman-Brown step is
    applied; SB is derived for correlations, so the Kendall ceiling is an approximation and is
    labelled as such wherever it is reported.
    """
    import itertools
    masks = sorted(raw)
    seeds = sorted({s for m in masks for s in raw[m]}, key=lambda x: (str(type(x)), str(x)))
    masks = [m for m in masks if all(s in raw[m] for s in seeds)]
    S = len(seeds)
    h = S // 2
    if S < 2 or len(masks) < 4:
        return np.nan
    seen, splits = set(), []
    for c in itertools.combinations(range(S), h):
        rest = tuple(sorted(set(range(S)) - set(c)))
        if len(rest) != h:
            continue
        k = frozenset([c, rest])
        if k in seen:
            continue
        seen.add(k)
        splits.append((c, rest))
    rs = []
    for a, b in splits[:max_splits]:
        va = np.array([np.mean([raw[m][seeds[i]] for i in a]) for m in masks])
        vb = np.array([np.mean([raw[m][seeds[i]] for i in b]) for m in masks])
        r = statfn(va, vb)
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return np.nan
    half, k = float(np.mean(rs)), S / h
    return float(k * half / (1 + (k - 1) * half))


def stratified_bootstrap(px, pg, o, strata, statfn, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    d0 = statfn(px, o) - statfn(pg, o)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        v = statfn(px[i], o[i]) - statfn(pg[i], o[i])
        if np.isfinite(v):
            diffs.append(v)
    diffs = np.array(diffs)
    if not len(diffs):
        return d0, np.nan, np.nan, np.nan
    return (d0, float((diffs <= 0).mean()),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def evaluate(campaign="M", masks=None, strata=None):
    """Score PREREG_M. `masks`/`strata` are overridable ONLY so the wiring can be smoke-tested
    against a draw that already has a committed answer, before campaign M has a single run
    (the pass-3 pattern that caught BLOCKERS #14). Confirmation always runs the defaults."""
    if masks is None:
        masks = m_masks()
        strata = np.array([m["subdraw"] for m in masks])
    strata = np.asarray(strata)
    rows = []
    for name, spec in PREREG_M.items():
        cfg = P7.CONFIGS[spec["config"]]
        t = spec["target"]
        est = cfg["scores"]()
        base = P7._graddot(cfg["baseline"])[t]
        raw = F.campaign_outcomes(campaign, "plain", targets=(t,))[t]
        obs = F.seed_mean(raw)
        out = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
        ok = np.isfinite(out)
        px = P7.mask_pred(est, masks)[ok]
        pg = P7.mask_pred(base, masks)[ok]
        o, st = out[ok], strata[ok]
        for sname, fn in STATS.items():
            c = ceiling(raw, fn)
            rho, rho_g = fn(px, o), fn(pg, o)
            p_abs = _p_onesided(sname, px, o)
            d0, pp, lo, hi = stratified_bootstrap(px, pg, o, st, fn)
            rows.append({
                "name": name, "target": t, "statistic": sname,
                "primary": sname == "kendall_tau_b", "n_masks": int(ok.sum()),
                "depth": RT.M_DEPTH, "lds": rho, "graddot_lds": rho_g, "ceiling": c,
                "ratio": rho / c if np.isfinite(c) and c else np.nan,
                "p_abs": p_abs, "alpha_abs": ALPHA_ABS,
                "paired_delta": d0, "paired_p": pp, "ci_lo": lo, "ci_hi": hi,
                "ci_width": hi - lo,
                "PASS_abs": bool(np.isfinite(rho) and np.isfinite(c) and rho / c >= 0.5
                                 and p_abs < ALPHA_ABS),
                "PASS_paired": bool(np.isfinite(d0) and d0 > 0 and pp < 0.05)})
    return pd.DataFrame(rows)


def grand_pooled():
    """SECONDARY: M pooled with the pre-existing clean OOS masks (K+L). Reported, not a bar."""
    rows = []
    for name, spec in PREREG_M.items():
        cfg = P7.CONFIGS[spec["config"]]
        t = spec["target"]
        est, base = cfg["scores"](), P7._graddot(cfg["baseline"])[t]
        px, pg, o, st = [], [], [], []
        for dr in ("K", "L"):
            ms = P7.masks_for(dr)
            obs = F.seed_mean(P7.raw_outcomes(dr, t))
            v = np.array([obs.get(m["mask_id"], np.nan) for m in ms])
            k = np.isfinite(v)
            px += list(P7.mask_pred(est, ms)[k]); pg += list(P7.mask_pred(base, ms)[k])
            o += list(v[k]); st += [dr] * int(k.sum())
        ms = m_masks()
        obs = F.seed_mean(F.campaign_outcomes("M", "plain", targets=(t,))[t])
        v = np.array([obs.get(m["mask_id"], np.nan) for m in ms])
        k = np.isfinite(v)
        px += list(P7.mask_pred(est, ms)[k]); pg += list(P7.mask_pred(base, ms)[k])
        o += list(v[k]); st += [m["subdraw"] for m, kk in zip(ms, k) if kk]
        px, pg, o, st = np.array(px), np.array(pg), np.array(o), np.array(st)
        for sname, fn in STATS.items():
            d0, pp, lo, hi = stratified_bootstrap(px, pg, o, st, fn)
            rows.append({"name": name, "scope": "K+L+M (mixed depth 10/10/2)",
                         "statistic": sname, "n_masks": len(o), "paired_delta": d0,
                         "paired_p": pp, "ci_lo": lo, "ci_hi": hi, "ci_width": hi - lo})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="M", choices=["M"])
    ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    df = evaluate("M")
    df.to_csv(os.path.join(RESULTS, "confirm_mseries.csv"), index=False)
    gp = grand_pooled()
    gp.to_csv(os.path.join(RESULTS, "confirm_mseries_grandpooled.csv"), index=False)
    pd.set_option("display.width", 220)
    print("=" * 118)
    print(f"CAMPAIGN M -- pass-7 resolving campaign. Family of 1, alpha_abs = {ALPHA_ABS:.4f}")
    print("144 masks x depth 2. PRIMARY statistic Kendall tau_b; Spearman secondary.")
    print("=" * 118)
    print(df[["name", "statistic", "primary", "n_masks", "lds", "graddot_lds", "ceiling",
              "ratio", "p_abs", "PASS_abs", "paired_delta", "ci_lo", "ci_hi", "ci_width",
              "paired_p", "PASS_paired"]].round(4).to_string(index=False))
    print("\nSECONDARY -- pooled with the pre-existing clean OOS masks (not a bar):")
    print(gp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
