"""PASS 9 -- is campaign N's PRIMARY number an attribution result or a training-set-size effect?

Zero GPU. This file exists because a leakage control on an unrelated estimator failed in a way that
implicated the pass-8 headline rather than the estimator.

THE OBSERVATION. `p9_datamodel_cluster.permutation_control` shuffles mask outcomes WITHIN stratum
and re-runs the whole leave-one-mask-out pipeline. Under a shuffle there is no attribution signal
left to find, so the control has to collapse to ~0. It returned Kendall **0.425** against a real
0.679. Shuffling within stratum destroys within-stratum structure but preserves each stratum's
MEAN, so the only thing a shuffled run can still track is between-stratum variation -- |S|, the
number of clusters the mask keeps, which sets the training-set size to 60/75/90 demos.

WHY THAT REACHES PASS 8. `p8_prereg.md` is explicit: "|S| is a stratum, not a covariate to pool
over. It sets the training-set size (60/75/90 demos), which moves the outcome directly ... Reported
per stratum and pooled with |S| controlled." But `confirm_nseries.evaluate` computes the PRIMARY
absolute bar as `rho = fn(pg, o)` over all 149 conditional masks POOLED, with no stratum control --
`st` is built there and then used only by the secondary paired analysis. The prereg promised a
control the primary did not apply.

WHAT THIS FILE MEASURES. Three things, all on campaign N's committed data and the frozen
`p7_pooled_oos._graddot("cached")` object:

  1. GradDot's LDS, ceiling and ratio POOLED (reproducing the committed 0.4747 / 0.6715 / 0.707)
     and WITHIN each |S| stratum, where training-set size is constant by construction.
  2. A permutation control on GradDot itself. GradDot is a FIXED estimator -- its scores are not
     fit to anything -- so a shuffled-outcome correlation cannot be leakage under any definition.
     Whatever pooled correlation survives the shuffle is the design's |S| effect, full stop.
  3. The same permutation control computed WITHIN stratum, which must collapse to ~0 and thereby
     shows the per-stratum numbers are clean.

This is a diagnosis of an existing committed result, not a preregistered test. No alpha, no bar,
no new hypothesis. It is reported with its full uncertainty and the conclusion is whatever it is.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import functionals as F  # noqa: E402
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import p8_masks as P8M  # noqa: E402
from if_repair.confirm_mseries import ceiling, STATS  # noqa: E402
from if_repair.confirm_nseries import (achieved_depth, analysis_depth,  # noqa: E402
                                       conditional_masks)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PRIMARY_STAT = "kendall_tau_b"
N_PERM = 2000
N_BOOT = 5000


def load(campaign="N", target="C5", depth_override=None):
    masks_all = P8M.manifest()["masks"]
    ms = conditional_masks(target, masks_all)
    raw_all = F.campaign_outcomes(campaign, "plain", targets=(target,))[target]
    raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in ms if m["mask_id"] in raw_all}
    depth, seeds = achieved_depth(raw, len(ms))
    d = depth_override or analysis_depth(depth)
    seeds = list(seeds)[:d]
    raw = {m: {s: v[s] for s in seeds} for m, v in raw.items() if all(s in v for s in seeds)}
    obs = F.seed_mean(raw)
    use = [m for m in ms if m["mask_id"] in obs]
    y = np.array([obs[m["mask_id"]] for m in use], float)
    pg = P7.mask_pred(P7._graddot("cached")[target], use)
    st = np.array([m["stratum"] for m in use])
    ok = np.isfinite(y) & np.isfinite(pg)
    use = [m for m, k in zip(use, ok) if k]
    raw = {m["mask_id"]: raw[m["mask_id"]] for m in use}
    return y[ok], pg[ok], st[ok], use, raw, d


def perm_pooled(pg, y, st, fn, n=N_PERM, seed=0):
    """Shuffle outcomes WITHIN stratum, correlate pooled. GradDot is fixed, so this is not
    leakage -- it is the |S| effect and nothing else."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        yp = y.copy()
        for s in set(st):
            j = np.where(st == s)[0]
            yp[j] = y[j][rng.permutation(len(j))]
        out[i] = fn(pg, yp)
    return out


def boot_ratio(pg, y, raw, ids, fn, n=N_BOOT, seed=0):
    """Bootstrap the RATIO, resampling masks and recomputing the ceiling on the same resample."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    m = len(y)
    for i in range(n):
        j = rng.integers(0, m, m)
        if np.std(pg[j]) == 0 or np.std(y[j]) == 0:
            out[i] = np.nan
            continue
        sub = {f"{ids[k]}#{q}": raw[ids[k]] for q, k in enumerate(j)}
        c = ceiling(sub, fn)
        out[i] = fn(pg[j], y[j]) / c if np.isfinite(c) and c else np.nan
    return out[np.isfinite(out)]


def evaluate(campaign="N", target="C5"):
    y, pg, st, use, raw, d = load(campaign, target)
    ids = [m["mask_id"] for m in use]
    rows = []
    for sname, fn in STATS.items():
        # ---- pooled (what the committed primary reports)
        c = ceiling(raw, fn)
        lds = fn(pg, y)
        perm = perm_pooled(pg, y, st, fn)
        br = boot_ratio(pg, y, raw, ids, fn)
        rows.append({
            "scope": "POOLED (committed primary)", "statistic": sname,
            "primary": sname == PRIMARY_STAT, "n_masks": len(y), "depth": d,
            "lds": lds, "ceiling": c, "ratio": lds / c,
            "ratio_ci_lo": float(np.percentile(br, 2.5)) if len(br) else np.nan,
            "ratio_ci_hi": float(np.percentile(br, 97.5)) if len(br) else np.nan,
            "perm_mean": float(perm.mean()), "perm_p97.5": float(np.percentile(perm, 97.5)),
            "lds_above_perm": lds - float(perm.mean()),
            "clears_half_bar": bool(lds / c >= 0.5),
        })
        # ---- within stratum (|S| constant by construction)
        for s in sorted(set(st)):
            j = st == s
            if j.sum() < 8 or np.std(pg[j]) == 0:
                continue
            raw_s = {m["mask_id"]: raw[m["mask_id"]] for m, k in zip(use, j) if k}
            c_s = ceiling(raw_s, fn)
            l_s = fn(pg[j], y[j])
            perm_s = perm_pooled(pg[j], y[j], st[j], fn)
            br_s = boot_ratio(pg[j], y[j], raw_s, [m["mask_id"] for m, k in zip(use, j) if k], fn)
            rows.append({
                "scope": f"within {s}", "statistic": sname, "primary": sname == PRIMARY_STAT,
                "n_masks": int(j.sum()), "depth": d,
                "lds": l_s, "ceiling": c_s, "ratio": l_s / c_s if np.isfinite(c_s) and c_s else np.nan,
                "ratio_ci_lo": float(np.percentile(br_s, 2.5)) if len(br_s) else np.nan,
                "ratio_ci_hi": float(np.percentile(br_s, 97.5)) if len(br_s) else np.nan,
                "perm_mean": float(perm_s.mean()),
                "perm_p97.5": float(np.percentile(perm_s, 97.5)),
                "lds_above_perm": l_s - float(perm_s.mean()),
                "clears_half_bar": bool(np.isfinite(c_s) and c_s and l_s / c_s >= 0.5),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="N")
    ap.add_argument("--target", default="C5")
    ap.add_argument("--out", default=os.path.join(RESULTS, "p9_stratum_control.csv"))
    a = ap.parse_args()
    df = evaluate(a.campaign, a.target)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df[df.primary].to_string(index=False))
        print()
        print(df[~df.primary].to_string(index=False))
    print(f"\n[p9/stratum-control] -> {a.out}")


if __name__ == "__main__":
    main()
