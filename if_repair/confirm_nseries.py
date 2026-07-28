"""Campaign N -- the pass-8 CLUSTER-GRAIN confirmation. Preregistered, computed once.

PREREG_N (if_repair/p8_prereg.md, frozen at 956c061 while campaign N had zero runs).

  N1  GradDot_dmean, target C5, cluster grain, conditional on target-in-mask, clears the
      ABSOLUTE half-ceiling bar: lds >= 0.5 x cluster-grain noise ceiling, both computed
      within campaign N. FAMILY OF ONE -> alpha_abs = 0.05.

Why an absolute bar rather than a paired one. Pass 7's HANDOFF open thread #4 asks whether the
half-ceiling bar is unreachable for every estimator anyone would try -- in which case it measures
the ceiling instead of discriminating hypotheses. Seven passes failed it at demo grain. If a
plain, uncorrected GradDot clears it as soon as the unit is 15 demos instead of 1, the bar was
fine and the grain was the problem.

The estimator and its baseline are imported from `p7_pooled_oos.CONFIGS`, not rebuilt, so this
scores the identical frozen object that pass 7's committed rows were computed from. Rebuilding it
is the gap BLOCKERS #14 came through.

TWO THINGS THIS FILE ENFORCES MECHANICALLY, because remembering them is not a control:

1. SCORE ONCE. `--out` refuses to overwrite an existing result file. p = 0.057 on campaign M is
   exactly the situation `p7_prereg.md` §7 exists to protect.
2. THE PREREGISTERED STOPPING RULE. Campaign N is time-boxed, so the analysis runs at the largest
   depth d for which ALL masks have d completed seeds. That prefix is read off the RUN DIRECTORY
   and never off the outcomes, and because the job list is seed-major it is always a complete,
   balanced design. `achieved_depth` below is the only place depth is decided.
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
from if_repair.confirm_mseries import ceiling, stratified_bootstrap, STATS, _p_onesided  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

PREREG_N = {
    "N1_graddot_absbar_C5": {
        "config": "graddot", "target": "C5",
        "why": "the pass-8 hypothesis of record. Plain GradDot_dmean scored +0.239 Kendall at "
               "cluster grain on Stage F (n=40 conditional), a draw it was never tuned against, "
               "while every self-influence correction from passes 4-7 reversed sign there. "
               "Campaign N is a fresh disjoint draw and tests the absolute bar, not the scan.",
    },
}
ALPHA_ABS = 0.05 / len(PREREG_N)
PRIMARY_STAT = "kendall_tau_b"


# ------------------------------------------------------------------ the stopping rule
def achieved_depth(raw_by_mask, n_expected):
    """Largest d such that EVERY expected mask has the first d seeds completed.

    Reads completion only -- never an outcome value. Seeds are ordered, and the seed-major job
    list means a complete prefix is a complete balanced design.
    """
    seeds = sorted({s for v in raw_by_mask.values() for s in v})
    d = 0
    for k in range(1, len(seeds) + 1):
        pref = seeds[:k]
        done = [m for m, v in raw_by_mask.items() if all(s in v for s in pref)]
        if len(done) >= n_expected:
            d = k
        else:
            break
    return d, seeds[:d]


def analysis_depth(achieved):
    """Largest EVEN depth <= achieved. Used for BOTH the LDS and the ceiling.

    DEFECT FOUND BEFORE ANY CONTRAST WAS COMPUTED (2026-07-27, campaign N still running, no
    result file written). The split-half ceiling recipe -- `confirm_mseries.ceiling`, the one
    PREREG_N names -- builds two DISJOINT EQUAL halves and drops any split whose remainder is the
    wrong size (`h = S // 2`, `if len(rest) != h: continue`). At odd S every split is dropped and
    it returns NaN. Verified: S=2,4,6 give a ceiling; S=3,5 give NaN. PREREG_N specifies depth 5,
    so scoring at the achieved depth would have produced a NaN bar and failed the primary for a
    mechanical reason rather than a scientific one.

    The fix is to analyse at the largest even depth. Both quantities use it, rather than only the
    ceiling, because the Spearman-Brown step extrapolates from half-depth to full S: a ceiling
    built on 4 seeds is the bar for a depth-4 outcome, and testing a depth-5 LDS against it would
    be ANTI-CONSERVATIVE -- a lower bar than the estimate deserves. Matching them is the
    conservative choice and keeps the ratio internally consistent.

    This changes no hypothesis, no target, no statistic and no direction. The full achieved depth
    is still reported as a secondary robustness read.
    """
    return 2 * (achieved // 2)


def conditional_masks(target, masks):
    """Masks that CONTAIN the target cluster.

    src/analysis.py names the unconditional cluster-grain number `INFLATED` for a structural
    reason: a mask that drops the target entirely has its outcome dominated by that removal
    rather than by the ordering of the demos it kept.
    """
    return [m for m in masks if target in m["clusters"]]


def evaluate(campaign="N"):
    masks_all = P8M.manifest()["masks"]
    rows = []
    for name, spec in PREREG_N.items():
        t = spec["target"]
        base = P7._graddot("cached")[t]
        raw_all = F.campaign_outcomes(campaign, "plain", targets=(t,))[t]
        ms = conditional_masks(t, masks_all)
        raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in ms if m["mask_id"] in raw_all}
        depth, seeds = achieved_depth(raw, len(ms))
        if depth == 0:
            raise SystemExit(
                f"campaign {campaign}: no depth is complete across all {len(ms)} conditional "
                f"masks for {t}; the preregistered stopping rule has nothing to analyse yet")
        d_even = analysis_depth(depth)
        if d_even == 0:
            raise SystemExit(
                f"campaign {campaign}: achieved depth {depth} has no even prefix; the split-half "
                "ceiling needs two disjoint equal halves (see analysis_depth)")
        seeds_full, seeds = list(seeds), list(seeds)[:d_even]
        raw = {m: {s: v[s] for s in seeds} for m, v in raw.items() if all(s in v for s in seeds)}
        obs = F.seed_mean(raw)
        use = [m for m in ms if m["mask_id"] in obs]
        o = np.array([obs[m["mask_id"]] for m in use], float)
        pg = P7.mask_pred(base, use)
        st = np.array([m["stratum"] for m in use])
        ok = np.isfinite(o) & np.isfinite(pg)
        o, pg, st, use = o[ok], pg[ok], st[ok], [m for m, k in zip(use, ok) if k]
        for sname, fn in STATS.items():
            c = ceiling(raw, fn)
            rho = fn(pg, o)
            p_abs = _p_onesided(sname, pg, o)
            rows.append({
                "name": name, "target": t, "statistic": sname,
                "primary": sname == PRIMARY_STAT, "grain": "cluster",
                "n_masks": len(o), "depth": d_even, "depth_achieved": depth,
                "seeds": ",".join(map(str, seeds)),
                "depth_note": (f"analysed at even depth {d_even} of {len(seeds_full)} achieved; "
                               "the split-half ceiling needs disjoint equal halves"),
                "lds": rho, "ceiling": c,
                "ratio": rho / c if np.isfinite(c) and c else np.nan,
                "p_abs": p_abs, "alpha_abs": ALPHA_ABS,
                "PASS_abs": bool(np.isfinite(rho) and np.isfinite(c) and c
                                 and rho / c >= 0.5 and p_abs < ALPHA_ABS),
                "ceiling_note": "Spearman-Brown is derived for correlations, so the Kendall "
                                "ceiling is an approximation" if sname == PRIMARY_STAT else "",
            })
    return pd.DataFrame(rows)


def secondary(campaign="N"):
    """Descriptive only -- no alpha, no bar. Does the Stage F sign reversal replicate?"""
    masks_all = P8M.manifest()["masks"]
    rows = []
    for cname, cfg in P7.CONFIGS.items():
        t = cfg["target"]
        est = cfg["scores"]()
        base = P7._graddot(cfg["baseline"])[t]
        raw_all = F.campaign_outcomes(campaign, "plain", targets=(t,))[t]
        ms = conditional_masks(t, masks_all)
        raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in ms if m["mask_id"] in raw_all}
        depth, seeds = achieved_depth(raw, len(ms))
        if analysis_depth(depth) == 0:
            continue
        seeds = list(seeds)[:analysis_depth(depth)]
        raw = {m: {s: v[s] for s in seeds} for m, v in raw.items() if all(s in v for s in seeds)}
        obs = F.seed_mean(raw)
        use = [m for m in ms if m["mask_id"] in obs]
        o = np.array([obs[m["mask_id"]] for m in use], float)
        px, pg = P7.mask_pred(est, use), P7.mask_pred(base, use)
        st = np.array([m["stratum"] for m in use])
        ok = np.isfinite(o) & np.isfinite(px) & np.isfinite(pg)
        o, px, pg, st = o[ok], px[ok], pg[ok], st[ok]
        for sname, fn in STATS.items():
            d0, pp, lo, hi = stratified_bootstrap(px, pg, o, st, fn)
            rows.append({"config": cname, "label": cfg["label"], "target": t,
                         "statistic": sname, "n_masks": len(o),
                         "depth": analysis_depth(depth), "depth_achieved": depth,
                         "lds": fn(px, o), "graddot_lds": fn(pg, o),
                         "paired_delta": d0, "paired_p": pp, "ci_lo": lo, "ci_hi": hi})
            for s in np.unique(st):
                j = st == s
                if j.sum() >= 8 and np.std(px[j]) > 0 and np.std(pg[j]) > 0:
                    rows.append({"config": cname, "label": cfg["label"], "target": t,
                                 "statistic": sname, "n_masks": int(j.sum()), "depth": depth,
                                 "stratum": s, "lds": fn(px[j], o[j]),
                                 "graddot_lds": fn(pg[j], o[j]),
                                 "paired_delta": fn(px[j], o[j]) - fn(pg[j], o[j])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="N")
    ap.add_argument("--out", default=os.path.join(RESULTS, "confirm_nseries.csv"))
    ap.add_argument("--i_understand_this_scores_once", action="store_true",
                    help="required acknowledgement; the result file is never overwritten")
    a = ap.parse_args()

    if os.path.exists(a.out):
        raise SystemExit(
            f"REFUSING to rescore: {a.out} already exists.\n"
            "PREREG_N fixes score-once with no optional stopping. If you believe the existing "
            "result is wrong, say so in the write-up and leave it on disk -- do not overwrite it.")
    if not a.i_understand_this_scores_once:
        raise SystemExit("pass --i_understand_this_scores_once (PREREG_N: computed once)")

    os.makedirs(RESULTS, exist_ok=True)
    df = evaluate(a.campaign)
    df.to_csv(a.out, index=False)
    sec = secondary(a.campaign)
    sec.to_csv(os.path.join(RESULTS, "confirm_nseries_secondary.csv"), index=False)

    pd.set_option("display.width", 220)
    print("=" * 110)
    print(f"CAMPAIGN N -- pass-8 cluster-grain confirmation. Family of 1, "
          f"alpha_abs = {ALPHA_ABS:.4f}")
    print(f"PRIMARY statistic {PRIMARY_STAT}; Spearman mandatory secondary.")
    print("=" * 110)
    print(df[["name", "statistic", "primary", "n_masks", "depth", "lds", "ceiling", "ratio",
              "p_abs", "PASS_abs"]].round(4).to_string(index=False))
    print("\nSECONDARY (descriptive, no bar) -- do the pass-4/7 corrections still reverse?")
    if len(sec):
        pooled = sec[sec.get("stratum").isna()] if "stratum" in sec else sec
        print(pooled[["config", "target", "statistic", "n_masks", "lds", "graddot_lds",
                      "paired_delta", "ci_lo", "ci_hi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
