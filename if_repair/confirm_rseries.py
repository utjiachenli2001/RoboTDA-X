"""Campaign R -- pass 10's SECOND-PARTITION confirmation. Preregistered, computed once.

PREREG_R (if_repair/p10_prereg.md, frozen at 257665a while campaign R had zero runs).

  R1  At k=3, campaign R's ratio point estimate falls INSIDE campaign O's committed 95% bootstrap CI.
  R2  The same at k=5.
  FAMILY OF TWO -> alpha = 0.025 one-sided each.

THE HYPOTHESIS IS AGREEMENT, AND A FAILURE IS THE INTERESTING OUTCOME. Pass 9's k=3 and k=5 rungs
rest on ONE committed partition of the corpus into groups, so they carry partition-sampling variance
that the k=15 rung structurally cannot -- a cluster has no composition freedom. `p9_prereg.md` named
this check and the curve came out close (0.356 vs 0.365). If a rung lands outside campaign O's
interval, pass 9's numbers are partly a one-partition artifact and its curve needs amending. Stating
the criterion in advance is what stops a disagreement being reframed as noise after the fact.

THE COMPARISON TARGETS ARE READ FROM THE COMMITTED CSV, NOT TYPED IN. Transcribing
[0.180, 0.550] and [0.204, 0.559] by hand into this file would put the pass's decision rule one typo
away from being wrong, so the bounds come from `results/confirm_oseries.csv` at run time and the file's
absence is a hard error rather than a fallback.

WHAT THIS CAMPAIGN CANNOT DO. It cannot resolve the grain trend. The k=15 conditional population is
capped at C(8,4) = 70 and pass 10's census exhausted it; no purchasable design tightens that rung
below a CI width of ~0.6, which still overlaps both sub-cluster rungs. Campaign R tests whether pass
9's two sub-cluster rungs are partition-robust, and nothing more.

THE SAME TWO MECHANICAL ENFORCEMENTS AS CAMPAIGNS N AND O:
1. SCORE ONCE. `--out` refuses to overwrite.
2. NO PARTIAL SCORE. The seed-major job list means k=3 completes before k=5, so there is a window in
   which only k=3 is analysable. Writing then would satisfy the never-overwrite guard forever and
   leave R2 permanently unanswerable. `missing_grains` refuses unless BOTH grains are ready.
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
from if_repair import p10_masks2 as P10M  # noqa: E402
from if_repair.confirm_mseries import ceiling, STATS  # noqa: E402
from if_repair.confirm_nseries import achieved_depth, analysis_depth  # noqa: E402
from if_repair.p9_stratum_control import boot_ratio, perm_pooled  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TARGET = P10M.TARGET
PRIMARY_STAT = "kendall_tau_b"
BAR = 0.5
PREREG_R = {"R1": {"k": 3}, "R2": {"k": 5}}
ALPHA = 0.05 / len(PREREG_R)


def campaign_o_targets(path=None):
    """{(k, statistic): (ci_lo, ci_hi, ratio)} from the committed campaign-O result."""
    path = path or os.path.join(RESULTS, "confirm_oseries.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- PREREG_R's decision rule is defined against campaign "
                         "O's committed intervals and cannot be evaluated without them")
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        rung = str(r["rung"])
        if not rung.startswith("k="):
            continue
        out[(int(rung.split("=")[1]), r["statistic"])] = (
            float(r["ratio_ci_lo"]), float(r["ratio_ci_hi"]), float(r["ratio"]))
    return out


def _outcomes(masks, target=TARGET):
    raw_all = F.campaign_outcomes("R", "plain", targets=(target,))[target]
    raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in masks if m["mask_id"] in raw_all}
    if not raw:
        return None
    depth, seeds = achieved_depth(raw, len(masks))
    d = analysis_depth(depth)
    if d == 0:
        return None
    seeds = list(seeds)[:d]
    raw = {m: {s: v[s] for s in seeds} for m, v in raw.items() if all(s in v for s in seeds)}
    obs = F.seed_mean(raw)
    use = [m for m in masks if m["mask_id"] in obs]
    y = np.array([obs[m["mask_id"]] for m in use], float)
    pg = P7.mask_pred(P7._graddot("cached")[target], use)
    st = np.array([m.get("stratum", "one") for m in use])
    ok = np.isfinite(y) & np.isfinite(pg)
    use = [m for m, q in zip(use, ok) if q]
    raw = {m["mask_id"]: raw[m["mask_id"]] for m in use}
    return use, raw, y[ok], pg[ok], st[ok], d


def missing_grains(df):
    """Preregistered grains with no scored rung. Same guard as campaign O."""
    scored = {r.split("=")[1] for r in df["rung"].unique() if str(r).startswith("k=")}
    wanted = {str(spec["k"]) for spec in PREREG_R.values()}
    return sorted(wanted - scored, key=int)


def evaluate():
    man = P10M.manifest()
    tgt = campaign_o_targets()
    rows = []
    for name, spec in PREREG_R.items():
        k = spec["k"]
        got = _outcomes(man["masks"][str(k)])
        if got is None:
            print(f"[confirm_R] {name} (k={k}): no complete even depth yet -- skipped")
            continue
        use, raw, y, pg, st, d = got
        ids = [m["mask_id"] for m in use]
        for sname, fn in STATS.items():
            c = ceiling(raw, fn)
            lds = fn(pg, y)
            ratio = lds / c if np.isfinite(c) and c else np.nan
            br = boot_ratio(pg, y, raw, ids, fn)
            perm = perm_pooled(pg, y, st, fn)
            o_lo, o_hi, o_ratio = tgt.get((k, sname), (np.nan, np.nan, np.nan))
            inside = bool(np.isfinite(ratio) and o_lo <= ratio <= o_hi)
            rows.append({
                "prereg": name, "rung": f"k={k}", "statistic": sname,
                "primary": sname == PRIMARY_STAT, "n_masks": len(y), "depth": d,
                "partition": "second (group_seed 20260730)",
                "lds": lds, "ceiling": c, "ratio": ratio,
                "ratio_ci_lo": float(np.percentile(br, 2.5)) if len(br) else np.nan,
                "ratio_ci_hi": float(np.percentile(br, 97.5)) if len(br) else np.nan,
                "ratio_sqrt": lds / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan,
                "campaignO_ratio": o_ratio, "campaignO_ci_lo": o_lo, "campaignO_ci_hi": o_hi,
                "ratio_diff_vs_O": ratio - o_ratio,
                "AGREES_inside_O_CI": inside,
                "alpha": ALPHA,
                "perm_null_mean": float(perm.mean()),
                "perm_null_p97.5": float(np.percentile(perm, 97.5)),
                "beats_perm_null": bool(lds > np.percentile(perm, 97.5)),
                "CLEARS_BAR_descriptive": bool(np.isfinite(ratio) and ratio >= BAR),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "confirm_rseries.csv"))
    ap.add_argument("--i_understand_this_scores_once", action="store_true")
    ap.add_argument("--allow_partial_score", action="store_true",
                    help="deliberate protocol deviation; recorded in the output")
    a = ap.parse_args()

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists. Campaign R is scored ONCE (PREREG_R). Refusing.")
    if not a.i_understand_this_scores_once:
        raise SystemExit("pass --i_understand_this_scores_once")

    df = evaluate()
    if df.empty:
        raise SystemExit("nothing scored -- the stopping rule has no complete even depth yet")
    missing = missing_grains(df)
    if missing and not a.allow_partial_score:
        raise SystemExit(
            f"REFUSING to score: grain(s) k={','.join(missing)} have no complete even depth yet, "
            f"but the result file is written ONCE. Scoring now would leave "
            f"k={','.join(missing)} permanently unanswerable (PREREG_R is a family of two). "
            f"Wait, or pass --allow_partial_score to record a deliberate deviation.")
    df["partial_score"] = bool(missing)

    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["prereg", "rung", "n_masks", "depth", "lds", "ceiling", "ratio", "campaignO_ratio",
            "campaignO_ci_lo", "campaignO_ci_hi", "ratio_diff_vs_O", "AGREES_inside_O_CI",
            "beats_perm_null"]
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(df[df.primary][cols].to_string(index=False))
    print(f"\n[confirm_R] -> {a.out}")


if __name__ == "__main__":
    main()
