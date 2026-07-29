"""Campaign O -- the pass-9 SUB-CLUSTER-GRAIN confirmation. Preregistered, computed once.

PREREG_O (if_repair/p9_prereg.md, frozen at 22b286a while campaign O had zero runs).

  O1  GradDot_dmean, target C5, k=3, training set fixed at 75 demos, conditioning "all C5 groups
      retained": the LOWER BOUND of the 95% bootstrap CI of the ratio is >= 0.5.
  O2  the same at k=5.
  FAMILY OF TWO -> alpha = 0.025 one-sided each.

WHY THE CI LOWER BOUND AND NOT A POINT ESTIMATE. Campaign N's PASS rule combined a p-value against
rho > 0 -- which at n ~ 150 is astronomically significant for any modest positive tau, and so never
the binding constraint -- with a point-estimate check that ratio >= 0.5. That puts alpha on the easy
half of the decision and none on the half carrying the claim. Here the bar itself is the inferential
object: masks are resampled and THE CEILING IS RECOMPUTED ON EACH RESAMPLE, so the interval carries
the uncertainty of the numerator and the denominator together.

WHY THE TRAINING SET IS FIXED AT 75 DEMOS. `p9_stratum_control` showed campaign N's pooled primary
is substantially a training-set-SIZE effect: GradDot is a fixed estimator, yet on outcomes shuffled
within stratum it still scores Kendall 0.353 pooled against a real 0.475, because |S| moves the
outcome directly and every prediction is a sum over kept demos. Campaign O removes that channel by
construction rather than adjusting for it afterwards -- every mask at every grain keeps exactly 75
demos, so there is no size variation left to confound.

THE k=15 RUNG IS FREE AND MUST BE DEPTH-MATCHED. It is campaign N's 5of9 conditional stratum, which
is already at 75 demos, re-derived here at DEPTH 2 in the same seed slots {4401, 4402} as campaign
O. The committed 0.7069 is at depth 4 and is used only as a regression check, never as a curve
point: `ratio` is not depth-invariant, and since the ceiling is a reliability r while the attainable
maximum is ~sqrt(r), lower depth inflates the ratio. Comparing a depth-2 rung against a depth-4 one
would confound grain with protocol.

THE SAME TWO MECHANICAL ENFORCEMENTS AS CAMPAIGN N:
1. SCORE ONCE. `--out` refuses to overwrite an existing result file.
2. THE PREREGISTERED STOPPING RULE. The analysis runs at the largest EVEN depth for which ALL masks
   of a grain have completed seeds, read off the RUN DIRECTORY and never off the outcomes.
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
from if_repair import p9_masks as P9M  # noqa: E402
from if_repair.confirm_mseries import ceiling, STATS  # noqa: E402
from if_repair.confirm_nseries import (achieved_depth, analysis_depth,  # noqa: E402
                                       conditional_masks)
from if_repair.p9_stratum_control import boot_ratio, perm_pooled  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TARGET = "C5"
PRIMARY_STAT = "kendall_tau_b"
BAR = 0.5
PREREG_O = {"O1": {"k": 3}, "O2": {"k": 5}}
ALPHA = 0.05 / len(PREREG_O)


def _score_rung(label, masks, raw, y, pg, st, d_even, source):
    rows = []
    for sname, fn in STATS.items():
        c = ceiling(raw, fn)
        lds = fn(pg, y)
        br = boot_ratio(pg, y, raw, [m["mask_id"] for m in masks], fn)
        perm = perm_pooled(pg, y, st, fn)
        lo = float(np.percentile(br, 2.5)) if len(br) else np.nan
        hi = float(np.percentile(br, 97.5)) if len(br) else np.nan
        rows.append({
            "rung": label, "source": source, "statistic": sname,
            "primary": sname == PRIMARY_STAT, "n_masks": len(y), "depth": d_even,
            "retained_demos": P9M.RETAINED_DEMOS,
            "lds": lds, "ceiling": c,
            "ratio": lds / c if np.isfinite(c) and c else np.nan,
            "ratio_ci_lo": lo, "ratio_ci_hi": hi,
            "ratio_sqrt": lds / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan,
            "perm_null_mean": float(perm.mean()),
            "perm_null_p97.5": float(np.percentile(perm, 97.5)),
            "beats_perm_null": bool(lds > np.percentile(perm, 97.5)),
            "CLEARS_BAR": bool(np.isfinite(lo) and lo >= BAR),
            "alpha": ALPHA,
        })
    return rows


def _outcomes(campaign, masks, target=TARGET, depth_cap=None):
    raw_all = F.campaign_outcomes(campaign, "plain", targets=(target,))[target]
    raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in masks if m["mask_id"] in raw_all}
    if not raw:
        return None
    depth, seeds = achieved_depth(raw, len(masks))
    d = analysis_depth(depth)
    if depth_cap:
        d = min(d, depth_cap)
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
    use = [m for m, k in zip(use, ok) if k]
    raw = {m["mask_id"]: raw[m["mask_id"]] for m in use}
    return use, raw, y[ok], pg[ok], st[ok], d


def evaluate():
    man = P9M.manifest()
    rows = []

    # ---- the two BOUGHT rungs (campaign O)
    for name, spec in PREREG_O.items():
        k = spec["k"]
        masks = man["masks"][str(k)]
        got = _outcomes("O", masks)
        if got is None:
            print(f"[confirm_O] {name} (k={k}): no complete even depth yet -- skipped")
            continue
        use, raw, y, pg, st, d = got
        rows += [dict(r, prereg=name) for r in
                 _score_rung(f"k={k}", use, raw, y, pg, st, d, "campaign O (fresh)")]

    # ---- the INHERITED rung, depth-matched (campaign N 5of9, already 75 demos)
    n5 = [m for m in conditional_masks(TARGET, P8M.manifest()["masks"])
          if m["stratum"] == "5of9"]
    got = _outcomes("N", n5, depth_cap=P9M.DEPTH)
    if got is not None:
        use, raw, y, pg, st, d = got
        rows += [dict(r, prereg="reference") for r in
                 _score_rung("k=15", use, raw, y, pg, st, d,
                             "campaign N 5of9, depth-matched")]
    return pd.DataFrame(rows)


def missing_grains(df):
    """Preregistered grains with no scored rung in `df`, as sorted strings.

    Extracted from main() so the guard is testable: the failure mode it prevents (a partial score
    consuming the one-shot write) cannot be exercised through main() without actually writing.
    """
    scored = {r.split("=")[1] for r in df["rung"].unique() if str(r).startswith("k=")}
    wanted = {str(spec["k"]) for spec in PREREG_O.values()}
    return sorted(wanted - scored, key=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "confirm_oseries.csv"))
    ap.add_argument("--i_understand_this_scores_once", action="store_true",
                    help="required acknowledgement; the result file is never overwritten")
    ap.add_argument("--allow_partial_score", action="store_true",
                    help="deliberate protocol deviation: score before every preregistered grain "
                         "has a complete even depth. Recorded in the output.")
    a = ap.parse_args()

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists. Campaign O is scored ONCE (PREREG_O). Refusing.")
    if not a.i_understand_this_scores_once:
        raise SystemExit("pass --i_understand_this_scores_once")

    df = evaluate()
    if df.empty:
        raise SystemExit("nothing scored -- the stopping rule has no complete even depth yet")

    # ---- do not let a partial campaign consume the one-shot score.
    missing = missing_grains(df)
    if missing and not a.allow_partial_score:
        raise SystemExit(
            f"REFUSING to score: grain(s) k={','.join(missing)} have no complete even depth yet, "
            f"but the result file is written ONCE. Scoring now would answer "
            f"{sorted(scored)} and leave k={','.join(missing)} permanently unanswerable "
            f"(PREREG_O is a family of two). Wait for the campaign, or pass "
            f"--allow_partial_score to record a deliberate protocol deviation.")
    df["partial_score"] = bool(missing)
    if missing:
        df["partial_note"] = (f"DEVIATION: scored while k={','.join(missing)} was incomplete; "
                              f"those grains are not answerable from this file")
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df[df.primary][["rung", "source", "n_masks", "depth", "lds", "ceiling", "ratio",
                              "ratio_ci_lo", "ratio_ci_hi", "perm_null_mean", "beats_perm_null",
                              "CLEARS_BAR"]].to_string(index=False))
    print(f"\n[confirm_O] -> {a.out}")


if __name__ == "__main__":
    main()
