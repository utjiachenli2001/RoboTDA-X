"""PASS 11 -- audit every committed ratio comparison for the BLOCKERS #46(b) ceiling effect. Zero GPU.

THE PROBLEM. This project's headline quantity is `ratio = LDS / ceiling`, and nine passes of
conclusions rest on comparing ratios between subsets, grains, configs and campaigns. Pass 10's k=15
census showed the denominator is not a stable yardstick at these sample sizes: two halves of ONE
population at ONE depth gave ceilings of 0.5211 and 0.6678 -- a 28% spread -- while the LDS stayed
flat at 0.258/0.265/0.280. The entire ratio gap between those halves came from the denominator.

So a ratio comparison can say "estimator A beats estimator B" when what actually happened is
"subset A was quieter than subset B". This module asks, for every committed comparison the repo
makes: which is it?

THE DECOMPOSITION. For two rows with ratios r = L/C,

    log(r_A / r_B) = log(L_A / L_B) - log(C_A / C_B)

so the gap splits additively in logs into an ESTIMATOR term and a CEILING term. Reporting each term's
share of the total absolute movement gives a direct answer: a comparison whose ceiling share exceeds
0.5 is being driven by the denominator rather than by the thing it claims to measure.

Logs are used rather than raw differences because the ratio is multiplicative in its two parts; a raw
difference would attribute movement to whichever term happened to be larger in absolute units.

WHAT THIS AUDIT CAN AND CANNOT CONCLUDE. It flags comparisons whose arithmetic is denominator-driven.
It does NOT automatically overturn them: a comparison can be denominator-driven and still land on the
right answer, and paired contrasts (which share masks and therefore share a ceiling) are structurally
immune. The output is a triage list, and each flagged row needs reading in context before any
committed conclusion is amended.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
STAT = "kendall_tau_b"
EPS = 1e-12

# A high ceiling SHARE is meaningless when nothing moved. TracIn vs TracIn_trunc_k1 sits at ratios
# 0.3897 vs 0.3949 with a 1.3% ceiling spread: the estimator term is ~0, so the share is 1.0 while
# the comparison is a near-tie nobody draws a conclusion from. A flag therefore needs BOTH a
# denominator-dominated split AND enough absolute movement for the split to matter. 0.05 in log units
# is ~5% of ratio, below which no conclusion in this repo turns.
MATERIAL_LOG_MOVE = 0.05


def decompose(lds_a, c_a, lds_b, c_b):
    """-> (ratio_gap, estimator_term, ceiling_term, ceiling_share) in log space."""
    if min(lds_a, lds_b) <= 0 or min(c_a, c_b) <= 0:
        return (np.nan,) * 4          # log decomposition is undefined through zero/negative LDS
    est = np.log(lds_a / lds_b)
    cei = -np.log(c_a / c_b)
    tot = abs(est) + abs(cei)
    return (est + cei, est, cei, abs(cei) / tot if tot > EPS else np.nan)


def _csv(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _pairs(df, label_col, source, kind, lds_col="lds", ceil_col="ceiling"):
    """All within-file pairwise comparisons on the primary statistic."""
    if df is None or lds_col not in df or ceil_col not in df:
        return []
    d = df[df.get("statistic", STAT) == STAT] if "statistic" in df else df
    d = d.dropna(subset=[lds_col, ceil_col])
    rows = []
    recs = d.to_dict("records")
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            gap, est, cei, share = decompose(a[lds_col], a[ceil_col], b[lds_col], b[ceil_col])
            if not np.isfinite(share):
                continue
            rows.append({
                "source": source, "kind": kind,
                "A": str(a.get(label_col, "?")), "B": str(b.get(label_col, "?")),
                "lds_A": a[lds_col], "lds_B": b[lds_col],
                "ceil_A": a[ceil_col], "ceil_B": b[ceil_col],
                "ratio_A": a[lds_col] / a[ceil_col], "ratio_B": b[lds_col] / b[ceil_col],
                "log_gap": gap, "estimator_term": est, "ceiling_term": cei,
                "ceiling_share": share,
                "abs_ceiling_term": abs(cei),
                "material": bool(abs(cei) >= MATERIAL_LOG_MOVE),
                "DENOMINATOR_DRIVEN": bool(share > 0.5 and abs(cei) >= MATERIAL_LOG_MOVE),
                "ceiling_spread_pct": 100 * abs(a[ceil_col] - b[ceil_col])
                / max(a[ceil_col], b[ceil_col]),
            })
    return rows


LABEL_CANDIDATES = ("subset", "scope", "rung", "attempt", "config", "name", "estimator",
                    "label", "arm", "grain", "variant", "method", "target", "draw", "stratum")
MAX_PAIRS_PER_FILE = 600


def _label_col(df):
    """The column that names what each row IS. Prefer known names, else first object column."""
    for c in LABEL_CANDIDATES:
        if c in df.columns and df[c].nunique() > 1:
            return c
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() > 1:
            return c
    return None


def eligible_files():
    """Every committed result CSV carrying BOTH an lds and a ceiling column."""
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.csv"))):
        try:
            cols = set(pd.read_csv(p, nrows=1).columns)
        except Exception:
            continue
        if "lds" in cols and "ceiling" in cols:
            out.append(p)
    return out


def audit():
    """Auto-discovered across the WHOLE back catalogue, not a hand-picked subset.

    An earlier version of this module audited five files chosen by hand -- all from passes 9 and 10 --
    and would have supported a completeness claim over 13% of the eligible surface. The back
    catalogue is precisely what needed auditing, so discovery is now automatic and coverage is
    reported.
    """
    rows, coverage = [], []
    for path in eligible_files():
        src = os.path.basename(path)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        lab = _label_col(df)
        if lab is None:
            coverage.append({"source": src, "rows": len(df), "pairs": 0,
                             "skipped": "no label column to name the rows"})
            continue
        got = _pairs(df, lab, src, "auto")
        truncated = len(got) > MAX_PAIRS_PER_FILE
        if truncated:
            got = sorted(got, key=lambda r: -r["ceiling_share"])[:MAX_PAIRS_PER_FILE]
        rows += got
        coverage.append({"source": src, "rows": len(df), "pairs": len(got),
                         "skipped": f"TRUNCATED to worst {MAX_PAIRS_PER_FILE}" if truncated else ""})
    df = pd.DataFrame(rows)
    cov = pd.DataFrame(coverage)
    if df.empty:
        return df, cov
    return df.sort_values("ceiling_share", ascending=False).reset_index(drop=True), cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p11_ceiling_audit.csv"))
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    df, cov = audit()
    if df.empty:
        raise SystemExit("no comparable rows found")
    cov.to_csv(os.path.join(RESULTS, "p11_ceiling_audit_coverage.csv"), index=False)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)

    df = df.sort_values(["DENOMINATOR_DRIVEN", "abs_ceiling_term"], ascending=False)
    n_flag = int(df.DENOMINATOR_DRIVEN.sum())
    elig = len(eligible_files())
    print(f"[p11/ceiling-audit] {len(df)} pairwise comparisons across {df.source.nunique()} of "
          f"{elig} audit-eligible result files "
          f"({100 * df.source.nunique() / elig:.0f}% coverage)")
    skipped = cov[cov.pairs == 0]
    if len(skipped):
        print(f"  {len(skipped)} file(s) yielded no comparisons: "
              f"{', '.join(skipped.source.tolist()[:6])}")
    print(f"  DENOMINATOR-DRIVEN (ceiling share > 0.5): {n_flag} "
          f"({100 * n_flag / len(df):.0f}%)")
    print(f"  median ceiling share: {df.ceiling_share.median():.3f}")
    print(f"  (a flag needs ceiling share > 0.5 AND >= {MATERIAL_LOG_MOVE} log-units of ceiling "
          f"movement; {int((~df.material).sum())} comparisons are immaterial near-ties)")
    print(f"  median ceiling spread between compared subsets: "
          f"{df.ceiling_spread_pct.median():.1f}%")
    print(f"\n  worst {a.top} by ceiling share:")
    cols = ["source", "A", "B", "ratio_A", "ratio_B", "ceiling_share", "abs_ceiling_term",
            "ceiling_spread_pct", "DENOMINATOR_DRIVEN"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 34):
        print(df.head(a.top)[cols].to_string(index=False))
    print(f"\n[p11/ceiling-audit] -> {a.out}")


if __name__ == "__main__":
    main()
