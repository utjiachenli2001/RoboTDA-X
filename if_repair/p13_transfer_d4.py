"""PASS 13 -- the cross-partition TRANSFER arm re-read at an UNBIASED ceiling. Descriptive.

#50 is the project's headline positive: the datamodel fit on campaign O and scored on campaign R's
independent partition reaches ratio 0.781 (k=3) and 0.754 (k=5), clearing the half-ceiling bar out of
partition at 4.0x and 2.4x GradDot. It was measured at depth 2.

#42 says the ratio is inflated at low depth, because the ceiling is a reliability r while the
attainable maximum is ~sqrt(r), and the inflation grows as r falls. #51 re-read the WITHIN-campaign
datamodel at depth 4 and it survived -- but it could not touch the transfer arm, because the SCORING
side of that arm is campaign R and campaign R had only depth-2 outcomes. Only the fit side could
improve, so the headline positive stayed on the inflated scale while everything around it moved to the
honest one.

Campaign S bought the missing half: two more seed slots on campaign R's identical 800 masks. This
module merges R (slots 4401-4402) with S (4403-4404) to read the transfer arm at depth 4, against a
ceiling computed at depth 4 on those same masks.

WHAT MOVES AND WHAT DOES NOT. The masks are unchanged, the fit is unchanged (campaign O at whatever
depth is specified), and the estimators are unchanged. Only the scoring side's depth changes, so any
movement is the denominator correcting -- not a fresh draw, not a different mask set.

Campaign R's preregistered scoring stays frozen: `confirm_rseries.csv` is untouched and this writes a
separate descriptive file carrying no alpha.
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
from if_repair import p9_grain as G  # noqa: E402
from if_repair import p9_masks as P9M  # noqa: E402
from if_repair import p10_masks2 as P10M  # noqa: E402
from if_repair.confirm_mseries import ceiling, STATS  # noqa: E402
from if_repair.p9_datamodel_cluster import loo_predict  # noqa: E402
from if_repair.p10_datamodel_subcluster import group_design  # noqa: E402
from if_repair.p11_transfer import _load, fit_on, coefficients_to_demo_scores  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TARGET = "C5"
PRIMARY_STAT = "kendall_tau_b"
BAR = 0.5
N_BOOT = 1500


def merged_R(target=TARGET):
    """Campaign R's masks with campaign S's extra seed slots merged in -- depth 4 on the same masks."""
    out = {}
    for c in ("R", "S"):
        try:
            got = F.campaign_outcomes(c, "plain", targets=(target,))[target]
        except Exception:
            continue
        for m, v in got.items():
            out.setdefault(m, {}).update(v)
    return out


def read_at(masks, raw_all, depth):
    raw = {m["mask_id"]: raw_all.get(m["mask_id"], {}) for m in masks}
    raw = {m: v for m, v in raw.items() if len(v) >= depth}
    raw = {m: dict(sorted(v.items())[:depth]) for m, v in raw.items()}
    if len(raw) < 10:
        return None
    obs = F.seed_mean(raw)
    use = [m for m in masks if m["mask_id"] in obs]
    y = np.array([obs[m["mask_id"]] for m in use], float)
    ok = np.isfinite(y)
    use = [m for m, q in zip(use, ok) if q]
    return use, {m["mask_id"]: raw[m["mask_id"]] for m in use}, y[ok]


def evaluate(k, model="ridge"):
    o_use, _, o_y, _ = _load("O", P9M.manifest()["masks"][str(k)])
    coef, gids, _ = fit_on(o_use, o_y, k, G.GROUP_SEED, model=model)
    demo_scores = coefficients_to_demo_scores(coef, gids, k, G.GROUP_SEED)

    r_masks = P10M.manifest()["masks"][str(k)]
    raw_all = merged_R()
    r_gids = [g["group_id"] for g in G.groups(k, seed=P10M.GROUP_SEED2)]
    rng = np.random.default_rng(0)
    rows = []

    for depth in (2, 4):
        got = read_at(r_masks, raw_all, depth)
        if got is None:
            print(f"[p13/transfer-d4] k={k} depth {depth}: not available yet -- skipped")
            continue
        use, raw, y = got
        arms = {
            "datamodel fit on O -> scored on R (TRANSFER)": P7.mask_pred(demo_scores, use),
            "datamodel fit on R -> scored on R (within, LOO)":
                loo_predict(group_design(use, r_gids), y, model=model)[0],
            "GradDot (campaign-independent)": P7.mask_pred(P7._graddot("cached")[TARGET], use),
        }
        for sname, fn in STATS.items():
            c = ceiling(raw, fn)
            for arm, p in arms.items():
                g = np.isfinite(p) & np.isfinite(y)
                lds = fn(p[g], y[g])
                bs = np.empty(N_BOOT)
                for i in range(N_BOOT):
                    j = rng.integers(0, g.sum(), g.sum())
                    bs[i] = fn(p[g][j], y[g][j])
                rows.append({
                    "grain": f"k={k}", "arm": arm, "statistic": sname,
                    "primary": sname == PRIMARY_STAT, "scoring_depth": depth,
                    "n_masks": int(g.sum()),
                    "lds": lds, "lds_se": float(bs.std(ddof=1)), "ceiling": c,
                    "ratio": lds / c if np.isfinite(c) and c else np.nan,
                    "ratio_sqrt": lds / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan,
                    "CLEARS_BAR": bool(np.isfinite(c) and c and lds / c >= BAR),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p13_transfer_depth4.csv"))
    a = ap.parse_args()
    df = pd.concat([evaluate(k) for k in (3, 5)], ignore_index=True)
    if df.empty:
        raise SystemExit("no outcomes at a usable depth yet")
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["grain", "arm", "scoring_depth", "n_masks", "lds", "lds_se", "ceiling", "ratio",
            "ratio_sqrt", "CLEARS_BAR"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 46):
        print(df[df.primary][cols].to_string(index=False))
    print(f"\n[p13/transfer-d4] -> {a.out}")


if __name__ == "__main__":
    main()
