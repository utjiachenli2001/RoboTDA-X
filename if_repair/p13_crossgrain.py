"""PASS 13 -- is the datamodel's k=5 transfer gap about GRAIN or about COEFFICIENT COUNT? Zero GPU.

BLOCKERS #50 found the datamodel transfers across an independent re-partition -- it attributes -- but
with an asymmetry nobody explained: at k=5 the transfer loses 32% of the within-campaign LDS
(0.4699 -> 0.3210, z = 3.7) while at k=3 the loss is undetectable (0.3287 -> 0.3057, z = 0.55). The
finer grain transfers BETTER, which is the opposite of the naive expectation.

Two hypotheses, and they are separable on data already on disk:

  (a) GRAIN. Something about 5-demo groups makes the fitted coefficients more tied to the particular
      grouping than 3-demo groups are.
  (b) COEFFICIENT COUNT. k=5 has 27 coefficients against 400 masks (14.8 per coefficient) versus k=3's
      45 against 400 (8.9). The more over-determined fit has more capacity to absorb ensemble-specific
      structure, so it loses more when that structure is taken away.

THE TEST. Transfer ACROSS GRAINS within one campaign: fit at k=3, map the 45 coefficients to per-demo
scores, and score campaign O's k=5 mask set -- then the reverse. The masks differ between grains (a
k=3 mask's demo set is not a union of k=5 groups), so this is out-of-sample in mask space, and it
crosses the grouping without crossing the partition.

  * If the loss follows the FITTING grain -- fits at k=5 degrade wherever they are scored -- the cause
    travels with the fit, which is (b), coefficient count.
  * If the loss follows the SCORING grain -- everything degrades when scored on k=5 masks -- the cause
    is in the target, which is (a), grain.

Reported descriptively. No alpha: #50 is already committed and this decomposes it rather than
re-testing it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import p9_grain as G  # noqa: E402
from if_repair import p9_masks as P9M  # noqa: E402
from if_repair.confirm_mseries import ceiling, STATS  # noqa: E402
from if_repair.p9_datamodel_cluster import loo_predict  # noqa: E402
from if_repair.p11_transfer import _load, fit_on, coefficients_to_demo_scores  # noqa: E402
from if_repair.p10_datamodel_subcluster import group_design  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PRIMARY_STAT = "kendall_tau_b"
N_BOOT = 1500


def evaluate(model="ridge"):
    man = P9M.manifest()
    data, fits = {}, {}
    for k in (3, 5):
        use, raw, y, d = _load("O", man["masks"][str(k)])
        data[k] = (use, raw, y, d)
        coef, gids, alpha = fit_on(use, y, k, G.GROUP_SEED, model=model)
        fits[k] = coefficients_to_demo_scores(coef, gids, k, G.GROUP_SEED)

    rng = np.random.default_rng(0)
    rows = []
    for score_k in (3, 5):
        use, raw, y, d = data[score_k]
        gids = [g["group_id"] for g in G.groups(score_k)]
        within, _ = loo_predict(group_design(use, gids), y, model=model)
        # The same-grain full-fit arm is IN-SAMPLE -- that model was fit on these exact masks, so
        # its number is a fit statistic and NOT a transfer number. Labelled, not dropped, because
        # seeing it beside the LOO arm shows how much the in-sample read inflates.
        arms = {
            f"fit k={score_k} (within, LOO -- the honest same-grain read)": within,
            f"fit k={score_k} (IN-SAMPLE, not a transfer number)":
                P7.mask_pred(fits[score_k], use),
            f"fit k={3 if score_k == 5 else 5} -> score k={score_k} (CROSS-GRAIN)":
                P7.mask_pred(fits[3 if score_k == 5 else 5], use),
            "GradDot (reference)": P7.mask_pred(P7._graddot("cached")["C5"], use),
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
                    "scored_on": f"k={score_k}", "arm": arm, "statistic": sname,
                    "primary": sname == PRIMARY_STAT, "n_masks": int(g.sum()), "depth": d,
                    "lds": lds, "lds_se": float(bs.std(ddof=1)), "ceiling": c,
                    "ratio": lds / c if np.isfinite(c) and c else np.nan,
                    "cross_grain": "CROSS-GRAIN" in arm,
                    "in_sample": "IN-SAMPLE" in arm,
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p13_crossgrain.csv"))
    a = ap.parse_args()
    df = evaluate()
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30,
                           "display.max_colwidth", 34):
        print(df[df.primary][["scored_on", "arm", "n_masks", "lds", "lds_se", "ceiling",
                              "ratio"]].to_string(index=False))
    print(f"\n[p13/crossgrain] -> {a.out}")


if __name__ == "__main__":
    main()
