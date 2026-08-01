"""PASS 14 -- do two independent partitions AGREE on which demonstrations matter? Zero GPU.

The project's central positive claim (#50, qualified by #53) is that the design-based datamodel
ATTRIBUTES: fit on campaign O it predicts campaign R's outcomes far better than any gradient
estimator. That is a claim about PREDICTION. It is not quite the claim the word "attribution" makes.

A model can transfer well while disagreeing about individual demonstrations. Suppose most of the
predictable variance lives in a coarse structure -- a few clusters that matter and the rest that do
not. Two fits could both capture that, predict each other's mask outcomes well, and still assign
quite different influence to any particular demo. The transfer test cannot tell those apart, because
it only ever looks at summed mask predictions.

**This module tests the literal claim.** Campaigns O and R are independent partitions of the same 135
demonstrations, sharing zero groups. Fit each separately, map each set of group coefficients down to
135 per-demo scores, and correlate the two vectors directly. If the datamodel is attributing to
demonstrations, two arbitrary re-groupings of the same corpus should infer similar per-demo influence.
If it is fitting mask-ensemble structure, they need not.

WHY THE MAPPING IS FAIR. Within a group the demos are perfectly collinear, so a group's coefficient is
spread evenly over its k demos -- the only identifiable split. The two partitions group different
demos together, so the two 135-vectors are NOT constrained to agree by construction: a demo's score
under O comes from one set of groupmates and under R from a disjoint set.

CONTROLS, because a raw correlation here is easy to over-read:
  * **Shuffle null.** Permute one partition's per-demo scores and re-correlate. Must collapse to ~0.
  * **Granularity ceiling.** Two fits at the SAME grain on disjoint halves of ONE campaign -- the
    best agreement achievable without crossing partitions. Cross-partition agreement should be read
    against that, not against 1.0.
  * **GradDot reference.** Its per-demo scores are fixed and campaign-independent, so it agrees with
    itself perfectly. It is not a competitor here; it is the reminder that self-agreement is trivial
    for anything that does not look at outcomes.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p9_grain as G  # noqa: E402
from if_repair import p9_masks as P9M  # noqa: E402
from if_repair import p10_masks2 as P10M  # noqa: E402
from if_repair.confirm_mseries import STATS  # noqa: E402
from if_repair.p11_transfer import _load, fit_on, coefficients_to_demo_scores  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
N_PERM = 500


def demo_vector(scores, ids):
    return np.array([scores.get(d, 0.0) for d in ids], float)


def fit_partition(campaign, masks, k, group_seed, model="ridge"):
    use, _, y, _ = _load(campaign, masks)
    coef, gids, _ = fit_on(use, y, k, group_seed, model=model)
    return coefficients_to_demo_scores(coef, gids, k, group_seed), use, y


def half_fit(campaign, masks, k, group_seed, seed=0, model="ridge"):
    """Two fits at the SAME grain on disjoint halves of ONE campaign -- the agreement ceiling."""
    use, _, y, _ = _load(campaign, masks)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(use))
    h = len(use) // 2
    out = []
    for sel in (idx[:h], idx[h:2 * h]):
        ms = [use[i] for i in sel]
        coef, gids, _ = fit_on(ms, y[sel], k, group_seed, model=model)
        out.append(coefficients_to_demo_scores(coef, gids, k, group_seed))
    return out


def evaluate(model="ridge"):
    ids = list(D.cache_for("bc_s10")["train_ids"])
    rng = np.random.default_rng(0)
    rows = []

    for k in (3, 5):
        o_scores, _, _ = fit_partition("O", P9M.manifest()["masks"][str(k)], k,
                                       G.GROUP_SEED, model=model)
        r_scores, _, _ = fit_partition("R", P10M.manifest()["masks"][str(k)], k,
                                       P10M.GROUP_SEED2, model=model)
        a, b = demo_vector(o_scores, ids), demo_vector(r_scores, ids)

        ha, hb = half_fit("O", P9M.manifest()["masks"][str(k)], k, G.GROUP_SEED, model=model)
        hva, hvb = demo_vector(ha, ids), demo_vector(hb, ids)

        # shuffle null on the cross-partition pair
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            null[i] = np.corrcoef(a, b[rng.permutation(len(b))])[0, 1]

        for label, x, z, note in (
            ("CROSS-PARTITION (O vs R, disjoint groupings)", a, b,
             "the literal attribution test -- two arbitrary re-groupings of the same 135 demos"),
            ("within-campaign halves (same grouping)", hva, hvb,
             "agreement ceiling: best achievable without crossing partitions"),
        ):
            rows.append({
                "grain": f"k={k}", "comparison": label, "n_demos": len(ids),
                "pearson": float(np.corrcoef(x, z)[0, 1]),
                "spearman": float(STATS["spearman"](x, z)),
                "kendall": float(STATS["kendall_tau_b"](x, z)),
                "shuffle_null_mean": float(null.mean()) if "CROSS" in label else np.nan,
                "shuffle_null_p97.5": float(np.percentile(null, 97.5)) if "CROSS" in label
                                      else np.nan,
                "note": note,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p14_agreement.csv"))
    a = ap.parse_args()
    df = evaluate()
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["grain", "comparison", "n_demos", "pearson", "spearman", "kendall",
            "shuffle_null_mean", "shuffle_null_p97.5"]
    with pd.option_context("display.width", 240, "display.max_columns", 30,
                           "display.max_colwidth", 46):
        print(df[cols].to_string(index=False))
    print("\n  read cross-partition against the within-campaign ceiling, not against 1.0")
    print(f"\n[p14/agreement] -> {a.out}")


if __name__ == "__main__":
    main()
