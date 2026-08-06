"""PASS 17 -- per-demonstration influence scores, both partitions, for the demo-gallery webpage.

WHAT THIS IS FOR. Everything the campaign measured was about masks and aggregate statistics. This
dumps the thing a person can actually look at: a score for each of the 135 demonstrations, saying how
much the datamodel thinks that demonstration mattered.

**AND IT DUMPS BOTH PARTITIONS, ON PURPOSE.** BLOCKERS #54 measured that two independent partitions
agree on per-demo influence at only ~0.47-0.52 -- far above a ~0 shuffle null, so the attribution is
real, but much weaker than the 4.8x predictive advantage suggests. **A gallery built from one
partition's ranking would present as settled exactly the quantity this project showed is its least
settled.** So both rankings ship, and the page shows where they disagree.

The scores are the k=3 fit, which is the grain where the transfer arm clears the bar at the unbiased
ceiling (#53). Group coefficients are spread evenly over each group's demos -- within a group the
demos are collinear and no other split is identifiable -- so a demo's score is its group's
coefficient divided by 3, and two demos in the same group of the same partition are tied by
construction. That is why the two partitions matter: they group the demos differently, so a demo's
two scores come from disjoint groupmates.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p9_grain as G  # noqa: E402
from if_repair import p9_masks as P9M  # noqa: E402
from if_repair import p10_masks2 as P10M  # noqa: E402
from if_repair.confirm_mseries import STATS  # noqa: E402
from if_repair.p11_transfer import _load, fit_on, coefficients_to_demo_scores  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
K = 3


def build():
    ids = list(D.cache_for("bc_s10")["train_ids"])
    _, by_c = dataset.train_pool()
    cluster_of = {d: c for c, ds in by_c.items() for d in ds}
    suite_of = dataset.suite_of_cluster()

    o_use, _, o_y, _ = _load("O", P9M.manifest()["masks"][str(K)])
    coef, gids, _ = fit_on(o_use, o_y, K, G.GROUP_SEED)
    s_o = coefficients_to_demo_scores(coef, gids, K, G.GROUP_SEED)

    r_use, _, r_y, _ = _load("R", P10M.manifest()["masks"][str(K)])
    coef_r, gids_r, _ = fit_on(r_use, r_y, K, P10M.GROUP_SEED2)
    s_r = coefficients_to_demo_scores(coef_r, gids_r, K, P10M.GROUP_SEED2)

    a = np.array([s_o[d] for d in ids])
    b = np.array([s_r[d] for d in ids])
    # z-score each partition so the two rankings are on a comparable scale
    az = (a - a.mean()) / a.std()
    bz = (b - b.mean()) / b.std()
    mean_z = (az + bz) / 2
    rank_o = np.argsort(np.argsort(-a))       # 0 = most influential
    rank_r = np.argsort(np.argsort(-b))

    rows = []
    for i, d in enumerate(ids):
        suite, task, demo = dataset.parse_did(d)
        rows.append({
            "demo_id": d, "suite": suite, "task": task, "demo": demo,
            "cluster": cluster_of.get(d), "cluster_suite": suite_of.get(cluster_of.get(d)),
            "score_O": float(a[i]), "score_R": float(b[i]),
            "z_O": float(az[i]), "z_R": float(bz[i]), "z_mean": float(mean_z[i]),
            "rank_O": int(rank_o[i]) + 1, "rank_R": int(rank_r[i]) + 1,
            "rank_gap": int(abs(rank_o[i] - rank_r[i])),
        })

    agree = {
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(STATS["spearman"](a, b)),
        "kendall": float(STATS["kendall_tau_b"](a, b)),
        "n_demos": len(ids), "grain_k": K,
        "note": ("Two INDEPENDENT partitions of the same 135 demos, sharing zero groups. This is the "
                 "honest strength of a per-demonstration claim on this corpus -- see BLOCKERS #54."),
    }
    out = {"agreement": agree, "demos": rows}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "p17_demo_scores.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main():
    out = build()
    ag = out["agreement"]
    rows = sorted(out["demos"], key=lambda r: -r["z_mean"])
    print(f"[p17] {ag['n_demos']} demos, k={ag['grain_k']}; cross-partition agreement: "
          f"pearson {ag['pearson']:.3f} spearman {ag['spearman']:.3f} kendall {ag['kendall']:.3f}")
    print("\n  TOP 8 by mean z (most influential):")
    for r in rows[:8]:
        print(f"    {r['z_mean']:+.2f}  O#{r['rank_O']:<4d} R#{r['rank_R']:<4d} gap={r['rank_gap']:<4d} "
              f"{r['cluster']} {r['suite']:15s} {r['task'][:44]:44s} {r['demo']}")
    print("\n  BOTTOM 8 (least influential):")
    for r in rows[-8:]:
        print(f"    {r['z_mean']:+.2f}  O#{r['rank_O']:<4d} R#{r['rank_R']:<4d} gap={r['rank_gap']:<4d} "
              f"{r['cluster']} {r['suite']:15s} {r['task'][:44]:44s} {r['demo']}")
    gaps = np.array([r["rank_gap"] for r in out["demos"]])
    print(f"\n  rank disagreement between partitions: median {np.median(gaps):.0f} places, "
          f"max {gaps.max()} (of {len(gaps)})")


if __name__ == "__main__":
    main()
