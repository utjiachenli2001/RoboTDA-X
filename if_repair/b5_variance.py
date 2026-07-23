"""B5 -- what the common-random-seed estimand is actually worth, measured rather than assumed.

The pass-1 handoff proposed re-running the mask sweep with common random seeds, on the premise
that "every outcome here is a seed mean over independently seeded retrains, so the mask-to-mask
contrast carries the full between-seed variance", and that init is ~72% of that variance.

READ src/train.py BEFORE BELIEVING THAT PREMISE. `torch.manual_seed(seed)` is called before the
model is built, and the number of RNG draws consumed by initialisation depends only on the
architecture -- which is frozen. So two masks trained at the SAME seed already start from
BIT-IDENTICAL weights. Stage G used seeds 401-410 for every mask, so the archived design is
ALREADY common-init. Whatever init contributes is a per-seed offset shared by all masks, and a
shared additive offset cannot move a rank correlation computed ACROSS masks.

What can still hurt the mask contrast is the INTERACTION: the part of a seed's effect that
depends on which demos are in the mask (batch composition, dropout draws, optimisation path).
That is not removable by sharing a seed, and it is what the ceiling is really made of.

So the experiment worth running is not "common seeds" -- that is already the case -- but a
DECOMPOSITION. Campaign C trains 8 masks x 3 inits x 3 orders with the two streams split
(retrain.train_one), which the repo's single --seed cannot express, and this module partitions
the variance into main effects (harmless) and mask interactions (harmful), then measures the
reliability the mask ranking would have under three designs.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402

D.add_repo_paths()
from lds import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def grid(campaign="C", target="C1", weighting="plain"):
    """-> (masks, inits, orders, Y[m,i,o]) from the campaign's per-frame losses."""
    raw = F.campaign_outcomes(campaign, weighting, targets=(target,))[target]
    masks = sorted(raw)
    inits = sorted({k[0] for m in masks for k in raw[m]})
    orders = sorted({k[1] for m in masks for k in raw[m]})
    Y = np.full((len(masks), len(inits), len(orders)), np.nan)
    for a, m in enumerate(masks):
        for b, i in enumerate(inits):
            for c, o in enumerate(orders):
                if (i, o) in raw[m]:
                    Y[a, b, c] = raw[m][(i, o)]
    return masks, inits, orders, Y


def anova(Y):
    """Balanced 3-way sums of squares with one observation per cell.

    The 3-way interaction is confounded with error at n=1 per cell and is reported as
    `resid_MxIxO`; every other term is exact.
    """
    M, I, O = Y.shape
    g = np.nanmean(Y)
    m_ = np.nanmean(Y, axis=(1, 2))
    i_ = np.nanmean(Y, axis=(0, 2))
    o_ = np.nanmean(Y, axis=(0, 1))
    mi = np.nanmean(Y, axis=2)
    mo = np.nanmean(Y, axis=1)
    io = np.nanmean(Y, axis=0)
    ss = {}
    ss["mask"] = I * O * np.sum((m_ - g) ** 2)
    ss["init"] = M * O * np.sum((i_ - g) ** 2)
    ss["order"] = M * I * np.sum((o_ - g) ** 2)
    ss["mask_x_init"] = O * np.sum((mi - m_[:, None] - i_[None, :] + g) ** 2)
    ss["mask_x_order"] = I * np.sum((mo - m_[:, None] - o_[None, :] + g) ** 2)
    ss["init_x_order"] = M * np.sum((io - i_[:, None] - o_[None, :] + g) ** 2)
    ss["total"] = np.nansum((Y - g) ** 2)
    ss["resid_MxIxO"] = ss["total"] - sum(ss[k] for k in
                                          ("mask", "init", "order", "mask_x_init",
                                           "mask_x_order", "init_x_order"))
    return ss


def design_reliability(Y, n_rep=2000, seed=0):
    """Spearman reliability of the MASK ranking under three seeding designs.

    common_seed  one (init, order) cell shared by every mask -- the archived Stage-G design.
    common_init  init shared by every mask, order drawn per mask.
    independent  both drawn per mask -- the design the pass-1 handoff assumed was in use.
    """
    rng = np.random.default_rng(seed)
    M, I, O = Y.shape
    out = {}

    pairs = [(a, b) for a, b in itertools.combinations(itertools.product(range(I), range(O)), 2)]
    rs = [spearman(Y[:, a[0], a[1]], Y[:, b[0], b[1]]) for a, b in pairs]
    out["common_seed"] = float(np.nanmean(rs))

    for name, share_init in (("common_init", True), ("independent", False)):
        acc = []
        for _ in range(n_rep):
            va, vb = np.empty(M), np.empty(M)
            ia, ib = rng.integers(0, I), rng.integers(0, I)
            for m in range(M):
                i1 = ia if share_init else rng.integers(0, I)
                i2 = ib if share_init else rng.integers(0, I)
                o1, o2 = rng.integers(0, O), rng.integers(0, O)
                if (i1, o1) == (i2, o2):                 # never correlate a cell with itself
                    o2 = (o2 + 1) % O
                va[m], vb[m] = Y[m, i1, o1], Y[m, i2, o2]
            r = spearman(va, vb)
            if np.isfinite(r):
                acc.append(r)
        out[name] = float(np.mean(acc))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="C")
    ap.add_argument("--targets", default=",".join(D.ALL_TARGETS))
    a = ap.parse_args()
    rows, rel = [], []
    for t in a.targets.split(","):
        try:
            masks, inits, orders, Y = grid(a.campaign, t)
        except Exception as e:                                  # noqa: BLE001
            print(f"[b5] {t}: {e}")
            continue
        if np.isnan(Y).any():
            print(f"[b5] {t}: incomplete grid ({int(np.isnan(Y).sum())} missing cells) -- skipped")
            continue
        ss = anova(Y)
        tot = ss["total"]
        rows.append({"target": t, "n_masks": len(masks), "n_init": len(inits),
                     "n_order": len(orders),
                     **{k: 100.0 * v / tot for k, v in ss.items() if k != "total"}})
        r = design_reliability(Y)
        rel.append({"target": t, **r})

    df = pd.DataFrame(rows)
    dr = pd.DataFrame(rel)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b5_variance.csv"), index=False)
    dr.to_csv(os.path.join(RESULTS, "b5_design_reliability.csv"), index=False)

    print("=" * 100)
    print("B5 -- variance decomposition of the held-out outcome (% of total SS), campaign C")
    print("=" * 100)
    cols = ["target", "mask", "init", "order", "mask_x_init", "mask_x_order",
            "init_x_order", "resid_MxIxO"]
    print(df[cols].round(2).to_string(index=False))
    print("\nHARMLESS (shared across masks): init + order + init_x_order")
    print("HARMFUL  (moves the mask ranking): mask_x_init + mask_x_order + resid")
    harm = df[["mask_x_init", "mask_x_order", "resid_MxIxO"]].sum(axis=1)
    ok = df[["init", "order", "init_x_order"]].sum(axis=1)
    print(pd.DataFrame({"target": df.target, "signal_mask_%": df["mask"].round(1),
                        "harmless_%": ok.round(1), "harmful_%": harm.round(1)})
          .to_string(index=False))
    print("\n" + "=" * 100)
    print("Reliability of the MASK ranking under three seeding designs (single retrain each)")
    print("=" * 100)
    print(dr.round(4).to_string(index=False))
    print("\ncommon_seed is what Stage G already does (same seed -> bit-identical init).")
    print("The gap common_seed - independent is what a 'common-random-seed retrain' could buy;")
    print("it is already banked, so a CRN re-run would recover none of it.")


if __name__ == "__main__":
    main()
