"""PASS 11 -- campaign O re-read at DEPTH 4. Descriptive; campaign O's own scoring stays frozen.

WHY THIS RUN EXISTS, HAVING BEEN DEFERRED TWICE AS A FOOTNOTE.

BLOCKERS #42: the project's `ratio = rho / ceiling` uses a Spearman-Brown RELIABILITY in the
denominator, while the largest correlation any predictor can have with an observation of reliability
r is about sqrt(r). So the ratio is inflated by ~1/sqrt(r), and inflated MORE as r falls -- i.e. at
lower seed depth. Measured directly on the k=15 rung: the same masks give ratio 0.666 at depth 2 and
0.496 at depth 4, because the ceiling rises from 0.451 to 0.521.

That asymmetry is why this was a footnote for two passes and why it is not one now:

- Pass 9/10's **negative** results (gradient attribution clears the bar nowhere) sit at depth 2, where
  the ratio is inflated. They were therefore measured on the most generous scale the design offers,
  and a stricter denominator can only push them further down. Nothing to test.
- Pass 11's **positive** result (#50, the datamodel clearing the bar) is also at depth 2. A positive
  claim measured on an inflated scale deserves the unbiased one. **That is the test.**

THE MERGE. `functionals.campaign_outcomes` takes a single campaign, and depth 4 on these masks spans
two: campaign O holds seed slots {4401, 4402} and campaign Q holds {4403, 4404} for the IDENTICAL 800
masks. Mask ids are shared, so the merge is a per-mask dict update and the depth-2 and depth-4 reads
differ only in depth -- no fresh draw, no confound with mask sampling.

WHAT IS AND IS NOT RE-READ. GradDot and the within-campaign datamodel are both re-scored at depth 4 on
campaign O's masks. The cross-partition TRANSFER arm (#50) cannot be fully re-read: campaign R is
depth 2 and no depth-4 outcomes exist for it, so only the fit side improves. That arm is reported with
the limitation stated rather than quietly presented as a depth-4 number.
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
from if_repair.confirm_mseries import ceiling, STATS  # noqa: E402
from if_repair.p9_datamodel_cluster import loo_predict  # noqa: E402
from if_repair.p10_datamodel_subcluster import group_design  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TARGET = "C5"
PRIMARY_STAT = "kendall_tau_b"
BAR = 0.5
N_BOOT = 1500


def merged_outcomes(campaigns=("O", "Q"), target=TARGET):
    """Per-mask {seed: value} across campaigns. Mask ids are shared, so this is a dict update."""
    out = {}
    for c in campaigns:
        try:
            got = F.campaign_outcomes(c, "plain", targets=(target,))[target]
        except Exception:
            continue
        for m, v in got.items():
            out.setdefault(m, {}).update(v)
    return out


def read_at(masks, raw_all, depth):
    """Seed-mean outcomes at exactly `depth` slots, and the masks that have them."""
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
    masks_all = P9M.manifest()["masks"][str(k)]
    raw_all = merged_outcomes()
    gids = [g["group_id"] for g in G.groups(k)]
    rng = np.random.default_rng(0)
    rows = []

    for depth in (2, 4):
        got = read_at(masks_all, raw_all, depth)
        if got is None:
            continue
        use, raw, y = got
        X = group_design(use, gids)
        pg = P7.mask_pred(P7._graddot("cached")[TARGET], use)
        pred_dm, _ = loo_predict(X, y, model=model)

        for sname, fn in STATS.items():
            c = ceiling(raw, fn)
            for arm, p in (("GradDot", pg), (f"datamodel_{model} (LOO, within)", pred_dm)):
                g = np.isfinite(p) & np.isfinite(y)
                lds = fn(p[g], y[g])
                bs = np.empty(N_BOOT)
                for i in range(N_BOOT):
                    j = rng.integers(0, g.sum(), g.sum())
                    bs[i] = fn(p[g][j], y[g][j])
                rows.append({
                    "grain": f"k={k}", "arm": arm, "statistic": sname,
                    "primary": sname == PRIMARY_STAT, "depth": depth,
                    "n_masks": int(g.sum()), "n_coef": X.shape[1],
                    "lds": lds, "lds_se": float(bs.std(ddof=1)), "ceiling": c,
                    "ratio": lds / c if np.isfinite(c) and c else np.nan,
                    "ratio_sqrt": lds / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan,
                    "CLEARS_BAR": bool(np.isfinite(c) and c and lds / c >= BAR),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p11_depth.csv"))
    a = ap.parse_args()
    df = pd.concat([evaluate(k) for k in (3, 5)], ignore_index=True)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["grain", "arm", "depth", "n_masks", "lds", "lds_se", "ceiling", "ratio", "ratio_sqrt",
            "CLEARS_BAR"]
    with pd.option_context("display.width", 240, "display.max_columns", 30,
                           "display.max_colwidth", 40):
        print(df[df.primary][cols].to_string(index=False))
    print(f"\n[p11/depth] -> {a.out}")


if __name__ == "__main__":
    main()
