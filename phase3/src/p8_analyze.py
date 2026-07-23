"""P8a -- variance decomposition of the seed noise into INIT / ORDER / (INTERACTION+RESIDUAL).

Phase 1 and 2 established that training-seed variance dominates data-composition variance. That
is the study's central claim, and so far it names no MECHANISM. P8a asks: WHICH RNG is it?

DESIGN: 12 Stage-G masks x (2 init seeds) x (2 order seeds) = 48 retrains, fully crossed.
The trainer is bit-identical to Phase-1's when init == order (proved in p8_bitcheck.json), so the
two factors are genuinely separable and not an artifact of a rewrite.

THE DECOMPOSITION. Within each mask, the 2x2 table of outcomes y[i][o] decomposes as

    y[i][o] = mu + a_i + b_o + (ab)_io          (a = INIT, b = ORDER)

    SS_init  = 2 * sum_i (ybar_i. - ybar)^2      (df 1)
    SS_order = 2 * sum_o (ybar_.o - ybar)^2      (df 1)
    SS_inter = sum_io (y_io - ybar_i. - ybar_.o + ybar)^2   (df 1)

With ONE replicate per cell the interaction and the residual are CONFOUNDED -- there is no
within-cell replication to separate them. This is DECLARED IN ADVANCE in the preregistration and
reported as a single INTERACTION+RESIDUAL term, never as a pure interaction.

DISCLOSED CONFOUND (also preregistered): torch.manual_seed drives BOTH parameter init AND the
dropout mask stream, so the INIT factor is "init + dropout", not init alone.

PREREGISTERED READ-OUT: descriptive, no pass/fail -- but declared in advance: if either factor
exceeds 70% of the total seed variance, it is NAMED the dominant source; if neither does, the
finding is "seed noise is not attributable to a single RNG factor".
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402

INIT_SEEDS = [701, 702]
ORDER_SEEDS = [801, 802]
OUTCOMES = ["neg_plain_loss", "logit_success"]
DOMINANCE = 0.70          # preregistered naming threshold
N_BOOT = 2000


def collect():
    jobs = json.load(open(os.path.join(P3_RESULTS, "p8a_jobs.json")))
    rows, missing = [], []
    for j in jobs:
        rd = j["run_dir"]
        oc = L.read_outcomes(rd, required=False)          # MARKER-GATED (P6.1)
        if oc is None:
            missing.append(os.path.basename(rd))
            continue
        for c, v in oc.items():
            rows.append({"run": os.path.basename(rd), "mask_id": j["mask_id"],
                         "init_seed": j["init_seed"], "order_seed": j["order_seed"],
                         "target": c, "success_rate": v["success_rate"],
                         "n_episodes": v["n_episodes"], "plain_loss": v["plain_loss"],
                         "transport_loss": v["transport_loss"],
                         "interaction_loss": v["interaction_loss"]})
    df = pd.DataFrame(rows)
    df["logit_success"] = L.logit_success_rowwise(df.success_rate.values, df.n_episodes.values)
    df["neg_plain_loss"] = -df.plain_loss
    return df, missing


def decompose_mask(y):
    """y: 2x2 array [init][order]. -> (ss_init, ss_order, ss_inter_plus_resid)."""
    gm = y.mean()
    ri = y.mean(axis=1)      # per-init means
    ro = y.mean(axis=0)      # per-order means
    ss_i = 2.0 * ((ri - gm) ** 2).sum()
    ss_o = 2.0 * ((ro - gm) ** 2).sum()
    ss_x = ((y - ri[:, None] - ro[None, :] + gm) ** 2).sum()
    return ss_i, ss_o, ss_x


def main():
    df, missing = collect()
    if missing:
        print(f"[P8a] WARNING: {len(missing)} runs missing outcomes: {missing[:6]}")
    masks = sorted(df.mask_id.unique())
    print(f"[P8a] {len(df)} rows, {len(masks)} masks, "
          f"{df.run.nunique()} runs (expect 48)")

    res = {}
    for oc in OUTCOMES:
        per_mask = []
        for m in masks:
            for t in dataset.clusters():
                sub = df[(df.mask_id == m) & (df.target == t)]
                piv = sub.pivot_table(index="init_seed", columns="order_seed", values=oc)
                if piv.shape != (2, 2) or piv.isna().any().any():
                    continue
                y = piv.loc[INIT_SEEDS, ORDER_SEEDS].values
                si, so, sx = decompose_mask(y)
                per_mask.append({"mask_id": m, "target": t, "ss_init": si, "ss_order": so,
                                 "ss_inter_resid": sx, "ss_total": si + so + sx,
                                 "cell_mean": float(y.mean()), "cell_range": float(y.max()-y.min())})
        PM = pd.DataFrame(per_mask)

        # aggregate: sum of squares pooled over (mask, target) cells, then fractions
        tot = PM[["ss_init", "ss_order", "ss_inter_resid"]].sum()
        frac = (tot / tot.sum()).to_dict()

        # bootstrap over MASKS (the resampling unit)
        rng = np.random.default_rng(0)
        boot = {k: [] for k in ("ss_init", "ss_order", "ss_inter_resid")}
        for _ in range(N_BOOT):
            mm = rng.choice(masks, size=len(masks), replace=True)
            sub = pd.concat([PM[PM.mask_id == m] for m in mm])
            s = sub[["ss_init", "ss_order", "ss_inter_resid"]].sum()
            s = s / s.sum()
            for k in boot:
                boot[k].append(float(s[k]))
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
              for k, v in boot.items()}

        # per-target breakdown (descriptive)
        by_t = {}
        for t in dataset.clusters():
            s = PM[PM.target == t][["ss_init", "ss_order", "ss_inter_resid"]].sum()
            if s.sum() > 0:
                by_t[t] = (s / s.sum()).to_dict()

        dom = None
        if frac["ss_init"] > DOMINANCE:
            dom = "INIT (parameter initialization + dropout stream)"
        elif frac["ss_order"] > DOMINANCE:
            dom = "ORDER (batch permutation)"

        res[oc] = {
            "n_cells": int(len(PM)),
            "variance_fraction": {"INIT": frac["ss_init"], "ORDER": frac["ss_order"],
                                  "INTERACTION_PLUS_RESIDUAL": frac["ss_inter_resid"]},
            "bootstrap_ci95_over_masks": {"INIT": ci["ss_init"], "ORDER": ci["ss_order"],
                                          "INTERACTION_PLUS_RESIDUAL": ci["ss_inter_resid"]},
            "per_target_variance_fraction": by_t,
            "DOMINANCE_THRESHOLD": DOMINANCE,
            "DOMINANT_FACTOR": dom,
            "READOUT": (f"{dom} exceeds {DOMINANCE:.0%} of the seed variance and is NAMED the "
                        f"dominant source." if dom else
                        f"NEITHER factor exceeds {DOMINANCE:.0%} of the seed variance: seed noise "
                        f"is NOT attributable to a single RNG factor."),
        }
        print(f"\n[P8a] {oc}: INIT {frac['ss_init']:.3f} "
              f"[{ci['ss_init'][0]:.3f},{ci['ss_init'][1]:.3f}]  "
              f"ORDER {frac['ss_order']:.3f} [{ci['ss_order'][0]:.3f},{ci['ss_order'][1]:.3f}]  "
              f"INTER+RESID {frac['ss_inter_resid']:.3f} "
              f"[{ci['ss_inter_resid'][0]:.3f},{ci['ss_inter_resid'][1]:.3f}]")
        print(f"       -> {res[oc]['READOUT']}")

    df.to_parquet(os.path.join(P3_RESULTS, "p8a_outcomes.parquet"), index=False)
    out = {
        "stage": "P8a seed-noise factorial decomposition",
        "n_retrains": int(df.run.nunique()), "n_missing": len(missing), "missing": missing,
        "masks": masks, "init_seeds": INIT_SEEDS, "order_seeds": ORDER_SEEDS,
        "design": "12 masks x 2 init x 2 order, fully crossed (48 retrains)",
        "factor_definitions": {
            "INIT": ("torch.manual_seed / np.random.seed -> parameter initialization AND the "
                     "dropout mask stream. DISCLOSED CONFOUND: dropout rides on this RNG, so the "
                     "factor is 'init + dropout', not init alone. Preregistered, not discovered."),
            "ORDER": "the CPU torch.Generator that draws the per-epoch batch permutation, only."},
        "interaction_note": ("With ONE replicate per (init, order) cell, the interaction and the "
                             "residual are confounded. They are reported as a single "
                             "INTERACTION+RESIDUAL term. Declared in advance."),
        "instrument_check": "phase3/results/p8_bitcheck.json -- the factorial trainer is "
                            "bit-identical to src/train.py when init == order",
        "results": res,
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p8a_variance_decomposition.json"), out)
    print(f"\n[P8a] -> phase3/results/p8a_variance_decomposition.json")


if __name__ == "__main__":
    main()
