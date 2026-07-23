"""P8b analysis -- do the two cheap repairs buy mask-ranking reliability?

THE COMPARISONS (12 masks, both outcomes):

  arm (i) is judged against a FREE baseline, at S = 1:
      baseline S=1 : mean pairwise Spearman of single-seed mask-score vectors (seeds 401,402,403)
      arm (i)  S=1 : the same, with every model replaced by its 3-checkpoint weight average
    Checkpoints already exist (TracIn needed them), so arm (i) costs NOTHING. Any gain is free.

  arm (ii) is judged against an EQUAL-SEED-COST baseline, at S = 3:
      baseline S=3 : Spearman( mean-OUTCOME(401,402,403), mean-OUTCOME(404,405,406) )
      arm (ii) S=3 : Spearman( action-ENSEMBLE(401,402,403), action-ENSEMBLE(404,405,406) )
    Both spend 3 trained models per estimate, so this isolates WHERE the averaging happens:
    in outcome space (what Phase 2 did) or in action space.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS, P2_RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman  # noqa: E402

SEEDS_A = [401, 402, 403]
SEEDS_B = [404, 405, 406]
OUTCOMES = ["neg_plain_loss", "logit_success"]


def masks12():
    return json.load(open(os.path.join(P3_RESULTS, "p8a_mask_selection.json")))["masks"]


def val(oc, v):
    if oc == "neg_plain_loss":
        return -v["plain_loss"]
    return float(L.logit_success_rowwise([v["success_rate"]], [v["n_episodes"]])[0])


def main():
    ms = masks12()
    clusters = dataset.clusters()

    # ---- baseline: the EXISTING single-seed runs (Stage G / G6), read marker-gated
    G6 = pd.read_parquet(os.path.join(P2_RESULTS, "stage_G6_outcomes.parquet"))
    G6 = G6[G6.mask_id.isin(ms)]

    # ---- arm (i): checkpoint-averaged runs
    ck = {}
    for m in ms:
        for s in SEEDS_A:
            d = os.path.join(P3_RUNS, "P8b", f"ckptavg_{m}_s{s}")
            oc = L.read_outcomes(d, required=False)
            if oc:
                ck[(m, s)] = oc

    # ---- arm (ii): action-ensemble runs
    ae = {}
    for m in ms:
        for tag in ("A", "B"):
            d = os.path.join(P3_RUNS, "P8b", f"actens_{m}_{tag}")
            oc = L.read_outcomes(d, required=False)
            if oc:
                ae[(m, tag)] = oc

    print(f"[P8b] masks={len(ms)} ckptavg_evals={len(ck)} actens_evals={len(ae)}")

    res = {}
    for oc in OUTCOMES:
        r = {}
        # ---------- S=1 baseline: mean pairwise rho over seeds A, per target, then mean
        b1, a1 = [], []
        for t in clusters:
            vec = {}
            for s in SEEDS_A:
                sub = G6[(G6.target == t) & (G6.seed == s)].set_index("mask_id")
                vec[s] = np.array([sub.loc[m, oc] if m in sub.index else np.nan for m in ms])
            rs = [spearman(vec[x], vec[y]) for x, y in itertools.combinations(SEEDS_A, 2)]
            b1.append(np.nanmean(rs))

            cvec = {}
            for s in SEEDS_A:
                cvec[s] = np.array([val(oc, ck[(m, s)][t]) if (m, s) in ck else np.nan
                                    for m in ms])
            rs = [spearman(cvec[x], cvec[y]) for x, y in itertools.combinations(SEEDS_A, 2)]
            a1.append(np.nanmean(rs))
        r["baseline_S1_single_seed"] = float(np.nanmean(b1))
        r["arm_i_S1_checkpoint_averaged"] = float(np.nanmean(a1))
        r["arm_i_gain"] = r["arm_i_S1_checkpoint_averaged"] - r["baseline_S1_single_seed"]

        # ---------- S=3 baseline (outcome averaging) vs arm (ii) (action ensembling)
        b3, a3 = [], []
        for t in clusters:
            sa = G6[(G6.target == t) & (G6.seed.isin(SEEDS_A))].groupby("mask_id")[oc].mean()
            sb = G6[(G6.target == t) & (G6.seed.isin(SEEDS_B))].groupby("mask_id")[oc].mean()
            va = np.array([sa.get(m, np.nan) for m in ms])
            vb = np.array([sb.get(m, np.nan) for m in ms])
            b3.append(spearman(va, vb))

            ea = np.array([val(oc, ae[(m, "A")][t]) if (m, "A") in ae else np.nan for m in ms])
            eb = np.array([val(oc, ae[(m, "B")][t]) if (m, "B") in ae else np.nan for m in ms])
            a3.append(spearman(ea, eb))
        r["baseline_S3_outcome_mean"] = float(np.nanmean(b3))
        r["arm_ii_S3_action_ensemble"] = float(np.nanmean(a3))
        r["arm_ii_gain"] = r["arm_ii_S3_action_ensemble"] - r["baseline_S3_outcome_mean"]

        res[oc] = r
        print(f"\n[P8b] {oc}")
        print(f"  baseline S=1 (single seed)          : {r['baseline_S1_single_seed']:+.3f}")
        print(f"  arm (i)  S=1 (checkpoint-averaged)  : {r['arm_i_S1_checkpoint_averaged']:+.3f}"
              f"   gain {r['arm_i_gain']:+.3f}   [FREE]")
        print(f"  baseline S=3 (outcome mean)         : {r['baseline_S3_outcome_mean']:+.3f}")
        print(f"  arm (ii) S=3 (action ensemble)      : {r['arm_ii_S3_action_ensemble']:+.3f}"
              f"   gain {r['arm_ii_gain']:+.3f}   [same 3-seed cost]")

    out = {
        "stage": "P8b variance-reduction probe",
        "n_retrains": 0,
        "masks": ms, "n_masks": len(ms),
        "arm_i": ("checkpoint-averaged policy: mean of the WEIGHTS of ckpt_2/3/4 of each existing "
                  "run. FREE -- the checkpoints already exist because TracIn needed them."),
        "arm_ii": ("action-ensemble policy: the MEAN ACTION of S models, stepped in lockstep "
                   "through one env. Costs S trained models -- the same as seed-ensembling."),
        "n_ckptavg_evals": len(ck), "n_actens_evals": len(ae),
        "determinism_gate": "phase3/results/p8b_determinism.json",
        "results": res,
        "reading": ("arm (i) is compared to the S=1 baseline because it is free; arm (ii) is "
                    "compared to the S=3 outcome-mean baseline because it spends the same 3 "
                    "seeds -- so it isolates WHERE the averaging happens (action space vs outcome "
                    "space), not how much it costs."),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p8b_variance_reduction.json"), out)
    print(f"\n[P8b] -> phase3/results/p8b_variance_reduction.json")


if __name__ == "__main__":
    main()
