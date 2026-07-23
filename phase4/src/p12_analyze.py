"""P12 -- deepen the BC ground truth to S = 10, and GATE the instrument.

The 24 Stage-G masks now have 10 training seeds: 401-406 (Phases 1-2) + 407-410 (Phase 4).
This script does TWO things and no more:

  1. INGEST the 96 new runs through the MARKER-GATED reader and merge them with the archived
     S=6 table -> phase4/results/p12_outcomes_S10.parquet (the S=10 ground truth).

  2. Run the PREREGISTERED INSTRUMENT GATE before any verdict exists:

         r1_S6      = mean pairwise single-seed Spearman over the C(6,2)=15 pairs of seeds
                      401-406, computed from S=6 DATA ONLY
         predicted  = SB(r1_S6, 10) = 10*r1 / (1 + 9*r1)
         measured   = SB-corrected 5v5 split-half ceiling on the S=10 data (126 distinct splits)
         GATE       : |predicted - measured| <= 0.10 on BOTH focal targets C1 and C5

     This is the check that FAILED by 0.61 for the diffusion policy (PHASE3_DEFECT.md) and
     PASSED to 0.014 for the BC-Transformer in Phase 2. It is the reason we can trust a
     ratio-to-ceiling test at all. If it fails, we STOP: write PHASE4_DEFECT.md, compute no
     verdict.

NO VERDICT IS COMPUTED HERE. P12 produces the instrument; P11 uses it.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, P2_RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402

SEEDS_OLD = [401, 402, 403, 404, 405, 406]      # Phases 1-2
SEEDS_NEW = [407, 408, 409, 410]                # Phase 4 (fallback: 407, 408 -> S=8)
FOCAL = ["C1", "C5"]
GATE_TOL = 0.10                                 # PREREGISTERED
OUTCOME = "neg_plain_loss"


def ingest(jobs):
    """MARKER-GATED ingestion of the new runs. A partially-failed run is REFUSED, not skipped."""
    rows, missing = [], []
    for j in jobs:
        rd = j["run_dir"]
        if not L.is_marked(rd, "probe"):
            missing.append(os.path.basename(rd))
            continue
        for c, v in L.read_outcomes(rd).items():
            rows.append({"stage": "P12_stage_G10", "run": os.path.basename(rd),
                         "mask_id": j["mask_id"], "seed": j["seed"], "target": c,
                         **{k: v[k] for k in ("success_rate", "n_episodes", "plain_loss",
                                              "transport_loss", "interaction_loss")}})
    return pd.DataFrame(rows), missing


def main():
    L.assert_prereg_locked()
    jobs = json.load(open(os.path.join(P4_RESULTS, "p12_jobs.json")))
    new, missing = ingest(jobs)
    if missing:
        print(f"[P12] {len(missing)} runs not yet complete: {missing[:8]}")
        if len(missing) == len(jobs):
            raise SystemExit("[P12] nothing ingested -- training has not finished")

    seeds_have = sorted(new.seed.unique().tolist())
    old = pd.read_parquet(os.path.join(P2_RESULTS, "stage_G6_outcomes.parquet"))
    old = old[["stage", "run", "mask_id", "seed", "target", "success_rate", "n_episodes",
               "plain_loss", "transport_loss", "interaction_loss"]]

    df = pd.concat([old, new], ignore_index=True)
    df["logit_success"] = L.logit_success_rowwise(df.success_rate.values, df.n_episodes.values)
    df["neg_plain_loss"] = -df.plain_loss
    df["neg_transport_loss"] = -df.transport_loss
    df["neg_interaction_loss"] = -df.interaction_loss

    S = SEEDS_OLD + seeds_have
    n_seed = df.groupby("mask_id").seed.nunique()
    print(f"[P12] merged: {len(df)} rows, {df.mask_id.nunique()} masks, seeds {S}, "
          f"seeds/mask min={n_seed.min()} max={n_seed.max()}")
    if n_seed.min() != n_seed.max():
        raise RuntimeError(f"ragged seed coverage across masks: {n_seed.to_dict()}")

    df.to_parquet(os.path.join(P4_RESULTS, "p12_outcomes_S10.parquet"), index=False)

    # ---------------------------------------------------------------- ceilings + INSTRUMENT GATE
    clusters = dataset.clusters()
    res, gate_fail = {}, []
    for t in clusters:
        sub = df[df.target == t]
        piv = sub.pivot_table(index="mask_id", columns="seed", values=OUTCOME)

        # predicted from S=6 DATA ONLY
        r1_s6, pairs_s6 = L.mean_pairwise_1seed_r(piv, SEEDS_OLD)
        pred10 = L.sb(r1_s6, len(S))

        # measured on the full S data
        r_half, r_full, splits, n_split = L.split_half_ceiling(piv, S, agg="mean")

        # the archived S=6 ceiling, for the record (Phase-2 P1 estimator, 10 distinct 3v3 splits)
        r3_s6, r6_s6, _, n3 = L.split_half_ceiling(piv, SEEDS_OLD, agg="mean")

        diff = abs(pred10 - r_full) if np.isfinite(pred10) and np.isfinite(r_full) else np.nan
        passed = bool(np.isfinite(diff) and diff <= GATE_TOL)
        e = {
            "target": t, "focal": t in FOCAL, "n_masks": int(piv.dropna().shape[0]),
            "n_seeds": len(S), "seeds": S,
            "r1_from_S6_only": r1_s6, "n_pairs_S6": len(pairs_s6),
            "predicted_ceiling_SB_from_S6": pred10,
            f"ceiling_{len(S)//2}v{len(S)//2}_splithalf_uncorrected": r_half,
            f"ceiling_{len(S)}seed_SB": r_full,
            "n_splits": n_split,
            "archived_ceiling_6seed_SB": r6_s6, "archived_r3_splithalf": r3_s6,
            "SB_consistency_abs_diff": diff,
            "SB_consistency_tol": GATE_TOL,
            "SB_consistency_PASS": passed,
            "bar_half_ceiling": 0.5 * r_full if np.isfinite(r_full) else np.nan,
            "per_split": splits,
        }
        res[t] = e
        if t in FOCAL and not passed:
            gate_fail.append(t)

    gate = {
        "gate": "P12 SB-consistency (instrument gate)",
        "definition": ("r1 from S=6 data only (15 pairs) -> SB to S=10; compared with the "
                       "measured SB-corrected 5v5 ceiling on the S=10 data"),
        "tolerance": GATE_TOL,
        "focal_targets": FOCAL,
        "PASS": len(gate_fail) == 0,
        "failed_focal_targets": gate_fail,
    }
    out = {"stage": "P12", "PRIMARY_OUTCOME": OUTCOME, "aggregator": "seed MEAN",
           "aggregator_justification": ("the BC-Transformer's executed-action outcome is NOT "
                                        "heavy-tailed; Phase-2's SB check on this instrument "
                                        "agreed to 0.014 (preregistered justification)"),
           "seeds": S, "S": len(S), "n_new_retrains": len(jobs), "n_missing": len(missing),
           "missing": missing, "INSTRUMENT_GATE": gate, "targets": res}
    L.atomic_write_json(os.path.join(P4_RESULTS, "p12_ceilings.json"), out)

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 100)
    print(f"P12 -- BC GROUND TRUTH AT S={len(S)}  (outcome: {OUTCOME}, seed MEAN)")
    print("=" * 100)
    print(f"{'target':7s} {'r1(S6)':>7s} {'pred10(SB)':>11s} {'meas 5v5':>9s} "
          f"{'meas10(SB)':>11s} {'|diff|':>7s} {'bar=.5c':>8s}  gate")
    for t in clusters:
        e = res[t]
        print(f"{t:7s} {e['r1_from_S6_only']:7.3f} {e['predicted_ceiling_SB_from_S6']:11.3f} "
              f"{e[f'ceiling_{len(S)//2}v{len(S)//2}_splithalf_uncorrected']:9.3f} "
              f"{e[f'ceiling_{len(S)}seed_SB']:11.3f} {e['SB_consistency_abs_diff']:7.3f} "
              f"{e['bar_half_ceiling']:8.3f}  "
              f"{'PASS' if e['SB_consistency_PASS'] else 'FAIL'}"
              f"{'  <-- FOCAL' if e['focal'] else ''}")
    print("-" * 100)
    print(f"INSTRUMENT GATE (focal C1/C5, tol {GATE_TOL}): "
          f"{'PASS' if gate['PASS'] else 'FAIL -> STOP, write PHASE4_DEFECT.md'}")
    print("=" * 100)
    if not gate["PASS"]:
        raise SystemExit(f"[P12] INSTRUMENT GATE FAILED on {gate_fail}. STOP. "
                         f"Write phase4/PHASE4_DEFECT.md before any verdict is computed.")


if __name__ == "__main__":
    main()
