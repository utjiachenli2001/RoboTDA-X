"""P9 -- the ATTRIBUTION-side cost law. How many ensemble members does a stable ranking need?

Phase-2 P4 priced the GROUND-TRUTH side in training seeds ("reliability is bought with seeds, not
episodes; a 0.8 ceiling on a near-floor cluster needs ~12.5x the seeds"). P9 prices the OTHER
side: the attribution itself is an average over E ensemble members, and that average is also
noisy. How large must E be before the per-demo RANKING stops moving?

MEASURED, for E in {2, 5, 10, 20}, per attributor:
  (a) split-half ranking reliability -- draw two DISJOINT E/2-member sub-ensembles, average each
      one's per-demo scores, and take the Spearman between the two score vectors over the 135
      demos. Averaged over splits (ALL disjoint splits when <= 200, else 200 drawn with rng(919)).
  (b) demo-grain LDS of the E-member ensemble mean vs the Stage-G6 6-seed ground truth
  (c) stability of the headline intrusion rate and of the top-15 composition across sub-ensembles

PREREGISTERED READ-OUT: the Spearman-Brown EXTRAPOLATION of (a) to the E needed for a 0.8 ranking
reliability, per attributor, EXPLICITLY LABELLED AN EXTRAPOLATION:
      E* = E * r_target (1 - r_E) / (r_E (1 - r_target))
computed from the LARGEST measured E (=20) and cross-checked against E=10.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, RESULTS, P2_RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

ATTRS = ["IF", "TRAK", "TracIn"]
E_LADDER = [2, 5, 10, 20]
FOCAL = ["C1", "C5"]
RNG = 919
MAX_SPLITS = 200
R_TARGET = 0.8


def spearman_brown_E(r_E, E, r_target=R_TARGET):
    """E needed for r_target, from a measured r_E at size E. AN EXTRAPOLATION."""
    if not np.isfinite(r_E) or r_E <= 0 or r_E >= 1:
        return np.nan
    return float(E * r_target * (1 - r_E) / (r_E * (1 - r_target)))


def load_members():
    old = pd.read_parquet(os.path.join(RESULTS, "influence_table_per_member.parquet"))
    old = old[old.functional == "plain"]
    new = pd.read_parquet(os.path.join(P3_RESULTS, "p9_influence_new_members.parquet"))
    df = pd.concat([old, new], ignore_index=True)
    ms = sorted(df.member.unique())
    print(f"[P9] {len(ms)} members: {ms[0]} .. {ms[-1]}")
    assert len(ms) == 20, f"expected E=20, got {len(ms)}"
    return df, ms


def score_vec(df, attr, target, members, demos):
    s = df[(df.attributor == attr) & (df.target == target) & (df.member.isin(members))]
    g = s.groupby("demo_id")["score"].mean()
    return np.array([g.get(d, np.nan) for d in demos])


def main():
    df, members = load_members()
    demos, by_c = dataset.train_pool()
    clusters = dataset.clusters()
    rng = np.random.default_rng(RNG)

    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    G6 = pd.read_parquet(os.path.join(P2_RESULTS, "stage_G6_outcomes.parquet"))
    p1 = json.load(open(os.path.join(P2_RESULTS, "p1_demo_grain.json")))
    ceil = {t: p1["all_targets"][t]["neg_plain_loss"]["ceiling_6seed_SB"] for t in clusters}
    obs = {t: G6[G6.target == t].groupby("mask_id")["neg_plain_loss"].mean().to_dict()
           for t in clusters}

    rows, rel_rows = [], []
    for E in E_LADDER:
        half = E // 2
        # disjoint half-splits of an E-subset drawn from the 20 members
        all_splits = []
        if E == 20:
            # all disjoint 10|10 splits of the fixed 20 = C(20,10)/2 = 92378 -> subsample
            for _ in range(MAX_SPLITS):
                p = list(rng.permutation(members))
                all_splits.append((p[:half], p[half:2 * half]))
        else:
            for _ in range(MAX_SPLITS):
                p = list(rng.permutation(members))
                all_splits.append((p[:half], p[half:2 * half]))

        for attr in ATTRS:
            # (a) split-half ranking reliability, per target then averaged
            per_target = {}
            for t in clusters:
                rs = []
                for A, B in all_splits:
                    a = score_vec(df, attr, t, A, demos)
                    b = score_vec(df, attr, t, B, demos)
                    r = spearman(a, b)
                    if np.isfinite(r):
                        rs.append(r)
                per_target[t] = float(np.mean(rs))
            r_mean = float(np.mean(list(per_target.values())))

            # (b) demo-grain LDS of the E-member ensemble mean (first E members, fixed)
            sub = members[:E]
            lds_focal = {}
            for t in FOCAL:
                sc = dict(zip(demos, score_vec(df, attr, t, sub, demos)))
                pred = np.array([sum(sc.get(d, 0.0) for d in m["demos"]) for m in gman])
                out = np.array([obs[t].get(m["mask_id"], np.nan) for m in gman])
                ok = np.isfinite(out)
                rho = spearman(pred[ok], out[ok])
                lds_focal[t] = {"lds": rho, "ratio": rho / ceil[t],
                                "p_onesided": spearman_p_onesided(rho, int(ok.sum()))}

            # (c) headline stability across sub-ensembles: intrusion + top-15 composition
            intr, top15 = [], []
            for A, _ in all_splits[:50]:
                sc = dict(zip(demos, score_vec(df, attr, "C1", A, demos)))
                ins = np.array([sc[d] for d in by_c["C1"]])
                outs = np.array([sc[d] for c in by_c if c != "C1" for d in by_c[c]])
                intr.append(float((outs > np.median(ins)).mean()))
                top15.append(frozenset(sorted(sc, key=lambda d: -sc[d])[:15]))
            jac = []
            for x, y in itertools.combinations(set(top15), 2):
                jac.append(len(x & y) / len(x | y))

            rel_rows.append({
                "E": E, "attributor": attr,
                "split_half_reliability_mean_over_9_targets": r_mean,
                "per_target_reliability": per_target,
                "SB_extrapolated_E_for_0.8": spearman_brown_E(r_mean, E),
                "demo_lds_C1": lds_focal["C1"]["lds"], "ratio_C1": lds_focal["C1"]["ratio"],
                "demo_lds_C5": lds_focal["C5"]["lds"], "ratio_C5": lds_focal["C5"]["ratio"],
                "intrusion_C1_mean": float(np.mean(intr)),
                "intrusion_C1_sd_across_subensembles": float(np.std(intr)),
                "top15_jaccard_mean_across_subensembles": float(np.mean(jac)) if jac else 1.0,
                "n_distinct_top15_sets": len(set(top15)),
            })
            print(f"[P9] E={E:2d} {attr:7s} split-half r={r_mean:+.3f}  "
                  f"SB E*(0.8)={spearman_brown_E(r_mean, E):7.1f}  "
                  f"LDS C1={lds_focal['C1']['lds']:+.3f} C5={lds_focal['C5']['lds']:+.3f}  "
                  f"top15 Jaccard={np.mean(jac) if jac else 1.0:.3f}", flush=True)

    T = pd.DataFrame(rel_rows)
    T.to_csv(os.path.join(P3_RESULTS, "p9_ensemble_cost_law.csv"), index=False)

    # ---- PREREGISTERED READ-OUT: SB extrapolation from the LARGEST measured E, cross-checked
    readout = {}
    for attr in ATTRS:
        r20 = float(T[(T.E == 20) & (T.attributor == attr)]
                    .split_half_reliability_mean_over_9_targets.iloc[0])
        r10 = float(T[(T.E == 10) & (T.attributor == attr)]
                    .split_half_reliability_mean_over_9_targets.iloc[0])
        readout[attr] = {
            "measured_reliability_at_E20": r20,
            "measured_reliability_at_E10": r10,
            "SB_EXTRAPOLATED_E_for_0.8_from_E20": spearman_brown_E(r20, 20),
            "SB_EXTRAPOLATED_E_for_0.8_from_E10_crosscheck": spearman_brown_E(r10, 10),
            "LABEL": "EXTRAPOLATION (Spearman-Brown), not a measurement.",
            "already_at_0.8_at_E20": bool(r20 >= R_TARGET),
        }

    out = {
        "stage": "P9 attribution-side ensemble cost law",
        "n_retrains": 10, "new_seeds": list(range(211, 221)), "E_total": 20,
        "ridge": "FROZEN Phase-1 default (ridge_rel = 1e-2) -- the ridge question is P6.5's",
        "E_ladder": E_LADDER,
        "n_splits_per_E": MAX_SPLITS, "split_seed": RNG,
        "PREREGISTERED_READOUT_SB_extrapolation_to_0.8": readout,
        "table": rel_rows,
        "reading": ("split_half_reliability is the Spearman between two DISJOINT E/2-member "
                    "sub-ensembles' per-demo score vectors -- i.e. the reliability of an "
                    "E/2-member ensemble. It is the attribution-side analogue of Phase-2 P4's "
                    "ground-truth reliability."),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p9_ensemble_cost_law.json"), out)

    print("\n" + "=" * 96)
    print("P9 -- ATTRIBUTION-SIDE COST LAW")
    print("=" * 96)
    print(f"{'attr':8s} {'E=2':>8s} {'E=5':>8s} {'E=10':>8s} {'E=20':>8s}   "
          f"{'E* for r=0.8 (EXTRAPOLATED)':>30s}")
    for attr in ATTRS:
        vals = [float(T[(T.E == e) & (T.attributor == attr)]
                      .split_half_reliability_mean_over_9_targets.iloc[0]) for e in E_LADDER]
        e_star = readout[attr]["SB_EXTRAPOLATED_E_for_0.8_from_E20"]
        print(f"{attr:8s} " + " ".join(f"{v:+8.3f}" for v in vals) +
              f"   {e_star:>30.1f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
