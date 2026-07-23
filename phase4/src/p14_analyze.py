"""P14 -- CONFIRMATORY HETEROGENEITY ON FRESH GROUND TRUTH.

Phase-3 P7 found C2 and C9 clear half-ceiling at some grains. But it found that DESCRIPTIVELY, on
the 24 Stage-G masks -- ground truth that had already been used for exploration, by a max over
three attributors and four grains. There is no confirmatory license in that. P14 buys one: 16
FRESH masks (seed 1104, verified to coincide with ZERO Stage-G masks), 4 fresh seeds, probes
restricted to C2's and C9's clusters. No analysis has ever touched this ground truth.

PRIMARY TEST (preregistered): the P11 CHAMPION, frozen identically -- TracIn, E=20, per-member
unit-L2 normalization -- on four cells: C2 at g in {1,3} and C9 at g in {1,5}. Criterion
rho >= 0.5 * measured 4-seed SB ceiling AND one-sided p < 0.0125 (Bonferroni-4).

DISCLOSED TENSION (in the preregistration, before any Phase-4 number existed): P7's qualifying
cells were qualified by a MAX OVER THREE ATTRIBUTORS. Under P7's primary rule (a),
    C2@g=1 qualified via TracIn (ratio 0.509)
    C2@g=3 qualified via **IF** (ratio 0.652) -- TracIn there scored only 0.277
    C9@g=1 qualified via TracIn (0.516)
    C9@g=5 qualified via TracIn (0.663)
so freezing the primary estimator to the champion makes C2@g=3 a GENUINELY ADVERSARIAL cell. We
preregister it anyway, because a per-cell attributor choice is exactly the multiple-comparison
move P6.5 exposed. The remedy is a SECOND declared family, not a post-hoc swap:

SECONDARY TEST (preregistered): per cell, the attributor that QUALIFIED it in P7 -- frozen from
phase3/results/p7_grain_ladder.csv, not from any Phase-4 datum. Bonferroni-8 (p < 0.00625).

POWER LIMIT, STATED UP FRONT: n = 16 masks. The critical rho is ~0.55 at alpha=0.0125 and ~0.59
at alpha=0.00625, while the half-ceiling bar will be ~0.45. SIGNIFICANCE binds, not the ratio.
The test is CONSERVATIVE: a FAIL here is WEAK evidence; a PASS is STRONG. This asymmetry is
restated wherever the verdict appears.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, P3_RESULTS, RESULTS

sys.path.insert(0, L.P3_SRC)
from p7_grain_ladder import coarse_pred, groups_random  # noqa: E402

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci  # noqa: E402

SEEDS = [411, 412, 413, 414]
MEMBERS = [f"ens_s{s}" for s in range(201, 221)]        # E = 20, the P11 champion ensemble
ALPHA_PRIMARY = 0.0125         # Bonferroni-4 (4 cells)
ALPHA_SECONDARY = 0.00625      # Bonferroni-8 (2 families x 4 cells)
OUTCOME = "neg_plain_loss"

# PREREGISTERED cells, and the P7-qualifying attributor for each (frozen from p7_grain_ladder.csv)
CELLS = [("C2", 1, "TracIn"), ("C2", 3, "IF"), ("C9", 1, "TracIn"), ("C9", 5, "TracIn")]


def critical_rho(n, alpha):
    """The smallest rho that reaches one-sided significance alpha at n masks (t approximation)."""
    tc = stats.t.isf(alpha, n - 2)
    return float(tc / np.sqrt(n - 2 + tc ** 2))


def ingest(jobs):
    rows, missing = [], []
    for j in jobs:
        rd = j["run_dir"]
        if not L.is_marked(rd, "probe"):
            missing.append(os.path.basename(rd))
            continue
        for c, v in L.read_outcomes(rd).items():
            rows.append({"run": os.path.basename(rd), "mask_id": j["mask_id"], "seed": j["seed"],
                         "target": c, "success_rate": v["success_rate"],
                         "n_episodes": v["n_episodes"], "plain_loss": v["plain_loss"],
                         "transport_loss": v["transport_loss"],
                         "interaction_loss": v["interaction_loss"]})
    return pd.DataFrame(rows), missing


def load_per_member():
    a = pd.read_parquet(os.path.join(RESULTS, "influence_table_per_member.parquet"))
    a = a[a.functional == "plain"][["attributor", "target", "demo_id", "member", "score"]]
    b = pd.read_parquet(os.path.join(P3_RESULTS, "p9_influence_new_members.parquet"))
    b = b[b.functional == "plain"][["attributor", "target", "demo_id", "member", "score"]]
    return pd.concat([a, b], ignore_index=True)


def main():
    L.assert_prereg_locked()
    man = json.load(open(os.path.join(P4_RESULTS, "p14_mask_manifest.json")))
    assert man["coincidence_check"]["PASS"], "P14 masks failed the Stage-G coincidence check"
    masks = man["masks"]
    jobs = json.load(open(os.path.join(P4_RESULTS, "p14_jobs.json")))

    df, missing = ingest(jobs)
    if missing:
        print(f"[P14] {len(missing)} runs not yet complete: {missing[:8]}")
        if len(missing) == len(jobs):
            raise SystemExit("[P14] nothing ingested -- training has not finished")
    df["neg_plain_loss"] = -df.plain_loss
    df["logit_success"] = L.logit_success_rowwise(df.success_rate.values, df.n_episodes.values)
    seeds = sorted(df.seed.unique().tolist())
    n_seed = df.groupby("mask_id").seed.nunique()
    if n_seed.min() != n_seed.max():
        raise RuntimeError(f"ragged seed coverage: {n_seed.to_dict()}")
    df.to_parquet(os.path.join(P4_RESULTS, "p14_outcomes.parquet"), index=False)
    print(f"[P14] ingested: {len(df)} rows, {df.mask_id.nunique()} fresh masks, seeds {seeds}, "
          f"targets {sorted(df.target.unique())}, {int(df.n_episodes.iloc[0])} episodes/target")

    # ---------------------------------------------------------------- ceilings (fresh, S=4)
    _, by_c = dataset.train_pool()
    pm = load_per_member()
    train_ids, _ = dataset.train_pool()

    ceil = {}
    for t in ("C2", "C9"):
        sub = df[df.target == t]
        piv = sub.pivot_table(index="mask_id", columns="seed", values=OUTCOME)
        r2, r4, splits, n_split = L.split_half_ceiling(piv, seeds, agg="mean")
        r1, _ = L.mean_pairwise_1seed_r(piv, seeds)
        ceil[t] = {"target": t, "n_masks": int(piv.dropna().shape[0]), "n_seeds": len(seeds),
                   "ceiling_2v2_splithalf_uncorrected": r2, "ceiling_4seed_SB": r4,
                   "n_splits": n_split, "r1_mean_pairwise": r1,
                   "predicted_4seed_SB_from_r1": L.sb(r1, len(seeds)),
                   "bar_half_ceiling": 0.5 * r4 if np.isfinite(r4) else np.nan,
                   "per_split": splits}

    n_masks = len(masks)
    crit_p = critical_rho(n_masks, ALPHA_PRIMARY)
    crit_s = critical_rho(n_masks, ALPHA_SECONDARY)
    print(f"\n[P14] fresh 4-seed ceilings: "
          f"C2 {ceil['C2']['ceiling_4seed_SB']:.3f} (bar {ceil['C2']['bar_half_ceiling']:.3f}), "
          f"C9 {ceil['C9']['ceiling_4seed_SB']:.3f} (bar {ceil['C9']['bar_half_ceiling']:.3f})")
    print(f"[P14] POWER LIMIT at n={n_masks}: critical rho = {crit_p:.3f} (primary, "
          f"alpha={ALPHA_PRIMARY}) / {crit_s:.3f} (secondary, alpha={ALPHA_SECONDARY})")

    # ---------------------------------------------------------------- the four cells
    cells, rows = {}, []
    for t, g, qual_attr in CELLS:
        obs = df[df.target == t].groupby("mask_id")[OUTCOME].mean()
        out_v = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
        c4 = ceil[t]["ceiling_4seed_SB"]
        # P7's PRIMARY grouping rule (a), reused verbatim (same function, same seed 707)
        groups = [gr for c in dataset.clusters() for gr in groups_random(by_c[c], g)]
        assert all(len(gr) == g for gr in groups), f"g={g}: unequal groups"

        entry = {"target": t, "grain": g, "ceiling_4seed_SB": c4, "bar": 0.5 * c4,
                 "n_masks": n_masks, "arms": {}}
        for arm, attr, alpha in (("PRIMARY_champion", "TracIn", ALPHA_PRIMARY),
                                 ("SECONDARY_p7_qualifier", qual_attr, ALPHA_SECONDARY)):
            sc = L.normalized_ensemble_scores(pm, attr, t, train_ids, MEMBERS, normalize=True)
            pred = np.array([coarse_pred(m["demos"], groups, sc, g) for m in masks])
            ok = np.isfinite(out_v) & np.isfinite(pred)
            rho = spearman(pred[ok], out_v[ok])
            n = int(ok.sum())
            p1 = spearman_p_onesided(rho, n)
            lo, hi = bootstrap_spearman_ci(pred[ok], out_v[ok])
            meets = L.meets_half_ceiling(rho, c4)
            sig = bool(np.isfinite(p1) and p1 < alpha)
            e = {"arm": arm, "attributor": attr, "alpha_bonferroni": alpha, "rho": rho,
                 "n_masks": n, "p_onesided": p1, "ci95": [lo, hi],
                 "ratio_to_ceiling": (rho / c4) if (np.isfinite(c4) and c4) else np.nan,
                 "ratio_exact": L.ratio_exact_str(rho, c4),
                 "meets_half_ceiling_EXACT": meets, "p_lt_alpha": sig,
                 "PASS": bool(meets and sig),
                 "critical_rho_at_this_alpha": critical_rho(n, alpha)}
            entry["arms"][arm] = e
            rows.append({"target": t, "grain": g, "arm": arm, "attributor": attr, "rho": rho,
                         "ceiling": c4, "bar": 0.5 * c4, "ratio": e["ratio_to_ceiling"],
                         "p_onesided": p1, "alpha": alpha, "PASS": e["PASS"]})
        cells[f"{t}_g{g}"] = entry
    pd.DataFrame(rows).to_csv(os.path.join(P4_RESULTS, "p14_lds_table.csv"), index=False)

    # ---------------------------------------------------------------- VERDICT
    prim_pass = [k for k, v in cells.items() if v["arms"]["PRIMARY_champion"]["PASS"]]
    seco_pass = [k for k, v in cells.items() if v["arms"]["SECONDARY_p7_qualifier"]["PASS"]]

    INTERP_PASS = ("the paper upgrades from 'attribution fails at demo grain' to 'attribution "
                   "faithfulness is target-dependent, confirmed out-of-sample -- and here is the "
                   "moderator question that opens.'")
    INTERP_FAIL = ("the P7 heterogeneity is flagged as likely exploration noise.")
    POWER = ("POWER LIMIT (preregistered, stated up front): n=16 masks; critical rho ~0.55 "
             "(primary) / ~0.59 (secondary) while the bar is ~0.45, so SIGNIFICANCE binds, not "
             "the ratio. This test is CONSERVATIVE: a FAIL is WEAK evidence, a PASS is STRONG.")

    out = {
        "stage": "P14", "preregistration_sha256": L.PREREG_SHA,
        "fresh_masks": {"K": n_masks, "mask_seed": man["mask_seed_used"],
                        "coincidence_with_stage_G": man["coincidence_check"]["n_coinciding"],
                        "manifest": os.path.join(P4_RESULTS, "p14_mask_manifest.json")},
        "seeds": seeds, "S": len(seeds), "n_retrains": len(jobs), "n_missing": len(missing),
        "probes": "restricted to C2 and C9 clusters (6 tasks x 10 rollouts = 60 episodes/model)",
        "PRIMARY_OUTCOME": OUTCOME, "aggregator": "seed MEAN",
        "PRIMARY_ESTIMATOR": "the P11 CHAMPION, frozen identically: TracIn, E=20, per-member "
                             "unit-L2 normalization",
        "SECONDARY_ESTIMATOR": "per cell, the P7-qualifying attributor (frozen from "
                               "phase3/results/p7_grain_ladder.csv): " +
                               ", ".join(f"{t}@g{g}->{a}" for t, g, a in CELLS),
        "grouping_rule": "P7 PRIMARY rule (a): random within cluster, default_rng(707), blocks of g",
        "criterion_primary": "rho >= 0.5 * measured 4-seed SB ceiling AND p < 0.0125 (Bonferroni-4)",
        "criterion_secondary": "same, at p < 0.00625 (Bonferroni-8)",
        "ceilings": ceil,
        "POWER_LIMIT": POWER,
        "critical_rho_primary": crit_p, "critical_rho_secondary": crit_s,
        "cells": cells,
        "PRIMARY_cells_passing": prim_pass,
        "SECONDARY_cells_passing": seco_pass,
        "VERDICT": "PASS" if prim_pass else "FAIL",
        "PREREGISTERED_INTERPRETATION": INTERP_PASS if prim_pass else INTERP_FAIL,
    }
    L.atomic_write_json(os.path.join(P4_RESULTS, "p14_verdict.json"), out)

    print("\n" + "=" * 110)
    print(f"P14 -- CONFIRMATORY HETEROGENEITY, FRESH GROUND TRUTH "
          f"({n_masks} new masks, S={len(seeds)})")
    print("=" * 110)
    print(f"{'cell':10s} {'ceiling':>8s} {'bar':>6s} | {'PRIMARY (champion=TracIn)':>32s} | "
          f"{'SECONDARY (P7 qualifier)':>32s}")
    for t, g, qa in CELLS:
        e = cells[f"{t}_g{g}"]
        p, s = e["arms"]["PRIMARY_champion"], e["arms"]["SECONDARY_p7_qualifier"]
        print(f"{t}@g={g:<5d} {e['ceiling_4seed_SB']:8.3f} {e['bar']:6.3f} | "
              f"rho={p['rho']:+.3f} r={p['ratio_to_ceiling']:+.2f} p={p['p_onesided']:.4f} "
              f"{'PASS' if p['PASS'] else 'fail'} | "
              f"{s['attributor']:6s} rho={s['rho']:+.3f} r={s['ratio_to_ceiling']:+.2f} "
              f"p={s['p_onesided']:.4f} {'PASS' if s['PASS'] else 'fail'}")
    print("-" * 110)
    print(f"P14 PRIMARY VERDICT: {'PASS on ' + ', '.join(prim_pass) if prim_pass else 'FAIL on all 4 cells'}")
    print(f"  -> {out['PREREGISTERED_INTERPRETATION']}")
    print(f"P14 SECONDARY: {'PASS on ' + ', '.join(seco_pass) if seco_pass else 'FAIL on all 4 cells'}")
    print(f"\n{POWER}")
    print("=" * 110)


if __name__ == "__main__":
    main()
