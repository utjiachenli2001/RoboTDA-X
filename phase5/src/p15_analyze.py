"""P15 -- DIFFUSION C1 REPLICATION AT S=10, with a STRICTER instrument gate.

P13's diffusion C1 PASS rested on an instrument that cleared its usability gate by only 0.003
(S=4->8 SB-consistency 0.147 vs a 0.15 tolerance), and a descriptive variant of that check would
have FAILED. Phase 4 §8: the result "should be replicated at greater S before it carries weight."

This stage deepens the diffusion ground truth to S=10 (seeds 601-608 + 609,610), firms the gate,
and gives the pass (or its failure) full weight. C5 is NOT re-tested -- it FAILED on a SOUND
instrument in P13 and is settled.

THE STRICTER GATE (preregistered, computed BEFORE any LDS, C1 only). C1 is USABLE iff ALL of:
  (i)   median SB-corrected ceiling (126 5v5 median split-halves, SB x2) >= 0.40
  (ii)  |SB(r1 over seeds 601-604, depth 10) - measured median SB ceiling| <= 0.15   (S=4 -> 10)
  (iii) |SB(r1 over seeds 601-608, depth 10) - measured median SB ceiling| <= 0.10   (S=8 -> 10)
Stricter than P13's C1 gate: deeper ceiling (126 vs 35 splits) AND a third check (iii) at a
tighter 0.10 tolerance. Gate failure => INSTRUMENT-UNUSABLE => the preregistered DOWNGRADE.

VERDICT (C1 only, champion identical to P13 bit-for-bit): TracIn, E=5 diffusion ensemble
(dpens_s621-625), frozen (t,eps) bank (sha 61aadccfef2f...), per-member unit-L2 normalization.
Criterion: rho >= 0.5 * S=10 median SB ceiling AND one-sided p < 0.025, n=24, exact rational.

SB CAVEAT: Spearman-Brown is EXACT for means and APPROXIMATE for medians. Both the uncorrected 5v5
and the SB-corrected value are reported; the bar uses the SB-corrected (higher) value.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5lib as L
from p5lib import P5_RESULTS, P4_RESULTS, P3_RESULTS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci, mask_pred_score  # noqa: E402

SEEDS_OLD = [601, 602, 603, 604, 605, 606, 607, 608]      # from p13_outcomes_S8.parquet
SEEDS_S4 = [601, 602, 603, 604]        # ORIGINAL preregistered S=4 SB base
SEEDS_S8 = [601, 602, 603, 604, 605, 606, 607, 608]       # S=8 SB base (tighter check)
SEEDS_NEW = [609, 610]                 # PREREGISTERED
FOCAL_VERDICT = "C1"                   # C1 ONLY (C5 settled, out of scope)
MEMBERS = [f"dpens_s{s}" for s in range(621, 626)]        # E = 5
ALPHA = 0.025
CEIL_MIN = 0.40                        # gate (i)
SB_TOL_S4 = 0.15                       # gate (ii)
SB_TOL_S8 = 0.10                       # gate (iii)  -- STRICTER
OUTCOME = "neg_plain_loss"
BASE_COLS = ["run", "mask_id", "seed", "target", "success_rate", "n_episodes",
             "plain_loss", "denoise_loss"]


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
                         "denoise_loss": v.get("denoise_loss", np.nan)})
    return pd.DataFrame(rows), missing


def main():
    L.assert_prereg_locked()
    clusters = dataset.clusters()
    masks = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]

    jobs = json.load(open(os.path.join(P5_RESULTS, "p15_jobs.json")))
    new, missing = ingest(jobs)
    if missing:
        print(f"[P15] {len(missing)} runs not yet complete: {missing[:8]}")
        if len(missing) == len(jobs):
            raise SystemExit("[P15] nothing ingested -- training has not finished")
        raise SystemExit(f"[P15] {len(missing)}/{len(jobs)} runs incomplete -- refusing partial "
                         f"verdict")

    old = pd.read_parquet(os.path.join(P4_RESULTS, "p13_outcomes_S8.parquet"))[BASE_COLS]
    df = pd.concat([old, new[BASE_COLS]], ignore_index=True)
    df["neg_plain_loss"] = -df.plain_loss
    df["neg_denoise_loss"] = -df.denoise_loss
    SEEDS = SEEDS_OLD + sorted(new.seed.unique().tolist())
    # NOTE: plain_loss (the verdict outcome) exists for ALL 9 targets at ALL seeds; success_rate is
    # present for all 9 at S<=8 (full battery) but only for C1 at S=9,10 (preregistered economy).
    npl = df.groupby(["mask_id", "target"])[OUTCOME].count()
    if npl.min() != len(SEEDS):
        raise RuntimeError(f"ragged {OUTCOME} coverage: min={npl.min()} != S={len(SEEDS)}")
    df.to_parquet(os.path.join(P5_RESULTS, "p15_outcomes_S10.parquet"), index=False)
    print(f"[P15] merged: {len(df)} rows, {df.mask_id.nunique()} masks, seeds {SEEDS} "
          f"(S={len(SEEDS)})")

    # ================================================================ 1. CEILINGS + THE GATE
    ceil = {}
    for t in clusters:
        sub = df[df.target == t]
        piv = sub.pivot_table(index="mask_id", columns="seed", values=OUTCOME)
        r5_med, r10_med, splits_med, n_split = L.split_half_ceiling(piv, SEEDS, agg="median")
        r5_mean, r10_mean, _, _ = L.split_half_ceiling(piv, SEEDS, agg="mean")

        r1_s4, np4, pred10_s4 = L.sb_consistency(piv, SEEDS_S4, len(SEEDS))
        r1_s8, np8, pred10_s8 = L.sb_consistency(piv, SEEDS_S8, len(SEEDS))
        r1_all, npall, pred10_all = L.sb_consistency(piv, SEEDS, len(SEEDS))

        diff_s4 = abs(pred10_s4 - r10_med)
        diff_s8 = abs(pred10_s8 - r10_med)
        gate_i = bool(np.isfinite(r10_med) and r10_med >= CEIL_MIN)
        gate_ii = bool(np.isfinite(diff_s4) and diff_s4 <= SB_TOL_S4)
        gate_iii = bool(np.isfinite(diff_s8) and diff_s8 <= SB_TOL_S8)
        usable = gate_i and gate_ii and gate_iii

        ceil[t] = {
            "target": t, "focal_verdict": t == FOCAL_VERDICT,
            "ceiling_median_5v5_uncorrected": r5_med,
            "ceiling_median_10seed_SB": r10_med,
            "n_splits": n_split,
            "ceiling_MEAN_10seed_SB_for_contrast": r10_mean,
            "ceiling_MEAN_5v5_uncorrected": r5_mean,
            "r1_from_S4_601_604": r1_s4, "n_pairs_S4": np4,
            "predicted_10seed_SB_from_S4": pred10_s4, "SB_consistency_S4to10_abs_diff": diff_s4,
            "r1_from_S8_601_608": r1_s8, "n_pairs_S8": np8,
            "predicted_10seed_SB_from_S8": pred10_s8, "SB_consistency_S8to10_abs_diff": diff_s8,
            "r1_all_10_seeds_DESCRIPTIVE": r1_all,
            "predicted_10seed_SB_from_all_DESCRIPTIVE": pred10_all,
            "SB_consistency_all_to_10_abs_diff_DESCRIPTIVE": abs(pred10_all - r10_med),
            "GATE_i_ceiling_ge_0.40": gate_i,
            "GATE_ii_S4to10_within_0.15": gate_ii,
            "GATE_iii_S8to10_within_0.10": gate_iii,
            "CEILING_USABLE": usable,
            "bar_half_ceiling": 0.5 * r10_med if np.isfinite(r10_med) else np.nan,
            "per_split": splits_med,
            "SB_caveat": "Spearman-Brown is EXACT for means, APPROXIMATE for medians",
        }

    print("\n" + "=" * 118)
    print(f"P15 -- STRICTER CEILING-USABILITY GATE (median aggregator, S={len(SEEDS)}) "
          f"-- computed BEFORE any LDS (verdict target: {FOCAL_VERDICT})")
    print("=" * 118)
    print(f"{'target':7s} {'5v5(med)':>9s} {'SB10(med)':>10s} {'SB10(mean)':>11s} "
          f"{'|S4->10|':>9s} {'|S8->10|':>9s} {'(i)':>5s} {'(ii)':>5s} {'(iii)':>6s}  usable")
    for t in clusters:
        e = ceil[t]
        print(f"{t:7s} {e['ceiling_median_5v5_uncorrected']:9.3f} "
              f"{e['ceiling_median_10seed_SB']:10.3f} {e['ceiling_MEAN_10seed_SB_for_contrast']:11.3f} "
              f"{e['SB_consistency_S4to10_abs_diff']:9.3f} {e['SB_consistency_S8to10_abs_diff']:9.3f} "
              f"{str(e['GATE_i_ceiling_ge_0.40']):>5s} {str(e['GATE_ii_S4to10_within_0.15']):>5s} "
              f"{str(e['GATE_iii_S8to10_within_0.10']):>6s}  "
              f"{'USABLE' if e['CEILING_USABLE'] else 'UNUSABLE'}"
              f"{'  <-- VERDICT TARGET' if e['focal_verdict'] else ''}")
    print("=" * 118)

    # ================================================================ 2. attribution (champion)
    pm = pd.read_parquet(os.path.join(P3_RESULTS, "p10_influence_per_member.parquet"))
    train_ids, _ = dataset.train_pool()
    bank_path = os.path.join(P3_RESULTS, "p10_noise_bank.json")
    bank_sha = L.sha256_file(bank_path)
    EXPECT_BANK_SHA = "61aadccfef2fb45300d611f262bdc285c6a8f9888ed907a1c48b112d8405bc17"
    if bank_sha != EXPECT_BANK_SHA:
        raise RuntimeError(f"noise bank sha mismatch: {bank_sha} != {EXPECT_BANK_SHA}")
    print(f"[P15] frozen noise bank verified (sha {bank_sha[:12]}...)")

    ATTRS = [("TracIn", True, "CHAMPION"), ("TracIn", False, "descriptive"),
             ("TRAK", True, "descriptive"), ("IF", True, "descriptive")]
    CH = "TracIn_diffE5_normalized"

    res, rows = {}, []
    for t in clusters:
        sub = df[df.target == t]
        piv = sub.pivot_table(index="mask_id", columns="seed", values=OUTCOME)[SEEDS]
        med = piv.median(axis=1)
        out_v = np.array([med.get(m["mask_id"], np.nan) for m in masks])
        res[t] = {**ceil[t], "estimators": {}}
        for attr, norm, fam in ATTRS:
            eid = f"{attr}_diffE5_{'normalized' if norm else 'unnormalized'}"
            sc = L.normalized_ensemble_scores(pm, attr, t, train_ids, MEMBERS, normalize=norm)
            pred = np.array([mask_pred_score(sc, m["demos"]) for m in masks])
            ok = np.isfinite(out_v) & np.isfinite(pred)
            rho = spearman(pred[ok], out_v[ok])
            n = int(ok.sum())
            p1 = spearman_p_onesided(rho, n)
            lo, hi = bootstrap_spearman_ci(pred[ok], out_v[ok])
            c10 = ceil[t]["ceiling_median_10seed_SB"]
            meets = L.meets_half_ceiling(rho, c10)
            sig = bool(np.isfinite(p1) and p1 < ALPHA)
            e = {"estimator": eid, "family": fam, "rho": rho, "n_masks": n, "p_onesided": p1,
                 "ci95": [lo, hi],
                 "ratio_to_ceiling": (rho / c10) if (np.isfinite(c10) and c10) else np.nan,
                 "ratio_exact": L.ratio_exact_str(rho, c10),
                 "meets_half_ceiling_EXACT": meets, "p_lt_0.025": sig,
                 "PASS": bool(meets and sig) if (fam == "CHAMPION" and t == FOCAL_VERDICT)
                 else None}
            res[t]["estimators"][eid] = e
            rows.append({"target": t, "verdict_target": t == FOCAL_VERDICT, "estimator": eid,
                         "family": fam, "rho": rho, "ceiling_median_SB": c10, "bar": 0.5 * c10,
                         "ratio": e["ratio_to_ceiling"], "p_onesided": p1,
                         "ceiling_usable": ceil[t]["CEILING_USABLE"], "PASS": e["PASS"]})
    pd.DataFrame(rows).to_csv(os.path.join(P5_RESULTS, "p15_lds_table.csv"), index=False)

    # ================================================================ 3. THE GATED VERDICT (C1)
    t = FOCAL_VERDICT
    e = res[t]["estimators"][CH]
    if not ceil[t]["CEILING_USABLE"]:
        reason = ("median ceiling < 0.40" if not ceil[t]["GATE_i_ceiling_ge_0.40"]
                  else "S=4->10 SB-consistency > 0.15" if not ceil[t]["GATE_ii_S4to10_within_0.15"]
                  else "S=8->10 SB-consistency > 0.10")
        verdict = {"VERDICT": "INSTRUMENT-UNUSABLE", "reason": reason,
                   "champion_rho_REPORTED_NOT_A_VERDICT": e["rho"]}
        overall = "INSTRUMENT-UNUSABLE (C1)"
        interp = ("P13's C1 PASS is downgraded in the paper to 'instrument-marginal, unresolved' "
                  "-- reported, not celebrated.")
    else:
        verdict = {"VERDICT": "PASS" if e["PASS"] else "FAIL",
                   "ceiling_median_10seed_SB": ceil[t]["ceiling_median_10seed_SB"],
                   "bar": 0.5 * ceil[t]["ceiling_median_10seed_SB"], "champion": e}
        overall = f"{verdict['VERDICT']} (C1)"
        if e["PASS"]:
            interp = ("the diffusion C1 result carries full weight: demo-grain faithfulness is "
                      "policy-class-dependent and the BC null does not generalize -- now "
                      "replicated at two seed depths on a sound instrument.")
        else:
            interp = ("P13's C1 PASS did not replicate at deeper S and is reported as fragile; the "
                      "external-validity question re-opens.")

    # ---- seed-MEAN brokenness series (descriptive): S=6 (Phase3), S=8 (Phase4), S=10 (here)
    mean_series = {
        "C1": {"S6_MEAN_SB_ceiling": None, "S8_MEAN_SB_ceiling":
               float(ceil["C1"]["ceiling_MEAN_10seed_SB_for_contrast"])},
    }
    try:
        p13 = json.load(open(os.path.join(P4_RESULTS, "p13_verdict.json")))
        for t2 in clusters:
            mean_series.setdefault(t2, {})
            mean_series[t2]["S8_MEAN_SB_ceiling"] = float(
                p13["all_targets_DESCRIPTIVE"][t2]["ceiling_MEAN_8seed_SB_for_contrast"])
            mean_series[t2]["S10_MEAN_SB_ceiling"] = float(
                ceil[t2]["ceiling_MEAN_10seed_SB_for_contrast"])
            mean_series[t2]["S10_MEDIAN_SB_ceiling"] = float(ceil[t2]["ceiling_median_10seed_SB"])
    except Exception as ex:
        mean_series["_note"] = f"S8 series read error: {ex}"

    out = {
        "stage": "P15", "preregistration_sha256": L.PREREG_SHA,
        "seeds": SEEDS, "S": len(SEEDS), "n_masks": len(masks),
        "PREREGISTERED_AGGREGATOR": "MEDIAN over seeds of the held-out executed-action L2",
        "PRIMARY_OUTCOME": OUTCOME,
        "VERDICT_TARGET": FOCAL_VERDICT,
        "CHAMPION": ("TracIn, E=5 diffusion ensemble (dpens_s621-625), frozen paired (t,eps) "
                     "noise bank, per-member unit-L2 normalization -- IDENTICAL to P13"),
        "noise_bank_sha256": bank_sha,
        "SB_caveat": "Spearman-Brown is EXACT for means and APPROXIMATE for medians; the bar uses "
                     "the SB-corrected value",
        "STRICTER_GATE": {
            "definition": ("(i) median SB ceiling >= 0.40 AND (ii) |SB(r1[601-604],10) - ceiling| "
                           "<= 0.15 AND (iii) |SB(r1[601-608],10) - ceiling| <= 0.10"),
            "ceiling_min": CEIL_MIN, "sb_tol_S4to10": SB_TOL_S4, "sb_tol_S8to10": SB_TOL_S8,
            "C1": {k: ceil["C1"][k] for k in (
                "ceiling_median_10seed_SB", "SB_consistency_S4to10_abs_diff",
                "SB_consistency_S8to10_abs_diff", "SB_consistency_all_to_10_abs_diff_DESCRIPTIVE",
                "GATE_i_ceiling_ge_0.40", "GATE_ii_S4to10_within_0.15",
                "GATE_iii_S8to10_within_0.10", "CEILING_USABLE")},
        },
        "criterion": "C1: rho >= 0.5 * median SB ceiling AND one-sided p < 0.025, n=24, exact rational",
        "C1_VERDICT": verdict,
        "VERDICT": overall,
        "PREREGISTERED_INTERPRETATION": interp,
        "seed_mean_brokenness_series": mean_series,
        "all_targets_DESCRIPTIVE": res,
        "n_new_retrains": len(jobs), "n_missing": len(missing),
    }
    L.atomic_write_json(os.path.join(P5_RESULTS, "p15_verdict.json"), out)

    print("\n" + "=" * 118)
    print(f"P15 -- DIFFUSION C1 REPLICATION (median aggregator, S={len(SEEDS)}, n=24 masks)")
    print("=" * 118)
    c10 = ceil[t]["ceiling_median_10seed_SB"]
    print(f"C1 ceiling(med,SB10)={c10:.3f} bar={0.5*c10:.3f} usable={ceil[t]['CEILING_USABLE']} "
          f"| CHAMPION rho={e['rho']:+.3f} ratio={e['ratio_to_ceiling']:.2f} p={e['p_onesided']:.4f}")
    print(f"P15 VERDICT: {overall}")
    print(f"  -> {interp}")
    print("-" * 118)
    print("seed-MEAN brokenness series (descriptive):")
    for t2 in ("C1", "C2", "C9"):
        s = mean_series.get(t2, {})
        print(f"  {t2}: MEAN SB ceiling  S8={s.get('S8_MEAN_SB_ceiling')}  "
              f"S10={s.get('S10_MEAN_SB_ceiling')}   (MEDIAN S10={s.get('S10_MEDIAN_SB_ceiling')})")
    print("=" * 118)


if __name__ == "__main__":
    main()
