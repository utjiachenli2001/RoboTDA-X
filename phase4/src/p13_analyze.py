"""P13 -- THE DIFFUSION SETTLEMENT. The follow-up PHASE3_REPORT §6.3 demands, run exactly.

Phase 3 could not answer the external-validity question because the INSTRUMENT broke, not the
attributor. The diffusion policy is deterministic given its weights but NOT seed-stable: its
executed action is a 5-step DDIM trajectory from a fixed latent through a MULTIMODAL denoiser, so
an occasional seed lands that latent in the wrong basin and its held-out L2 blows up 3-5x. The
seed MEAN of a heavy-tailed outcome is a broken estimator -- 4 of 9 mean ceilings came out
NEGATIVE, the SB consistency check missed by 0.61, and reliability DROPPED as seeds were added.
The median restored ceilings to 0.57-0.77, but it was a POST-HOC aggregator, so Phase 3 refused
to call it a result and demanded exactly three things of a follow-up:

    "A follow-up must preregister the median aggregator, S >= 6, and an up-front
     ceiling-usability gate."                                        -- PHASE3_REPORT §6.3

This stage is that follow-up. All three are preregistered (sha fca37f54...):

  1. AGGREGATOR: the MEDIAN over seeds of the held-out executed-action L2 per mask. Declared
     BEFORE any S=8 datum existed.
  2. S = 8: seeds 601-606 (Phase 3) + 607, 608 (Phase 4), on the same 24 masks, frozen config.
  3. UP-FRONT CEILING-USABILITY GATE, evaluated BEFORE any LDS is computed for a target:
        (a) measured median SB-corrected ceiling >= 0.40, AND
        (b) the S=4 -> 8 SB-consistency check holds within 0.15
     A target failing either is INSTRUMENT-UNUSABLE -- NOT a FAIL. The distinction is the whole
     point of the stage: Phase 3's mean-aggregated FAIL was uninformative precisely because it
     was a measurement failure wearing a verdict's clothes.

SB CAVEAT (disclosed in the preregistration, restated here): Spearman-Brown is EXACT for means
and only APPROXIMATE for medians. Both the uncorrected 4v4 split-half and the SB-corrected value
are reported; the BAR uses the SB-corrected (higher, conservative-for-passing) value.

CHAMPION: TracIn over the E=5 diffusion ensemble with the FROZEN paired (t, epsilon) noise bank,
per-member unit-L2 normalization. TracIn's split-half reliability across those members is 0.927
(p10_attr_stability.json) -- it is the only attributor with consistent signal for this class. No
sweeps.

CRITERION: focal C1/C5, rho >= 0.5 * median ceiling, one-sided p < 0.025 (Bonferroni-2).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, P3_RESULTS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci, mask_pred_score  # noqa: E402

SEEDS_OLD = [601, 602, 603, 604, 605, 606]
SEEDS_S4 = [601, 602, 603, 604]        # the ORIGINAL preregistered S=4 set (the SB base)
SEEDS_NEW = [607, 608]
FOCAL = ["C1", "C5"]
MEMBERS = [f"dpens_s{s}" for s in range(621, 626)]      # E = 5
ALPHA = 0.025                                          # Bonferroni-2
CEIL_MIN = 0.40                                        # PREREGISTERED gate (a)
SB_TOL = 0.15                                          # PREREGISTERED gate (b)
OUTCOME = "neg_plain_loss"


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

    jobs = json.load(open(os.path.join(P4_RESULTS, "p13_jobs.json")))
    new, missing = ingest(jobs)
    if missing:
        print(f"[P13] {len(missing)} runs not yet complete: {missing[:8]}")
        if len(missing) == len(jobs):
            raise SystemExit("[P13] nothing ingested -- training has not finished")

    old = pd.read_parquet(os.path.join(P3_RESULTS, "p10b_outcomes_S6.parquet"))
    old = old[["run", "mask_id", "seed", "target", "success_rate", "n_episodes", "plain_loss",
               "denoise_loss"]]
    df = pd.concat([old, new], ignore_index=True)
    df["logit_success"] = L.logit_success_rowwise(df.success_rate.values, df.n_episodes.values)
    df["neg_plain_loss"] = -df.plain_loss
    df["neg_denoise_loss"] = -df.denoise_loss
    SEEDS = SEEDS_OLD + sorted(new.seed.unique().tolist())
    n_seed = df.groupby("mask_id").seed.nunique()
    if n_seed.min() != n_seed.max():
        raise RuntimeError(f"ragged seed coverage: {n_seed.to_dict()}")
    df.to_parquet(os.path.join(P4_RESULTS, "p13_outcomes_S8.parquet"), index=False)
    print(f"[P13] merged: {len(df)} rows, {df.mask_id.nunique()} masks, seeds {SEEDS} (S={len(SEEDS)})")

    # ================================================================ 1. CEILINGS + THE GATE
    # Computed BEFORE any LDS. The gate decides whether a target may HAVE a verdict at all.
    ceil = {}
    for t in clusters:
        sub = df[df.target == t]
        piv = sub.pivot_table(index="mask_id", columns="seed", values=OUTCOME)

        r4_med, r8_med, splits_med, n_split = L.split_half_ceiling(piv, SEEDS, agg="median")
        r4_mean, r8_mean, _, _ = L.split_half_ceiling(piv, SEEDS, agg="mean")

        # the SB base: SINGLE-seed reliability on the ORIGINAL S=4 seeds only (aggregator-free)
        r1_s4, pairs_s4 = L.mean_pairwise_1seed_r(piv, SEEDS_S4)
        pred8_from_s4 = L.sb(r1_s4, len(SEEDS))
        # descriptive: the same, based on all C(8,2)=28 pairs
        r1_all, pairs_all = L.mean_pairwise_1seed_r(piv, SEEDS)
        pred8_from_all = L.sb(r1_all, len(SEEDS))

        sb_diff = abs(pred8_from_s4 - r8_med)
        gate_a = bool(np.isfinite(r8_med) and r8_med >= CEIL_MIN)
        gate_b = bool(np.isfinite(sb_diff) and sb_diff <= SB_TOL)
        usable = gate_a and gate_b

        ceil[t] = {
            "target": t, "focal": t in FOCAL,
            "ceiling_median_4v4_uncorrected": r4_med,
            "ceiling_median_8seed_SB": r8_med,
            "n_splits": n_split,
            "ceiling_MEAN_8seed_SB_for_contrast": r8_mean,
            "r1_from_S4_only": r1_s4, "n_pairs_S4": len(pairs_s4),
            "predicted_8seed_SB_from_S4": pred8_from_s4,
            "SB_consistency_abs_diff": sb_diff,
            "r1_all_8_seeds_DESCRIPTIVE": r1_all,
            "predicted_8seed_SB_from_all_DESCRIPTIVE": pred8_from_all,
            "GATE_a_ceiling_ge_0.40": gate_a,
            "GATE_b_SB_consistency_within_0.15": gate_b,
            "CEILING_USABLE": usable,
            "bar_half_ceiling": 0.5 * r8_med if np.isfinite(r8_med) else np.nan,
            "per_split": splits_med,
            "SB_caveat": "Spearman-Brown is EXACT for means, APPROXIMATE for medians",
        }

    print("\n" + "=" * 108)
    print(f"P13 -- UP-FRONT CEILING-USABILITY GATE (median aggregator, S={len(SEEDS)}) "
          f"-- computed BEFORE any LDS")
    print("=" * 108)
    print(f"{'target':7s} {'4v4(med)':>9s} {'SB8(med)':>9s} {'SB8(mean)':>10s} {'r1(S4)':>7s} "
          f"{'pred8':>7s} {'|diff|':>7s} {'>=0.40':>7s} {'<=0.15':>7s}  usable")
    for t in clusters:
        e = ceil[t]
        print(f"{t:7s} {e['ceiling_median_4v4_uncorrected']:9.3f} "
              f"{e['ceiling_median_8seed_SB']:9.3f} {e['ceiling_MEAN_8seed_SB_for_contrast']:10.3f} "
              f"{e['r1_from_S4_only']:7.3f} {e['predicted_8seed_SB_from_S4']:7.3f} "
              f"{e['SB_consistency_abs_diff']:7.3f} "
              f"{str(e['GATE_a_ceiling_ge_0.40']):>7s} {str(e['GATE_b_SB_consistency_within_0.15']):>7s}  "
              f"{'USABLE' if e['CEILING_USABLE'] else 'UNUSABLE'}"
              f"{'  <-- FOCAL' if e['focal'] else ''}")
    print("=" * 108)

    # ================================================================ 2. attribution (champion)
    pm = pd.read_parquet(os.path.join(P3_RESULTS, "p10_influence_per_member.parquet"))
    train_ids, _ = dataset.train_pool()
    bank = json.load(open(os.path.join(P3_RESULTS, "p10_noise_bank.json")))
    print(f"[P13] frozen noise bank: {os.path.join(P3_RESULTS, 'p10_noise_bank.json')} "
          f"(sha {L.sha256_file(os.path.join(P3_RESULTS, 'p10_noise_bank.json'))[:12]}...), "
          f"keys={list(bank)[:6]}")

    ATTRS = [("TracIn", True, "CHAMPION"), ("TracIn", False, "descriptive"),
             ("TRAK", True, "descriptive"), ("IF", True, "descriptive")]

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
            c8 = ceil[t]["ceiling_median_8seed_SB"]
            meets = L.meets_half_ceiling(rho, c8)
            sig = bool(np.isfinite(p1) and p1 < ALPHA)
            e = {"estimator": eid, "family": fam, "rho": rho, "n_masks": n, "p_onesided": p1,
                 "ci95": [lo, hi],
                 "ratio_to_ceiling": (rho / c8) if (np.isfinite(c8) and c8) else np.nan,
                 "ratio_exact": L.ratio_exact_str(rho, c8),
                 "meets_half_ceiling_EXACT": meets, "p_lt_0.025": sig,
                 "PASS": bool(meets and sig) if fam == "CHAMPION" else None}
            res[t]["estimators"][eid] = e
            rows.append({"target": t, "focal": t in FOCAL, "estimator": eid, "family": fam,
                         "rho": rho, "ceiling_median_SB": c8, "bar": 0.5 * c8,
                         "ratio": e["ratio_to_ceiling"], "p_onesided": p1,
                         "ceiling_usable": ceil[t]["CEILING_USABLE"], "PASS": e["PASS"]})
    pd.DataFrame(rows).to_csv(os.path.join(P4_RESULTS, "p13_lds_table.csv"), index=False)

    # ================================================================ 3. THE GATED VERDICT
    CH = "TracIn_diffE5_normalized"
    focal_verdict = {}
    for t in FOCAL:
        e = res[t]["estimators"][CH]
        if not ceil[t]["CEILING_USABLE"]:
            focal_verdict[t] = {"VERDICT": "INSTRUMENT-UNUSABLE",
                                "reason": ("median ceiling < 0.40" if not ceil[t]["GATE_a_ceiling_ge_0.40"]
                                           else "S=4->8 SB-consistency check missed by > 0.15"),
                                "ceiling_median_8seed_SB": ceil[t]["ceiling_median_8seed_SB"],
                                "SB_consistency_abs_diff": ceil[t]["SB_consistency_abs_diff"],
                                "champion_rho_REPORTED_NOT_A_VERDICT": e["rho"]}
        else:
            focal_verdict[t] = {"VERDICT": "PASS" if e["PASS"] else "FAIL",
                                "ceiling_median_8seed_SB": ceil[t]["ceiling_median_8seed_SB"],
                                "bar": 0.5 * ceil[t]["ceiling_median_8seed_SB"],
                                "champion": e}

    vs = {t: focal_verdict[t]["VERDICT"] for t in FOCAL}
    usable = [t for t in FOCAL if vs[t] != "INSTRUMENT-UNUSABLE"]
    passed = [t for t in FOCAL if vs[t] == "PASS"]

    if "C1" in passed:
        interp = ("demo-grain faithfulness is policy-class-dependent, and the BC null does not "
                  "generalize.")
        overall = "PASS (C1)"
    elif len(usable) == 2 and not passed:
        interp = ("the null extends to the diffusion class on this corpus -- the external-validity "
                  "question closes in the null direction.")
        overall = "FAIL (both focal targets, usable ceilings)"
    elif passed:                                   # C5 only -- not the median-diagnostic hypothesis
        interp = ("PASS on C5 only. The preregistered PASS sentence is stated for C1 (the "
                  "median-diagnostic hypothesis). A C5-only pass is reported as exactly that and "
                  "is NOT averaged with any other verdict type.")
        overall = "PASS (C5 only)"
    else:
        interp = ("MIXED: report exactly that. No averaging over verdict types.")
        overall = f"MIXED ({', '.join(f'{t}={vs[t]}' for t in FOCAL)})"

    # ================================================================ 4. C5 DIAGNOSIS (EXPLORATORY)
    diag = {"LABEL": "EXPLORATORY / DESCRIPTIVE -- instrument characterization, NOT a verdict"}
    for t in ("C1", "C5"):
        sub = df[df.target == t]
        piv = sub.pivot_table(index="mask_id", columns="seed", values=OUTCOME)[SEEDS]
        pl = sub.pivot_table(index="mask_id", columns="seed", values="plain_loss")[SEEDS]
        med = piv.median(axis=1)
        sr = sub.pivot_table(index="mask_id", columns="seed", values="success_rate")[SEEDS]
        diag[t] = {
            "median_outcome_distribution": {
                "min": float(med.min()), "q25": float(med.quantile(.25)),
                "median": float(med.median()), "q75": float(med.quantile(.75)),
                "max": float(med.max()), "iqr": float(med.quantile(.75) - med.quantile(.25))},
            "between_mask_sd_SIGNAL": float(med.std()),
            "within_mask_seed_sd_NOISE_median_over_masks": float(pl.std(axis=1).median()),
            "signal_over_noise": float(med.std() / pl.std(axis=1).median()),
            "success_floor_mean_success_rate": float(sr.values.mean()),
            "success_rate_max_over_masks_seeds": float(sr.values.max()),
            "n_mask_seed_cells_with_zero_success": int((sr.values == 0).sum()),
            "heavy_tail_median_max_over_min_across_seeds": float((pl.max(1) / pl.min(1)).median()),
            "attributor_scores": {k: {"rho": v["rho"], "ratio": v["ratio_to_ceiling"],
                                      "p_onesided": v["p_onesided"]}
                                  for k, v in res[t]["estimators"].items()},
            "ceiling_median_8seed_SB": ceil[t]["ceiling_median_8seed_SB"],
        }
    diag["C5_vs_C1_summary"] = {
        "signal_sd_ratio_C5_over_C1": (diag["C5"]["between_mask_sd_SIGNAL"]
                                       / diag["C1"]["between_mask_sd_SIGNAL"]),
        "SN_ratio_C5_over_C1": (diag["C5"]["signal_over_noise"] / diag["C1"]["signal_over_noise"]),
    }

    out = {
        "stage": "P13", "preregistration_sha256": L.PREREG_SHA,
        "seeds": SEEDS, "S": len(SEEDS), "n_masks": len(masks),
        "PREREGISTERED_AGGREGATOR": "MEDIAN over seeds of the held-out executed-action L2",
        "PRIMARY_OUTCOME": OUTCOME,
        "CHAMPION": ("TracIn, E=5 diffusion ensemble (dpens_s621-625), frozen paired (t,eps) "
                     "noise bank, per-member unit-L2 normalization"),
        "frozen_config": os.path.join(P3_RESULTS, "p10_config_frozen.json"),
        "noise_bank_sha256": L.sha256_file(os.path.join(P3_RESULTS, "p10_noise_bank.json")),
        "SB_caveat": "Spearman-Brown is EXACT for means and APPROXIMATE for medians; the bar uses "
                     "the SB-corrected (higher, conservative-for-passing) value",
        "CEILING_USABILITY_GATE": {
            "ceiling_min": CEIL_MIN, "sb_tolerance": SB_TOL,
            "definition": ("(a) median SB-corrected ceiling >= 0.40 AND (b) |SB(r1 from seeds "
                           "601-604, 8) - measured median SB ceiling| <= 0.15"),
            "per_target": {t: {"CEILING_USABLE": ceil[t]["CEILING_USABLE"],
                               "gate_a": ceil[t]["GATE_a_ceiling_ge_0.40"],
                               "gate_b": ceil[t]["GATE_b_SB_consistency_within_0.15"]}
                           for t in clusters}},
        "criterion": "focal C1/C5: rho >= 0.5 * median ceiling AND one-sided p < 0.025 (Bonf-2)",
        "FOCAL_VERDICT": focal_verdict,
        "VERDICT": overall,
        "PREREGISTERED_INTERPRETATION": interp,
        "C5_DIAGNOSIS_EXPLORATORY": diag,
        "all_targets_DESCRIPTIVE": res,
        "n_new_retrains": len(jobs), "n_missing": len(missing),
    }
    L.atomic_write_json(os.path.join(P4_RESULTS, "p13_verdict.json"), out)

    print("\n" + "=" * 108)
    print(f"P13 -- DIFFUSION SETTLEMENT  (median aggregator, S={len(SEEDS)}, n=24 masks)")
    print("=" * 108)
    print(f"{'target':7s} {'ceil(med,SB)':>12s} {'bar':>7s} {'usable':>8s} | "
          f"{'CHAMPION rho':>12s} {'ratio':>6s} {'p1':>8s}  verdict")
    for t in clusters:
        e = res[t]["estimators"][CH]
        u = "USABLE" if ceil[t]["CEILING_USABLE"] else "UNUSABLE"
        v = focal_verdict[t]["VERDICT"] if t in FOCAL else "-"
        print(f"{t:7s} {ceil[t]['ceiling_median_8seed_SB']:12.3f} "
              f"{ceil[t]['bar_half_ceiling']:7.3f} {u:>8s} | {e['rho']:+12.3f} "
              f"{e['ratio_to_ceiling']:6.2f} {e['p_onesided']:8.4f}  {v}")
    print("-" * 108)
    print(f"P13 VERDICT: {overall}")
    print(f"  -> {interp}")
    print("=" * 108)
    print("C5 DIAGNOSIS (EXPLORATORY, not a verdict):")
    for t in ("C1", "C5"):
        d = diag[t]
        print(f"  {t}: signal sd={d['between_mask_sd_SIGNAL']:.3f}  "
              f"noise sd={d['within_mask_seed_sd_NOISE_median_over_masks']:.3f}  "
              f"S/N={d['signal_over_noise']:.2f}  "
              f"mean success={d['success_floor_mean_success_rate']:.3f}  "
              f"zero-success cells={d['n_mask_seed_cells_with_zero_success']}/{24*len(SEEDS)}")
    print("=" * 108)


if __name__ == "__main__":
    main()
