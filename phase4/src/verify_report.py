"""Verify EVERY number in PHASE4_REPORT.md against the artifact it claims to come from.

No number survives in the report that the files do not back. The report's claimed value is written
here as a LITERAL; the artifact's value is read from disk; they must agree. Mirrors
phase3/src/verify_report.py.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, P4

FAILS, N = [], 0


def chk(name, got, want, tol=5e-3):
    global N
    N += 1
    try:
        ok = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        ok = (got == want)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:56s} artifact={got}  report={want}")
    if not ok:
        FAILS.append({"check": name, "artifact": got, "report": want})


J = lambda p: json.load(open(os.path.join(P4_RESULTS, p)))  # noqa: E731

print("=" * 104)
print("PHASE-4 REPORT VERIFICATION -- every reported number vs its artifact")
print("=" * 104)

# ---------------------------------------------------------------- preregistration
print("\n-- preregistration lock --")
chk("prereg sha256 on disk == the one the report cites", L.sha256_file(L.PREREG), L.PREREG_SHA)
chk("prereg .sha256 sidecar matches",
    open(os.path.join(P4, "preregistration_phase4.sha256")).read().split()[0], L.PREREG_SHA)

# ---------------------------------------------------------------- P12
print("\n-- P12: the instrument --")
p12 = J("p12_ceilings.json")
chk("P12 S", p12["S"], 10)
chk("P12 instrument gate PASS", p12["INSTRUMENT_GATE"]["PASS"], True)
chk("P12 n failed focal targets", len(p12["INSTRUMENT_GATE"]["failed_focal_targets"]), 0)
chk("P12 all 9 targets pass the gate",
    sum(p12["targets"][t]["SB_consistency_PASS"] for t in p12["targets"]), 9)
for t, r1, pred, meas5, meas10, diff, bar in (
        ("C1", 0.750, 0.968, 0.906, 0.951, 0.017, 0.475),
        ("C5", 0.648, 0.948, 0.883, 0.938, 0.010, 0.469)):
    e = p12["targets"][t]
    chk(f"P12 {t} r1 from S=6 only", e["r1_from_S6_only"], r1)
    chk(f"P12 {t} predicted 10-seed SB", e["predicted_ceiling_SB_from_S6"], pred)
    chk(f"P12 {t} measured 5v5", e["ceiling_5v5_splithalf_uncorrected"], meas5)
    chk(f"P12 {t} measured 10-seed SB (ceiling)", e["ceiling_10seed_SB"], meas10)
    chk(f"P12 {t} SB-consistency |diff|", e["SB_consistency_abs_diff"], diff)
    chk(f"P12 {t} bar = half ceiling", e["bar_half_ceiling"], bar)
chk("P12 worst miss over all 9 targets (C9)",
    max(p12["targets"][t]["SB_consistency_abs_diff"] for t in p12["targets"]), 0.043)
chk("P12 n distinct 5v5 splits", p12["targets"]["C1"]["n_splits"], 126)

# ---------------------------------------------------------------- P11
print("\n-- P11: champion FAIL, secondary PASS --")
p11 = J("p11_verdict.json")
chk("P11 VERDICT (champion)", p11["VERDICT"], "FAIL")
chk("P11 champion pass on any focal", p11["CHAMPION_PASS_any_focal"], False)
chk("P11 SECONDARY pass on any focal", p11["SECONDARY_PASS_any_focal"], True)
chk("P11 S (analysis seed count)", p11["S"], 10)
for t, ce, bar, crho, crat, cp, srho, srat, sp in (
        ("C1", 0.951, 0.475, 0.409, 0.43, 0.0237, 0.513, 0.54, 0.0052),
        ("C5", 0.938, 0.469, 0.304, 0.32, 0.0741, 0.329, 0.35, 0.0584)):
    f = p11["focal"][t]
    chk(f"P11 {t} ceiling", f["ceiling"], ce)
    chk(f"P11 {t} bar", f["bar"], bar)
    chk(f"P11 {t} CHAMPION rho", f["CHAMPION"]["rho"], crho)
    chk(f"P11 {t} CHAMPION ratio", f["CHAMPION"]["ratio_to_ceiling"], crat)
    chk(f"P11 {t} CHAMPION p", f["CHAMPION"]["p_onesided"], cp, 1e-3)
    chk(f"P11 {t} SECONDARY rho", f["SECONDARY"]["rho"], srho)
    chk(f"P11 {t} SECONDARY ratio", f["SECONDARY"]["ratio_to_ceiling"], srat)
    chk(f"P11 {t} SECONDARY p", f["SECONDARY"]["p_onesided"], sp, 1e-3)
chk("P11 C1 SECONDARY PASS", p11["focal"]["C1"]["SECONDARY"]["PASS"], True)
chk("P11 C1 SECONDARY meets half-ceiling (EXACT)",
    p11["focal"]["C1"]["SECONDARY"]["meets_half_ceiling_EXACT"], True)
chk("P11 C1 CHAMPION meets half-ceiling (EXACT)",
    p11["focal"]["C1"]["CHAMPION"]["meets_half_ceiling_EXACT"], False)
chk("P11 C1 CHAMPION p < 0.025 (significant but below bar)",
    p11["focal"]["C1"]["CHAMPION"]["p_lt_alpha"], True)
D = p11["all_targets_DESCRIPTIVE"]
chk("P11 C1 TracIn unnormalized (descriptive)",
    D["C1"]["estimators"]["TracIn_E20_unnormalized"]["rho"], 0.419)
chk("P11 C1 GradDot_dmean rho (descriptive)",
    D["C1"]["estimators"]["GradDot_E20_dmean"]["rho"], 0.593)
chk("P11 C1 GradDot_dmean ratio (descriptive)",
    D["C1"]["estimators"]["GradDot_E20_dmean"]["ratio_to_ceiling"], 0.62)
chk("P11 C1 GradDot_dmean p (descriptive)",
    D["C1"]["estimators"]["GradDot_E20_dmean"]["p_onesided"], 0.0011, 1e-3)
sv = J("p11_gram_selfvalidation.json")
chk("P11 gram self-validation n_pass", sv["n_pass"], 180)
chk("P11 gram self-validation n_checks", sv["n_checks"], 180)
chk("P11 gram self-validation ALL_PASS", sv["ALL_PASS"], True)
chk("P11 gram min Spearman", sv["min_spearman"], 1.000000, 1e-6)
chk("P11 gram worst max rel diff < 1e-9", sv["worst_max_rel_diff"] < 1e-9, True)
chk("P11 probe-leak models checked", sv["probe_leak_guard"]["n_models_checked"], 10)
chk("P11 probe-leak probe ids", sv["probe_leak_guard"]["n_probe_ids"], 90)
chk("P11 probe-leak clean", sv["probe_leak_guard"]["clean"], True)

# ---------------------------------------------------------------- P13
print("\n-- P13: diffusion settlement --")
p13 = J("p13_verdict.json")
chk("P13 S", p13["S"], 8)
chk("P13 VERDICT", p13["VERDICT"], "PASS (C1)")
chk("P13 C1 verdict", p13["FOCAL_VERDICT"]["C1"]["VERDICT"], "PASS")
chk("P13 C5 verdict", p13["FOCAL_VERDICT"]["C5"]["VERDICT"], "FAIL")
c1 = p13["FOCAL_VERDICT"]["C1"]["champion"]
chk("P13 C1 rho", c1["rho"], 0.477)
chk("P13 C1 ratio", c1["ratio_to_ceiling"], 0.84)
chk("P13 C1 p", c1["p_onesided"], 0.0092, 1e-3)
chk("P13 C1 median ceiling (SB)", p13["FOCAL_VERDICT"]["C1"]["ceiling_median_8seed_SB"], 0.569)
chk("P13 C1 bar", p13["FOCAL_VERDICT"]["C1"]["bar"], 0.285)
c5 = p13["FOCAL_VERDICT"]["C5"]["champion"]
chk("P13 C5 rho", c5["rho"], 0.072)
chk("P13 C5 ratio", c5["ratio_to_ceiling"], 0.11)
chk("P13 C5 p", c5["p_onesided"], 0.369, 1e-3)
chk("P13 C5 median ceiling (SB)", p13["FOCAL_VERDICT"]["C5"]["ceiling_median_8seed_SB"], 0.664)
G = p13["all_targets_DESCRIPTIVE"]
chk("P13 C1 SB-consistency |diff| (gate margin 0.003)", G["C1"]["SB_consistency_abs_diff"], 0.147)
chk("P13 C1 gate b PASS", G["C1"]["GATE_b_SB_consistency_within_0.15"], True)
chk("P13 C5 SB-consistency |diff|", G["C5"]["SB_consistency_abs_diff"], 0.014)
chk("P13 C2 SB-consistency |diff|", G["C2"]["SB_consistency_abs_diff"], 0.221)
chk("P13 C2 CEILING_USABLE = False", G["C2"]["CEILING_USABLE"], False)
chk("P13 C2 champion rho (would have looked like a near-pass)",
    G["C2"]["estimators"]["TracIn_diffE5_normalized"]["rho"], 0.310)
chk("P13 C2 champion ratio", G["C2"]["estimators"]["TracIn_diffE5_normalized"]["ratio_to_ceiling"],
    0.54)
chk("P13 C1 descriptive all-8 SB check WOULD FAIL the gate (>0.15)",
    abs(G["C1"]["predicted_8seed_SB_from_all_DESCRIPTIVE"] - G["C1"]["ceiling_median_8seed_SB"]),
    0.215)
chk("P13 C5 descriptive all-8 SB check passes",
    abs(G["C5"]["predicted_8seed_SB_from_all_DESCRIPTIVE"] - G["C5"]["ceiling_median_8seed_SB"]),
    0.093)
# the MEAN ceilings are still broken (negative) at S=8
chk("P13 C1 MEAN ceiling still NEGATIVE at S=8", G["C1"]["ceiling_MEAN_8seed_SB_for_contrast"],
    -0.113)
chk("P13 C2 MEAN ceiling NEGATIVE", G["C2"]["ceiling_MEAN_8seed_SB_for_contrast"], -0.049)
chk("P13 C9 MEAN ceiling NEGATIVE", G["C9"]["ceiling_MEAN_8seed_SB_for_contrast"], -0.023)
chk("P13 C5 MEAN ceiling", G["C5"]["ceiling_MEAN_8seed_SB_for_contrast"], 0.021)
chk("P13 n distinct 4v4 splits", G["C1"]["n_splits"], 35)
d = p13["C5_DIAGNOSIS_EXPLORATORY"]
chk("P13 C1 signal sd", d["C1"]["between_mask_sd_SIGNAL"], 0.136)
chk("P13 C5 signal sd", d["C5"]["between_mask_sd_SIGNAL"], 0.092)
chk("P13 C1 noise sd", d["C1"]["within_mask_seed_sd_NOISE_median_over_masks"], 0.537)
chk("P13 C5 noise sd", d["C5"]["within_mask_seed_sd_NOISE_median_over_masks"], 0.580)
chk("P13 C1 S/N", d["C1"]["signal_over_noise"], 0.253)
chk("P13 C5 S/N", d["C5"]["signal_over_noise"], 0.158)
chk("P13 C1 mean success", d["C1"]["success_floor_mean_success_rate"], 0.229)
chk("P13 C5 mean success (HIGHER than C1's)", d["C5"]["success_floor_mean_success_rate"], 0.268)

# ---------------------------------------------------------------- P14
print("\n-- P14: fresh heterogeneity --")
p14 = J("p14_verdict.json")
chk("P14 VERDICT", p14["VERDICT"], "FAIL")
chk("P14 primary cells passing", len(p14["PRIMARY_cells_passing"]), 0)
chk("P14 secondary cells passing", len(p14["SECONDARY_cells_passing"]), 0)
chk("P14 n fresh masks", p14["fresh_masks"]["K"], 16)
chk("P14 mask seed", p14["fresh_masks"]["mask_seed"], 1104)
chk("P14 coincidences with Stage G", p14["fresh_masks"]["coincidence_with_stage_G"], 0)
chk("P14 S", p14["S"], 4)
chk("P14 critical rho (primary)", p14["critical_rho_primary"], 0.557)
chk("P14 C2 fresh ceiling", p14["ceilings"]["C2"]["ceiling_4seed_SB"], 0.837)
chk("P14 C9 fresh ceiling", p14["ceilings"]["C9"]["ceiling_4seed_SB"], 0.851)
for cell, prho, pp, srho, sp, sattr in (
        ("C2_g1", 0.241, 0.184, 0.241, 0.184, "TracIn"),
        ("C2_g3", -0.191, 0.761, 0.224, 0.203, "IF"),
        ("C9_g1", 0.006, 0.491, 0.006, 0.491, "TracIn"),
        ("C9_g5", -0.162, 0.725, -0.162, 0.725, "TracIn")):
    e = p14["cells"][cell]
    chk(f"P14 {cell} PRIMARY rho", e["arms"]["PRIMARY_champion"]["rho"], prho)
    chk(f"P14 {cell} PRIMARY p", e["arms"]["PRIMARY_champion"]["p_onesided"], pp, 1e-3)
    chk(f"P14 {cell} PRIMARY PASS", e["arms"]["PRIMARY_champion"]["PASS"], False)
    chk(f"P14 {cell} SECONDARY attributor", e["arms"]["SECONDARY_p7_qualifier"]["attributor"],
        sattr)
    chk(f"P14 {cell} SECONDARY rho", e["arms"]["SECONDARY_p7_qualifier"]["rho"], srho)
    chk(f"P14 {cell} SECONDARY PASS", e["arms"]["SECONDARY_p7_qualifier"]["PASS"], False)
man = J("p14_mask_manifest.json")
chk("P14 mask coincidence check PASS", man["coincidence_check"]["PASS"], True)
chk("P14 mask seed increments", man["seed_increments_due_to_collision"], 0)
chk("P14 demos per mask", man["demos_per_mask"], 68)

# the P7 -> P14 point-estimate collapse (P7 values read from the PHASE-3 artifact, read-only)
print("\n-- P14: the P7 -> P14 collapse (P7 values read from phase3/results/p7_grain_ladder.csv) --")
p7 = pd.read_csv(os.path.join(L.P3_RESULTS, "p7_grain_ladder.csv"))
p7 = p7[p7.rule == "random"]


def p7ratio(t, g, a):
    r = p7[(p7.target == t) & (p7.g == g) & (p7.attributor == a)]
    return float(r.ratio.iloc[0])


for t, g, a, want_p7, want_p4 in (("C2", 1, "TracIn", 0.51, 0.29), ("C2", 3, "IF", 0.65, 0.27),
                                  ("C9", 1, "TracIn", 0.52, 0.01), ("C9", 5, "TracIn", 0.66, -0.19)):
    chk(f"P7 {t}@g={g} ({a}) ratio", p7ratio(t, g, a), want_p7, 1e-2)
    arm = ("SECONDARY_p7_qualifier" if a == "IF" else "PRIMARY_champion")
    chk(f"P14 {t}@g={g} ({a}) ratio", p14["cells"][f"{t}_g{g}"]["arms"][arm]["ratio_to_ceiling"],
        want_p4, 1e-2)

# ---------------------------------------------------------------- SYNTHETIC + budget
print("\n-- SYNTHETIC + budget --")
syn = J("SYNTHETIC_p4_unit_tests.json")
chk("SYNTHETIC n_pass", syn["n_pass"], 32)
chk("SYNTHETIC n_tests", syn["n_tests"], 32)
chk("SYNTHETIC ALL_PASS", syn["ALL_PASS"], True)
b = J("budget_actual.json")
chk("budget TOTAL retrains", b["TOTAL_retrains"], 208)
chk("budget TOTAL ok", b["TOTAL_ok"], 208)
chk("budget TOTAL failed", b["TOTAL_failed"], 0)
chk("budget TOTAL gpu_h", b["TOTAL_gpu_h"], 32.50, 0.05)
chk("budget TOTAL episodes", b["TOTAL_episodes"], 42720)
chk("budget alert tripped", b["ALERT_TRIPPED"], False)
chk("budget cuts", b["cuts"], "NONE")
for s, r, g, e in (("P12", 96, 11.95, 25920), ("P13", 48, 15.35, 12960), ("P14", 64, 5.18, 3840)):
    row = [x for x in b["stages"] if x["stage"] == s][0]
    chk(f"budget {s} retrains", row["retrains_actual"], r)
    chk(f"budget {s} gpu_h", row["gpu_h_actual"], g, 0.02)
    chk(f"budget {s} episodes", row["episodes_actual"], e)
    chk(f"budget {s} failures", row["n_fail"], 0)
chk("budget P11 attribution gpu_h", b["P11_attribution_gpu_h"], 0.02, 0.01)

# ---------------------------------------------------------------- out
print("\n" + "=" * 104)
print(f"REPORT VERIFICATION: {N - len(FAILS)}/{N} numbers verified against artifacts, "
      f"{len(FAILS)} MISMATCHES")
print("=" * 104)
for f in FAILS:
    print(f"  MISMATCH: {f}")

L.atomic_write_json(os.path.join(P4_RESULTS, "report_verification.json"), {
    "report": "phase4/PHASE4_REPORT.md",
    "preregistration_sha256": L.PREREG_SHA,
    "n_checks": N, "n_verified": N - len(FAILS), "n_mismatches": len(FAILS),
    "ALL_VERIFIED": len(FAILS) == 0, "mismatches": FAILS,
})
sys.exit(1 if FAILS else 0)
