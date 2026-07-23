"""Verify EVERY headline number in PHASE3_REPORT.md against the artifact it claims to come from.

No number in the report may survive that the files do not back. Mirrors phase2/src/verify_report.py.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS

FAILS = []
N = 0


def chk(name, got, want, tol=1e-3):
    global N
    N += 1
    ok = (got == want) if isinstance(want, (bool, str, int)) and not isinstance(want, bool) is False \
        else None
    try:
        ok = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        ok = (got == want)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:58s} artifact={got}  report={want}")
    if not ok:
        FAILS.append((name, got, want))


J = lambda p: json.load(open(os.path.join(P3_RESULTS, p)))  # noqa: E731

print("=" * 100)
print("PHASE-3 REPORT VERIFICATION -- every headline number vs its artifact")
print("=" * 100)

print("\n-- P6 --")
chk("p6 marker sweep violations", J("p6_marker_sweep.json")["n_violations"], 0)
chk("p6 marker sweep run dirs", J("p6_marker_sweep.json")["n_run_dirs_scanned"], 730)
plc = J("p6_probe_leak_check.json")
chk("p6 leaking model-artifact pairs", plc["n_leaking_model_artifact_pairs"], 0)
chk("p6 Q490 probe-demo inclusions",
    sum(g["n_probe_demos_in_training_set"] for g in plc["loaded_gun_confirmed"]), 300)
g6 = J("p6_g6_integrity.json")
chk("p6 G6 dirs", g6["host_check"]["n_dirs"], 96)
chk("p6 G6 bad dirs", g6["host_check"]["n_bad_dirs"], 0)
chk("p6 G6 spot rows matching", g6["n_spot_rows_matching"], 5)
nc = J("p6_no_change.json")
chk("p6 no-change n_checks", nc["n_checks"], 125)
chk("p6 no-change mismatches", nc["n_mismatches"], 0)

print("\n-- P6.5 lambda sweep --")
ext = J("p6_lambda_sweep_extended.json")
chk("C1 LDS at default ridge", ext["demo_lds_at_default_ridge_1e-2"]["C1"], 0.256, 1e-3)
chk("C5 LDS at default ridge", ext["demo_lds_at_default_ridge_1e-2"]["C5"], 0.380, 1e-3)
chk("C1 oracle-tuned max", ext["ORACLE_TUNED_MAX_extended"]["C1"]["best_demo_lds"], 0.504, 1e-3)
chk("C1 oracle-tuned ratio", ext["ORACLE_TUNED_MAX_extended"]["C1"]["ratio"], 0.54, 1e-2)
chk("C1 oracle-tuned p", ext["ORACLE_TUNED_MAX_extended"]["C1"]["p_onesided"], 0.0060, 1e-3)
chk("C5 oracle-tuned max", ext["ORACLE_TUNED_MAX_extended"]["C5"]["best_demo_lds"], 0.432, 1e-3)
cv = ext["MANDATORY_CROSS_VALIDATION_extended"]
chk("CV tune C1 -> eval C5 rho", cv["tune_on_C1__evaluate_on_C5"]["heldout_demo_lds"], 0.397, 1e-3)
chk("CV tune C5 -> eval C1 rho", cv["tune_on_C5__evaluate_on_C1"]["heldout_demo_lds"], 0.341, 1e-3)
chk("CV any PASS (must be False)",
    any(v["heldout_PASS_ratio_AND_p"] for v in cv.values()), False)
chk("lambda->inf convergence gap C1",
    ext["convergence_check"]["C1"]["max_abs_gap_vs_limit"], 0.0, 1e-9)

print("\n-- P7 grain ladder --")
p7 = J("p7_grain_ladder.json")
chk("C1 smallest qualifying g (None)",
    str(p7["PREREGISTERED_READOUT_focal"]["C1"]["SMALLEST_QUALIFYING_g"]), "None")
chk("C5 smallest qualifying g (None)",
    str(p7["PREREGISTERED_READOUT_focal"]["C5"]["SMALLEST_QUALIFYING_g"]), "None")
chk("DTW-random mean at g=3", p7["dtw_minus_random_mean_over_9_targets"]["3"], 0.062, 5e-3)
chk("DTW wins at g=3 (of 9)", p7["dtw_beats_random_n_targets_of_9"]["3"], 6)

print("\n-- P8a --")
p8 = J("p8a_variance_decomposition.json")
vl = p8["results"]["neg_plain_loss"]["variance_fraction"]
chk("P8a INIT share (L2)", vl["INIT"], 0.718, 2e-3)
chk("P8a ORDER share (L2)", vl["ORDER"], 0.151, 2e-3)
vs = p8["results"]["logit_success"]["variance_fraction"]
chk("P8a INIT share (success)", vs["INIT"], 0.454, 2e-3)
chk("P8a n_retrains", p8["n_retrains"], 48)
bc = J("p8_bitcheck.json")
chk("P8 bitcheck factorial==stock",
    bc["ASSERTION_1_factorial_equals_stock_when_init_eq_order"]["IDENTICAL"], True)
chk("P8 phase-1 reproduces archived",
    bc["INFORMATIONAL_phase1_reproducible_across_time"]["IDENTICAL"], True)

print("\n-- P8b --")
b = J("p8b_variance_reduction.json")["results"]["neg_plain_loss"]
chk("P8b baseline S=1", b["baseline_S1_single_seed"], 0.593, 2e-3)
chk("P8b arm(i) ckpt-avg", b["arm_i_S1_checkpoint_averaged"], 0.596, 2e-3)
chk("P8b baseline S=3", b["baseline_S3_outcome_mean"], 0.819, 2e-3)
chk("P8b arm(ii) action-ens", b["arm_ii_S3_action_ensemble"], 0.799, 2e-3)
chk("P8b strong gate", J("p8b_determinism_strong.json")["GATE"], True)
chk("P8b strong gate informative", J("p8b_determinism_strong.json")["GATE_IS_INFORMATIVE"], True)

print("\n-- P9 --")
p9 = J("p9_ensemble_cost_law.json")["PREREGISTERED_READOUT_SB_extrapolation_to_0.8"]
chk("P9 IF reliability at E=10", p9["IF"]["measured_reliability_at_E10"], 0.199, 2e-3)
chk("P9 IF E* for 0.8", p9["IF"]["SB_EXTRAPOLATED_E_for_0.8_from_E20"], 234.9, 1.0)
chk("P9 TRAK E* for 0.8", p9["TRAK"]["SB_EXTRAPOLATED_E_for_0.8_from_E20"], 111.1, 1.0)
chk("P9 TracIn reliability at E=20", p9["TracIn"]["measured_reliability_at_E20"], 0.831, 2e-3)
T = pd.read_csv(os.path.join(P3_RESULTS, "p9_ensemble_cost_law.csv"))
chk("P9 IF top-15 Jaccard at E=10",
    float(T[(T.E == 10) & (T.attributor == "IF")].top15_jaccard_mean_across_subensembles.iloc[0]),
    0.222, 2e-3)

print("\n-- P10 --")
cal = J("p10_calibration.json")
chk("P10 calib runs used", cal["runs_used"], 7)
chk("P10 calib best success", cal["BEST_SUCCESS"], 1.00, 1e-6)
chk("P10 viable", cal["VIABLE"], True)
chk("P10 determinism gate", J("p10_determinism.json")["GATE"], True)
a0 = J("p10a_gate0.json")["results"]
chk("P10a C1 margin (pts)", a0["C1"]["margin_pts"], 9.7, 0.1)
chk("P10a C5 margin (pts)", a0["C5"]["margin_pts"], 7.6, 0.1)
s4 = J("p10_verdict_S4_PREREGISTERED.json")["focal_verdict"]
chk("P10 S=4 C1 ceiling", s4["C1"]["ceiling_4seed_measured"], 0.040, 2e-3)
chk("P10 S=4 PASS (must be False)",
    any(v["any_preregistered_attributor_PASS"] for v in s4.values()), False)
s6 = J("p10_verdict_S6.json")
chk("P10 S=6 C1 ceiling", s6["focal_verdict"]["C1"]["ceiling_6seed_measured"], 0.079, 2e-3)
chk("P10 S=6 ceiling usable (must be False)", s6["CEILING_IS_USABLE"], False)
chk("P10 S=6 SB gap C1", s6["focal_verdict"]["C1"]["SB_consistency_gap"], 0.612, 5e-3)
md = J("p10_diagnostic_median.json")
chk("P10 median ceiling C1", md["focal_verdict"]["C1"]["ceiling_median_6seed"], 0.568, 2e-3)
chk("P10 median ceiling C5", md["focal_verdict"]["C5"]["ceiling_median_6seed"], 0.718, 2e-3)
chk("P10 median C1 rho", md["focal_verdict"]["C1"]["best_rho"], 0.420, 2e-3)
chk("P10 median C1 ratio", md["focal_verdict"]["C1"]["best_ratio"], 0.74, 1e-2)
chk("P10 median C1 p", md["focal_verdict"]["C1"]["best_p_onesided"], 0.0205, 1e-3)
chk("P10 median C5 rho", md["focal_verdict"]["C5"]["best_rho"], 0.060, 2e-3)
st = J("p10_attr_stability.json")["diffusion"]
chk("P10 diffusion TracIn attr stability", st["TracIn"]["mean_over_9_targets"], 0.927, 2e-3)
chk("P10 diffusion IF attr stability", st["IF"]["mean_over_9_targets"], 0.521, 2e-3)

print("\n-- budget --")
bud = J("budget_actual.json")
chk("total retrains", bud["total_retrains"], 230)
chk("total GPU-h", bud["total_gpu_h"], 66.17, 0.01)
chk("total episodes", bud["total_episodes"], 74130)
chk("alert tripped (must be False)", bud["alert_tripped"], False)
chk("all orchestrated runs ok",
    sum(s["fail"] for s in bud["per_stage"]), 0)

print("\n" + "=" * 100)
print(f"{N} checks, {len(FAILS)} FAILURES")
for f in FAILS:
    print("  MISMATCH:", f)
print("ALL REPORT NUMBERS VERIFIED" if not FAILS else "REPORT HAS UNBACKED NUMBERS")
print("=" * 100)
L.atomic_write_json(os.path.join(P3_RESULTS, "report_verification.json"),
                    {"n_checks": N, "n_failures": len(FAILS), "failures": FAILS,
                     "ALL_VERIFIED": len(FAILS) == 0})
sys.exit(0 if not FAILS else 1)
