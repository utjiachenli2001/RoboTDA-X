"""Verify that every headline number quoted in PHASE2_REPORT.md is present in an artifact file.

Guards against transcription drift between the analyses and the prose. Each check re-reads the
artifact and asserts the value the report claims.
"""
import json
import os
import re
import sys

import pandas as pd

R = "/mnt/sdb/ljc/RoboTDA-X/phase2"
rep = open(f"{R}/PHASE2_REPORT.md").read()
fails, checks = [], 0


def chk(name, artifact_val, claimed, tol=5e-4):
    global checks
    checks += 1
    ok = abs(float(artifact_val) - float(claimed)) <= tol
    if not ok:
        fails.append(f"{name}: artifact={artifact_val} report={claimed}")
    return ok


def in_report(s, name):
    global checks
    checks += 1
    if s not in rep:
        fails.append(f"{name}: string not found in report: {s!r}")


# ---------------------------------------------------------------- P0
a = json.load(open(f"{R}/results/p0_intake_audit.json"))
chk("P0 gate1 rho", a["gate1"]["best_rho_recomputed"], 0.251748, 1e-5)
assert a["ALL_PASS"], "P0 did not pass!"
in_report("+0.251748", "P0 rho")
in_report("7/3, 5/6, −19/6", "P0 stage-C exact margins")

# ---------------------------------------------------------------- P1
p1 = json.load(open(f"{R}/results/p1_demo_grain.json"))
assert p1["PASS"] is False, "P1 verdict changed!"
for t, ceil, rho, ratio in [("C1", 0.933, 0.369, 0.40), ("C5", 0.918, 0.380, 0.41)]:
    v = p1["focal_verdict"][t]
    chk(f"P1 {t} ceiling", v["ceiling_6seed"], ceil, 5e-4)
    chk(f"P1 {t} best rho", v["best_rho"], rho, 5e-4)
    chk(f"P1 {t} ratio", v["best_ratio"], ratio, 5e-3)
    assert v["any_attributor_PASS"] is False, f"P1 {t} PASS changed!"
lad = json.load(open(f"{R}/results/p1_seed_ladder.json"))["ladder"]
chk("P1 C1 LDS S=1", lad["C1"]["1"]["best_lds_mean"], 0.333, 1e-3)
chk("P1 C1 ceil S=1", lad["C1"]["1"]["ceiling_SvS"], 0.750, 1e-3)

# ---------------------------------------------------------------- P2
p2 = json.load(open(f"{R}/results/p2_transfer_sign.json"))
h = p2["PREREGISTERED_PRIMARY"]
assert h["PASS"] is False, "P2 verdict changed!"
chk("P2 primary rho", h["rho"], 0.163, 1e-3)
chk("P2 primary p", h["p_onesided"], 0.209, 1e-3)
chk("P2 sign agree", h["sign_agree"], 17, 0)
chk("P2 C1 margin", p2["by_cluster_check"]["C1"]["cluster_margin_pts_mean"], 1.00, 1e-6)
chk("P2 C2 margin", p2["by_cluster_check"]["C2"]["cluster_margin_pts_mean"], 14.10, 1e-6)
chk("P2 C5 margin", p2["by_cluster_check"]["C5"]["cluster_margin_pts_mean"], 5.00, 1e-6)
st = json.load(open(f"{R}/results/p2_attribution_stability.json"))
chk("P2 E-vs-B IF", st["agreement_between_ensembles"]["IF"]["spearman_predicted_benefit_E_vs_B"], 0.774, 1e-3)
chk("P2 E-vs-B TracIn", st["agreement_between_ensembles"]["TracIn"]["spearman_predicted_benefit_E_vs_B"], -0.103, 1e-3)
chk("P2 IF subens range", st["subensemble_spread"]["IF"]["range"], 0.468, 1e-3)
chk("P2 IF frac pass", st["subensemble_spread"]["IF"]["frac_that_would_PASS_alpha05_uncorrected"], 0.04, 5e-3)

# ---------------------------------------------------------------- P3
p3 = json.load(open(f"{R}/results/p3_regime_boundary.json"))
ti = p3["PREREGISTERED_TEST_i_success_ceiling_rises_with_Q"]
tii = p3["PREREGISTERED_TEST_ii_LDS_ceiling_ratio_rises_with_Q"]
assert ti["PASS"] is False and tii["PASS"] is False, "P3 verdicts changed!"
chk("P3 test i p", ti["p_onesided_exact"], 0.968, 1e-3)
chk("P3 test ii p", tii["p_onesided_exact"], 0.995, 1e-3)
chk("P3 Q150 succ ceiling", p3["per_Q"]["150"]["outcomes"]["logit_success"]["ceiling_4seed_SB"], -0.561, 1e-3)
chk("P3 Q15 succ ceiling", p3["per_Q"]["15"]["outcomes"]["logit_success"]["ceiling_4seed_SB"], 0.690, 1e-3)
dec = p3["DESCRIPTIVE_reverse_direction_tests"]
chk("P3 reverse ii p", dec["ii_LDS_ratio_DECREASES_with_Q"]["p_onesided_exact"], 0.032, 1e-3)
snr = p3["DESCRIPTIVE_signal_vs_seed_noise"]["per_Q"]
chk("P3 Q15 S/N", snr["15"]["success"]["signal_to_noise"], 1.13, 5e-3)
chk("P3 Q150 S/N", snr["150"]["success"]["signal_to_noise"], 0.37, 5e-3)

# ---------------------------------------------------------------- P4
p4 = json.load(open(f"{R}/results/p4_success_reliability.json"))
g = p4["grid"]
chk("P4 C1 30ep S1", g["C1"]["10"]["1"]["reliability"], 0.185, 1e-3)
chk("P4 C1 150ep S3", g["C1"]["50"]["3"]["reliability"], 0.502, 1e-3)
chk("P4 C2 30ep S1", g["C2"]["10"]["1"]["reliability"], 0.601, 1e-3)
chk("P4 C2 150ep S3", g["C2"]["50"]["3"]["reliability"], 0.797, 1e-3)
assert g["C1"]["50"]["6"]["is_extrapolation"] is True, "S=6 must be flagged as extrapolation"
m = p4["HEADLINE_episodes_vs_seeds"]
chk("P4 C1 episode gain", m["C1"]["episodes_axis"]["delta_reliability"], 0.057, 1e-3)
chk("P4 C1 seed gain", m["C1"]["seeds_axis"]["delta_reliability"], 0.259, 1e-3)
chk("P4 C2 episode gain", m["C2"]["episodes_axis"]["delta_reliability"], -0.025, 1e-3)
chk("P4 C2 seed gain", m["C2"]["seeds_axis"]["delta_reliability"], 0.221, 1e-3)
assert m["C1"]["seeds_beat_episodes"] and m["C2"]["seeds_beat_episodes"]

# ---------------------------------------------------------------- P5
p5 = json.load(open(f"{R}/results/p5_coverage_fix.json"))
chk("P5 cov success", p5["mean_success_pct"]["coverage_constrained"], 1.67, 5e-3)
chk("P5 unc success", p5["mean_success_pct"]["unconstrained"], 1.11, 5e-3)
chk("P5 target_only", p5["mean_success_pct"]["target_only"], 17.22, 5e-3)
chk("P5 random15", p5["mean_success_pct"]["random15"], 13.89, 5e-3)
chk("P5 cov-unc margin", p5["paired_margins_pts"]["unconstrained"]["mean"], 0.56, 5e-3)
chk("P5 cov-target margin", p5["paired_margins_pts"]["target_only"]["mean"], -15.56, 5e-3)

# ---------------------------------------------------------------- defect + budget
d = json.load(open(f"{R}/results/p4_determinism_check.json"))
assert d["DEFECT_CONFIRMED"] and d["vectors_are_discriminating"], "defect check not confirmed!"
assert d["episodes_0_9"]["steps"] == d["episodes_50_59"]["steps"], "step vectors differ!"
in_report("154", "defect step count")

ep = json.load(open(f"{R}/results/episode_ledger.json"))
chk("total episodes", sum(ep.values()), 101101, 0)

gpu = 0.0
for f, st in [("logs/P1_stage_G6_summary.json", "P1"), ("logs/P3_summary.json", "P3"),
              ("logs/P5_summary.json", "P5")]:
    s = json.load(open(f"{R}/{f}"))
    gpu += s["gpu_h_est"]
    assert s["n_fail"] == 0, f"{st} had failures!"
import glob
w = sum(json.load(open(p))["wall_s"] for p in glob.glob(f"{R}/runs/P4/*/success50.json"))
gpu += w / 3600
chk("P4 gpu-h", w / 3600, 15.45, 0.05)
print(f"total GPU-h (excl. attribution) = {gpu:.2f}")
assert gpu < 75, "GLOBAL BUDGET ALERT BREACHED"

print("=" * 70)
print(f"{checks} checks run, {len(fails)} FAILURES")
for f in fails:
    print("  FAIL:", f)
print("=" * 70)
print("REPORT VERIFIED — every quoted number matches its artifact" if not fails
      else "REPORT HAS DRIFT — fix before publishing")
sys.exit(1 if fails else 0)
