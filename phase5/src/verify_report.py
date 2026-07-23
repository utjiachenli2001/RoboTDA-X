"""Verify EVERY number in PHASE5_REPORT.md against the artifact it claims to come from.

The report's claimed value is written here as a LITERAL; the artifact's value is read from disk;
they must agree. No number survives in the report that the files do not back. Mirrors
phase4/src/verify_report.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5lib as L
from p5lib import P5_RESULTS, P5

FAILS, N = [], 0


def chk(name, got, want, tol=5e-3):
    global N
    N += 1
    try:
        ok = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        ok = (got == want)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:58s} artifact={got}  report={want}")
    if not ok:
        FAILS.append({"check": name, "artifact": got, "report": want})


J = lambda p: json.load(open(os.path.join(P5_RESULTS, p)))  # noqa: E731

print("=" * 108)
print("PHASE-5 REPORT VERIFICATION -- every reported number vs its artifact")
print("=" * 108)

# ---------------------------------------------------------------- preregistration lock
print("\n-- preregistration lock --")
chk("prereg sha256 on disk == cited", L.sha256_file(L.PREREG), L.PREREG_SHA)
chk("prereg .sha256 sidecar matches",
    open(os.path.join(P5, "preregistration_phase5.sha256")).read().split()[0], L.PREREG_SHA)

# ---------------------------------------------------------------- SYNTHETIC
print("\n-- SYNTHETIC unit tests --")
syn = J("SYNTHETIC_p5_unit_tests.json")
chk("SYNTHETIC all pass", syn["ALL_PASS"], True)
chk("SYNTHETIC n_pass", syn["n_pass"], 15)
chk("SYNTHETIC n_tests", syn["n_tests"], 15)

# ---------------------------------------------------------------- P15 gate + verdict
print("\n-- P15: diffusion C1 replication at S=10 --")
p15 = J("p15_verdict.json")
chk("P15 S", p15["S"], 10)
chk("P15 VERDICT", p15["VERDICT"], "PASS (C1)")
g = p15["STRICTER_GATE"]["C1"]
chk("P15 C1 median SB10 ceiling", g["ceiling_median_10seed_SB"], 0.830)
chk("P15 C1 gate(ii) S4->10 |diff|", g["SB_consistency_S4to10_abs_diff"], 0.070)
chk("P15 C1 gate(iii) S8->10 |diff|", g["SB_consistency_S8to10_abs_diff"], 0.010)
chk("P15 C1 gate(iii) tol strictness", 0.010 <= 0.10, True)
chk("P15 C1 descriptive all-seed |diff|",
    g["SB_consistency_all_to_10_abs_diff_DESCRIPTIVE"], 0.016)
chk("P15 C1 CEILING_USABLE", g["CEILING_USABLE"], True)
ch = p15["C1_VERDICT"]["champion"]
chk("P15 C1 champion rho", ch["rho"], 0.479)
chk("P15 C1 champion ratio", ch["ratio_to_ceiling"], 0.577)
chk("P15 C1 champion p_onesided", ch["p_onesided"], 0.0089)
chk("P15 C1 meets half ceiling EXACT", ch["meets_half_ceiling_EXACT"], True)
chk("P15 C1 bar", p15["C1_VERDICT"]["bar"], 0.415)
chk("P15 noise bank sha", p15["noise_bank_sha256"],
    "61aadccfef2fb45300d611f262bdc285c6a8f9888ed907a1c48b112d8405bc17")
# P13 rho for the replication-strength comparison
p13rho = json.load(open(os.path.join(L.P4_RESULTS, "p13_verdict.json"))
                   )["all_targets_DESCRIPTIVE"]["C1"]["estimators"]["TracIn_diffE5_normalized"]["rho"]
chk("P13 C1 champion rho (for near-identity claim)", p13rho, 0.477)

# seed-mean brokenness series
print("\n-- P15: seed-mean brokenness series --")
ms = p15["seed_mean_brokenness_series"]
chk("C1 MEAN SB ceiling S8", ms["C1"]["S8_MEAN_SB_ceiling"], -0.113)
chk("C1 MEAN SB ceiling S10", ms["C1"]["S10_MEAN_SB_ceiling"], 0.201)
chk("C1 MEDIAN SB ceiling S10", ms["C1"]["S10_MEDIAN_SB_ceiling"], 0.830)
chk("C9 MEAN SB ceiling S10 still negative", ms["C9"]["S10_MEAN_SB_ceiling"], -0.022)
# S6 mean ceiling from Phase-3 artifact
s6 = json.load(open(os.path.join(L.ROOT, "phase3", "results", "p10_verdict_S6.json"))
               )["focal_verdict"]["C1"]["ceiling_6seed_measured"]
chk("C1 MEAN ceiling S6 (Phase-3 measured)", s6, 0.079)

# ---------------------------------------------------------------- P16 breadth
print("\n-- P16: GradDot breadth on 7 non-focal BC targets --")
p16 = J("p16_verdict.json")
chk("P16 k_primary", p16["k_primary"], 0)
chk("P16 k_secondary", p16["k_secondary"], 0)
chk("P16 self-val C1 exact match", p16["SELF_VALIDATION"]["C1_exact_match"], True)
chk("P16 self-val C1 GradDot recomputed", p16["SELF_VALIDATION"]["C1_GradDot_recomputed"],
    0.5130434782608695, tol=1e-15)
chk("P16 probe-leak n_models", p16["SELF_VALIDATION"]["probe_leak_guard"]["n_models_checked"], 20)
chk("P16 critical rho n24 Bonf7", p16["critical_rho_primary_n24"], 0.493)
chk("P16 alpha primary", p16["alpha_primary_Bonferroni7"], 0.05 / 7, tol=1e-9)
for t, rho, ratio, p1, meets, sig in (
        ("C2", 0.368, 0.397, 0.0385, False, False),
        ("C3", -0.075, -0.099, 0.6358, False, False),
        ("C4", 0.430, 0.487, 0.0179, False, False),
        ("C6", 0.157, 0.163, 0.2326, False, False),
        ("C7", 0.331, 0.349, 0.0569, False, False),
        ("C8", -0.178, -0.188, 0.7977, False, False),
        ("C9", 0.354, 0.396, 0.0449, False, False)):
    e = p16["per_target"][t]["estimators"]["GradDot_E20_normalized"]
    chk(f"P16 {t} GradDot rho", e["rho"], rho)
    chk(f"P16 {t} GradDot ratio", e["ratio_to_ceiling"], ratio)
    chk(f"P16 {t} GradDot p", e["p_onesided"], p1)
    chk(f"P16 {t} meets half ceiling", e["meets_half_ceiling_EXACT"], meets)
    chk(f"P16 {t} p<alpha", e["p_lt_alpha"], sig)

# ---------------------------------------------------------------- P17 exploratory
print("\n-- P17: EXPLORATORY diffusion GradDot --")
p17 = J("p17_exploratory.json")
chk("P17 label present", "EXPLORATORY" in p17["LABEL"], True)
for t, gd, tr10 in (("C1", 0.414, 0.479), ("C5", 0.186, 0.117)):
    r = p17["rows"][t]
    chk(f"P17 {t} GradDot(diff) rho", r["GradDot_diffE5_normalized"]["rho"], gd)
    chk(f"P17 {t} TracIn S10 rho", r["TracIn_diffE5_normalized_S10"]["rho"], tr10)
chk("P17 C1: TracIn > GradDot (diffusion order)",
    p17["rows"]["C1"]["TracIn_diffE5_normalized_S10"]["rho"]
    > p17["rows"]["C1"]["GradDot_diffE5_normalized"]["rho"], True)

# ---------------------------------------------------------------- budget
print("\n-- budget --")
b = json.load(open(os.path.join(P5_RESULTS, "budget_actual.json")))
chk("budget total retrains", b["total_retrains"], 48)
chk("budget total fail", b["total_fail"], 0)
chk("budget total gpu_h", b["total_gpu_h_actual"], 8.46, tol=0.02)
chk("budget total episodes", b["total_episodes"], 1440)
chk("budget alert not tripped", b["budget_alert_tripped"], False)

# ---------------------------------------------------------------- summary
out = {"n_checks": N, "n_fail": len(FAILS), "ALL_VERIFIED": len(FAILS) == 0, "failures": FAILS}
L.atomic_write_json(os.path.join(P5_RESULTS, "report_verification.json"), out)
print("\n" + "=" * 108)
print(f"VERIFICATION: {N - len(FAILS)}/{N} checks pass, {len(FAILS)} FAILED")
print("=" * 108)
if FAILS:
    raise SystemExit("REPORT VERIFICATION FAILED")
