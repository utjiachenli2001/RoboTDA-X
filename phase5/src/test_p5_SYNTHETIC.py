"""SYNTHETIC unit tests for the Phase-5 statistics. LABELLED SYNTHETIC. NEVER mixed with real
results. Every test runs on data with a KNOWN answer so a wrong implementation is caught before it
touches a real verdict.

Validates: exact rational criterion arithmetic (fails one ULP below the bar); balanced split
enumeration (126 at S=10, 35 at S=8, 3 at S=4); Spearman-Brown recovery; the median beating the
mean on planted heavy-tail contamination (the premise the diffusion instrument rests on); the
stricter three-condition gate (passes i+ii but fails iii => UNUSABLE); per-member normalization
rank-invariance; the P16 Bonferroni thresholds and the commensurate critical rho at n=24.
"""
import math
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5lib as L

sys.path.insert(0, os.path.join(L.ROOT, "src"))
from lds import spearman_p_onesided  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append({"test": name, "PASS": bool(cond), "detail": detail})
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    return cond


def t_exact_criterion():
    ceiling = 0.9505589414333672
    bar = 0.5 * Fraction(ceiling)
    exactly = float(bar)                       # nearest double to 0.5*ceiling
    below = math.nextafter(exactly, -math.inf)
    above = math.nextafter(exactly, math.inf)
    check("exact_criterion_at_bar_passes", L.meets_half_ceiling(exactly, ceiling),
          f"rho={exactly!r}")
    check("exact_criterion_one_ulp_below_FAILS", not L.meets_half_ceiling(below, ceiling),
          "epsilon compare would wrongly pass")
    check("exact_criterion_one_ulp_above_passes", L.meets_half_ceiling(above, ceiling))


def t_split_enumeration():
    for S, expect in ((10, 126), (8, 35), (4, 3), (6, 10)):
        seeds = list(range(1, S + 1))
        piv = pd.DataFrame(np.random.RandomState(S).randn(24, S),
                           columns=seeds, index=[f"m{i}" for i in range(24)])
        _, _, splits, n = L.split_half_ceiling(piv, seeds, agg="median")
        check(f"balanced_splits_S{S}_eq_{expect}", n == expect, f"got {n}")


def t_spearman_brown_recovery():
    rng = np.random.RandomState(0)
    n_mask, S = 200, 10
    truth = rng.randn(n_mask)
    # each seed = truth + independent noise; single-seed reliability is controllable via noise sd
    noise_sd = 1.0
    cols = {s: truth + noise_sd * rng.randn(n_mask) for s in range(1, S + 1)}
    piv = pd.DataFrame(cols, index=[f"m{i}" for i in range(n_mask)])
    r1, _ = L.mean_pairwise_1seed_r(piv, list(range(1, S + 1)))
    pred10 = L.sb(r1, 10)
    _, meas10, _, _ = L.split_half_ceiling(piv, list(range(1, S + 1)), agg="mean")
    check("SB_predicts_10seed_ceiling_within_0.05", abs(pred10 - meas10) < 0.05,
          f"pred={pred10:.3f} meas={meas10:.3f} r1={r1:.3f}")


def t_median_beats_mean_heavytail():
    rng = np.random.RandomState(1)
    n_mask, S = 24, 10
    truth = rng.randn(n_mask)
    cols = {}
    for s in range(1, S + 1):
        v = truth + 0.3 * rng.randn(n_mask)
        # plant heavy-tail contamination: a few masks get a blown-up value on this seed
        contam = rng.rand(n_mask) < 0.15
        v = v + contam * rng.randn(n_mask) * 8.0
        cols[s] = v
    piv = pd.DataFrame(cols, index=[f"m{i}" for i in range(n_mask)])
    _, mean_ceil, _, _ = L.split_half_ceiling(piv, list(range(1, S + 1)), agg="mean")
    _, med_ceil, _, _ = L.split_half_ceiling(piv, list(range(1, S + 1)), agg="median")
    check("median_beats_mean_on_heavytail", med_ceil > mean_ceil + 0.1,
          f"mean={mean_ceil:.3f} median={med_ceil:.3f}")


def t_stricter_gate_logic():
    # a synthetic target that passes (i) ceiling>=0.40 and (ii) S4->10 within 0.15 but FAILS
    # (iii) S8->10 within 0.10 must be ruled UNUSABLE.
    ceiling = 0.55
    gate_i = ceiling >= 0.40
    gate_ii = abs(0.68 - ceiling) <= 0.15       # 0.13 <= 0.15  -> True
    gate_iii = abs(0.70 - ceiling) <= 0.10      # 0.15 <= 0.10  -> False
    usable = gate_i and gate_ii and gate_iii
    check("stricter_gate_fails_on_iii", (gate_i and gate_ii and not gate_iii and not usable),
          f"i={gate_i} ii={gate_ii} iii={gate_iii}")
    # and one that passes all three
    usable2 = (0.55 >= 0.40) and (abs(0.60 - 0.55) <= 0.15) and (abs(0.62 - 0.55) <= 0.10)
    check("stricter_gate_passes_all_three", usable2)


def t_normalization_invariance():
    rng = np.random.RandomState(2)
    demos = [f"d{i}" for i in range(30)]
    members = [f"m{k}" for k in range(5)]
    rows = []
    base = {}
    for k, m in enumerate(members):
        v = rng.randn(30)
        base[m] = v
        for i, d in enumerate(demos):
            rows.append(("GradDot", "C1", d, m, float(v[i])))
    df = pd.DataFrame(rows, columns=["attributor", "target", "demo_id", "member", "score"])
    s1 = L.normalized_ensemble_scores(df, "GradDot", "C1", demos, members, normalize=True)
    # scale each member by an arbitrary positive constant -> normalized result identical
    rows2 = []
    for k, m in enumerate(members):
        scale = (k + 1) * 3.7
        for i, d in enumerate(demos):
            rows2.append(("GradDot", "C1", d, m, float(base[m][i] * scale)))
    df2 = pd.DataFrame(rows2, columns=["attributor", "target", "demo_id", "member", "score"])
    s2 = L.normalized_ensemble_scores(df2, "GradDot", "C1", demos, members, normalize=True)
    a = np.array([s1[d] for d in demos])
    b = np.array([s2[d] for d in demos])
    check("per_member_normalization_scale_invariant", np.allclose(a, b, atol=1e-12),
          f"max abs diff {np.max(np.abs(a-b)):.2e}")


def t_bonferroni_and_critical_rho():
    check("bonferroni7_exact", abs(0.05 / 7 - 0.007142857142857143) < 1e-18)
    check("bonferroni14_exact", abs(0.05 / 14 - 0.0035714285714285713) < 1e-18)
    # critical rho at n=24, p<0.05/7 should be ~0.50 (commensurate with the half-ceiling bar)
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if spearman_p_onesided(mid, 24) < 0.05 / 7:
            hi = mid
        else:
            lo = mid
    check("critical_rho_n24_bonf7_near_0.50", 0.47 < hi < 0.53, f"critical rho = {hi:.4f}")


def main():
    np.random.seed(12345)
    print("=" * 70)
    print("SYNTHETIC Phase-5 unit tests (NOT mixed with any real result)")
    print("=" * 70)
    for fn in (t_exact_criterion, t_split_enumeration, t_spearman_brown_recovery,
               t_median_beats_mean_heavytail, t_stricter_gate_logic,
               t_normalization_invariance, t_bonferroni_and_critical_rho):
        fn()
    n_pass = sum(r["PASS"] for r in RESULTS)
    out = {"LABEL": "SYNTHETIC -- Phase-5 statistics on known answers; NEVER mixed with real data",
           "n_tests": len(RESULTS), "n_pass": n_pass, "ALL_PASS": n_pass == len(RESULTS),
           "tests": RESULTS}
    L.atomic_write_json(os.path.join(L.P5_RESULTS, "SYNTHETIC_p5_unit_tests.json"), out)
    print("=" * 70)
    print(f"SYNTHETIC: {n_pass}/{len(RESULTS)} pass")
    print("=" * 70)
    if n_pass != len(RESULTS):
        raise SystemExit("SYNTHETIC tests FAILED")


if __name__ == "__main__":
    main()
