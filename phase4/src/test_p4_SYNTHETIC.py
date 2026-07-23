"""SYNTHETIC unit tests for the NEW Phase-4 machinery. Labelled SYNTHETIC; never mixed with
any real result.

Phase 4 introduces four pieces of statistics that no prior phase validated, and all four decide
verdicts:

  1. EXACT RATIONAL CRITERION ARITHMETIC (p4lib.meets_half_ceiling). Must agree with the exact
     rational comparison of the two doubles, including at an exact tie and at a one-ULP miss --
     where a float comparison with an epsilon would give the wrong answer.
  2. SPEARMAN-BROWN (p4lib.sb) and the k-vs-k split-half ceiling (p4lib.split_half_ceiling),
     which must recover a KNOWN reliability on data built with a known signal/noise ratio, and
     must enumerate exactly the right number of distinct splits (126 at S=10, 35 at S=8, 3 at
     S=4, 10 at S=6).
  3. The MEDIAN aggregator arm of the ceiling -- the heavy-tail remedy P13 preregisters. On a
     synthetic heavy-tailed outcome it must BEAT the mean arm, and on a clean Gaussian outcome
     it must roughly MATCH it. (If the median did not beat the mean on a planted heavy tail, the
     remedy would not be a remedy.)
  4. PER-MEMBER SCALE NORMALIZATION (p4lib.normalized_ensemble_scores): a member whose scores are
     rescaled by any positive constant must not change the normalized ensemble ranking, and a
     single huge-norm member must NOT be allowed to dominate (which is exactly what it does
     without normalization -- the P6.5 finding, reproduced here on synthetic data).

Plus the P7 coarse predictor's endpoint identity at g=1, which P14 relies on.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS

sys.path.insert(0, L.P3_SRC)
from p7_grain_ladder import coarse_pred, groups_random  # noqa: E402

sys.path.insert(0, os.path.join(L.ROOT, "src"))
from lds import spearman, mask_pred_score  # noqa: E402

R = []


def check(name, ok, detail=""):
    R.append({"test": name, "PASS": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


# ---------------------------------------------------------------- 1. exact criterion arithmetic
def t1_exact_arithmetic():
    print("\n[1] EXACT RATIONAL CRITERION ARITHMETIC")
    # exact tie: rho is EXACTLY half the ceiling in binary -> must PASS (>=)
    ceil = 0.5
    rho = 0.25
    check("exact tie 0.25 vs 0.5*0.5 -> PASS (>=)", L.meets_half_ceiling(rho, ceil) is True,
          f"rho={rho!r} ceil={ceil!r}")

    # one ULP below the tie -> must FAIL. A naive `rho >= 0.5*ceil - 1e-9` would wrongly pass.
    rho_lo = np.nextafter(0.25, -np.inf)
    ok = (L.meets_half_ceiling(rho_lo, ceil) is False)
    check("one ULP below the bar -> FAIL", ok, f"rho={rho_lo!r} (bar={0.5*ceil!r})")

    # one ULP above -> must PASS
    rho_hi = np.nextafter(0.25, np.inf)
    check("one ULP above the bar -> PASS", L.meets_half_ceiling(rho_hi, ceil) is True,
          f"rho={rho_hi!r}")

    # a case where 0.5*ceil is NOT exactly representable: ceil = 0.1 -> bar = 0.05
    # 0.1/2 in binary is exact (halving is exact), so build a harder one: compare against the
    # exact rational, not against a recomputed float.
    from fractions import Fraction
    rng = np.random.default_rng(4)
    agree = 0
    for _ in range(20000):
        r = float(rng.uniform(-1, 1))
        c = float(rng.uniform(0.01, 1.0))
        exact = Fraction(r) >= Fraction(1, 2) * Fraction(c)
        if L.meets_half_ceiling(r, c) == exact:
            agree += 1
    check("20000 random (rho, ceiling): agrees with exact rational", agree == 20000,
          f"{agree}/20000")

    # non-finite must never pass
    check("NaN ceiling -> FAIL", L.meets_half_ceiling(0.9, np.nan) is False)
    check("NaN rho -> FAIL", L.meets_half_ceiling(np.nan, 0.9) is False)


# ---------------------------------------------------------------- 2. Spearman-Brown + ceiling
def t2_sb_and_ceiling():
    print("\n[2] SPEARMAN-BROWN AND THE SPLIT-HALF CEILING")
    # SB algebra
    check("sb(r,1) == r", abs(L.sb(0.37, 1) - 0.37) < 1e-12)
    check("sb identity: sb(r,2) == 2r/(1+r)", abs(L.sb(0.4, 2) - 2 * 0.4 / 1.4) < 1e-12)
    # composition: sb(sb(r,2) as a half-value) round-trips
    r1 = 0.3
    r5 = L.sb(r1, 5)
    check("sb_half_to_full(sb(r1,5)) == sb(r1,10)",
          abs(L.sb_half_to_full(r5) - L.sb(r1, 10)) < 1e-9,
          f"{L.sb_half_to_full(r5):.6f} vs {L.sb(r1,10):.6f}")

    # split enumeration counts: C(2k,k)/2 distinct balanced splits
    rng = np.random.default_rng(0)
    for S, expect in ((4, 3), (6, 10), (8, 35), (10, 126)):
        seeds = list(range(600, 600 + S))
        piv = pd.DataFrame(rng.normal(size=(24, S)), columns=seeds,
                           index=[f"M{i:03d}" for i in range(24)])
        _, _, vals, n = L.split_half_ceiling(piv, seeds, agg="mean")
        check(f"S={S}: enumerates exactly {expect} distinct {S//2}v{S//2} splits", n == expect,
              f"got {n}")

    # RECOVERY: build outcomes with a known per-seed reliability and check SB recovers it.
    # y[mask,seed] = signal[mask] + noise ; 1-seed reliability r1 ~ var_s/(var_s+var_n).
    rng = np.random.default_rng(7)
    n_masks, S = 24, 10
    seeds = list(range(401, 401 + S))
    sig = rng.normal(size=n_masks)
    for var_n in (0.5, 1.0, 2.0):
        Y = sig[:, None] + rng.normal(scale=np.sqrt(var_n), size=(n_masks, S))
        piv = pd.DataFrame(Y, columns=seeds, index=[f"M{i:03d}" for i in range(n_masks)])
        r1, _ = L.mean_pairwise_1seed_r(piv, seeds)
        r5, r10_sb, _, _ = L.split_half_ceiling(piv, seeds, agg="mean")
        pred10 = L.sb(r1, 10)
        # SB from r1 must predict the measured 10-seed SB ceiling closely on CLEAN data.
        # (This is exactly the P12 instrument gate, run on data where it MUST pass.)
        check(f"SB consistency on clean data (var_n={var_n}): |pred-meas| < 0.10",
              abs(pred10 - r10_sb) < 0.10,
              f"r1={r1:.3f} pred10={pred10:.3f} meas10={r10_sb:.3f} "
              f"diff={abs(pred10-r10_sb):.3f}")


# ---------------------------------------------------------------- 3. the median remedy
def t3_median_vs_mean():
    print("\n[3] THE MEDIAN AGGREGATOR (P13's preregistered heavy-tail remedy)")
    rng = np.random.default_rng(11)
    n_masks, S = 24, 8
    seeds = list(range(601, 601 + S))
    sig = rng.normal(size=n_masks)
    idx = [f"G{i:03d}" for i in range(n_masks)]

    # (a) CLEAN Gaussian noise: mean and median arms should roughly MATCH (mean slightly better).
    Y = sig[:, None] + rng.normal(scale=1.0, size=(n_masks, S))
    piv = pd.DataFrame(Y, columns=seeds, index=idx)
    _, c_mean, _, _ = L.split_half_ceiling(piv, seeds, agg="mean")
    _, c_med, _, _ = L.split_half_ceiling(piv, seeds, agg="median")
    check("clean data: median ceiling within 0.15 of the mean ceiling",
          abs(c_mean - c_med) < 0.15, f"mean={c_mean:.3f} median={c_med:.3f}")

    # (b) HEAVY-TAILED contamination -- the planted diffusion pathology: with prob 0.15 a seed's
    #     value blows up (the DDIM wrong-basin event). The MEDIAN arm must BEAT the mean arm.
    Y = sig[:, None] + rng.normal(scale=0.5, size=(n_masks, S))
    blow = rng.random((n_masks, S)) < 0.15
    Y = Y + blow * rng.normal(loc=8.0, scale=4.0, size=(n_masks, S))
    piv = pd.DataFrame(Y, columns=seeds, index=idx)
    _, c_mean, _, _ = L.split_half_ceiling(piv, seeds, agg="mean")
    _, c_med, _, _ = L.split_half_ceiling(piv, seeds, agg="median")
    check("HEAVY TAIL: the median ceiling BEATS the mean ceiling", c_med > c_mean,
          f"mean={c_mean:.3f} median={c_med:.3f} gain={c_med-c_mean:+.3f}")
    check("HEAVY TAIL: the mean ceiling is degraded (< 0.8)", c_mean < 0.8,
          f"mean={c_mean:.3f}")


# ---------------------------------------------------------------- 4. per-member normalization
def t4_normalization():
    print("\n[4] PER-MEMBER SCALE NORMALIZATION")
    rng = np.random.default_rng(3)
    demos = [f"d{i:03d}" for i in range(135)]
    members = [f"m{i}" for i in range(5)]
    truth = rng.normal(size=135)

    def frame(scale):
        rows = []
        for k, m in enumerate(members):
            s = truth + rng.normal(scale=0.7, size=135)      # each member: signal + its own noise
            s = s * scale[k]
            for d, v in zip(demos, s):
                rows.append(("TracIn", "C1", d, m, float(v)))
        return pd.DataFrame(rows, columns=["attributor", "target", "demo_id", "member", "score"])

    # (a) rescale-invariance: multiplying members by arbitrary positive constants must not move
    #     the NORMALIZED ensemble ranking.
    df1 = frame([1.0] * 5)
    piv = df1.pivot_table(index="demo_id", columns="member", values="score")
    df2 = df1.copy()
    fac = {"m0": 100.0, "m1": 0.01, "m2": 3.0, "m3": 1.0, "m4": 50.0}
    df2["score"] = df2.apply(lambda r: r["score"] * fac[r["member"]], axis=1)

    a = L.normalized_ensemble_scores(df1, "TracIn", "C1", demos, members, normalize=True)
    b = L.normalized_ensemble_scores(df2, "TracIn", "C1", demos, members, normalize=True)
    rho = spearman([a[d] for d in demos], [b[d] for d in demos])
    check("normalized ensemble is INVARIANT to per-member rescaling (Spearman == 1)",
          abs(rho - 1.0) < 1e-9, f"rho={rho:.12f}")

    # (b) WITHOUT normalization, one huge-norm member dominates -- the P6.5 finding, synthetic.
    u = L.normalized_ensemble_scores(df2, "TracIn", "C1", demos, members, normalize=False)
    m0 = dict(zip(piv.index, piv["m0"].values * 100.0))
    rho_dom = spearman([u[d] for d in demos], [m0[d] for d in demos])
    rho_norm = spearman([b[d] for d in demos], [m0[d] for d in demos])
    check("UNNORMALIZED mean is dominated by the huge-norm member (rho > 0.95)", rho_dom > 0.95,
          f"rho(unnorm, m0)={rho_dom:.3f}")
    check("NORMALIZED mean is NOT dominated by it (rho < unnorm)", rho_norm < rho_dom,
          f"rho(norm, m0)={rho_norm:.3f} < {rho_dom:.3f}")

    # (c) a missing member cell must RAISE, never silently shorten the ensemble
    df3 = df1[~((df1.member == "m2") & (df1.demo_id == "d007"))]
    try:
        L.normalized_ensemble_scores(df3, "TracIn", "C1", demos, members)
        check("missing (member, demo) cell RAISES", False, "did not raise")
    except RuntimeError:
        check("missing (member, demo) cell RAISES", True)


# ---------------------------------------------------------------- 5. coarse predictor endpoint
def t5_coarse_predictor():
    print("\n[5] THE P7 COARSE PREDICTOR (P14 reuses it verbatim)")
    rng = np.random.default_rng(5)
    clusters = [f"C{i}" for i in range(1, 10)]
    by_c = {c: [f"{c}/demo_{i}" for i in range(15)] for c in clusters}
    all_demos = [d for c in clusters for d in by_c[c]]
    scores = {d: float(rng.normal()) for d in all_demos}
    mask = list(rng.choice(all_demos, 68, replace=False))

    for g in (1, 3, 5, 15):
        groups = [gr for c in clusters for gr in groups_random(by_c[c], g)]
        assert all(len(gr) == g for gr in groups), f"g={g}: unequal groups"
        cp = coarse_pred(mask, groups, scores, g)
        if g == 1:
            dp = mask_pred_score(scores, mask)
            check("g=1 coarse predictor == the plain demo predictor (EXACT)",
                  abs(cp - dp) < 1e-9, f"coarse={cp:.9f} demo={dp:.9f}")
        check(f"g={g}: predictor is finite", np.isfinite(cp), f"{cp:.4f}")

    # a mask containing EVERY demo: coarse pred == sum of all scores, for every g
    total = sum(scores.values())
    for g in (1, 3, 5, 15):
        groups = [gr for c in clusters for gr in groups_random(by_c[c], g)]
        cp = coarse_pred(all_demos, groups, scores, g)
        check(f"g={g}: full mask -> sum of all scores (EXACT)", abs(cp - total) < 1e-9,
              f"{cp:.9f} vs {total:.9f}")


def main():
    print("=" * 78)
    print("SYNTHETIC UNIT TESTS -- Phase-4 statistics  (SYNTHETIC DATA ONLY, NEVER MIXED)")
    print("=" * 78)
    t1_exact_arithmetic()
    t2_sb_and_ceiling()
    t3_median_vs_mean()
    t4_normalization()
    t5_coarse_predictor()

    n_pass = sum(r["PASS"] for r in R)
    print("\n" + "=" * 78)
    print(f"SYNTHETIC: {n_pass}/{len(R)} PASS")
    print("=" * 78)
    L.atomic_write_json(os.path.join(P4_RESULTS, "SYNTHETIC_p4_unit_tests.json"),
                        {"LABEL": "SYNTHETIC -- synthetic data only, never mixed with any real "
                                  "result", "n_tests": len(R), "n_pass": n_pass,
                         "ALL_PASS": n_pass == len(R), "tests": R})
    return 0 if n_pass == len(R) else 1


if __name__ == "__main__":
    sys.exit(main())
