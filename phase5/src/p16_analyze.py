"""P16 -- GRADDOT BREADTH TEST on the 7 non-focal BC targets. ZERO retrains.

The paper's strongest positive sentence rests on ONE target: on C1, GradDot (the exact lambda->inf
limit, E=20, per-member normalized) reached 0.54 of a 0.951 ceiling, p=0.0052 (P11 secondary). Is
that C1-specific? P16 measures the breadth of that estimator on the 7 targets never confirmatorily
tested at demo grain with GradDot. C1 and C5 are EXCLUDED as spent (P11 tested both).

Ground truth: the EXISTING S=10 BC outcomes (phase4/results/p12_outcomes_S10.parquet), held-out L2,
seed MEAN. Ceilings: the per-target S=10 ceilings from phase4/results/p12_ceilings.json. No retrains.

PRIMARY (frozen bit-for-bit to P11's secondary): GradDot = raw gradient dot product K toward each
target's plain-L2 functional, E=20 members (201-220), per-member unit-L2 normalization. Gram caches:
phase3/results/p6_gram_cache.npz (201-210) + phase4/results/p11_gram_cache_new_members.npz (211-220)
-- the SAME two caches P11's graddot_frame read. Bonferroni-7: p < 0.05/7.
SECONDARY (preregistered): GradDot_dmean = K_m / mean(diag(G_m)) averaged over members (the literal
lambda->inf ensemble-MEAN form). Bonferroni-14: p < 0.05/14.

CRITERION per target: rho >= 0.5 * that target's S=10 ceiling (EXACT rational) AND one-sided p <
the family's Bonferroni threshold. n = 24 masks. At n=24, p<0.00714 the critical rho is ~0.50, so
the half-ceiling bar and significance are COMMENSURATE -- both components reported; PASS needs both.

SELF-VALIDATION before the verdict: (1) recompute P11's C1 GradDot from the cache path used here
and confirm rho = +0.5130434782608695 EXACTLY; (2) probe-leak guard on all 20 members.
"""
import glob
import json
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5lib as L
from p5lib import P5_RESULTS, P4_RESULTS, P3_RESULTS, RESULTS, RUNS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci, mask_pred_score  # noqa: E402

TARGETS = ["C2", "C3", "C4", "C6", "C7", "C8", "C9"]      # the 7 non-focal; C1/C5 spent
MEMBERS = [f"ens_s{s}" for s in range(201, 221)]          # E = 20
ALPHA_PRIMARY = 0.05 / 7                                  # Bonferroni-7
ALPHA_SECONDARY = 0.05 / 14                               # Bonferroni-14
OUTCOME = "neg_plain_loss"
C1_GRADDOT_ARCHIVED = 0.5130434782608695                 # P11 archived C1 GradDot_E20_normalized


def graddot_frame(train_ids, clusters):
    """IDENTICAL to phase4/src/p11_analyze.py:graddot_frame -- the two Gram caches, bit-for-bit."""
    rows = []
    for path in (os.path.join(P3_RESULTS, "p6_gram_cache.npz"),
                 os.path.join(P4_RESULTS, "p11_gram_cache_new_members.npz")):
        Z = np.load(path, allow_pickle=True)
        K, G = Z["K"], Z["G"]
        mem, tids, tgts = list(Z["members"]), list(Z["train_ids"]), list(Z["targets"])
        assert tids == list(train_ids), f"{path}: train_ids order differs"
        assert tgts == list(clusters), f"{path}: target order differs"
        d = np.array([np.mean(np.diag(G[m])) for m in range(K.shape[0])])
        for mi, m in enumerate(mem):
            for j, c in enumerate(tgts):
                for i, t in enumerate(tids):
                    rows.append(("GradDot", c, t, str(m), float(K[mi, i, j])))
                    rows.append(("GradDot_dmean", c, t, str(m), float(K[mi, i, j] / d[mi])))
    return pd.DataFrame(rows, columns=["attributor", "target", "demo_id", "member", "score"])


def member_run_dirs():
    """201-210 -> runs/stage_E; 211-220 -> phase3/runs/P9. Returns list of run dirs with demos."""
    dirs = []
    for s in range(201, 211):
        d = os.path.join(RUNS, "stage_E", f"ens_s{s}")
        assert os.path.isdir(d), f"missing member run dir {d}"
        dirs.append(d)
    for s in range(211, 221):
        d = os.path.join(L.P3_RUNS, "P9", f"ens_s{s}")
        assert os.path.isdir(d), f"missing member run dir {d}"
        dirs.append(d)
    return dirs


def compute_graddot(gd, attr, target, train_ids):
    return L.normalized_ensemble_scores(gd, attr, target, train_ids, MEMBERS,
                                        normalize=(attr == "GradDot"))


def main():
    L.assert_prereg_locked()
    clusters = dataset.clusters()
    train_ids, _ = dataset.train_pool()
    masks = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    assert len(masks) == 24

    # ---------------------------------------------------------------- probe-leak guard (all 20)
    heldout = set(dataset.heldout_pool()[0])
    dirs = member_run_dirs()
    guard = L.assert_no_probe_leak(dirs, heldout, context="P16 GradDot, all 20 BC members")
    print(f"[P16] probe-leak guard PASSED: {guard}")

    # ---------------------------------------------------------------- ground truth (S=10, mean)
    C = json.load(open(os.path.join(P4_RESULTS, "p12_ceilings.json")))
    if not C["INSTRUMENT_GATE"]["PASS"]:
        raise SystemExit("[P16] P12 instrument gate FAILED -- no ground truth.")
    df = pd.read_parquet(os.path.join(P4_RESULTS, "p12_outcomes_S10.parquet"))
    SEEDS = C["seeds"]
    ceilings = {t: C["targets"][t][f"ceiling_{len(SEEDS)}seed_SB"] for t in clusters}
    obs = {t: df[df.target == t].groupby("mask_id")[OUTCOME].mean().to_dict() for t in clusters}
    print(f"[P16] ground truth: p12 S={len(SEEDS)}, seed MEAN, held-out L2")

    gd = graddot_frame(train_ids, clusters)

    # ---------------------------------------------------------------- SELF-VALIDATION: C1
    sc_c1 = compute_graddot(gd, "GradDot", "C1", train_ids)
    pred_c1 = np.array([mask_pred_score(sc_c1, m["demos"]) for m in masks])
    out_c1 = np.array([obs["C1"].get(m["mask_id"], np.nan) for m in masks])
    ok = np.isfinite(pred_c1) & np.isfinite(out_c1)
    rho_c1 = spearman(pred_c1[ok], out_c1[ok])
    c1_ok = (Fraction(float(rho_c1)) == Fraction(float(C1_GRADDOT_ARCHIVED)))
    print(f"[P16] SELF-VALIDATION C1 GradDot: rho={rho_c1!r} archived={C1_GRADDOT_ARCHIVED!r} "
          f"exact_match={c1_ok}")
    if not c1_ok:
        raise SystemExit(f"[P16] C1 GradDot self-validation FAILED: {rho_c1} != "
                         f"{C1_GRADDOT_ARCHIVED}. The cache is a different estimator. STOP.")

    # ---------------------------------------------------------------- the 7-target breadth test
    FAMILIES = [("GradDot", "GradDot_E20_normalized", "PRIMARY", ALPHA_PRIMARY),
                ("GradDot_dmean", "GradDot_E20_dmean", "SECONDARY", ALPHA_SECONDARY)]
    res, rows = {}, []
    for t in TARGETS:
        out_v = np.array([obs[t].get(m["mask_id"], np.nan) for m in masks])
        res[t] = {"ceiling": ceilings[t], "bar_half_ceiling": 0.5 * ceilings[t], "estimators": {}}
        for attr, eid, fam, alpha in FAMILIES:
            sc = compute_graddot(gd, attr, t, train_ids)
            pred = np.array([mask_pred_score(sc, m["demos"]) for m in masks])
            okm = np.isfinite(out_v) & np.isfinite(pred)
            rho = spearman(pred[okm], out_v[okm])
            n = int(okm.sum())
            p1 = spearman_p_onesided(rho, n)
            lo, hi = bootstrap_spearman_ci(pred[okm], out_v[okm])
            meets = L.meets_half_ceiling(rho, ceilings[t])
            sig = bool(np.isfinite(p1) and p1 < alpha)
            e = {"estimator": eid, "family": fam, "rho": rho, "n_masks": n, "p_onesided": p1,
                 "ci95": [lo, hi],
                 "ratio_to_ceiling": (rho / ceilings[t]) if ceilings[t] else np.nan,
                 "ratio_exact": L.ratio_exact_str(rho, ceilings[t]),
                 "meets_half_ceiling_EXACT": meets, "alpha_bonferroni": alpha, "p_lt_alpha": sig,
                 "PASS": bool(meets and sig)}
            res[t]["estimators"][eid] = e
            rows.append({"target": t, "estimator": eid, "family": fam, "rho": rho,
                         "ceiling": ceilings[t], "bar": 0.5 * ceilings[t],
                         "ratio": e["ratio_to_ceiling"], "p_onesided": p1, "alpha": alpha,
                         "meets_half_ceiling": meets, "p_lt_alpha": sig, "PASS": e["PASS"]})
    pd.DataFrame(rows).to_csv(os.path.join(P5_RESULTS, "p16_lds_table.csv"), index=False)

    # ---------------------------------------------------------------- count-based verdict
    primary_pass = [t for t in TARGETS if res[t]["estimators"]["GradDot_E20_normalized"]["PASS"]]
    secondary_pass = [t for t in TARGETS if res[t]["estimators"]["GradDot_E20_dmean"]["PASS"]]
    k = len(primary_pass)
    if k >= 2:
        interp = (f"GradDot demo-grain faithfulness is not C1-specific; the paper may state that "
                  f"attribution-done-right works on a minority-to-plurality of targets, with the "
                  f"count (k={k} of 7 non-focal BC targets, plus C1 = {k+1} of 8 tested).")
    elif k == 0:
        interp = ("the achievability result is C1-specific on this corpus; the paper must scope it "
                  "that way (k=0 of the 7 non-focal BC targets pass GradDot).")
    else:
        interp = (f"k={k} of 7 non-focal BC targets pass GradDot; report the count plainly, no "
                  f"stronger sentence than the count supports.")

    # critical-rho note (commensurate bar)
    crit = _critical_rho(24, ALPHA_PRIMARY)
    out = {
        "stage": "P16", "preregistration_sha256": L.PREREG_SHA,
        "ground_truth": f"phase4/results/p12_outcomes_S10.parquet, S={len(SEEDS)}, seed MEAN, "
                        f"held-out L2",
        "targets": TARGETS, "n_masks": 24, "PRIMARY_OUTCOME": OUTCOME,
        "PRIMARY": "GradDot_E20_normalized (raw gradient dot product, E=20, unit-L2 per member)",
        "SECONDARY": "GradDot_E20_dmean (K_m / mean(diag G_m), E=20)",
        "alpha_primary_Bonferroni7": ALPHA_PRIMARY,
        "alpha_secondary_Bonferroni14": ALPHA_SECONDARY,
        "critical_rho_primary_n24": crit,
        "commensurate_note": ("at n=24 and p<0.00714 the critical rho ~ 0.50, so the half-ceiling "
                              "bar and significance are commensurate; both components are reported "
                              "and a PASS requires BOTH"),
        "SELF_VALIDATION": {
            "C1_GradDot_recomputed": float(rho_c1),
            "C1_GradDot_archived": C1_GRADDOT_ARCHIVED,
            "C1_exact_match": bool(c1_ok),
            "probe_leak_guard": guard},
        "PRIMARY_PASS_targets": primary_pass, "k_primary": k,
        "SECONDARY_PASS_targets": secondary_pass, "k_secondary": len(secondary_pass),
        "PREREGISTERED_INTERPRETATION": interp,
        "per_target": res,
    }
    L.atomic_write_json(os.path.join(P5_RESULTS, "p16_verdict.json"), out)

    print("\n" + "=" * 104)
    print(f"P16 -- GRADDOT BREADTH (BC, S={len(SEEDS)}, held-out L2, seed mean, n=24 masks)")
    print(f"critical rho (n=24, p<{ALPHA_PRIMARY:.5f}) ~ {crit:.3f}  |  bar = 0.5*ceiling")
    print("=" * 104)
    print(f"{'target':7s} {'ceiling':>8s} {'bar':>7s} | "
          f"{'PRIMARY rho':>12s} {'ratio':>6s} {'p1':>8s} {'>=bar':>6s} {'sig':>4s} {'PASS':>5s} | "
          f"{'dmean rho':>10s} {'ratio':>6s} {'p1':>8s} {'PASS':>5s}")
    for t in TARGETS:
        p = res[t]["estimators"]["GradDot_E20_normalized"]
        s = res[t]["estimators"]["GradDot_E20_dmean"]
        print(f"{t:7s} {ceilings[t]:8.3f} {0.5*ceilings[t]:7.3f} | "
              f"{p['rho']:+12.3f} {p['ratio_to_ceiling']:6.2f} {p['p_onesided']:8.4f} "
              f"{str(p['meets_half_ceiling_EXACT']):>6s} {str(p['p_lt_alpha']):>4s} "
              f"{('PASS' if p['PASS'] else 'fail'):>5s} | "
              f"{s['rho']:+10.3f} {s['ratio_to_ceiling']:6.2f} {s['p_onesided']:8.4f} "
              f"{('PASS' if s['PASS'] else 'fail'):>5s}")
    print("-" * 104)
    print(f"PRIMARY (Bonferroni-7): k = {k} of 7 pass  [{', '.join(primary_pass) or 'none'}]")
    print(f"SECONDARY (Bonferroni-14): {len(secondary_pass)} of 7 pass  "
          f"[{', '.join(secondary_pass) or 'none'}]")
    print(f"  -> {interp}")
    print("=" * 104)


def _critical_rho(n, alpha):
    """Smallest rho whose one-sided p < alpha at sample size n (bisection on spearman_p_onesided)."""
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if spearman_p_onesided(mid, n) < alpha:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == "__main__":
    main()
