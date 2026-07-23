"""P5 analysis (DIAGNOSTIC, n=1 target -- do not generalize beyond C5).

Phase 1's RQ4 catastrophe: influence-top-15 selection on C5 scored -16.1 pts vs target-only, and
the selector had taken 9 outsiders for a 7-task cluster. Was that the influence SCORES, or the
missing COVERAGE constraint?

Paired at the SAME seeds (501-503), on C5's probe tasks at 20 rollouts/task:
    coverage-constrained  (NEW, 3 retrains)   vs  unconstrained influence-top-15 (Phase-1 runs)
    coverage-constrained                      vs  target-only                    (Phase-1 runs)
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np
from scipy import stats

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT, RUNS  # noqa: E402

P2 = os.path.join(ROOT, "phase2")
SEEDS = [501, 502, 503]
TARGET = "C5"


def succ(run_dir):
    p = os.path.join(run_dir, "outcomes.json")
    if not os.path.exists(p):
        return None
    return Fraction(json.load(open(p))["outcomes"][TARGET]["success_rate"]).limit_denominator(10**6)


def main():
    arms = {
        "coverage_constrained": [os.path.join(P2, "runs/P5", f"{TARGET}_influence_cov15_s{s}")
                                 for s in SEEDS],
        "unconstrained": [os.path.join(RUNS, "stage_I", f"{TARGET}_influence_top15_s{s}")
                          for s in SEEDS],
        "target_only": [os.path.join(RUNS, "stage_I", f"{TARGET}_target_only_s{s}")
                        for s in SEEDS],
        "random15": [os.path.join(RUNS, "stage_I", f"{TARGET}_random15_s{s}") for s in SEEDS],
    }
    per = {}
    for k, ds in arms.items():
        v = [succ(d) for d in ds]
        if any(x is None for x in v):
            print(f"[P5] {k}: MISSING outcomes -> did not run")
            per[k] = None
            continue
        per[k] = v

    sel = json.load(open(f"{P2}/results/p5_selection.json"))
    out = {"stage": "P5", "target": TARGET, "seeds": SEEDS,
           "DIAGNOSTIC_ONLY": "n=1 target; do NOT generalize beyond C5",
           "selection": {"attributor": sel["attributor"],
                         "unconstrained_outsiders": sel["unconstrained_outsiders"],
                         "coverage_constrained_outsiders": sel["coverage_constrained_outsiders"],
                         "overlap_demos": sel["n_overlap"], "n_tasks": sel["n_tasks"]},
           "per_arm_success_pct": {k: ([float(x) * 100 for x in v] if v else None)
                                   for k, v in per.items()},
           "mean_success_pct": {k: (float(sum(v) / len(v)) * 100 if v else None)
                                for k, v in per.items()},
           "paired_margins_pts": {}}

    base = per["coverage_constrained"]
    if base:
        for other in ("unconstrained", "target_only", "random15"):
            if not per[other]:
                out["paired_margins_pts"][other] = "did not run"
                continue
            d = [(a - b) * 100 for a, b in zip(base, per[other])]
            m = sum(d) / len(d)
            t = stats.ttest_1samp([float(x) for x in d], 0.0)
            out["paired_margins_pts"][other] = {
                "per_seed": [float(x) for x in d],
                "mean_exact": str(m), "mean": float(m),
                "t_p_twosided": float(t.pvalue), "n": len(d),
            }
    json.dump(out, open(f"{P2}/results/p5_coverage_fix.json", "w"), indent=1, default=float)

    print("=" * 88)
    print("P5 -- RQ4 COVERAGE FIX ON C5 (DIAGNOSTIC, n=1 target -- do not generalize)")
    print("=" * 88)
    print(f"  attributor={sel['attributor']}, C5 has {sel['n_tasks']} tasks, B=15")
    print(f"  unconstrained  : {sel['unconstrained_outsiders']}/15 outsiders (Phase 1)")
    print(f"  coverage-fixed : {sel['coverage_constrained_outsiders']}/15 outsiders, "
          f"all {sel['n_tasks']} tasks covered ({sel['n_overlap']}/15 demos shared)")
    print("-" * 88)
    for k in ("coverage_constrained", "unconstrained", "target_only", "random15"):
        m = out["mean_success_pct"][k]
        print(f"  {k:22s} {m:6.2f}%" if m is not None else f"  {k:22s}  did not run")
    print("-" * 88)
    for other, v in out["paired_margins_pts"].items():
        if isinstance(v, dict):
            print(f"  coverage-fixed - {other:14s} = {v['mean']:+6.2f} pts "
                  f"(per-seed {[round(x,1) for x in v['per_seed']]}, p={v['t_p_twosided']:.3f})")
    print("=" * 88)
    print("  Phase-1 reference: unconstrained influence-top-15 was -16.1 pts vs target-only.")
    print("=" * 88)


if __name__ == "__main__":
    main()
