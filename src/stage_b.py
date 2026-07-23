"""STAGE B -- GATE 0 (STOP/GO).

Design (spec §4):
  target-only (the target cluster's 15 demos) vs co-train (all 135), 5 seeds each, PAIRED BY
  SEED. Each model is evaluated on EVERY task of the target cluster x 20 rollouts
  (for C1 that is the full libero_goal suite: 10 tasks x 20 = 200 episodes/model).

Criterion:
  mean paired margin (co-train - target-only, averaged over the cluster's tasks)
    >= +5 success points AND one-sided paired t-test p < 0.05 (df=4, t > 2.132)

On FAIL for C1: repeat the SAME paired design for C2 and C5. If no target passes ->
GATE0_FAIL.md and HALT. If some pass -> proceed, noting heterogeneity is itself a finding.
"""
import os
import sys
import json
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import ROOT, RUNS, RESULTS
import dataset
import orchestrator as O

SEEDS = [101, 102, 103, 104, 105]
N_ROLLOUTS = 20
MARGIN_THRESHOLD = 5.0      # success POINTS (percent)
ALPHA = 0.05
T_CRIT = 2.132              # one-sided, df=4


def jobs_for(target):
    ids, by_c = dataset.train_pool()
    out = []
    for cond, demos in (("target", by_c[target]), ("cotrain", ids)):
        for s in SEEDS:
            out.append({
                "run_dir": os.path.join(RUNS, "stage_B", f"{target}_{cond}_s{s}"),
                "demos": demos, "seed": s, "n_rollouts": N_ROLLOUTS,
                "eval": "cluster_tasks", "target": target, "workers": 12,
                "cond": cond,
            })
    return out


def analyze(target):
    """Per-seed paired margins from cluster_eval.json artifacts. Returns the verdict dict."""
    rows = {}
    for cond in ("target", "cotrain"):
        for s in SEEDS:
            p = os.path.join(RUNS, "stage_B", f"{target}_{cond}_s{s}", "cluster_eval.json")
            if not os.path.exists(p):
                rows[(cond, s)] = None
                continue
            rows[(cond, s)] = json.load(open(p))

    missing = [k for k, v in rows.items() if v is None]
    per_seed = []
    for s in SEEDS:
        t, c = rows[("target", s)], rows[("cotrain", s)]
        if t is None or c is None:
            continue
        tasks = sorted(t["per_task_success"])
        tv = np.array([t["per_task_success"][k] for k in tasks]) * 100
        cv = np.array([c["per_task_success"][k] for k in tasks]) * 100
        per_seed.append({
            "seed": s,
            "target_only_success_pct": float(tv.mean()),
            "cotrain_success_pct": float(cv.mean()),
            "margin_pts": float(cv.mean() - tv.mean()),
            "n_tasks": len(tasks),
            "per_task_margin_pts": {k: float(cv[i] - tv[i]) for i, k in enumerate(tasks)},
        })

    margins = np.array([r["margin_pts"] for r in per_seed])
    res = {"target": target, "n_seeds": len(per_seed), "seeds": SEEDS,
           "n_rollouts_per_task": N_ROLLOUTS, "missing_runs": [f"{c}_s{s}" for c, s in missing],
           "per_seed": per_seed,
           "mean_margin_pts": float(margins.mean()) if len(margins) else None,
           "sd_margin_pts": float(margins.std(ddof=1)) if len(margins) > 1 else None,
           "mean_target_only_pct": float(np.mean([r["target_only_success_pct"] for r in per_seed]))
                                   if per_seed else None,
           "mean_cotrain_pct": float(np.mean([r["cotrain_success_pct"] for r in per_seed]))
                               if per_seed else None}
    if len(margins) >= 2:
        t, p_two = stats.ttest_1samp(margins, 0.0)
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
        res.update({"t_stat": float(t), "p_onesided": float(p_one), "df": len(margins) - 1})
        # Success rates are exact rationals (k / n_rollouts), so the mean margin is an exact
        # rational too. Comparing it to the +5.0 threshold in binary floating point can flip a
        # margin that is EXACTLY 5 to "below 5" (C5's margin is exactly 175/7/5 = 5, but the
        # float sum evaluates to 4.999999999999999). Evaluate the preregistered criterion in
        # exact arithmetic. This fixes HOW the criterion is evaluated; the criterion itself is
        # unchanged.
        from fractions import Fraction
        exact = []
        for r in per_seed:
            per = r["per_task_margin_pts"]
            exact.append(sum(Fraction(round(v * 100), 100) for v in per.values()) / len(per))
        exact_mean = sum(exact) / len(exact)
        res["exact_mean_margin_pts"] = float(exact_mean)
        res["pass_margin"] = bool(exact_mean >= Fraction(int(MARGIN_THRESHOLD)))
        res["pass_ttest"] = bool(p_one < ALPHA and t > 0)
        res["PASS"] = bool(res["pass_margin"] and res["pass_ttest"])
    else:
        res.update({"t_stat": None, "p_onesided": None, "PASS": None})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="C1", help="comma list; C1 first, fallbacks C2,C5")
    ap.add_argument("--analyze_only", action="store_true")
    a = ap.parse_args()
    targets = a.targets.split(",")

    all_res = {}
    for tgt in targets:
        if not a.analyze_only:
            jobs = jobs_for(tgt)
            O.run_jobs(jobs, f"stage_B_{tgt}")
        all_res[tgt] = analyze(tgt)
        r = all_res[tgt]
        print(f"\n=== GATE 0 / target {tgt} ===")
        for ps in r["per_seed"]:
            print(f"  seed {ps['seed']}: target-only={ps['target_only_success_pct']:.1f}%  "
                  f"co-train={ps['cotrain_success_pct']:.1f}%  "
                  f"margin={ps['margin_pts']:+.1f} pts")
        if r["PASS"] is not None:
            print(f"  mean margin = {r['mean_margin_pts']:+.2f} pts (SD {r['sd_margin_pts']:.2f}), "
                  f"t={r['t_stat']:.3f}, one-sided p={r['p_onesided']:.4f}")
            print(f"  criterion: margin >= +{MARGIN_THRESHOLD} pts -> {r['pass_margin']}; "
                  f"p < {ALPHA} -> {r['pass_ttest']}")
            print(f"  VERDICT: {'PASS' if r['PASS'] else 'FAIL'}")

    out = {"gate": "GATE 0 (Stage B)", "criterion":
           f"mean paired margin >= +{MARGIN_THRESHOLD} pts AND one-sided paired t-test p < {ALPHA}",
           "targets": all_res,
           "any_pass": any(v["PASS"] for v in all_res.values() if v["PASS"] is not None)}
    json.dump(out, open(os.path.join(RESULTS, "stage_B_gate0.json"), "w"), indent=1)
    print(f"\n[stage_B] any target passed: {out['any_pass']}")
    return out


if __name__ == "__main__":
    main()
