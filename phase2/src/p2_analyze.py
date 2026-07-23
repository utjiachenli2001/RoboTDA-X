"""P2 analysis: does attribution predict the SIGN and RANK of per-task transfer?

Phase 1: C1's cluster-level co-train null is CANCELLATION -- co-training helps 4 of its tasks by
+10..+23 pts and hurts 3 by -11..-22. This asks whether attribution knew which was which.

Measured margin (per task) = mean over seeds 101-105 of
      cotrain per_task_success - target-only per_task_success        (percentage points)
read from the EXISTING Phase-1 Stage-B artifacts (cluster_eval.json already stores
per_task_success at 20 rollouts/task) -- 0 retrains, 0 new episodes.

Predicted benefit (per task):
  PRIMARY   sum of OUTSIDER demo attributions toward that task's L2 functional
  SECONDARY mean(outsider) - mean(insider)

PREREGISTERED TEST: Spearman(predicted, measured) POOLED over the 27 tasks, one-sided alpha=.05
(critical rho ~ 0.32 at n=27). Headline estimator = IF on the Stage-E ensemble. Plus the
sign-agreement rate with an exact two-sided binomial test against 0.5.
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT, RUNS  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci  # noqa: E402

P2 = os.path.join(ROOT, "phase2")
TARGETS = ["C1", "C2", "C5"]
SEEDS = [101, 102, 103, 104, 105]
ATTRS = ["IF", "TRAK", "TracIn"]
PRIMARY_ATTR = "IF"          # preregistered headline estimator
PRIMARY_ENS = "stageE"


def measured_margins():
    """-> DataFrame(task_key, cluster, margin_pts, cotrain_pct, target_pct) from Stage-B."""
    rows = []
    for c in TARGETS:
        per = {}
        for cond in ("target", "cotrain"):
            for s in SEEDS:
                p = os.path.join(RUNS, "stage_B", f"{c}_{cond}_s{s}", "cluster_eval.json")
                ce = json.load(open(p))
                for t, v in ce["per_task_success"].items():
                    per.setdefault((t, cond), []).append(Fraction(v).limit_denominator(10**6))
        tasks = sorted({t for (t, _) in per})
        for t in tasks:
            co, tg = per[(t, "cotrain")], per[(t, "target")]
            assert len(co) == len(tg) == 5, (t, len(co), len(tg))
            m = sum(c_ - t_ for c_, t_ in zip(co, tg)) / 5 * 100      # exact rationals
            rows.append({"task_key": f"{c}/{t}", "cluster": c, "task": t,
                         "margin_pts": float(m), "margin_exact": str(m),
                         "cotrain_pct": float(sum(co) / 5 * 100),
                         "target_pct": float(sum(tg) / 5 * 100),
                         "per_seed_margin": [float((c_ - t_) * 100) for c_, t_ in zip(co, tg)]})
    df = pd.DataFrame(rows)
    assert len(df) == 27, len(df)
    return df


def predicted(ens):
    """-> DataFrame(task_key, attributor, pred_sum_outsider, pred_mean_diff)."""
    f = os.path.join(P2, "results", f"per_task_influence_{ens}.parquet")
    inf = pd.read_parquet(f)
    rows = []
    for (attr, task), g in inf.groupby(["attributor", "task"]):
        out = g[~g.is_insider].score
        ins = g[g.is_insider].score
        rows.append({"task_key": task, "attributor": attr,
                     "pred_sum_outsider": float(out.sum()),
                     "pred_mean_diff": float(out.mean() - ins.mean()),
                     "n_outsider": int(len(out)), "n_insider": int(len(ins))})
    return pd.DataFrame(rows)


def test(pred, meas, label):
    ok = np.isfinite(pred) & np.isfinite(meas)
    p, m = np.asarray(pred)[ok], np.asarray(meas)[ok]
    n = len(p)
    rho = spearman(p, m)
    p1 = spearman_p_onesided(rho, n)
    lo, hi = bootstrap_spearman_ci(p, m)
    agree = int(np.sum(np.sign(p) == np.sign(m)))
    binom = stats.binomtest(agree, n, 0.5, alternative="two-sided").pvalue if n else np.nan
    return {"label": label, "n": n, "rho": rho, "p_onesided": p1, "ci95": [lo, hi],
            "sign_agree": agree, "sign_agree_rate": agree / n if n else np.nan,
            "sign_binom_p_twosided": float(binom),
            "PASS_pooled_alpha05": bool(np.isfinite(p1) and p1 < 0.05 and rho > 0)}


def main():
    meas = measured_margins()
    meas.to_csv(f"{P2}/results/p2_measured_margins.csv", index=False)

    res = {"measured": {"n_tasks": 27,
                        "source": "runs/stage_B/*/cluster_eval.json (per_task_success, 20 rollouts, seeds 101-105)",
                        "n_retrains": 0, "n_new_episodes": 0},
           "by_cluster_check": {}}
    for c in TARGETS:
        s = meas[meas.cluster == c]
        res["by_cluster_check"][c] = {
            "cluster_margin_pts_mean": float(s.margin_pts.mean()),
            "n_helped": int((s.margin_pts > 0).sum()), "n_hurt": int((s.margin_pts < 0).sum()),
            "max_help": float(s.margin_pts.max()), "max_hurt": float(s.margin_pts.min()),
        }

    res["tests"] = {}
    for ens in ["stageE", "stageB"]:
        f = os.path.join(P2, "results", f"per_task_influence_{ens}.parquet")
        if not os.path.exists(f):
            res["tests"][ens] = "did not run"
            continue
        pr = predicted(ens)
        res["tests"][ens] = {}
        for attr in ATTRS:
            sub = pr[pr.attributor == attr].merge(meas, on="task_key")
            e = {}
            for pcol in ("pred_sum_outsider", "pred_mean_diff"):
                e[pcol] = {"pooled": test(sub[pcol].values, sub.margin_pts.values, "pooled27")}
                for c in TARGETS:
                    s = sub[sub.cluster == c]
                    e[pcol][c] = test(s[pcol].values, s.margin_pts.values, c)
            res["tests"][ens][attr] = e
            sub.to_csv(f"{P2}/results/p2_pred_vs_meas_{ens}_{attr}.csv", index=False)

    hp = res["tests"].get(PRIMARY_ENS)
    if isinstance(hp, dict):
        h = hp[PRIMARY_ATTR]["pred_sum_outsider"]["pooled"]
        res["PREREGISTERED_PRIMARY"] = {
            "estimator": f"{PRIMARY_ATTR} on {PRIMARY_ENS}, pred=sum of outsider attributions",
            "test": "Spearman(predicted, measured margin) pooled over 27 tasks, one-sided a=0.05",
            **h, "PASS": h["PASS_pooled_alpha05"],
        }
    json.dump(res, open(f"{P2}/results/p2_transfer_sign.json", "w"), indent=1, default=float)

    print("=" * 96)
    print("P2 -- PER-TASK TRANSFER-SIGN PREDICTION (27 tasks; 0 retrains, 0 new episodes)")
    print("=" * 96)
    for c in TARGETS:
        b = res["by_cluster_check"][c]
        print(f"  {c}: cluster margin {b['cluster_margin_pts_mean']:+6.2f} pts | "
              f"{b['n_helped']} tasks helped (max {b['max_help']:+.1f}), "
              f"{b['n_hurt']} hurt (max {b['max_hurt']:+.1f})   <- cancellation check")
    print("-" * 96)
    for ens in ["stageE", "stageB"]:
        if not isinstance(res["tests"].get(ens), dict):
            print(f"  {ens}: did not run")
            continue
        for attr in ATTRS:
            e = res["tests"][ens][attr]["pred_sum_outsider"]["pooled"]
            star = " <-- PREREGISTERED PRIMARY" if (ens == PRIMARY_ENS and attr == PRIMARY_ATTR) else ""
            print(f"  {ens:7s} {attr:7s} pooled n={e['n']} rho={e['rho']:+.3f} "
                  f"p1={e['p_onesided']:.4f} sign={e['sign_agree']}/{e['n']} "
                  f"({e['sign_agree_rate']*100:.0f}%, binom p={e['sign_binom_p_twosided']:.3f}) "
                  f"{'PASS' if e['PASS_pooled_alpha05'] else 'fail'}{star}")
    if "PREREGISTERED_PRIMARY" in res:
        print("=" * 96)
        print(f"PREREGISTERED VERDICT (P2): "
              f"{'PASS' if res['PREREGISTERED_PRIMARY']['PASS'] else 'FAIL'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
