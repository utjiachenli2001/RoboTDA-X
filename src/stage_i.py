"""STAGE I -- RQ4 confirmation: does influence-based selection beat the alternatives? (spec §8)

At budget B=15 demos, 4 conditions x 3 seeds x 2 focal targets (C1, C5) = 24 retrains:
  1 target_only      : the target's 15 insiders
  2 influence_top15  : top-15 of the full 135 by the best LDS-validated attributor toward the
                       target (plain functional)
  3 random15         : 15 random of the 135 (a FRESH random draw per training seed)
  4 similarity_top15 : the 15 nearest by mean DTW to the target's training demos
                       (self-pairs excluded)

Evaluated on the target's 3 probe tasks x 20 rollouts. Margins of (2) vs each of (1)/(3)/(4)
are paired by seed. A NEGATIVE result (influence <= similarity) is reported plainly -- it is a
finding, not a failure.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS, RESULTS
import dataset
import orchestrator as O

SEEDS = [501, 502, 503]
TARGETS = ["C1", "C5"]
B = 15
N_ROLLOUTS = 20
CONDS = ["target_only", "influence_top15", "random15", "similarity_top15"]


def influence_top(target, k=B):
    import pandas as pd
    p = os.path.join(RESULTS, "influence_table.parquet")
    best = json.load(open(os.path.join(RESULTS, "best_attributor_by_target.json")))
    attr = best[target]
    df = pd.read_parquet(p)
    sub = df[(df.attributor == attr) & (df.functional == "plain") & (df.target == target)]
    return list(sub.nlargest(k, "score").demo_id), attr


def similarity_top(target, k=B):
    """15 nearest by MEAN DTW to the target's training demos (self-pairs excluded)."""
    import moderators as MO
    z = MO.build()
    ids = [str(x) for x in z["demo_ids"]]
    D = z["dtw"]
    _, by_c = dataset.train_pool()
    tgt_idx = [ids.index(d) for d in by_c[target]]
    means = []
    for i, d in enumerate(ids):
        cols = [j for j in tgt_idx if j != i]          # exclude the self-pair
        means.append(float(D[i, cols].mean()))
    order = np.argsort(means)                           # nearest first
    return [ids[i] for i in order[:k]]


def random15(seed, k=B):
    ids, _ = dataset.train_pool()
    rng = np.random.default_rng(10_000 + seed)          # fresh draw per training seed
    return [ids[i] for i in rng.choice(len(ids), k, replace=False)]


def build_jobs():
    _, by_c = dataset.train_pool()
    jobs = []
    for t in TARGETS:
        infl, attr = influence_top(t)
        sim = similarity_top(t)
        sel = {"target_only": by_c[t], "influence_top15": infl, "similarity_top15": sim}
        for cond in CONDS:
            for s in SEEDS:
                demos = random15(s) if cond == "random15" else sel[cond]
                assert len(demos) == B, f"{cond}: {len(demos)}"
                jobs.append({
                    "run_dir": os.path.join(RUNS, "stage_I", f"{t}_{cond}_s{s}"),
                    "demos": list(demos), "seed": s, "n_rollouts": N_ROLLOUTS,
                    "eval": "probe", "clusters": [t], "workers": 8,
                    "cond": cond, "target": t,
                })
    return jobs


def composition():
    """How many outsiders does each selection rule pick? (context for the margins)"""
    _, by_c = dataset.train_pool()
    out = {}
    for t in TARGETS:
        infl, attr = influence_top(t)
        sim = similarity_top(t)
        ins = set(by_c[t])
        out[t] = {
            "best_attributor": attr,
            "influence_top15_outsiders": int(sum(1 for d in infl if d not in ins)),
            "similarity_top15_outsiders": int(sum(1 for d in sim if d not in ins)),
            "influence_top15": infl, "similarity_top15": sim,
        }
    return out


def analyze():
    from scipy import stats
    res = {"budget_B": B, "targets": TARGETS, "seeds": SEEDS,
           "n_rollouts_per_probe_task": N_ROLLOUTS,
           "selection_composition": composition(), "by_target": {}}
    for t in TARGETS:
        per = {c: {} for c in CONDS}
        for c in CONDS:
            for s in SEEDS:
                p = os.path.join(RUNS, "stage_I", f"{t}_{c}_s{s}", "outcomes.json")
                if os.path.exists(p):
                    per[c][s] = 100 * json.load(open(p))["outcomes"][t]["success_rate"]
        entry = {"success_pct": {c: per[c] for c in CONDS},
                 "mean_success_pct": {c: (float(np.mean(list(per[c].values())))
                                          if per[c] else None) for c in CONDS},
                 "sd_success_pct": {c: (float(np.std(list(per[c].values()), ddof=1))
                                        if len(per[c]) > 1 else None) for c in CONDS},
                 "margins_influence_vs": {}}
        for other in ("target_only", "random15", "similarity_top15"):
            m = [per["influence_top15"][s] - per[other][s]
                 for s in SEEDS if s in per["influence_top15"] and s in per[other]]
            if m:
                t_stat, p_two = (stats.ttest_1samp(m, 0.0) if len(m) > 1 else (np.nan, np.nan))
                entry["margins_influence_vs"][other] = {
                    "per_seed_pts": m, "mean_pts": float(np.mean(m)),
                    "sd_pts": float(np.std(m, ddof=1)) if len(m) > 1 else None,
                    "t": float(t_stat) if np.isfinite(t_stat) else None,
                    "p_twosided": float(p_two) if np.isfinite(p_two) else None,
                }
        res["by_target"][t] = entry
    json.dump(res, open(os.path.join(RESULTS, "stage_I_rq4.json"), "w"), indent=1)

    print("\n=== STAGE I (RQ4): data selection at budget B=15 ===")
    for t in TARGETS:
        e = res["by_target"][t]
        comp = res["selection_composition"][t]
        print(f"\n  target {t} (best attributor: {comp['best_attributor']}; "
              f"influence-top15 picks {comp['influence_top15_outsiders']} outsiders, "
              f"similarity-top15 picks {comp['similarity_top15_outsiders']})")
        for c in CONDS:
            m, sd = e["mean_success_pct"][c], e["sd_success_pct"][c]
            print(f"    {c:>18}: {m:.1f}% +- {sd if sd else 0:.1f}" if m is not None
                  else f"    {c:>18}: did not run")
        for other, v in e["margins_influence_vs"].items():
            print(f"    influence - {other:<16} = {v['mean_pts']:+.1f} pts "
                  f"(SD {v['sd_pts'] if v['sd_pts'] else 0:.1f}, per-seed {v['per_seed_pts']})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze_only", action="store_true")
    a = ap.parse_args()
    if not a.analyze_only:
        O.run_jobs(build_jobs(), "stage_I")
    analyze()


if __name__ == "__main__":
    main()
