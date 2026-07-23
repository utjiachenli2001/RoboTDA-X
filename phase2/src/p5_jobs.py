"""P5 (optional, diagnostic): RQ4 coverage fix on C5.

Phase 1: influence-top-15 selection on C5 scored -16.1 pts vs target-only. The selector picked
9 outsiders for a 7-task target, leaving too few insiders to cover C5's tasks. Was the failure
the INFLUENCE SCORES, or the missing COVERAGE CONSTRAINT?

  unconstrained  = Phase-1's top-15 by influence           (EXISTS: runs/stage_I/C5_influence_top15_s50{1,2,3})
  coverage-fixed = top-15 by influence SUBJECT TO >= 1 training demo per C5 task
  target_only    = Phase-1's C5 target-only               (EXISTS: runs/stage_I/C5_target_only_s50{1,2,3})

Only the coverage-fixed arm is new: 3 retrains (seeds 501-503), paired against the existing runs
at the SAME seeds. Same attributor (C5 -> TRAK, from best_attributor_by_target.json), same
functional (plain), same B=15, same eval (C5 probe tasks x 20 rollouts) as Stage I.

DIAGNOSTIC ONLY, n=1 target. Do not generalize beyond C5.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import RESULTS, ROOT  # noqa: E402
import dataset  # noqa: E402

TARGET = "C5"
B = 15
SEEDS = [501, 502, 503]          # identical to Stage I -> paired comparison
N_ROLLOUTS = 20
P2 = os.path.join(ROOT, "phase2")


def coverage_constrained(target=TARGET, k=B):
    """Top-k by influence, but reserve one slot for each of the target's tasks.

    Greedy: (1) for every task of the target cluster, take that task's HIGHEST-influence training
    demo -> guarantees >= 1 demo per task; (2) fill the remaining k - n_tasks slots with the
    highest-influence demos among all those not yet chosen (insider or outsider).
    """
    best = json.load(open(os.path.join(RESULTS, "best_attributor_by_target.json")))
    attr = best[target]
    df = pd.read_parquet(os.path.join(RESULTS, "influence_table.parquet"))
    sub = df[(df.attributor == attr) & (df.functional == "plain") & (df.target == target)]
    sub = sub.sort_values("score", ascending=False)

    _, by_c = dataset.train_pool()
    insiders = set(by_c[target])
    task_of = {d: dataset.parse_did(d)[1] for d in insiders}
    tasks = sorted(set(task_of.values()))

    chosen = []
    for t in tasks:                                        # (1) one slot per task
        cand = sub[sub.demo_id.isin([d for d in insiders if task_of[d] == t])]
        assert len(cand), f"no training demo for task {t}"
        chosen.append(cand.iloc[0].demo_id)
    for d in sub.demo_id:                                  # (2) fill by influence rank
        if len(chosen) >= k:
            break
        if d not in chosen:
            chosen.append(d)
    assert len(chosen) == k
    covered = {task_of[d] for d in chosen if d in insiders}
    assert covered == set(tasks), (covered, tasks)
    n_out = sum(1 for d in chosen if d not in insiders)
    return chosen, attr, tasks, n_out


if __name__ == "__main__":
    sel, attr, tasks, n_out = coverage_constrained()
    unc = json.load(open(os.path.join(RESULTS, "stage_I_rq4.json")))
    uc = unc["selection_composition"][TARGET]

    jobs = [{"run_dir": os.path.join(P2, "runs/P5", f"{TARGET}_influence_cov15_s{s}"),
             "demos": sel, "seed": s, "n_rollouts": N_ROLLOUTS,
             "eval": "probe", "clusters": [TARGET], "workers": 8}
            for s in SEEDS]
    json.dump(jobs, open(f"{P2}/results/p5_jobs.json", "w"), indent=1)
    json.dump({"target": TARGET, "attributor": attr, "B": B, "seeds": SEEDS,
               "n_tasks": len(tasks), "tasks": tasks,
               "coverage_constrained": sel,
               "coverage_constrained_outsiders": n_out,
               "unconstrained": uc["influence_top15"],
               "unconstrained_outsiders": uc["influence_top15_outsiders"],
               "n_overlap": len(set(sel) & set(uc["influence_top15"]))},
              open(f"{P2}/results/p5_selection.json", "w"), indent=1)
    print(f"P5 {TARGET}: attributor={attr}, {len(tasks)} tasks")
    print(f"  unconstrained  : {uc['influence_top15_outsiders']}/15 outsiders (Phase 1)")
    print(f"  coverage-fixed : {n_out}/15 outsiders, all {len(tasks)} tasks covered")
    print(f"  overlap: {len(set(sel) & set(uc['influence_top15']))}/15 demos")
    print(f"  wrote {len(jobs)} jobs")
