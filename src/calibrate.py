"""Policy/optimization calibration (spec §3: 'tune briefly on C1 only, then freeze').

Establishes the achievable success ceiling of the frozen architecture before any gate is run,
so that a 0% Gate-0 result can be attributed to data, not to an under-trained policy.
Writes results/calibration.json. NOT part of any reported study result.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS, RESULTS
import dataset
import train as T
import rollout as R


def demos_of_task(suite, task, n):
    """First n demos of a single task (deterministic: demo_0..demo_{n-1})."""
    return [dataset.did(suite, task, f"demo_{i}") for i in range(n)]


def run_case(name, demos, tasks, seed, steps, n_rollouts, workers, out):
    run_dir = os.path.join(RUNS, "_calib", name)
    os.makedirs(run_dir, exist_ok=True)
    cfg = T.load_cfg()
    cfg["total_steps"] = steps
    t0 = time.time()
    if not bootstrap.is_done(run_dir, "train"):
        meta = T.train(run_dir, demos, seed, cfg)
        bootstrap.write_done(run_dir, "ok\n", name="train")
    else:
        meta = json.load(open(os.path.join(run_dir, "train_meta.json")))
    tr_s = time.time() - t0
    t0 = time.time()
    succ, info = R.run_rollouts(os.path.join(run_dir, "final.pt"), tasks, n_rollouts, workers)
    sr = {t: float(np.mean(v)) for t, v in succ.items()}
    rec = {"name": name, "n_demos": len(demos), "steps": steps, "seed": seed,
           "n_windows": meta["n_windows"], "train_s": tr_s, "rollout_s": time.time() - t0,
           "final_loss": meta["final_loss"], "per_task_success": sr,
           "mean_success": float(np.mean(list(sr.values()))) if sr else None,
           "n_errors": info["n_errors"]}
    print(f"[calib] {name}: demos={len(demos)} steps={steps} loss={rec['final_loss']:.3f} "
          f"success={rec['mean_success']} train={tr_s:.0f}s roll={rec['rollout_s']:.0f}s", flush=True)
    if info["n_errors"]:
        print("  ERRORS:", info["errors"][0]["err"][:400], flush=True)
    out.append(rec)
    # per-case artifact file: parallel calibration jobs must not race on one shared json
    d = os.path.join(RESULTS, "calib")
    os.makedirs(d, exist_ok=True)
    json.dump(rec, open(os.path.join(d, f"{name}.json"), "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    out = []
    suite = "libero_goal"
    task = "open_the_middle_drawer_of_the_cabinet"

    if a.case == "single_task_50":
        # CEILING TEST: one task, all 50 demos. If this does not succeed, something is broken.
        run_case(f"singletask50_s{a.steps}_s{a.seed}", demos_of_task(suite, task, 50),
                 [(suite, task)], a.seed, a.steps, a.n_rollouts, a.workers, out)
    elif a.case == "single_task_10":
        run_case(f"singletask10_s{a.steps}_s{a.seed}", demos_of_task(suite, task, 10),
                 [(suite, task)], a.seed, a.steps, a.n_rollouts, a.workers, out)
    elif a.case == "c1_full":
        # all 10 goal tasks x 50 demos = 500 (Stage C Q=500 condition)
        from clusters import suite_task_names
        tasks = sorted(suite_task_names("libero_goal"))
        demos = [d for t in tasks for d in demos_of_task(suite, t, 50)]
        run_case(f"c1full500_s{a.steps}_s{a.seed}", demos,
                 [(suite, t) for t in tasks], a.seed, a.steps, a.n_rollouts, a.workers, out)
    elif a.case == "c1_pool15":
        _, by_c = dataset.train_pool()
        probes = dataset.probe_tasks()["C1"]
        run_case(f"c1pool15_s{a.steps}_s{a.seed}", by_c["C1"],
                 [(suite, t) for t in probes], a.seed, a.steps, a.n_rollouts, a.workers, out)
    elif a.case == "cotrain135":
        ids, _ = dataset.train_pool()
        probes = dataset.probe_tasks()["C1"]
        run_case(f"cotrain135_s{a.steps}_s{a.seed}", ids,
                 [(suite, t) for t in probes], a.seed, a.steps, a.n_rollouts, a.workers, out)
    else:
        raise SystemExit(f"unknown case {a.case}")


if __name__ == "__main__":
    main()
