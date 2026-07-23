"""P8b determinism gate, STRENGTHENED -- on a DISCRIMINATING task.

WHY THIS EXISTS. The first-pass gate (p8b_determinism.json) passed, but on a DEGENERATE case:
all three C1 episodes failed at the full 600-step horizon, so the step counts matched trivially
([600,600,600] twice). A model that fails everything is bit-identical to itself for uninteresting
reasons. Phase 2 hit exactly this trap when confirming its 50-init-state defect and solved it the
same way: re-run the check on a checkpoint/task that ACTUALLY SUCCEEDS SOMETIMES, so that the
step count at which each success occurs is a real, information-bearing quantity.

This script:
  1. reads the P8b arm-(i)/arm-(ii) outcomes that have already been computed,
  2. picks the (cluster, probe task) with the HIGHEST success rate -- i.e. the most discriminating
     one available,
  3. replays 6 episodes twice under BOTH repaired policies,
  4. requires bit-identical success flags AND step counts, and requires that at least one episode
     actually SUCCEEDED (otherwise the check is degenerate again and is reported as such).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
import p8b_eval as P8B  # noqa: E402


def best_task():
    """The probe task with the highest success among the already-evaluated P8b runs."""
    best = (None, None, -1.0, None)      # (cluster, task, succ, run_dir)
    for m in P8B.masks12():
        for s in P8B.SEEDS_A:
            d = os.path.join(P3_RUNS, "P8b", f"ckptavg_{m}_s{s}")
            oc = L.read_outcomes(d, required=False)
            if not oc:
                continue
            for c, v in oc.items():
                for t, sv in (v.get("per_task_success") or {}).items():
                    if sv is not None and sv > best[2]:
                        best = (c, t, float(sv), d)
    return best


def replay(run_dir, suite, task, n=6):
    import multiprocessing as mp
    import rollout as RO
    mean, std = dataset.norm_stats()
    D = dataset.obj_pad_dim()
    ctx = mp.get_context("spawn")
    out = []
    for _ in range(2):
        with ctx.Pool(1, initializer=RO._init_worker,
                      initargs=(os.path.join(run_dir, "final.pt"), D, mean, std)) as pool:
            r = list(pool.imap_unordered(RO._rollout_task,
                                         [(suite, task, list(range(n)), 600)]))[0]
        out.append((r["success"], r["steps"]))
    return out


def replay_ens(mask, suite, task, n=6):
    import multiprocessing as mp
    mean, std = dataset.norm_stats()
    D = dataset.obj_pad_dim()
    cks = [os.path.join(P8B.g6_run_dir(mask, s), "final.pt") for s in P8B.SEEDS_A]
    ctx = mp.get_context("spawn")
    out = []
    for _ in range(2):
        with ctx.Pool(1, initializer=P8B._init_worker_ens, initargs=(cks, D, mean, std)) as pool:
            r = list(pool.imap_unordered(P8B._rollout_task_ens,
                                         [(suite, task, list(range(n)), 600)]))[0]
        out.append((r["success"], r["steps"]))
    return out


def main():
    c, task, succ, rd = best_task()
    if c is None:
        raise SystemExit("no P8b outcomes yet")
    suite = dataset.suite_of_cluster()[c]
    mask = os.path.basename(rd).split("_")[1]
    print(f"[strong] most discriminating probe task: {c}/{task} "
          f"(success {succ:.2f} on {os.path.basename(rd)})")

    a = replay(rd, suite, task)
    b = replay_ens(mask, suite, task)

    a_id = a[0] == a[1]
    b_id = b[0] == b[1]
    a_disc = any(a[0][0])
    b_disc = any(b[0][0])

    out = {
        "stage": "P8b determinism gate -- STRENGTHENED on a discriminating task",
        "why": ("The first-pass gate matched on a DEGENERATE case: all C1 episodes failed at the "
                "full 600-step horizon, so identical step counts ([600,600,600]) proved nothing. "
                "A determinism check is only informative on a task the policy sometimes SOLVES, "
                "because then the step count at which each success occurs is a real quantity."),
        "task": f"{c}/{task}", "task_success_rate": succ,
        "arm_i_checkpoint_average": {
            "run_dir": rd,
            "run1_success": a[0][0], "run1_steps": a[0][1],
            "run2_success": a[1][0], "run2_steps": a[1][1],
            "bit_identical": bool(a_id),
            "discriminating": bool(a_disc),
        },
        "arm_ii_action_ensemble": {
            "mask": mask, "members": P8B.SEEDS_A,
            "run1_success": b[0][0], "run1_steps": b[0][1],
            "run2_success": b[1][0], "run2_steps": b[1][1],
            "bit_identical": bool(b_id),
            "discriminating": bool(b_disc),
        },
        "GATE": bool(a_id and b_id),
        "GATE_IS_INFORMATIVE": bool(a_disc or b_disc),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p8b_determinism_strong.json"), out)
    print(f"[strong] arm(i)  identical={a_id} discriminating={a_disc} steps={a[0][1]}")
    print(f"[strong] arm(ii) identical={b_id} discriminating={b_disc} steps={b[0][1]}")
    print(f"[strong] GATE={out['GATE']}  INFORMATIVE={out['GATE_IS_INFORMATIVE']}")
    return 0 if out["GATE"] else 1


if __name__ == "__main__":
    sys.exit(main())
