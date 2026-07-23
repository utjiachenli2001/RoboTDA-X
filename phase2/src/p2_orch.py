"""Phase-2 orchestrator wrapper.

Identical scheduling/GPU-gating/resumability to Phase 1's src/orchestrator.py -- the ONLY
change is that logs and the stage summary are redirected to phase2/logs/, so Phase 2 never
writes outside phase2/ (Phase-1 artifacts stay read-only).

Usage: python p2_orch.py --jobs <jobs.json> --stage <name>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
import orchestrator as O  # noqa: E402

P2_LOGS = "/mnt/sdb/ljc/RoboTDA-X/phase2/logs"
P2_SRC = os.path.dirname(os.path.abspath(__file__))
os.makedirs(P2_LOGS, exist_ok=True)
O.LOGS = P2_LOGS          # run_jobs resolves LOGS from the orchestrator module namespace

_base_job_cmd = O.job_cmd


def job_cmd_with_heldout(job, gpu_id):
    """P3 needs BOTH the held-out L2 and the full-suite success. rollout.py's cluster_tasks
    mode writes only the latter, so the cheap losses-only pass is chained in FRONT of it:
    train -> heldout_losses -> cluster_tasks. Because rollout.py writes clustereval.marker
    last, a job that the orchestrator counts as done necessarily has both artifacts.
    """
    cmd = _base_job_cmd(job, gpu_id)
    if not job.get("heldout_losses"):
        return cmd
    rd = job["run_dir"]
    ck = job.get("ckpt", "final.pt")
    add = f"{O.PY} {P2_SRC}/p3_heldout_losses.py --run_dir {rd} --ckpt {ck} && "
    # splice after train.py, before rollout.py (both are inside the single bash -c '...')
    marker = f"{O.PY} {O.SRC}/rollout.py"
    assert marker in cmd, cmd
    return cmd.replace(marker, add + marker, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--gpus", default=None,
                    help="comma list, subset of 4,5,6,7 (default all). Lets a long training "
                         "stage share the box with an attribution job on a reserved GPU.")
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    if any(j.get("heldout_losses") for j in jobs):
        O.job_cmd = job_cmd_with_heldout
    gpus = tuple(int(x) for x in a.gpus.split(",")) if a.gpus else O.G.ALLOWED
    for g in gpus:
        if g not in O.G.ALLOWED:                    # HARD RULE, enforced here too
            raise ValueError(f"GPU {g} is OUT OF BOUNDS (allowed: {O.G.ALLOWED})")
    s = O.run_jobs(jobs, a.stage, gpus=gpus)
    print(json.dumps({k: v for k, v in s.items() if k != "results"}, indent=1))
