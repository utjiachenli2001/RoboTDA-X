"""Job orchestrator: runs (train -> probe battery) jobs across the allowed GPUs 4-7.

Contract:
  * resumable -- a job whose run_dir already has probe.marker is SKIPPED
  * GPU gating -- a job is only launched on a GPU that is idle by the hard-rule definition
    (mem < 1000 MiB and util < 10%); if no GPU is idle, sleep 300 s and re-check
  * crash isolation -- each job is a subprocess; a failure is recorded, the stage continues
  * every job writes its own log under logs/<stage>/<run>.log

jobs.json = [{"run_dir":..., "demos": [...], "seed": int,
              "n_rollouts": int, "clusters": [..] | null, "steps": int | null}, ...]
"""
import os
import sys
import json
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import ROOT, LOGS, is_done
import gpu as G

SRC = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


EVAL_MARKER = {"probe": "probe", "suite": "suite", "cluster_tasks": "clustereval"}


def eval_mode(job):
    return job.get("eval", "probe")


def job_done(job):
    return is_done(job["run_dir"], EVAL_MARKER[eval_mode(job)])


def job_cmd(job, gpu_id):
    """train (skips if train.marker) then evaluate (skips if the eval marker exists).

    eval = "probe" -> 3 probe tasks/cluster x n_rollouts + held-out losses
    eval = "suite" -> every task of job['suite'] x n_rollouts (Gate 0 / Stage C)
    """
    rd = job["run_dir"]
    demos_json = os.path.join(rd, "demos.json")
    steps = f"--steps {job['steps']}" if job.get("steps") else ""
    ck = f"--ckpt {job.get('ckpt', 'final.pt')}"
    m = eval_mode(job)
    if m == "suite":
        ev = f"--suite {job['suite']}"
    elif m == "cluster_tasks":
        ev = f"--cluster_tasks {job['target']}"
    else:
        ev = f"--clusters {','.join(job['clusters'])}" if job.get("clusters") else ""
    inner = (
        f"{PY} {SRC}/train.py --run_dir {rd} --demos {demos_json} "
        f"--seed {job['seed']} {steps} && "
        f"{PY} {SRC}/rollout.py --run_dir {rd} {ck} "
        f"--n_rollouts {job['n_rollouts']} --workers {job.get('workers', 12)} {ev}"
    )
    env = f"CUDA_VISIBLE_DEVICES={gpu_id}"
    return f"{env} bash -c '{inner}'"


def prepare(job):
    os.makedirs(job["run_dir"], exist_ok=True)
    p = os.path.join(job["run_dir"], "demos.json")
    if not os.path.exists(p):
        json.dump({"demos": job["demos"], "seed": job["seed"]}, open(p, "w"), indent=1)


def run_jobs(jobs, stage, gpus=G.ALLOWED, poll=10):
    logdir = os.path.join(LOGS, stage)
    os.makedirs(logdir, exist_ok=True)
    todo = [j for j in jobs if not job_done(j)]
    done_already = len(jobs) - len(todo)
    print(f"[{stage}] {len(jobs)} jobs, {done_already} already complete, {len(todo)} to run",
          flush=True)
    for j in todo:
        prepare(j)

    running = {}      # gpu -> (proc, job, t0, logfile)
    results = []
    t_stage = time.time()
    while todo or running:
        # --- reap finished
        for g, (p, j, t0, lf) in list(running.items()):
            if p.poll() is not None:
                ok = (p.returncode == 0) and job_done(j)
                rec = {"run_dir": j["run_dir"], "seed": j["seed"], "gpu": g,
                       "wall_s": time.time() - t0, "rc": p.returncode, "ok": ok}
                results.append(rec)
                lf.close()
                status = "OK " if ok else "FAIL"
                print(f"[{stage}] {status} {os.path.basename(j['run_dir'])} "
                      f"gpu{g} {rec['wall_s']:.0f}s "
                      f"({len(results)}/{len(results)+len(todo)+len(running)-1} done)", flush=True)
                del running[g]

        # --- launch on idle GPUs (hard rule: verify idleness EVERY launch)
        if todo:
            st = G.query()
            for g in gpus:
                if not todo:
                    break
                if g in running:
                    continue
                if not G.is_idle(g, st):
                    continue
                j = todo.pop(0)
                name = os.path.basename(j["run_dir"])
                lf = open(os.path.join(logdir, f"{name}.log"), "w")
                p = subprocess.Popen(job_cmd(j, g), shell=True, stdout=lf,
                                     stderr=subprocess.STDOUT, cwd=SRC)
                running[g] = (p, j, time.time(), lf)
                print(f"[{stage}] launch {name} on gpu{g}", flush=True)
                time.sleep(3)     # let CUDA init register before the next idleness query

        if todo and not running:
            # nothing running and nothing launchable => all allowed GPUs are busy (someone else)
            print(f"[{stage}] no idle GPU; sleeping {G.SLEEP_S}s", flush=True)
            time.sleep(G.SLEEP_S)
        else:
            time.sleep(poll)

    wall = time.time() - t_stage
    n_ok = sum(r["ok"] for r in results)
    summary = {"stage": stage, "n_jobs": len(jobs), "n_skipped": done_already,
               "n_run": len(results), "n_ok": n_ok, "n_fail": len(results) - n_ok,
               "wall_s": wall, "gpu_h_est": sum(r["wall_s"] for r in results) / 3600.0,
               "results": results}
    json.dump(summary, open(os.path.join(LOGS, f"{stage}_summary.json"), "w"), indent=1)
    print(f"[{stage}] COMPLETE: {n_ok}/{len(results)} ok, {len(results)-n_ok} failed, "
          f"wall={wall/60:.1f} min, gpu-h={summary['gpu_h_est']:.2f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    run_jobs(jobs, a.stage)


if __name__ == "__main__":
    main()
