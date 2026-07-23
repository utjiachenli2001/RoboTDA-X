"""Phase-4 orchestrator. Structurally identical to phase3/src/orch3.py.

HARD RULE (never regressed): GPUs 4-7 only; a job launches only on a GPU that is idle
(mem < 1000 MiB AND util < 10%), re-verified immediately before EVERY launch; if none is idle,
sleep 300 s and re-check.

Completion is MARKER-GATED: a job counts as done only when its eval marker exists -- never
merely because outcomes.json is on disk. This is what makes the stage resumable without risking
the silent ingestion of a partially-failed run.

Trainers are the FROZEN prior-phase ones, invoked in place (src/train.py for the BC-Transformer,
phase3/src/train_diffusion.py for the diffusion policy). Phase 4 does not fork or retune them;
it only writes into phase4/runs/.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_LOGS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import gpu as G  # noqa: E402

PY = "/home/ljc/miniconda/envs/robotda_x/bin/python"
SRC = os.path.join(L.ROOT, "src")
P3SRC = L.P3_SRC
EVAL_MARKER = {"probe": "probe", "suite": "suite", "cluster_tasks": "clustereval"}


def job_done(job):
    return L.is_marked(job["run_dir"], EVAL_MARKER[job.get("eval", "probe")])


def job_cmd(job, gpu_id):
    rd = job["run_dir"]
    dj = os.path.join(rd, "demos.json")
    trainer = job.get("trainer", "stock")
    steps = f"--steps {job['steps']}" if job.get("steps") else ""

    if trainer == "diffusion":
        tr = (f"{PY} {P3SRC}/train_diffusion.py --run_dir {rd} --demos {dj} "
              f"--seed {job['seed']} {steps}")
    else:
        tr = f"{PY} {SRC}/train.py --run_dir {rd} --demos {dj} --seed {job['seed']} {steps}"

    m = job.get("eval", "probe")
    if m == "suite":
        ev = f"--suite {job['suite']}"
    elif m == "cluster_tasks":
        ev = f"--cluster_tasks {job['target']}"
    else:
        ev = f"--clusters {','.join(job['clusters'])}" if job.get("clusters") else ""

    roll = f"{P3SRC}/rollout_diffusion.py" if trainer == "diffusion" else f"{SRC}/rollout.py"
    ro = (f"{PY} {roll} --run_dir {rd} --ckpt {job.get('ckpt', 'final.pt')} "
          f"--n_rollouts {job['n_rollouts']} --workers {job.get('workers', 12)} {ev}")
    return f"CUDA_VISIBLE_DEVICES={gpu_id} bash -c '{tr} && {ro}'"


def prepare(job):
    os.makedirs(job["run_dir"], exist_ok=True)
    p = os.path.join(job["run_dir"], "demos.json")
    if not os.path.exists(p):
        payload = {"demos": job["demos"], "seed": job["seed"]}
        json.dump(payload, open(p, "w"), indent=1)


def run_jobs(jobs, stage, gpus=(4, 5, 6, 7), poll=10):
    logdir = os.path.join(P4_LOGS, stage)
    os.makedirs(logdir, exist_ok=True)
    todo = [j for j in jobs if not job_done(j)]
    skipped = len(jobs) - len(todo)
    print(f"[{stage}] {len(jobs)} jobs, {skipped} already complete, {len(todo)} to run", flush=True)
    for j in todo:
        prepare(j)

    running, results = {}, []
    t0s = time.time()
    while todo or running:
        for g, (p, j, t0, lf) in list(running.items()):
            if p.poll() is not None:
                ok = (p.returncode == 0) and job_done(j)
                results.append({"run_dir": j["run_dir"], "gpu": g, "wall_s": time.time() - t0,
                                "rc": p.returncode, "ok": ok})
                lf.close()
                print(f"[{stage}] {'OK  ' if ok else 'FAIL'} "
                      f"{os.path.basename(j['run_dir'])} gpu{g} {time.time()-t0:.0f}s "
                      f"({len(results)}/{len(jobs)-skipped})", flush=True)
                del running[g]

        if todo:
            st = G.query()                       # HARD RULE: re-verify idleness every launch
            for g in gpus:
                if not todo or g in running:
                    continue
                if not G.is_idle(g, st):
                    continue
                j = todo.pop(0)
                name = os.path.basename(j["run_dir"])
                lf = open(os.path.join(logdir, f"{name}.log"), "w")
                p = subprocess.Popen(job_cmd(j, g), shell=True, stdout=lf,
                                     stderr=subprocess.STDOUT, cwd=os.path.dirname(
                                         os.path.abspath(__file__)))
                running[g] = (p, j, time.time(), lf)
                print(f"[{stage}] launch {name} on gpu{g}", flush=True)
                time.sleep(3)

        if todo and not running:
            print(f"[{stage}] no idle GPU; sleeping {G.SLEEP_S}s", flush=True)
            time.sleep(G.SLEEP_S)
        else:
            time.sleep(poll)

    n_ok = sum(r["ok"] for r in results)
    summary = {"stage": stage, "n_jobs": len(jobs), "n_skipped": skipped, "n_run": len(results),
               "n_ok": n_ok, "n_fail": len(results) - n_ok, "wall_s": time.time() - t0s,
               "gpu_h": sum(r["wall_s"] for r in results) / 3600.0, "results": results}
    L.atomic_write_json(os.path.join(P4_LOGS, f"{stage}_summary.json"), summary)
    print(f"[{stage}] COMPLETE: {n_ok}/{len(results)} ok, {len(results)-n_ok} FAILED, "
          f"wall={summary['wall_s']/60:.1f} min, gpu-h={summary['gpu_h']:.2f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    run_jobs(json.load(open(a.jobs)), a.stage)


if __name__ == "__main__":
    main()
