"""P4 driver: shard the eval-only models across the allowed GPUs, honouring the hard rule.

A shard is launched ONLY on a GPU that is idle by the hard-rule definition (mem < 1000 MiB and
util < 10%); if none is idle, sleep 300 s and re-check. Resumable: each model's p4.marker makes
p4_eval.py skip it, so a relaunch continues where it stopped.

  --arm a : the 48 EXISTING Stage-G models (24 masks x seeds 401,402)   [P4a, core]
  --arm b : the 96 NEW P1 models          (24 masks x seeds 403-406)    [P4b, conditional]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT, RUNS  # noqa: E402
import gpu as G  # noqa: E402

P2 = os.path.join(ROOT, "phase2")
SRC = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ARMS = {"a": (os.path.join(RUNS, "stage_G"), [401, 402]),
        "b": (os.path.join(P2, "runs/stage_G6"), [403, 404, 405, 406])}


def models_for(arm):
    root, seeds = ARMS[arm]
    out = []
    for d in sorted(os.listdir(root)):
        mask, s = d.rsplit("_s", 1)
        if int(s) not in seeds:
            continue
        ck = os.path.join(root, d, "final.pt")
        if not os.path.exists(ck):
            print(f"[P4] MISSING checkpoint, skipping: {ck}")
            continue
        out.append({"name": d, "ckpt": ck, "mask_id": mask, "seed": int(s)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b"], default="a")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--gpus", default=None,
                    help="comma list, subset of 4,5,6,7 (default all). Restrict this so P4 does "
                         "not contend with a training stage running on the other GPUs.")
    a = ap.parse_args()

    from p4_eval import N_ROLLOUTS                     # 50, per PHASE2_DEFECT.md -- NOT 90
    models = models_for(a.arm)
    print(f"[P4] arm={a.arm}: {len(models)} models, "
          f"{len(models)*6*N_ROLLOUTS} episodes (6 probe tasks x {N_ROLLOUTS} rollouts)",
          flush=True)

    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else list(G.ALLOWED)
    for g in gpus:
        if g not in G.ALLOWED:                          # HARD RULE
            raise ValueError(f"GPU {g} is OUT OF BOUNDS (allowed: {G.ALLOWED})")
    shards = {g: models[i::len(gpus)] for i, g in enumerate(gpus)}
    logdir = os.path.join(P2, "logs", f"P4{a.arm}")
    os.makedirs(logdir, exist_ok=True)

    running, t0 = {}, time.time()
    pending = list(gpus)
    while pending or running:
        for g, (p, lf) in list(running.items()):
            if p.poll() is not None:
                print(f"[P4] shard gpu{g} exited rc={p.returncode}", flush=True)
                lf.close()
                del running[g]
        if pending:
            st = G.query()
            for g in list(pending):
                if not shards[g]:
                    pending.remove(g)
                    continue
                if not G.is_idle(g, st):       # HARD RULE: verified at every launch
                    continue
                mf = os.path.join(P2, "results", f"p4{a.arm}_shard_gpu{g}.json")
                json.dump(shards[g], open(mf, "w"), indent=1)
                lf = open(os.path.join(logdir, f"gpu{g}.log"), "w")
                cmd = (f"CUDA_VISIBLE_DEVICES={g} {PY} {SRC}/p4_eval.py "
                       f"--models {mf} --workers {a.workers}")
                p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, cwd=SRC)
                running[g] = (p, lf)
                pending.remove(g)
                print(f"[P4] launch shard of {len(shards[g])} models on gpu{g}", flush=True)
                time.sleep(3)
            if pending and not running:
                print(f"[P4] no idle GPU; sleeping {G.SLEEP_S}s", flush=True)
                time.sleep(G.SLEEP_S)
        time.sleep(10)

    done = len([d for d in os.listdir(os.path.join(P2, "runs/P4"))
                if os.path.exists(os.path.join(P2, "runs/P4", d, "p4.marker"))])
    print(f"[P4] arm={a.arm} COMPLETE: {done} models have p4.marker, "
          f"wall={(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
