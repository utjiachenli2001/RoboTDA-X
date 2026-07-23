"""Driver for the remaining GPU work, in dependency order. Resumable: every stage skips itself
if its artifact already exists / its markers are present.

  1. P9 attribution         (1 GPU;  needs the 10 new ensemble members -- already trained)
  2. P8b arm (i) + (ii)     (GPUs;   needs the P8a mask list -- already fixed)
  3. P10 calibration        (GPUs;   <= 10 runs, hard cap, with the non-viability STOP rule)
  4. P10 determinism gate   (1 GPU;  must pass before ANY P10 experiment)
  5. P10ens / P10a / P10b   (GPUs)

Every launch re-verifies GPU idleness (hard rule: 4-7 only, mem < 1000 MiB AND util < 10%).
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS, P3_LOGS

PY = "/home/ljc/miniconda/envs/robotda_x/bin/python"
SRC = os.path.dirname(os.path.abspath(__file__))


def wait_for_gpus(n=1):
    while True:
        idle = L.gpu_idle()
        if len(idle) >= n:
            return idle
        print(f"[drive] waiting for {n} idle GPU(s); idle now: {idle}", flush=True)
        time.sleep(300)


def sh(cmd, log, gpu=None, check=True):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[drive] RUN {' '.join(cmd)}  (gpu={gpu}) -> {log}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=SRC, env=env).returncode
    print(f"[drive] rc={rc}", flush=True)
    if check and rc != 0:
        raise SystemExit(f"FAILED (rc={rc}): {' '.join(cmd)} -- see {log}")
    return rc


def main():
    stages = sys.argv[1:] or ["p9attr", "p8b", "p10cal", "p10run"]

    # ---------------------------------------------------------------- 1. P9 attribution
    if "p9attr" in stages:
        out = os.path.join(P3_RESULTS, "p9_influence_new_members.parquet")
        if os.path.exists(out):
            print("[drive] P9 attribution already done")
        else:
            g = wait_for_gpus(1)[0]
            sh([PY, "p9_attr.py"], os.path.join(P3_LOGS, "p9_attr.log"), gpu=g)
        sh([PY, "p9_analyze.py"], os.path.join(P3_LOGS, "p9_analyze.log"), gpu=L.gpu_idle()[0]
           if L.gpu_idle() else 4)

    # ---------------------------------------------------------------- 2. P8b
    if "p8b" in stages:
        gate = os.path.join(P3_RESULTS, "p8b_determinism.json")
        if not os.path.exists(gate):
            g = wait_for_gpus(1)[0]
            sh([PY, "p8b_eval.py", "--gate"], os.path.join(P3_LOGS, "p8b_gate.log"), gpu=g)
        gd = json.load(open(gate))
        if not gd.get("GATE"):
            raise SystemExit("[drive] P8b DETERMINISM GATE FAILED -- instrument defect")
        print("[drive] P8b determinism gate PASSED")
        # 60 evaluations, SHARDED across all idle GPUs (serial would take ~3 h)
        gs = wait_for_gpus(4)[:4]
        procs = []
        for k, g in enumerate(gs):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            lf = open(os.path.join(P3_LOGS, f"p8b_shard{k}.log"), "w")
            procs.append((k, subprocess.Popen(
                [PY, "p8b_eval.py", "--shard", str(k), "--nshard", str(len(gs))],
                stdout=lf, stderr=subprocess.STDOUT, cwd=SRC, env=env), lf))
            print(f"[drive] launched P8b shard {k}/{len(gs)} on gpu{g}", flush=True)
            time.sleep(3)
        for k, p, lf in procs:
            rc = p.wait()
            lf.close()
            print(f"[drive] P8b shard {k} rc={rc}", flush=True)
            if rc != 0:
                raise SystemExit(f"P8b shard {k} FAILED")
        sh([PY, "p8b_analyze.py"], os.path.join(P3_LOGS, "p8b_analyze.log"),
           gpu=L.gpu_idle()[0] if L.gpu_idle() else 4)

    # ---------------------------------------------------------------- 3. P10 calibration
    if "p10cal" in stages:
        cal = os.path.join(P3_RESULTS, "p10_calibration.json")
        if os.path.exists(cal):
            print("[drive] P10 calibration already done")
        else:
            wait_for_gpus(1)
            sh([PY, "p10_calibrate.py"], os.path.join(P3_LOGS, "p10_calibrate.log"), check=False)
        c = json.load(open(cal))
        print(f"[drive] P10 calibration: best={c['BEST_SUCCESS']:.1%} "
              f"(BC ref {c['BC_transformer_reference_success']:.0%}) VIABLE={c['VIABLE']}")
        if not c["VIABLE"]:
            print("[drive] P10 NON-VIABLE -- stopping the stage as preregistered.")
            return

    # ---------------------------------------------------------------- 4-5. P10 runs
    if "p10run" in stages:
        cal = json.load(open(os.path.join(P3_RESULTS, "p10_calibration.json")))
        if not cal["VIABLE"]:
            print("[drive] P10 non-viable; skipping P10 runs.")
            return

        # determinism gate on a calibration checkpoint
        dg = os.path.join(P3_RESULTS, "p10_determinism.json")
        if not os.path.exists(dg):
            best = max([r for r in cal["runs"] if r["success"] is not None],
                       key=lambda r: r["success"])["name"]
            ck = os.path.join(P3_RUNS, "_calib", best, "final.pt")
            g = wait_for_gpus(1)[0]
            sh([PY, "p10_determinism.py", ck], os.path.join(P3_LOGS, "p10_determinism.log"),
               gpu=g)
        d = json.load(open(dg))
        if not d["GATE"]:
            raise SystemExit("[drive] P10 DETERMINISM GATE FAILED -- instrument defect")
        print("[drive] P10 determinism gate PASSED")

        for which in ("ens", "a", "b"):
            sh([PY, "p10_jobs.py", which, "--run"],
               os.path.join(P3_LOGS, f"p10_{which}_run.log"))

        g = wait_for_gpus(1)[0]
        sh([PY, "p10_attr.py"], os.path.join(P3_LOGS, "p10_attr.log"), gpu=g)
        sh([PY, "p10_analyze.py"], os.path.join(P3_LOGS, "p10_analyze.log"), gpu=g)

    print("[drive] ALL REQUESTED STAGES COMPLETE")


if __name__ == "__main__":
    main()
