"""P10 calibration -- bounded hyperparameter search for the diffusion policy. HARD CAP: 10 runs.

PROTOCOL (mirrors Phase-1's BC-Transformer calibration EXACTLY, so the two are comparable):
    50 demos of the single task  libero_goal/open_the_middle_drawer_of_the_cabinet
    evaluate on that same task, 20 rollouts
    the BC-Transformer reached 70% here (src/calibrate.py case single_task_50)

PREREGISTERED STOP RULE: if after <= 10 runs the best config cannot EXCEED 40% success, STOP the
stage and report NON-VIABILITY. Do not run the corpus on a broken policy. A non-viable P10 is
reported as "P10 did not run: policy non-viable" -- it is NOT a null result about attribution.

The search is STAGED (not a free-for-all): learning rate, then step budget, then chunk length,
then the eval-time DDIM step count (which needs no retraining). The winning config is frozen to
p10_config_frozen.json with a SHA-256 BEFORE P10a/P10b launch.

NB the calibration models are trained on demo_0..demo_49 of the task -- which includes the
Phase-2 probe demos. That is deliberate: it reproduces Phase-1's calibration protocol exactly, so
the 70% reference is a like-for-like comparison. These models are used ONLY for the viability
check and NEVER for attribution; p3lib.assert_no_probe_leak would refuse them if they were.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS, P3_LOGS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402

PY = "/home/ljc/miniconda/envs/robotda_x/bin/python"
P3SRC = os.path.dirname(os.path.abspath(__file__))
SUITE = "libero_goal"
TASK = "open_the_middle_drawer_of_the_cabinet"
SEED = 101
N_ROLLOUTS = 20
MAX_RUNS = 10
BC_REFERENCE = 0.70
STOP_THRESHOLD = 0.40
CAL = os.path.join(P3_RUNS, "_calib")


def demos_50():
    return [f"{SUITE}/{TASK}/demo_{i}" for i in range(50)]


def one_run(name, cfg_over, gpu, n_ddim=None):
    """Train + single-task eval. Returns (success, wall_s). Resumable via markers."""
    rd = os.path.join(CAL, name)
    os.makedirs(rd, exist_ok=True)
    dj = os.path.join(rd, "demos.json")
    if not os.path.exists(dj):
        json.dump({"demos": demos_50(), "seed": SEED}, open(dj, "w"), indent=1)

    ev = os.path.join(rd, "eval_singletask.json")
    if os.path.exists(ev) and L.is_marked(rd, "train"):
        r = json.load(open(ev))
        print(f"[cal] SKIP {name}: success={r['success']:.3f}")
        return r["success"], r.get("wall_s", 0.0)

    t0 = time.time()
    nd = f"--n_ddim {n_ddim}" if n_ddim else ""
    inner = (
        f"{PY} {P3SRC}/train_diffusion.py --run_dir {rd} --demos {dj} --seed {SEED} "
        f"--cfg_json '{json.dumps(cfg_over)}' && "
        f"{PY} {P3SRC}/p10_calibrate.py --eval_only --run_dir {rd} {nd}"
    )
    log = os.path.join(P3_LOGS, f"calib_{name}.log")
    with open(log, "w") as f:
        rc = subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} bash -c '{inner}'", shell=True,
                            stdout=f, stderr=subprocess.STDOUT, cwd=P3SRC).returncode
    if rc != 0 or not os.path.exists(ev):
        print(f"[cal] FAILED {name} (rc={rc}) -- see {log}")
        return None, time.time() - t0
    r = json.load(open(ev))
    print(f"[cal] {name}: success={r['success']:.3f} ({time.time()-t0:.0f}s)")
    return r["success"], time.time() - t0


def eval_only(run_dir, n_ddim=None, tag=None):
    """Roll out the single calibration task. tag -> eval_<tag>.json (else eval_singletask.json)."""
    import rollout_diffusion as RD
    succ, info = RD.run_rollouts(os.path.join(run_dir, "final.pt"), [(SUITE, TASK)],
                                 N_ROLLOUTS, workers=1, n_ddim=n_ddim)
    if info["n_errors"]:
        raise RuntimeError(f"rollout errors: {info['errors'][:1]}")
    s = float(np.mean(succ[TASK]))
    meta = json.load(open(os.path.join(run_dir, "train_meta.json")))
    fname = f"eval_{tag}.json" if tag else "eval_singletask.json"
    L.atomic_write_json(os.path.join(run_dir, fname), {
        "task": TASK, "n_rollouts": N_ROLLOUTS, "success": s,
        "per_episode": succ[TASK], "n_ddim": n_ddim or meta["cfg"]["n_ddim_steps"],
        "cfg": meta["cfg"], "params_M": meta["params_M"],
        "train_wall_s": meta["wall_s"], "rollout_wall_s": info["wall_s"]})
    print(f"[cal-eval] {run_dir}: success={s:.3f} ({info['wall_s']:.0f}s)")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--n_ddim", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.eval_only:
        eval_only(a.run_dir, a.n_ddim, a.tag)
        return

    runs, budget = [], MAX_RUNS

    def gpus(n):
        while True:
            idle = L.gpu_idle()
            if len(idle) >= n:
                return idle[:n]
            if idle:
                return idle          # take what we can rather than stall
            print("[cal] no idle GPU; sleeping 300 s")
            time.sleep(300)

    def do_parallel(specs):
        """specs: [(name, cfg)]. Runs them CONCURRENTLY across idle GPUs. -> {name: success}"""
        nonlocal budget
        specs = [s for s in specs if budget > 0 or _cached(s[0])]
        out = {}
        pending = [s for s in specs if not _cached(s[0])]
        for s in specs:
            if _cached(s[0]):
                out[s[0]] = _load(s[0])
        while pending:
            gs = gpus(min(len(pending), 4))
            batch, pending = pending[:len(gs)], pending[len(gs):]
            procs = []
            for (name, cfg), g in zip(batch, gs):
                if budget <= 0:
                    print(f"[cal] BUDGET EXHAUSTED -- skipping {name}")
                    continue
                budget -= 1
                procs.append((name, cfg, _launch(name, cfg, g)))
                time.sleep(3)
            for name, cfg, p in procs:
                p.wait()
                s = _load(name)
                out[name] = s
                runs.append({"name": name, "cfg": cfg, "success": s})
                print(f"[cal] {name}: success={s if s is None else f'{s:.3f}'}")
        return out

    def _cached(name):
        return os.path.exists(os.path.join(CAL, name, "eval_singletask.json"))

    def _load(name):
        p = os.path.join(CAL, name, "eval_singletask.json")
        return json.load(open(p))["success"] if os.path.exists(p) else None

    def _launch(name, cfg, gpu):
        rd = os.path.join(CAL, name)
        os.makedirs(rd, exist_ok=True)
        dj = os.path.join(rd, "demos.json")
        if not os.path.exists(dj):
            json.dump({"demos": demos_50(), "seed": SEED}, open(dj, "w"), indent=1)
        # The cfg goes in a FILE, never on the command line: this command is run inside
        # `bash -c '...'`, so a quoted JSON string would have its quotes eaten by the outer shell
        # (that bug ate the first three calibration launches).
        cf = os.path.join(rd, "cfg_override.json")
        json.dump(cfg, open(cf, "w"), indent=1)
        inner = (f"{PY} {P3SRC}/train_diffusion.py --run_dir {rd} --demos {dj} --seed {SEED} "
                 f"--cfg_file {cf} && "
                 f"{PY} {P3SRC}/p10_calibrate.py --eval_only --run_dir {rd}")
        log = open(os.path.join(P3_LOGS, f"calib_{name}.log"), "w")
        print(f"[cal] launch {name} on gpu{gpu}", flush=True)
        return subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} bash -c '{inner}'", shell=True,
                                stdout=log, stderr=subprocess.STDOUT, cwd=P3SRC)

    base = {"total_steps": 8000, "h_chunk": 8, "n_train_steps": 100, "n_ddim_steps": 10}

    # ---- stage 1: learning rate (diffusion typically wants more than BC's 1e-4)
    print("=" * 80 + "\n[cal] STAGE 1: learning rate\n" + "=" * 80)
    lrs = [1e-4, 3e-4, 1e-3]
    r1 = do_parallel([(f"dp_lr{lr:g}", {**base, "lr": lr}) for lr in lrs])
    s1 = {lr: r1.get(f"dp_lr{lr:g}") for lr in lrs}
    ok1 = [k for k, v in s1.items() if v is not None]
    if not ok1:
        raise SystemExit("[cal] ALL stage-1 calibration runs FAILED to produce an eval -- this is "
                         "a harness failure, not a policy result. Inspect phase3/logs/calib_*.log. "
                         "Do NOT report non-viability from this: no model was trained.")
    best_lr = max(ok1, key=lambda k: s1[k])
    print(f"[cal] best lr = {best_lr:g} (success {s1[best_lr]:.3f})")

    # ---- stage 2: step budget
    print("=" * 80 + "\n[cal] STAGE 2: step budget\n" + "=" * 80)
    r2 = do_parallel([(f"dp_lr{best_lr:g}_st{st}", {**base, "lr": best_lr, "total_steps": st})
                      for st in (16000, 24000)])
    s2 = {8000: s1[best_lr]}
    for st in (16000, 24000):
        v = r2.get(f"dp_lr{best_lr:g}_st{st}")
        if v is not None:
            s2[st] = v
    best_st = max(s2, key=lambda k: s2[k])
    print(f"[cal] best steps = {best_st} (success {s2[best_st]:.3f})")

    # ---- stage 3: action-chunk length
    print("=" * 80 + "\n[cal] STAGE 3: action-chunk length\n" + "=" * 80)
    r3 = do_parallel([(f"dp_h{hc}", {**base, "lr": best_lr, "total_steps": best_st,
                                     "h_chunk": hc}) for hc in (4, 16)])
    s3 = {8: s2[best_st]}
    for hc in (4, 16):
        v = r3.get(f"dp_h{hc}")
        if v is not None:
            s3[hc] = v
    best_h = max(s3, key=lambda k: s3[k])
    print(f"[cal] best h_chunk = {best_h} (success {s3[best_h]:.3f})")

    best_cfg = {**base, "lr": best_lr, "total_steps": best_st, "h_chunk": best_h}
    best_succ = max(v for v in [s1[best_lr], s2[best_st], s3[best_h]] if v is not None)

    # ---- stage 4: DDIM step count. EVAL-ONLY (no retraining), so it does NOT consume the
    # 10-run training budget. This is the single biggest lever on P10b's cost: the rollout does
    # one denoiser call PER DDIM STEP PER ENV STEP, so halving the steps nearly halves the 96-run
    # P10b bill. We pick the SMALLEST step count whose success is within one rollout-SE of the
    # best (SE = sqrt(p(1-p)/20) at 20 rollouts), and record the wall time of each.
    print("=" * 80 + "\n[cal] STAGE 4: DDIM step count (eval-only, free)\n" + "=" * 80)
    best_name = max([r for r in runs if r["success"] is not None],
                    key=lambda r: r["success"])["name"]
    best_rd = os.path.join(CAL, best_name)
    ddim = {}
    for nd in (5, 10, 20):
        ev = os.path.join(best_rd, f"eval_ddim{nd}.json")
        if not os.path.exists(ev):
            g = gpus(1)[0]
            log = os.path.join(P3_LOGS, f"calib_ddim{nd}.log")
            with open(log, "w") as f:
                subprocess.run(
                    f"CUDA_VISIBLE_DEVICES={g} {PY} {P3SRC}/p10_calibrate.py --eval_only "
                    f"--run_dir {best_rd} --n_ddim {nd} --tag ddim{nd}",
                    shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=P3SRC)
        if os.path.exists(ev):
            r = json.load(open(ev))
            ddim[nd] = {"success": r["success"], "rollout_wall_s": r["rollout_wall_s"]}
            print(f"[cal] n_ddim={nd:2d}: success={r['success']:.3f} "
                  f"rollout={r['rollout_wall_s']:.0f}s")
    best_nd = 10
    if ddim:
        top = max(v["success"] for v in ddim.values())
        se = float(np.sqrt(max(top * (1 - top), 1e-6) / N_ROLLOUTS))
        ok = [nd for nd, v in ddim.items() if v["success"] >= top - se]
        best_nd = min(ok) if ok else 10
        print(f"[cal] best success {top:.3f}, SE {se:.3f} -> smallest n_ddim within 1 SE: "
              f"{best_nd}")
    best_cfg["n_ddim_steps"] = best_nd
    if ddim and best_nd in ddim:
        best_succ = max(best_succ, ddim[best_nd]["success"])

    viable = best_succ > STOP_THRESHOLD
    out = {
        "stage": "P10 calibration",
        "protocol": (f"50 demos of {SUITE}/{TASK}, single-task eval, {N_ROLLOUTS} rollouts -- "
                     f"IDENTICAL to Phase-1's BC-Transformer calibration (src/calibrate.py "
                     f"case single_task_50)"),
        "BC_transformer_reference_success": BC_REFERENCE,
        "PREREGISTERED_STOP_THRESHOLD": STOP_THRESHOLD,
        "max_runs_allowed": MAX_RUNS, "runs_used": len(runs), "budget_remaining": budget,
        "runs": runs,
        "stage1_lr": {str(k): v for k, v in s1.items()},
        "stage2_steps": {str(k): v for k, v in s2.items()},
        "stage3_h_chunk": {str(k): v for k, v in s3.items()},
        "BEST_CONFIG": best_cfg,
        "BEST_SUCCESS": best_succ,
        "VIABLE": bool(viable),
        "VERDICT": ("VIABLE -- proceed to P10a/P10b" if viable else
                    "NON-VIABLE -- STOP the stage and report non-viability"),
        "comparison_to_BC": (f"diffusion {best_succ:.1%} vs BC-Transformer {BC_REFERENCE:.0%} "
                             f"on the identical single-task 50-demo sanity"),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p10_calibration.json"), out)

    if viable:
        frozen = {"cfg": best_cfg, "calibrated_on": out["protocol"],
                  "best_single_task_success": best_succ,
                  "frozen_before": "P10a and P10b launch. No hyperparameter changes after this.",
                  "runs_used": len(runs)}
        L.atomic_write_json(os.path.join(P3_RESULTS, "p10_config_frozen.json"), frozen)
        frozen["sha256"] = L.sha256_file(os.path.join(P3_RESULTS, "p10_config_frozen.json"))
        L.atomic_write_json(os.path.join(P3_RESULTS, "p10_config_frozen.json"), frozen)
        print(f"\n[cal] FROZEN config -> p10_config_frozen.json  sha={frozen['sha256'][:16]}")

    print("\n" + "=" * 80)
    print("P10 CALIBRATION")
    print("=" * 80)
    for r in runs:
        s = f"{r['success']:.3f}" if r["success"] is not None else "FAILED"
        print(f"  {r['name']:24s} success={s}")
    print(f"\n  BEST: {best_cfg}")
    print(f"  BEST SUCCESS: {best_succ:.1%}   (BC-Transformer reference: {BC_REFERENCE:.0%})")
    print(f"  STOP THRESHOLD: {STOP_THRESHOLD:.0%}")
    print(f"  VERDICT: {out['VERDICT']}")
    print("=" * 80)
    return 0 if viable else 2


if __name__ == "__main__":
    sys.exit(main())
