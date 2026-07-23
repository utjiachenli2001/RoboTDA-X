"""Closed-loop rollout for the P10 diffusion policy.

A faithful copy of src/rollout.py -- SAME LIBERO convention (env.reset -> set_init_state ->
5 settle steps -> policy loop), SAME 600-step horizon, SAME fixed init states, SAME CTX=10
window construction -- with exactly two changes:

  1. the policy is the DiffusionPolicy, acting via the deterministic DDIM sampler (eta=0, fixed
     initial latent), executing the FIRST action of the predicted chunk and replanning each step;
  2. THE P6.1 FIX IS APPLIED: the outcome artifact is written ATOMICALLY (tmp + os.replace) and
     ONLY AFTER a successful run. On a rollout error we raise WITHOUT writing outcomes.json, so a
     partially-failed run can never leave a complete-looking artifact behind. (src/rollout.py
     writes the artifact BEFORE it raises -- the latent defect P6 exists to close.)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join("/mnt/sdb/ljc/RoboTDA-X", "src"))
import bootstrap  # noqa: F401,E402
from bootstrap import write_done  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L  # noqa: E402

HORIZON = 600
SETTLE_STEPS = 5
_W = {}


def _init_worker(ckpt_path, obj_pad_dim, mean, std, n_ddim):
    import torch
    torch.set_num_threads(1)
    import evaluate_diffusion as EVD
    _W["torch"] = torch
    _W["model"] = EVD.load_model(ckpt_path, device="cuda")
    _W["obj_pad_dim"] = obj_pad_dim
    _W["mean"] = mean
    _W["std"] = std
    _W["n_ddim"] = n_ddim
    import taskemb
    _W["emb"] = taskemb.load()


def _rollout_task(job):
    suite, task, ep_idx, horizon = job
    t0 = time.time()
    try:
        import libero_env as LE
        from libero.libero import benchmark
        model, torch_ = _W["model"], _W["torch"]
        D, mean, std, nd = _W["obj_pad_dim"], _W["mean"], _W["std"], _W["n_ddim"]
        lang = torch_.from_numpy(_W["emb"][task][None, :]).cuda()

        B = benchmark.get_benchmark_dict()[suite]()
        names = [t.name for t in B.tasks]
        ti = names.index(task)
        init_states = np.asarray(B.get_task_init_states(ti))

        env = LE.make_env(LE.get_bddl_path(suite, task), horizon=horizon + 50, seed=0)
        CTX = 10
        succ, steps_used = [], []
        for ep in ep_idx:
            env.reset()
            obs = env.set_init_state(init_states[ep % init_states.shape[0]])
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(np.zeros(7))
            f = (LE.featurize(obs, D) - mean) / std
            win = [f] * CTX
            ok, t = False, 0
            for t in range(1, horizon + 1):
                s = torch_.from_numpy(np.stack(win[-CTX:])[None]).float().cuda()
                with torch_.autocast("cuda", dtype=torch_.bfloat16):
                    a = model.act(s, lang, n_steps=nd)      # deterministic DDIM
                obs, _, _, _ = env.step(a[0].float().cpu().numpy())
                if LE.check_success(env):
                    ok = True
                    break
                win.append((LE.featurize(obs, D) - mean) / std)
            succ.append(bool(ok))
            steps_used.append(int(t))
        try:
            env.close()
        except Exception:
            pass
        return {"suite": suite, "task": task, "success": succ, "steps": steps_used,
                "wall_s": time.time() - t0, "err": None}
    except Exception:
        import traceback
        return {"suite": suite, "task": task, "success": [], "steps": [],
                "wall_s": time.time() - t0, "err": traceback.format_exc()[-1200:]}


def run_rollouts(ckpt_path, task_list, n_rollouts, workers=12, horizon=HORIZON, n_ddim=None,
                 ep_offset=0):
    import multiprocessing as mp
    import dataset
    import diffusion_policy as DP
    n_ddim = n_ddim or DP.N_DDIM_STEPS
    mean, std = dataset.norm_stats()
    D = dataset.obj_pad_dim()
    ep_idx = list(range(ep_offset, ep_offset + n_rollouts))
    jobs = [(s, t, ep_idx, horizon) for (s, t) in task_list]
    ctx = mp.get_context("spawn")
    out, errs = {}, []
    t0 = time.time()
    with ctx.Pool(min(workers, len(jobs)), initializer=_init_worker,
                  initargs=(ckpt_path, D, mean, std, n_ddim)) as pool:
        for r in pool.imap_unordered(_rollout_task, jobs):
            if r["err"]:
                errs.append(r)
                print(f"[rolloutDP] ERROR {r['task']}: {r['err'][:300]}", flush=True)
            else:
                out[r["task"]] = r["success"]
    return out, {"wall_s": time.time() - t0, "n_errors": len(errs), "errors": errs}


def probe_battery(run_dir, ckpt="final.pt", n_rollouts=10, workers=12, clusters=None,
                  horizon=HORIZON, n_ddim=None):
    import dataset
    import evaluate_diffusion as EVD
    if L.is_marked(run_dir, "probe"):
        print(f"[probeDP] SKIP (probe.marker): {run_dir}")
        return L.read_artifact(run_dir, "outcomes.json")

    t0 = time.time()
    probes = dataset.probe_tasks()
    suite_of = dataset.suite_of_cluster()
    cl = clusters or dataset.clusters()
    task_list = [(suite_of[c], t) for c in cl for t in probes[c]]

    ckpt_path = os.path.join(run_dir, ckpt)
    succ, info = run_rollouts(ckpt_path, task_list, n_rollouts, workers, horizon, n_ddim)

    # ---- P6.1 FIX: on error, RAISE WITHOUT WRITING the artifact.
    if info["n_errors"]:
        L.atomic_write_json(os.path.join(run_dir, "rollout_errors.json"), info["errors"])
        raise RuntimeError(f"{info['n_errors']} rollout task errors in {run_dir} -- "
                           f"outcomes.json deliberately NOT written (see p3lib docstring)")

    model = EVD.load_model(ckpt_path, device="cuda")
    losses = EVD.heldout_losses(model, device="cuda", n_ddim=n_ddim)

    out = {}
    for c in cl:
        s = [x for t in probes[c] for x in succ.get(t, [])]
        out[c] = {
            "success_rate": float(np.mean(s)) if s else None,
            "n_episodes": len(s),
            "per_task_success": {t: float(np.mean(succ[t])) if t in succ else None
                                 for t in probes[c]},
            **{k: losses[c][k] for k in ("plain_loss", "transport_loss", "interaction_loss",
                                         "denoise_loss", "denoise_transport",
                                         "denoise_interaction")},
        }
    meta = {"ckpt": ckpt, "policy": "diffusion", "n_rollouts": n_rollouts, "horizon": horizon,
            "n_ddim_steps": n_ddim or __import__("diffusion_policy").N_DDIM_STEPS,
            "rollout_wall_s": info["wall_s"], "n_errors": 0,
            "total_wall_s": time.time() - t0}
    L.atomic_write_json(os.path.join(run_dir, "outcomes.json"),
                        {"outcomes": out, "meta": meta})
    write_done(run_dir, json.dumps(meta) + "\n", name="probe")
    return {"outcomes": out, "meta": meta}


def cluster_eval(run_dir, cluster, ckpt="final.pt", n_rollouts=20, workers=12, horizon=HORIZON,
                 n_ddim=None):
    import dataset
    if L.is_marked(run_dir, "clustereval"):
        print(f"[clustevalDP] SKIP (marker): {run_dir}")
        return L.read_artifact(run_dir, "cluster_eval.json")
    t0 = time.time()
    man = {c["cluster"]: c for c in dataset.manifest()["clusters"]}[cluster]
    suite, tasks = man["suite"], sorted(man["tasks"])
    succ, info = run_rollouts(os.path.join(run_dir, ckpt), [(suite, t) for t in tasks],
                              n_rollouts, workers, horizon, n_ddim)
    if info["n_errors"]:
        L.atomic_write_json(os.path.join(run_dir, "cluster_errors.json"), info["errors"])
        raise RuntimeError(f"{info['n_errors']} cluster rollout errors in {run_dir}")
    per_task = {t: float(np.mean(succ[t])) for t in tasks if t in succ}
    out = {"cluster": cluster, "suite": suite, "n_tasks": len(tasks), "n_rollouts": n_rollouts,
           "per_task_success": per_task,
           "cluster_success": float(np.mean(list(per_task.values()))) if per_task else None,
           "n_episodes": sum(len(succ.get(t, [])) for t in tasks),
           "horizon": horizon, "wall_s": time.time() - t0, "n_errors": 0, "policy": "diffusion"}
    L.atomic_write_json(os.path.join(run_dir, "cluster_eval.json"), out)
    write_done(run_dir, json.dumps({k: out[k] for k in
                                    ("cluster", "cluster_success", "n_episodes", "wall_s")}) + "\n",
               name="clustereval")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--cluster_tasks", default=None)
    ap.add_argument("--n_ddim", type=int, default=None)
    a = ap.parse_args()
    if a.cluster_tasks:
        r = cluster_eval(a.run_dir, a.cluster_tasks, a.ckpt, a.n_rollouts, a.workers,
                         n_ddim=a.n_ddim)
        print(f"[clustevalDP] {a.run_dir} {a.cluster_tasks} "
              f"success={r['cluster_success']:.4f} ({r['n_episodes']} eps, {r['wall_s']:.0f}s)")
        return
    cl = a.clusters.split(",") if a.clusters else None
    r = probe_battery(a.run_dir, a.ckpt, a.n_rollouts, a.workers, cl, n_ddim=a.n_ddim)
    print(f"[probeDP] {a.run_dir} wall={r['meta']['total_wall_s']:.0f}s")
    for c, v in r["outcomes"].items():
        print(f"  {c}: succ={v['success_rate']} L2={v['plain_loss']:.3f} "
              f"denoise={v['denoise_loss']:.4f}")


if __name__ == "__main__":
    main()
