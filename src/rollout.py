"""Closed-loop rollout runner + probe battery.

Follows LIBERO's official evaluation convention (libero/lifelong/metric.py):
    env.reset() -> env.set_init_state(init_states[i]) -> 5 zero-action settle steps -> policy loop.

Init states are LIBERO's own fixed per-task init states, indices 0..R-1. Every model in the
study is therefore evaluated from the IDENTICAL initial conditions, so cross-model success
differences are not contaminated by initial-state sampling noise.

Horizon: 600 steps (LIBERO default eval max_steps). Episode ends early on success.

Parallelism: one process per env (MuJoCo is single-threaded); each worker holds its own copy
of the policy on the job's single visible GPU. CUDA_VISIBLE_DEVICES must already be set by
the caller so that workers only ever see the assigned GPU.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS, write_done, is_done

HORIZON = 600
SETTLE_STEPS = 5

_W = {}   # per-worker globals


def _init_worker(ckpt_path, obj_pad_dim, mean, std):
    import torch
    torch.set_num_threads(1)
    import evaluate as EV
    _W["torch"] = torch
    _W["model"] = EV.load_model(ckpt_path, device="cuda")
    _W["obj_pad_dim"] = obj_pad_dim
    _W["mean"] = mean
    _W["std"] = std
    import taskemb
    _W["emb"] = taskemb.load()


def _rollout_task(job):
    """job = (suite, task, ep_indices, horizon). Returns dict with per-episode success."""
    suite, task, ep_idx, horizon = job
    t0 = time.time()
    try:
        import torch
        import libero_env as LE
        from libero.libero import benchmark
        model, torch_ = _W["model"], _W["torch"]
        D, mean, std = _W["obj_pad_dim"], _W["mean"], _W["std"]
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
            win = [f] * CTX                                   # left-pad with first frame
            ok, t = False, 0
            for t in range(1, horizon + 1):
                s = torch_.from_numpy(np.stack(win[-CTX:])[None]).float().cuda()
                with torch_.autocast("cuda", dtype=torch_.bfloat16):
                    a = model.act(s, lang)
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


def run_rollouts(ckpt_path, task_list, n_rollouts, workers=14, horizon=HORIZON,
                 ep_offset=0):
    """task_list = [(suite, task), ...]. Returns {task: [bool,...]} plus timing."""
    import multiprocessing as mp
    import dataset
    mean, std = dataset.norm_stats()
    D = dataset.obj_pad_dim()
    ep_idx = list(range(ep_offset, ep_offset + n_rollouts))
    jobs = [(s, t, ep_idx, horizon) for (s, t) in task_list]
    ctx = mp.get_context("spawn")
    out, errs = {}, []
    t0 = time.time()
    with ctx.Pool(min(workers, len(jobs)), initializer=_init_worker,
                  initargs=(ckpt_path, D, mean, std)) as pool:
        for r in pool.imap_unordered(_rollout_task, jobs):
            if r["err"]:
                errs.append(r)
                print(f"[rollout] ERROR {r['task']}: {r['err'][:300]}", flush=True)
            else:
                out[r["task"]] = r["success"]
    return out, {"wall_s": time.time() - t0, "n_errors": len(errs), "errors": errs}


def probe_battery(run_dir, ckpt="final.pt", n_rollouts=10, workers=14, clusters=None,
                  horizon=HORIZON):
    """Spec §3: 3 probe tasks/cluster x n_rollouts + held-out plain/transport/interaction losses.

    Writes run_dir/outcomes.json:
        {cluster: {success_rate, n_episodes, plain_loss, transport_loss, interaction_loss}}
    """
    import dataset
    import evaluate as EV
    if is_done(run_dir, "probe"):
        print(f"[probe] SKIP (probe.marker): {run_dir}")
        return json.load(open(os.path.join(run_dir, "outcomes.json")))

    t0 = time.time()
    probes = dataset.probe_tasks()
    suite_of = dataset.suite_of_cluster()
    cl = clusters or dataset.clusters()
    task_list, owner = [], {}
    for c in cl:
        for t in probes[c]:
            task_list.append((suite_of[c], t))
            owner[t] = c

    ckpt_path = os.path.join(run_dir, ckpt)
    succ, info = run_rollouts(ckpt_path, task_list, n_rollouts, workers, horizon)

    model = EV.load_model(ckpt_path, device="cuda")
    losses = EV.heldout_losses(model, device="cuda")

    out = {}
    for c in cl:
        s = [x for t in probes[c] for x in succ.get(t, [])]
        out[c] = {
            "success_rate": float(np.mean(s)) if s else None,
            "n_episodes": len(s),
            "per_task_success": {t: float(np.mean(succ[t])) if t in succ else None
                                 for t in probes[c]},
            **{k: losses[c][k] for k in ("plain_loss", "transport_loss", "interaction_loss")},
        }
    meta = {"ckpt": ckpt, "n_rollouts": n_rollouts, "horizon": horizon,
            "rollout_wall_s": info["wall_s"], "n_errors": info["n_errors"],
            "total_wall_s": time.time() - t0}
    json.dump({"outcomes": out, "meta": meta}, open(os.path.join(run_dir, "outcomes.json"), "w"), indent=1)
    if info["n_errors"]:
        json.dump(info["errors"], open(os.path.join(run_dir, "rollout_errors.json"), "w"), indent=1)
        raise RuntimeError(f"{info['n_errors']} rollout task errors in {run_dir}")
    write_done(run_dir, json.dumps(meta) + "\n", name="probe")
    return {"outcomes": out, "meta": meta}


def suite_eval(run_dir, suite, ckpt="final.pt", n_rollouts=20, workers=14, horizon=HORIZON):
    """Evaluate on EVERY task of a suite (Gate 0 / Stage C use the full libero_goal suite:
    10 tasks x 20 rollouts = 200 episodes per model). Writes run_dir/suite_outcomes.json."""
    from clusters import suite_task_names
    if is_done(run_dir, "suite"):
        print(f"[suite] SKIP (suite.marker): {run_dir}")
        return json.load(open(os.path.join(run_dir, "suite_outcomes.json")))
    t0 = time.time()
    tasks = sorted(suite_task_names(suite))
    succ, info = run_rollouts(os.path.join(run_dir, ckpt), [(suite, t) for t in tasks],
                              n_rollouts, workers, horizon)
    per_task = {t: float(np.mean(succ[t])) for t in tasks if t in succ}
    out = {"suite": suite, "n_tasks": len(tasks), "n_rollouts": n_rollouts,
           "per_task_success": per_task,
           "suite_success": float(np.mean(list(per_task.values()))) if per_task else None,
           "n_episodes": sum(len(succ.get(t, [])) for t in tasks),
           "horizon": horizon, "wall_s": time.time() - t0, "n_errors": info["n_errors"]}
    json.dump(out, open(os.path.join(run_dir, "suite_outcomes.json"), "w"), indent=1)
    if info["n_errors"]:
        json.dump(info["errors"], open(os.path.join(run_dir, "suite_errors.json"), "w"), indent=1)
        raise RuntimeError(f"{info['n_errors']} suite rollout errors in {run_dir}")
    write_done(run_dir, json.dumps({k: out[k] for k in
                                    ("suite", "suite_success", "n_episodes", "wall_s")}) + "\n",
               name="suite")
    return out


def cluster_eval(run_dir, cluster, ckpt="final.pt", n_rollouts=20, workers=14, horizon=HORIZON):
    """Evaluate on EVERY task of one cluster (Gate 0 / Stage C).

    For C1 this is exactly the full libero_goal suite (10 tasks x 20 rollouts = 200 episodes),
    as the spec requires; it also generalizes to the C2/C5 fallback targets.
    Writes run_dir/cluster_eval.json.
    """
    import dataset
    if is_done(run_dir, "clustereval"):
        print(f"[clustereval] SKIP (marker): {run_dir}")
        return json.load(open(os.path.join(run_dir, "cluster_eval.json")))
    t0 = time.time()
    man = {c["cluster"]: c for c in dataset.manifest()["clusters"]}[cluster]
    suite, tasks = man["suite"], sorted(man["tasks"])
    succ, info = run_rollouts(os.path.join(run_dir, ckpt), [(suite, t) for t in tasks],
                              n_rollouts, workers, horizon)
    per_task = {t: float(np.mean(succ[t])) for t in tasks if t in succ}
    out = {"cluster": cluster, "suite": suite, "n_tasks": len(tasks),
           "n_rollouts": n_rollouts, "per_task_success": per_task,
           "cluster_success": float(np.mean(list(per_task.values()))) if per_task else None,
           "n_episodes": sum(len(succ.get(t, [])) for t in tasks),
           "horizon": horizon, "wall_s": time.time() - t0, "n_errors": info["n_errors"]}
    json.dump(out, open(os.path.join(run_dir, "cluster_eval.json"), "w"), indent=1)
    if info["n_errors"]:
        json.dump(info["errors"], open(os.path.join(run_dir, "cluster_errors.json"), "w"), indent=1)
        raise RuntimeError(f"{info['n_errors']} cluster rollout errors in {run_dir}")
    write_done(run_dir, json.dumps({k: out[k] for k in
                                    ("cluster", "cluster_success", "n_episodes", "wall_s")}) + "\n",
               name="clustereval")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--clusters", default=None, help="comma list, default all 9")
    ap.add_argument("--suite", default=None,
                    help="if set, evaluate EVERY task of this suite instead of the probe battery")
    ap.add_argument("--cluster_tasks", default=None,
                    help="if set, evaluate EVERY task of this cluster (Gate 0 / Stage C)")
    a = ap.parse_args()
    if a.cluster_tasks:
        r = cluster_eval(a.run_dir, a.cluster_tasks, a.ckpt, a.n_rollouts, a.workers)
        print(f"[clustereval] {a.run_dir} {a.cluster_tasks} "
              f"success={r['cluster_success']:.4f} ({r['n_episodes']} eps, {r['wall_s']:.0f}s)")
        return
    if a.suite:
        r = suite_eval(a.run_dir, a.suite, a.ckpt, a.n_rollouts, a.workers)
        print(f"[suite] {a.run_dir} {a.suite} success={r['suite_success']:.4f} "
              f"({r['n_episodes']} eps, {r['wall_s']:.0f}s)")
        return
    cl = a.clusters.split(",") if a.clusters else None
    r = probe_battery(a.run_dir, a.ckpt, a.n_rollouts, a.workers, cl)
    o = r["outcomes"]
    print(f"[probe] {a.run_dir} wall={r['meta']['total_wall_s']:.0f}s")
    for c, v in o.items():
        print(f"  {c}: succ={v['success_rate']} ({v['n_episodes']} eps) "
              f"plain={v['plain_loss']:.3f} transp={v['transport_loss']:.3f} "
              f"inter={v['interaction_loss']:.3f}")


if __name__ == "__main__":
    main()
