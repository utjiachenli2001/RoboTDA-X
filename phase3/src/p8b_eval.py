"""P8b -- do CHEAP estimator repairs buy mask-ranking reliability more cheaply than training seeds?

Phase-2 P4's finding was that ground-truth reliability is bought with SEEDS, not EPISODES -- and
seeds are expensive. So: can we get the seeds' benefit WITHOUT paying for them? Two repairs, both
requiring ZERO new training:

  ARM (i)  CHECKPOINT-AVERAGED POLICY -- average the WEIGHTS of the last 3 checkpoints
           (ckpt_2, ckpt_3, ckpt_4) of each EXISTING run. These checkpoints already exist because
           TracIn needed them, so this repair is FREE. Does averaging along the training
           trajectory recover any of what averaging across seeds buys?

  ARM (ii) ACTION-ENSEMBLE POLICY -- average the ACTION OUTPUTS of S models at each rollout step
           (one env, S policies stepped in lockstep). This costs S trained models, i.e. exactly
           what seed-ensembling costs -- so the question is whether ensembling in ACTION space
           beats ensembling in OUTCOME space at the SAME seed budget.

THE COMPARISONS (all over the same 12 masks, both outcomes):
    baseline S=1 : mean pairwise Spearman of single-seed mask-score vectors (seeds 401,402,403)
    arm (i)  S=1 : the same, with every model replaced by its checkpoint-average   <- FREE
    baseline S=3 : Spearman( mean-outcome(401,402,403), mean-outcome(404,405,406) )
    arm (ii) S=3 : Spearman( action-ens(401,402,403),   action-ens(404,405,406) )  <- same cost

So arm (i) is judged against a free baseline, and arm (ii) against an equal-seed-cost baseline.

DETERMINISM GATE (preregistered): before either arm is trusted, one repeated episode must replay
BIT-IDENTICALLY (same success flag AND same step count). A failure is an instrument defect -> STOP.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS, RUNS, P2_RUNS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import bootstrap  # noqa: F401,E402
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402

SEEDS_A = [401, 402, 403]
SEEDS_B = [404, 405, 406]
CKPT_AVG = ["ckpt_2.pt", "ckpt_3.pt", "ckpt_4.pt"]     # the last 3 (preregistered)
N_ROLLOUTS = 10


def g6_run_dir(mask_id, seed):
    """Seeds 401-402 are Phase-1 Stage G; 403-406 are Phase-2 Stage G6."""
    p1 = os.path.join(RUNS, "stage_G", f"{mask_id}_s{seed}")
    return p1 if os.path.isdir(p1) else os.path.join(P2_RUNS, "stage_G6", f"{mask_id}_s{seed}")


def masks12():
    return json.load(open(os.path.join(P3_RESULTS, "p8a_mask_selection.json")))["masks"]


# ---------------------------------------------------------------- arm (i): checkpoint average
def make_ckpt_avg(mask_id, seed):
    """Average the last 3 checkpoints' weights into a new run dir. Returns that dir."""
    src = g6_run_dir(mask_id, seed)
    dst = os.path.join(P3_RUNS, "P8b", f"ckptavg_{mask_id}_s{seed}")
    os.makedirs(dst, exist_ok=True)
    fp = os.path.join(dst, "final.pt")
    if os.path.exists(fp):
        return dst
    cks = [torch.load(os.path.join(src, c), map_location="cpu", weights_only=False)
           for c in CKPT_AVG]
    avg = {}
    for k in cks[0]["model"]:
        ts = [c["model"][k].float() for c in cks]
        avg[k] = torch.stack(ts).mean(0).to(cks[0]["model"][k].dtype)
    torch.save({"model": avg, "step": cks[-1]["step"], "epoch": cks[-1]["epoch"],
                "cfg": cks[-1]["cfg"], "state_dim": cks[-1]["state_dim"],
                "provenance": {"arm": "checkpoint_average", "source_run": src,
                               "checkpoints": CKPT_AVG}}, fp)
    json.dump({"demos": json.load(open(os.path.join(src, "demos.json")))["demos"], "seed": seed},
              open(os.path.join(dst, "demos.json"), "w"), indent=1)
    return dst


# ---------------------------------------------------------------- arm (ii): action ensemble
class ActionEnsemble:
    """S policies stepped in lockstep through ONE env; the executed action is their MEAN."""

    def __init__(self, ckpt_paths, device="cuda"):
        self.models = [EV.load_model(p, device=device) for p in ckpt_paths]

    @torch.no_grad()
    def act(self, states, lang):
        a = torch.stack([m.act(states, lang) for m in self.models])     # (S,B,7)
        return a.mean(0).clamp(-1, 1)

    @torch.no_grad()
    def l2(self, states, lang, actions):
        a = torch.stack([m.mean_action(states, lang) for m in self.models]).mean(0)
        return ((a - actions) ** 2).sum(-1)


_W = {}


def _init_worker_ens(ckpt_paths, obj_pad_dim, mean, std):
    torch.set_num_threads(1)
    _W["model"] = ActionEnsemble(ckpt_paths, device="cuda")
    _W["obj_pad_dim"], _W["mean"], _W["std"] = obj_pad_dim, mean, std
    import taskemb
    _W["emb"] = taskemb.load()


def _rollout_task_ens(job):
    suite, task, ep_idx, horizon = job
    t0 = time.time()
    try:
        import libero_env as LE
        from libero.libero import benchmark
        model = _W["model"]
        D, mean, std = _W["obj_pad_dim"], _W["mean"], _W["std"]
        lang = torch.from_numpy(_W["emb"][task][None, :]).cuda()
        B = benchmark.get_benchmark_dict()[suite]()
        names = [t.name for t in B.tasks]
        init_states = np.asarray(B.get_task_init_states(names.index(task)))
        env = LE.make_env(LE.get_bddl_path(suite, task), horizon=horizon + 50, seed=0)
        CTX, succ, steps = 10, [], []
        for ep in ep_idx:
            env.reset()
            obs = env.set_init_state(init_states[ep % init_states.shape[0]])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))
            f = (LE.featurize(obs, D) - mean) / std
            win = [f] * CTX
            ok, t = False, 0
            for t in range(1, horizon + 1):
                s = torch.from_numpy(np.stack(win[-CTX:])[None]).float().cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a = model.act(s, lang)
                obs, _, _, _ = env.step(a[0].float().cpu().numpy())
                if LE.check_success(env):
                    ok = True
                    break
                win.append((LE.featurize(obs, D) - mean) / std)
            succ.append(bool(ok))
            steps.append(int(t))
        try:
            env.close()
        except Exception:
            pass
        return {"task": task, "success": succ, "steps": steps, "wall_s": time.time()-t0,
                "err": None}
    except Exception:
        import traceback
        return {"task": task, "success": [], "steps": [], "wall_s": time.time()-t0,
                "err": traceback.format_exc()[-1200:]}


def ensemble_probe(run_dir, ckpt_paths, n_rollouts=N_ROLLOUTS, workers=12):
    """Probe battery for an ACTION-ENSEMBLE policy. Marker-gated + atomic (P6.1)."""
    import multiprocessing as mp
    if L.is_marked(run_dir, "probe"):
        return L.read_artifact(run_dir, "outcomes.json")
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    probes = dataset.probe_tasks()
    suite_of = dataset.suite_of_cluster()
    cl = dataset.clusters()
    tasks = [(suite_of[c], t) for c in cl for t in probes[c]]
    mean, std = dataset.norm_stats()
    D = dataset.obj_pad_dim()
    ep = list(range(n_rollouts))
    jobs = [(s, t, ep, 600) for (s, t) in tasks]
    ctx = mp.get_context("spawn")
    succ, errs = {}, []
    with ctx.Pool(min(workers, len(jobs)), initializer=_init_worker_ens,
                  initargs=(ckpt_paths, D, mean, std)) as pool:
        for r in pool.imap_unordered(_rollout_task_ens, jobs):
            if r["err"]:
                errs.append(r)
                print(f"[p8b-ens] ERROR {r['task']}: {r['err'][:200]}", flush=True)
            else:
                succ[r["task"]] = r["success"]
    if errs:
        L.atomic_write_json(os.path.join(run_dir, "rollout_errors.json"), errs)
        raise RuntimeError(f"{len(errs)} rollout errors in {run_dir}")

    ens = ActionEnsemble(ckpt_paths, device="cuda")
    bank = EV.heldout_bank("base")
    l2 = np.empty(bank.n)
    with torch.no_grad():
        for i in range(0, bank.n, 1024):
            s = torch.from_numpy(bank.S[i:i+1024]).cuda()
            l = torch.from_numpy(bank.L[i:i+1024]).cuda()
            a = torch.from_numpy(bank.A[i:i+1024]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l2[i:i+1024] = ens.l2(s, l, a).float().cpu().numpy()
    _, by_c = dataset.heldout_pool()
    id2k = {d: k for k, d in enumerate(bank.ids)}
    tr, it = bank.masks["base"]["transport"], bank.masks["base"]["interaction"]
    out = {}
    for c in cl:
        rows = np.concatenate([np.nonzero(bank.owner == id2k[d])[0] for d in by_c[c]])
        s = [x for t in probes[c] for x in succ.get(t, [])]
        q, tt, ii = l2[rows], tr[rows], it[rows]
        out[c] = {"success_rate": float(np.mean(s)), "n_episodes": len(s),
                  "plain_loss": float(q.mean()),
                  "transport_loss": float((q*tt).sum()/max(tt.sum(), 1)),
                  "interaction_loss": float((q*ii).sum()/max(ii.sum(), 1))}
    meta = {"arm": "action_ensemble", "n_members": len(ckpt_paths), "members": ckpt_paths,
            "n_rollouts": n_rollouts, "total_wall_s": time.time()-t0}
    L.atomic_write_json(os.path.join(run_dir, "outcomes.json"), {"outcomes": out, "meta": meta})
    from bootstrap import write_done
    write_done(run_dir, json.dumps(meta) + "\n", name="probe")
    return {"outcomes": out, "meta": meta}


# ---------------------------------------------------------------- determinism gate
def determinism_gate():
    """Replay ONE episode twice under each repaired policy; require bit-identical outcomes."""
    m = masks12()[0]
    res = {}

    # arm (i) -- use the worker directly so STEP COUNTS are captured too. rollout.run_rollouts
    # discards them, and success flags alone are not a real determinism test: a near-floor model
    # matches trivially by failing every episode at the full horizon.
    d = make_ckpt_avg(m, 401)
    import rollout as RO
    probes = dataset.probe_tasks()["C1"][:1]
    suite = dataset.suite_of_cluster()["C1"]
    mean, std = dataset.norm_stats()
    D = dataset.obj_pad_dim()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    ci = []
    for _ in range(2):
        with ctx.Pool(1, initializer=RO._init_worker,
                      initargs=(os.path.join(d, "final.pt"), D, mean, std)) as pool:
            rr = list(pool.imap_unordered(RO._rollout_task,
                                          [(suite, probes[0], list(range(3)), 600)]))[0]
        ci.append((rr["success"], rr["steps"]))
    res["arm_i_checkpoint_average"] = {
        "run1_success": ci[0][0], "run1_steps": ci[0][1],
        "run2_success": ci[1][0], "run2_steps": ci[1][1],
        "bit_identical": ci[0] == ci[1]}

    # arm (ii)
    cks = [os.path.join(g6_run_dir(m, s), "final.pt") for s in SEEDS_A]
    mean, std = dataset.norm_stats()
    D = dataset.obj_pad_dim()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    outs = []
    for _ in range(2):
        with ctx.Pool(1, initializer=_init_worker_ens, initargs=(cks, D, mean, std)) as pool:
            rr = list(pool.imap_unordered(_rollout_task_ens,
                                          [(suite, probes[0], list(range(3)), 600)]))[0]
        outs.append((rr["success"], rr["steps"]))
    res["arm_ii_action_ensemble"] = {
        "run1_success": outs[0][0], "run1_steps": outs[0][1],
        "run2_success": outs[1][0], "run2_steps": outs[1][1],
        "bit_identical": outs[0] == outs[1]}

    res["GATE"] = bool(res["arm_i_checkpoint_average"]["bit_identical"]
                       and res["arm_ii_action_ensemble"]["bit_identical"])
    res["meaning"] = ("Both repaired policies must be DETERMINISTIC (same success flags AND same "
                      "step counts on a replay), or their mask rankings would carry rollout noise "
                      "that the baseline does not have.")
    L.atomic_write_json(os.path.join(P3_RESULTS, "p8b_determinism.json"), res)
    print(f"[p8b] determinism gate: arm(i)={res['arm_i_checkpoint_average']['bit_identical']} "
          f"arm(ii)={res['arm_ii_action_ensemble']['bit_identical']} GATE={res['GATE']}")
    return res["GATE"]


def work_items():
    """Every P8b evaluation, as a flat list, so it can be SHARDED across GPUs.

    arm (i):  12 masks x 3 seeds  = 36 checkpoint-averaged probe batteries
    arm (ii): 12 masks x 2 halves = 24 action-ensemble probe batteries
    """
    ms = masks12()
    items = []
    for m in ms:
        for s in SEEDS_A:
            items.append(("i", m, s))
    for m in ms:
        for tag in ("A", "B"):
            items.append(("ii", m, tag))
    return items


def run_item(it):
    arm, m, k = it
    if arm == "i":
        d = make_ckpt_avg(m, k)
        import rollout as RO
        RO.probe_battery(d, "final.pt", N_ROLLOUTS, workers=12)
        print(f"[p8b-i] {m} s{k} done", flush=True)
    else:
        seeds = SEEDS_A if k == "A" else SEEDS_B
        rd = os.path.join(P3_RUNS, "P8b", f"actens_{m}_{k}")
        cks = [os.path.join(g6_run_dir(m, s), "final.pt") for s in seeds]
        ensemble_probe(rd, cks, N_ROLLOUTS, workers=12)
        print(f"[p8b-ii] {m} {k} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--shard", type=int, default=None, help="this shard index")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--arm", choices=["i", "ii"], default=None)
    a = ap.parse_args()

    if a.gate:
        sys.exit(0 if determinism_gate() else 1)

    items = work_items()
    if a.arm:
        items = [i for i in items if i[0] == a.arm]
    if a.shard is not None:
        items = [it for k, it in enumerate(items) if k % a.nshard == a.shard]
    print(f"[p8b] shard {a.shard}/{a.nshard}: {len(items)} evaluations", flush=True)
    for it in items:
        run_item(it)
    print(f"[p8b] shard {a.shard} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
