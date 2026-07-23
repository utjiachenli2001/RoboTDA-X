"""B10 -- the rollout-visited-states target functional (the last B2 functional, finally unblocked).

B2 proposed three redesigned functionals; two (ens_var, fail_div) were run in pass 3, the third
was blocked because LIBERO/robosuite were not installed. They are now installed (mujoco 3.1.6,
robosuite 1.4.0, libero 0.1.1; the config is redirected to the venv's assets via
runs/libero_cfg/config.yaml, no repo file edited). This module runs it.

THE IDEA. The study's plain functional weights every held-out demo frame equally. But at
deployment the policy only ever encounters the states its OWN rollouts visit, and on this corpus
the policy is broken -- it rarely reaches the demo's late-task states at all. So attribution
aimed at "loss on held-out demo frames" may be scoring states the policy never sees. The
rollout-visited weighting fixes the test side to the states that actually occur under the policy:

    roll out the 5 regenerated ensemble members on the held-out tasks, collect every visited
    (featurized, frozen-normalized) state, and weight each held-out frame by the KERNEL DENSITY
    of that visited-state cloud at the frame. Frames near where the policy actually goes get
    weight; frames it never reaches get ~0.

Like ens_var/fail_div, the weighting is an ENSEMBLE property computed ONCE, not per mask -- so
the only GPU cost is the rollouts. The outcome side is free: campaign A/B/I already store
per-frame held-out losses, so `functionals.campaign_outcomes` gives the reweighted outcome and
its split-half ceiling with no new training.

Per-cluster clouds. A cluster's held-out frames are weighted by that cluster's OWN rollout
states (the tasks differ across clusters), so the density is measured in the right neighbourhood.

The weighting is registered into functionals.py by writing `runs/heldout_weights.npz` with new
keys `rollout` / `rollout_q75`, which `functionals.weights()` already knows how to read.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RUNS = os.path.join(HERE, "runs")
VISITED = os.path.join(RUNS, "rollout_visited.npz")
N_EPISODES = 1         # one rollout per (member, task) -- the cloud only needs state COVERAGE
HORIZON = 250
SUBSAMPLE = 2          # keep every 2nd visited frame (states are highly autocorrelated)
WORKERS = 6            # fewer processes -> less GPU contention on the per-step batch-1 forward
N_MEMBERS = 3          # 3 of the 5 regen members; the broken policies visit near-identical states

# LIBERO needs these set before import; bootstrap reads LIBERO_CONFIG_PATH.
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(RUNS, "libero_cfg"))
os.environ.setdefault("MUJOCO_GL", "osmesa")


_W = {}


def _init_member(ckpt):
    """Load one member's model ONCE per worker process (not per job)."""
    import torch
    torch.set_num_threads(1)
    sys.path[:0] = [os.path.join(D.ROOT, "src"), os.path.join(D.ROOT, "phase3", "src")]
    import bootstrap  # noqa: F401
    import dataset
    import evaluate as EV
    import taskemb
    _W["torch"] = torch
    _W["model"] = EV.load_model(ckpt, device="cuda")
    _W["emb"] = taskemb.load()
    _W["mean"], _W["std"] = dataset.norm_stats()
    _W["Dpad"] = dataset.obj_pad_dim()


def _worker(job):
    """(suite, task, cluster) -> visited normalized states, using the preloaded member model."""
    suite, task, cluster = job
    try:
        import numpy as _np
        import libero_env as LE
        from libero.libero import benchmark
        torch = _W["torch"]
        model, emb = _W["model"], _W["emb"]
        mean, std, Dpad = _W["mean"], _W["std"], _W["Dpad"]

        lang = torch.from_numpy(emb[task][None, :]).cuda()
        Bm = benchmark.get_benchmark_dict()[suite]()
        ti = [t.name for t in Bm.tasks].index(task)
        init = _np.asarray(Bm.get_task_init_states(ti))
        env = LE.make_env(LE.get_bddl_path(suite, task), horizon=HORIZON + 50, seed=0)
        CTX = 10
        visited, nsucc = [], 0
        for ep in range(N_EPISODES):
            env.reset()
            obs = env.set_init_state(init[ep % init.shape[0]])
            for _ in range(5):
                obs, _, _, _ = env.step(_np.zeros(7))
            f = (LE.featurize(obs, Dpad) - mean) / std
            win = [f] * CTX
            for t in range(1, HORIZON + 1):
                s = torch.from_numpy(_np.stack(win[-CTX:])[None]).float().cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a = model.act(s, lang)
                obs, _, _, _ = env.step(a[0].float().cpu().numpy())
                fv = (LE.featurize(obs, Dpad) - mean) / std
                win.append(fv)
                if t % SUBSAMPLE == 0:
                    visited.append(fv.astype(_np.float32))
                if LE.check_success(env):
                    nsucc += 1
                    break
        try:
            env.close()
        except Exception:
            pass
        V = _np.stack(visited) if visited else _np.zeros((0, len(mean)), _np.float32)
        return {"cluster": cluster, "task": task, "V": V, "n_succ": nsucc, "err": None}
    except Exception:
        import traceback
        return {"cluster": cluster, "task": task, "V": None, "n_succ": 0,
                "err": traceback.format_exc()[-1500:]}


def collect_visited(force=False):
    if os.path.exists(VISITED) and not force:
        z = np.load(VISITED, allow_pickle=True)
        return {k: z[k] for k in z.files if k.startswith("V_")}, json.loads(str(z["meta"]))
    D.add_repo_paths()
    import dataset
    _, by_c = dataset.heldout_pool()
    members = sorted(glob.glob(os.path.join(RUNS, "regen", "ens_s*", "final.pt")))[:N_MEMBERS]
    tasks_by_cluster = {c: sorted({tuple(d.split("/")[:2]) for d in by_c[c]})
                        for c in dataset.clusters()}
    task_jobs = [(suite, task, c) for c, ts in tasks_by_cluster.items()
                 for (suite, task) in ts]
    print(f"[b10] {len(members)} members x {len(task_jobs)} tasks "
          f"= {len(members)*len(task_jobs)} rollouts, {WORKERS} workers", flush=True)
    t0 = time.time()
    per_cluster = {}
    meta = {"n_episodes": N_EPISODES, "horizon": HORIZON, "subsample": SUBSAMPLE,
            "members": [os.path.basename(os.path.dirname(m)) for m in members],
            "n_jobs": len(members) * len(task_jobs), "errors": [], "success_by_cluster": {}}
    done = 0
    for ck in members:
        # one pool per member: the model is loaded ONCE per worker via the initializer
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_member,
                                 initargs=(ck,)) as ex:
            for r in ex.map(_worker, task_jobs):
                done += 1
                if r["err"]:
                    meta["errors"].append(f"{r['cluster']}/{r['task']}: {r['err'][-200:]}")
                    continue
                per_cluster.setdefault(r["cluster"], []).append(r["V"])
                meta["success_by_cluster"][r["cluster"]] = \
                    meta["success_by_cluster"].get(r["cluster"], 0) + r["n_succ"]
                if done % 30 == 0:
                    print(f"[b10] {done}/{meta['n_jobs']} ({time.time()-t0:.0f}s)", flush=True)
    clouds = {f"V_{c}": np.concatenate(vs, 0) for c, vs in per_cluster.items() if vs}
    os.makedirs(RUNS, exist_ok=True)
    np.savez_compressed(VISITED, meta=json.dumps(meta), **clouds)
    print(f"[b10] visited-state clouds: "
          f"{ {c.replace('V_',''): int(v.shape[0]) for c, v in clouds.items()} } "
          f"({time.time()-t0:.0f}s, {len(meta['errors'])} errors)", flush=True)
    return clouds, meta


def build_weighting(clouds, bandwidth=None, device="cuda"):
    """-> (F,) frame weight = per-cluster kernel density of the visited cloud at each held-out frame.

    Density is a Gaussian kernel sum over visited states, computed per cluster on the frames that
    belong to that cluster. Bandwidth defaults to the median pairwise distance within the cloud
    (a standard heuristic), so the scale is set by the data, not chosen.
    """
    import torch
    from if_repair import functionals as F
    fm = F.frame_meta()
    bank_S = _heldout_states()               # (F, 128) frozen-normalized
    w = np.zeros(bank_S.shape[0], np.float64)
    MIN_CLOUD = 200                          # below this, the density estimate is unreliable
    covered, fell_back = [], []
    for c in sorted(set(fm["cluster_of_row"])):
        rows = np.nonzero(fm["cluster_of_row"] == c)[0]
        V = clouds.get(f"V_{c}")
        # FALLBACK: a cluster whose rollouts mostly failed (fd leak / missing asset) gets the
        # UNIFORM weighting on its frames rather than a zero. That keeps the functional defined
        # everywhere; such clusters are reported as `uniform-fallback` and not read as a rollout
        # result. C1-C5 (the dev targets) have full clouds and never fall back.
        if V is None or V.shape[0] < MIN_CLOUD or len(rows) == 0:
            w[rows] = 1.0
            fell_back.append(c)
            continue
        covered.append(c)
        Xf = torch.from_numpy(bank_S[rows]).float().to(device)
        Vt = torch.from_numpy(V).float().to(device)
        # bandwidth = median nearest-neighbour distance in V (subsampled for speed)
        m = min(2000, Vt.shape[0])
        idx = torch.randperm(Vt.shape[0], device=device)[:m]
        dd = torch.cdist(Vt[idx], Vt[idx])
        dd.fill_diagonal_(float("inf"))
        h = float(dd.min(dim=1).values.median()) if bandwidth is None else bandwidth
        h = max(h, 1e-3)
        # density at each frame: mean over V of exp(-||x-v||^2 / 2h^2), in chunks
        dens = torch.empty(len(rows), device=device)
        for i in range(0, len(rows), 256):
            d2 = torch.cdist(Xf[i:i+256], Vt) ** 2
            dens[i:i+256] = torch.exp(-d2 / (2 * h * h)).mean(dim=1)
        d = dens.double().cpu().numpy()
        # normalise each cluster's weights to mean 1 so the density scale (which depends on the
        # cloud size and bandwidth) does not change the cluster's overall contribution -- only
        # the RELATIVE weighting of its frames, which is the whole point.
        w[rows] = d / d.mean() if d.mean() > 0 else 1.0
    print(f"[b10] rollout weighting: {len(covered)} clusters from real clouds {covered}; "
          f"{len(fell_back)} uniform-fallback {fell_back}", flush=True)
    return w, covered, fell_back


def _heldout_states():
    """The frozen-normalized 128-d state of every held-out frame, in bank row order."""
    D.add_repo_paths()
    import evaluate as EV
    bank = EV.heldout_bank("base")
    return bank.S[:, -1, :].astype(np.float64)      # last (current) frame of each window


def register_weighting(w):
    """Add `rollout` (+ q75) to runs/heldout_weights.npz so functionals.weights() serves them."""
    from if_repair import functionals as F
    cache = dict(np.load(F.WEIGHT_CACHE)) if os.path.exists(F.WEIGHT_CACHE) else {}
    cache["rollout"] = w.astype(np.float64)
    np.savez_compressed(F.WEIGHT_CACHE, **cache)
    F.weights.cache_clear()
    # extend the known soft-weightings so q75 works
    if "rollout" not in F.SOFT_WEIGHTINGS:
        F.SOFT_WEIGHTINGS = tuple(F.SOFT_WEIGHTINGS) + ("rollout",)


def main():
    import argparse
    import pandas as pd
    from if_repair import functionals as F
    from if_repair import b8_maskdraw as B8
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    clouds, meta = collect_visited(force=a.force)
    print(f"[b10] rollout success totals by cluster: {meta['success_by_cluster']}")
    w, covered, fell_back = build_weighting(clouds)
    register_weighting(w)
    np.save(os.path.join(RESULTS, "b10_rollout_weight.npy"), w)

    weightings = ("plain", "rollout", "rollout_q75")
    # STEP 3: gate on each of the three campaigns
    rows = []
    for camp in ("A", "B", "I"):
        for wn in weightings:
            try:
                gate = F.ceiling_table("campaign", weightings=(wn,), targets=("C1", "C5"),
                                       campaign=camp)
            except Exception as e:                 # noqa: BLE001
                print(f"[b10] gate {camp}/{wn}: {e}")
                continue
            for _, r in gate.iterrows():
                rows.append({"campaign": camp, **r.to_dict()})
    gate_df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    gate_df.to_csv(os.path.join(RESULTS, "b10_ceilings.csv"), index=False)
    print("\n=== B10 STEP 3: ceiling gate for the rollout functional (>=0.4) ===")
    print(gate_df.pivot_table(index=["campaign", "target"], columns="weighting",
                              values="ceiling").round(3).to_string())

    # STEP 4: GradDot on the rollout functional vs plain, on campaign A (dev), then paired
    # difference vs plain on the I-series (out of sample).
    from if_repair import b2_functionals as B2
    _, ens = B2.build_ensemble(tuple(sorted(set(F.WEIGHTINGS) | {"rollout", "rollout_q75"})),
                               force=True)
    from if_repair import retrain as RT
    from p6_lambda_extend import scores_graddot
    from lds import spearman
    imasks = [{"mask_id": m["mask_id"], "demos": m["demos"]}
              for m in RT.fresh_demo_masks(seed=9973, prefix="I")[0]]

    print("\n=== B10 STEP 4: rollout functional vs plain (GradDot_dmean, ratio-to-ceiling) ===")
    res = []
    for camp, label in (("A", "dev (archived masks)"), ("I", "I-series (fresh)")):
        gm = ([{"mask_id": m["mask_id"], "demos": m["demos"]} for m in D.demo_masks()]
              if camp == "A" else imasks)
        for t in ("C1", "C5"):
            for wn in weightings:
                Z = ens[wn]; sc = scores_graddot(Z, normalize_per_member=True)[t]
                raw = F.campaign_outcomes(camp, wn, targets=(t,))[t]
                c = F.split_half_ceiling(raw)["ceiling"]; obs = F.seed_mean(raw)
                pred = np.array([sum(sc.get(d, 0.0) for d in m["demos"]) for m in gm])
                out = np.array([obs.get(m["mask_id"], np.nan) for m in gm])
                ok = np.isfinite(out)
                rho = spearman(pred[ok], out[ok])
                res.append({"where": label, "target": t, "weighting": wn,
                            "lds": rho, "ceiling": c, "ratio": rho / c})
    rf = pd.DataFrame(res)
    rf.to_csv(os.path.join(RESULTS, "b10_scores.csv"), index=False)
    print(rf.pivot_table(index=["where", "target"], columns="weighting",
                         values="ratio").round(3).to_string())


if __name__ == "__main__":
    main()
