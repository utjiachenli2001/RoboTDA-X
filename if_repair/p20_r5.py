"""R5 -- does attribution-based selection buy anything? Preregistered in `p20_prereg_r5.md`.

Campaign U measured that attribution is poorly MEASURABLE. It never asked what that costs. R5 asks
the deployable question: given a 370-demonstration pool and a budget of 25, does selecting by
influence beat selecting at random -- on held-out loss, and on task success?

WHY CHECKPOINTS ARE SAVED HERE AND NOWHERE ELSE. Campaign U deliberately keeps no weights (18 GB
for nothing downstream). R5's primary outcome is ROLLOUT SUCCESS, and `src/rollout.py` takes a
checkpoint path, so these 65 policies are the only ones written to disk.

WHY TASK COVERAGE IS ENFORCED ON EVERY ARM. A selection that silently drops a whole task would win
or lose for a reason that has nothing to do with attribution -- it would be measuring composition,
not influence. Coverage is repaired by swapping in the highest-ranked demonstration of each missing
task, and the swap count is reported per arm. This makes the test HARDER, deliberately.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p18_corpus as C  # noqa: E402
from if_repair import p18_campaign_u as U  # noqa: E402
from if_repair import p18_eval as EVT  # noqa: E402
from if_repair import p18_gram as GR  # noqa: E402
from if_repair import p18_score as S  # noqa: E402
from if_repair import retrain as R  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import train as TR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "runs", "r5")
OUT_CSV = os.path.join(RESULTS, "confirm_r5.csv")

POOL = 370
BUDGET = 25
SEEDS = (8101, 8102, 8103, 8104, 8105)
N_RANDOM = 10
RANDOM_SEED = 20260813
ROLLOUT_SEEDS = (8101, 8102, 8103)        # 3 of 5 rolled out; 100 episodes each
N_ROLLOUTS = 10                            # per task, x10 tasks = 100 episodes per policy


# --------------------------------------------------------------------------- scores
def datamodel_per_demo(pool=POOL, fit_partition="A"):
    """Per-demo datamodel influence: refit exactly as `p18_score.datamodel_prediction` does, and
    return the coefficient -> per-demo mapping instead of the mask predictions."""
    table, _ = S.outcome_table()
    fit_masks = U.build(pool, fit_partition)[0]
    fit_sigs = [m["sig"] for m in fit_masks if m["sig"] in table[(pool, fit_partition)]]
    G = len(C.groups(pool, fit_partition))
    fit_sigs = fit_sigs[:min(len(fit_sigs), int(round(9.2 * G * G / (G - 5))))]
    gids = [g["group_id"] for g in C.groups(pool, fit_partition)]
    gi = {g: k for k, g in enumerate(gids)}
    by_sig = {m["sig"]: m for m in fit_masks}
    X = np.zeros((len(fit_sigs), len(gids)))
    for r, s in enumerate(fit_sigs):
        for g in by_sig[s]["groups"]:
            X[r, gi[g]] = 1.0
    y = np.array([table[(pool, fit_partition)][s][0] for s in fit_sigs])
    Xc, yc = X - X.mean(0), y - y.mean()
    beta = np.linalg.solve(Xc.T @ Xc + np.eye(len(gids)), Xc.T @ yc)
    out = {}
    for g, k in gi.items():
        for d in C.index(pool, fit_partition)[g]["demos"]:
            # NEGATED: beta predicts the OUTCOME (held-out loss), so a demo that LOWERS loss has a
            # negative coefficient. Influence is oriented so that HIGHER = better to keep.
            out[d] = -beta[k] / C.GROUP_SIZE
    return out


def _cover(sel, scores, pool=POOL):
    """Repair a selection so all 10 tasks appear, swapping in each missing task's best demo."""
    sel = list(sel)
    swaps = 0
    while True:
        have = {d.split("/")[1] for d in sel}
        missing = [t for t in C.tasks() if t not in have]
        if not missing:
            return sel, swaps
        t = missing[0]
        cand = max((d for d in C.rung_demos(pool) if d.split("/")[1] == t),
                   key=lambda d: scores[d])
        # drop the lowest-ranked demo of the most over-represented task
        cnt = {}
        for d in sel:
            cnt[d.split("/")[1]] = cnt.get(d.split("/")[1], 0) + 1
        worst_task = max(cnt, key=lambda k: cnt[k])
        drop = min((d for d in sel if d.split("/")[1] == worst_task), key=lambda d: scores[d])
        sel.remove(drop)
        sel.append(cand)
        swaps += 1


def selections():
    """-> {arm: (sorted demo ids, n_coverage_swaps)}."""
    dm = datamodel_per_demo()
    # GradDot orientation: score_i = <grad NLL(demo_i), grad L2(eval bank)>. A positive dot means
    # descending demo i's loss also descends the held-out loss, i.e. training on it HELPS. So the
    # raw score is already oriented "higher = better to keep" and needs no negation -- unlike the
    # datamodel, whose beta predicts the loss itself and is negated in datamodel_per_demo().
    gd = GR.load(POOL, "H1")
    ids = list(C.rung_demos(POOL))
    arms = {}
    top = sorted(ids, key=lambda d: dm[d], reverse=True)[:BUDGET]
    bot = sorted(ids, key=lambda d: dm[d])[:BUDGET]
    gtop = sorted(ids, key=lambda d: gd[d], reverse=True)[:BUDGET]
    arms["DM-top"] = _cover(top, dm)
    arms["DM-bottom"] = _cover(bot, {d: -v for d, v in dm.items()})
    arms["GD-top"] = _cover(gtop, gd)
    rng = np.random.default_rng(RANDOM_SEED)
    for i in range(N_RANDOM):
        r = [ids[j] for j in rng.permutation(len(ids))[:BUDGET]]
        arms[f"RANDOM-{i:02d}"] = _cover(r, {d: 0.0 for d in ids})
    return {k: (sorted(v[0]), v[1]) for k, v in arms.items()}


# --------------------------------------------------------------------------- run
def run(worker=0, nworkers=1, steps=None):
    import time
    cfg = TR.load_cfg()
    if steps:
        cfg["total_steps"] = steps
    os.makedirs(CKPT, exist_ok=True)
    sel = selections()
    jobs = [(a, s) for a in sel for s in SEEDS]
    mine = [j for k, j in enumerate(jobs) if k % nworkers == worker]
    fidx = EVT.frame_index()
    print(f"[r5] worker {worker}/{nworkers}: {len(mine)} of {len(jobs)} policies", flush=True)
    t0 = time.time()
    for arm, seed in mine:
        out = os.path.join(CKPT, f"{arm}_s{seed}.pt")
        res = os.path.join(CKPT, f"{arm}_s{seed}.json")
        if os.path.exists(res):
            continue
        demos, _ = sel[arm]
        model, meta = R.train_one(demos, seed, seed, cfg)
        l2, nll = EVT.heldout_frame_losses(model)
        agg = EVT.aggregate(l2, nll, fidx)
        torch.save({"state_dim": dataset.state_dim(), "cfg": cfg,
                    "model": model.state_dict()}, out)
        json.dump({"arm": arm, "seed": seed, "n_demos": len(demos),
                   "plain_loss": agg["plain_loss"], "final_train_loss": meta["final_loss"],
                   "demos": demos}, open(res, "w"))
        del model
        torch.cuda.empty_cache()
        print(f"[r5] {arm} s{seed} loss={agg['plain_loss']:.5f} "
              f"({(time.time() - t0) / 60:.1f}m)", flush=True)
    print(f"[r5] worker {worker} DONE ({(time.time() - t0) / 60:.1f} m)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--rollouts", action="store_true")
    ap.add_argument("--n-rollouts", type=int, default=N_ROLLOUTS)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    if a.rollouts:
        run_rollouts_all(a.worker, a.nworkers, a.n_rollouts, a.procs)
        return
    if a.plan:
        sel = selections()
        dm = datamodel_per_demo()
        print(f"[r5] pool {POOL}, budget {BUDGET}, {len(sel)} arms x {len(SEEDS)} seeds "
              f"= {len(sel) * len(SEEDS)} policies")
        for arm, (d, sw) in sel.items():
            tasks = len({x.split('/')[1] for x in d})
            print(f"  {arm:12s} n={len(d)} tasks={tasks} coverage_swaps={sw} "
                  f"mean_dm_score={np.mean([dm[x] for x in d]):+.6f}")
        ov = set(sel["DM-top"][0]) & set(sel["DM-bottom"][0])
        print(f"[r5] DM-top n DM-bottom overlap: {len(ov)} (should be 0)")
        os.makedirs(RESULTS, exist_ok=True)
        json.dump({k: {"demos": v[0], "coverage_swaps": v[1]} for k, v in sel.items()},
                  open(os.path.join(RESULTS, "p20_r5_selections.json"), "w"), indent=1)
    if a.run:
        run(a.worker, a.nworkers, a.steps)




# --------------------------------------------------------------------------- rollouts
def rollout_one(arm, seed, n_rollouts=N_ROLLOUTS, workers=14):
    """100 episodes for one policy: 10 libero_goal tasks x n_rollouts, via src/rollout.py."""
    import rollout as RO
    ck = os.path.join(CKPT, f"{arm}_s{seed}.pt")
    out = os.path.join(CKPT, f"{arm}_s{seed}.rollout.json")
    if os.path.exists(out):
        return json.load(open(out))
    tasks = [(C.SUITE, t) for t in C.tasks()]
    succ, info = RO.run_rollouts(ck, tasks, n_rollouts, workers=workers)
    per = {t: float(np.mean(v)) for t, v in succ.items()}
    rec = {"arm": arm, "seed": seed, "n_rollouts": n_rollouts,
           "per_task": per, "n_tasks": len(per),
           "success_rate": float(np.mean(list(per.values()))) if per else float("nan"),
           "n_episodes": int(sum(len(v) for v in succ.values())),
           "wall_s": info["wall_s"], "n_errors": info["n_errors"]}
    json.dump(rec, open(out, "w"), indent=1)
    return rec


def run_rollouts_all(worker=0, nworkers=1, n_rollouts=N_ROLLOUTS, procs=14):
    import time
    sel = selections()
    jobs = [(a, s) for a in sel for s in ROLLOUT_SEEDS]
    mine = [j for k, j in enumerate(jobs) if k % nworkers == worker]
    print(f"[r5/roll] worker {worker}/{nworkers}: {len(mine)} of {len(jobs)} policies", flush=True)
    t0 = time.time()
    for k, (arm, seed) in enumerate(mine):
        r = rollout_one(arm, seed, n_rollouts=n_rollouts, workers=procs)
        print(f"[r5/roll] {k+1}/{len(mine)} {arm} s{seed} success={r['success_rate']:.3f} "
              f"({r['n_episodes']} eps, {r['wall_s']:.0f}s, errs={r['n_errors']}) "
              f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    print(f"[r5/roll] worker {worker} DONE ({(time.time()-t0)/60:.1f} m)", flush=True)


if __name__ == "__main__":
    main()
