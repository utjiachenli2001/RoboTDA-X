"""R5b -- a powered replication of the selection-vs-random comparison. Prereg: p21_prereg_r5b.md.

R5 stands as scored. R5b does not re-score it: 20 random replicates instead of 10, 5 rolled-out
seeds instead of 3, 400 episodes instead of 100, and a DIFFERENCE-OF-MEANS test instead of a
percentile band -- because DM-top is one fixed selection, not a draw from the random distribution,
so asking whether it is an OUTLIER among random draws answers a question nobody posed.

The first 10 random replicates are BYTE-IDENTICAL to R5's: `selections()` consumes one RNG stream
in order, so extending the loop from 10 to 20 appends without disturbing the prefix. Asserted in
`verify_prefix()` rather than assumed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import p18_corpus as C  # noqa: E402
from if_repair import p20_r5 as R5  # noqa: E402

RESULTS = R5.RESULTS
CKPT = R5.CKPT
OUT = os.path.join(RESULTS, "confirm_r5b.csv")

N_RANDOM_B = 20
SEEDS = R5.SEEDS                      # all 5 seeds are rolled out in R5b
N_ROLLOUTS_B = 40                     # per task, x10 tasks = 400 episodes per policy
ALPHA = 0.025
N_BOOT = 20000


def selections_b():
    dm = R5.datamodel_per_demo()
    gd = R5.GR.load(R5.POOL, "H1")
    ids = list(C.rung_demos(R5.POOL))
    arms = {}
    arms["DM-top"] = R5._cover(sorted(ids, key=lambda d: dm[d], reverse=True)[:R5.BUDGET], dm)
    arms["DM-bottom"] = R5._cover(sorted(ids, key=lambda d: dm[d])[:R5.BUDGET],
                                  {d: -v for d, v in dm.items()})
    arms["GD-top"] = R5._cover(sorted(ids, key=lambda d: gd[d], reverse=True)[:R5.BUDGET], gd)
    rng = np.random.default_rng(R5.RANDOM_SEED)
    for i in range(N_RANDOM_B):
        r = [ids[j] for j in rng.permutation(len(ids))[:R5.BUDGET]]
        arms[f"RANDOM-{i:02d}"] = R5._cover(r, {d: 0.0 for d in ids})
    return {k: (sorted(v[0]), v[1]) for k, v in arms.items()}


def verify_prefix():
    """R5b's first 10 random replicates must equal R5's exactly, or the two are not comparable."""
    a, b = R5.selections(), selections_b()
    bad = [k for k in a if a[k][0] != b[k][0]]
    return bad


def train(worker=0, nworkers=1):
    import time
    import torch
    from if_repair import p18_eval as EVT
    from if_repair import retrain as R
    import dataset
    import train as TR
    cfg = TR.load_cfg()
    sel = selections_b()
    jobs = [(a, s) for a in sel for s in SEEDS]
    mine = [j for k, j in enumerate(jobs) if k % nworkers == worker]
    fidx = EVT.frame_index()
    t0 = time.time()
    todo = [(a, s) for a, s in mine
            if not os.path.exists(os.path.join(CKPT, f"{a}_s{s}.json"))]
    print(f"[r5b] worker {worker}/{nworkers}: {len(todo)} new of {len(mine)}", flush=True)
    for arm, seed in todo:
        demos, _ = sel[arm]
        model, meta = R.train_one(demos, seed, seed, cfg)
        l2, nll = EVT.heldout_frame_losses(model)
        agg = EVT.aggregate(l2, nll, fidx)
        torch.save({"state_dim": dataset.state_dim(), "cfg": cfg,
                    "model": model.state_dict()}, os.path.join(CKPT, f"{arm}_s{seed}.pt"))
        json.dump({"arm": arm, "seed": seed, "n_demos": len(demos),
                   "plain_loss": agg["plain_loss"], "final_train_loss": meta["final_loss"],
                   "demos": demos}, open(os.path.join(CKPT, f"{arm}_s{seed}.json"), "w"))
        del model
        torch.cuda.empty_cache()
        print(f"[r5b] {arm} s{seed} loss={agg['plain_loss']:.5f} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"[r5b] worker {worker} train DONE ({(time.time()-t0)/60:.1f} m)", flush=True)


def rollouts(worker=0, nworkers=1, procs=10):
    """400 episodes per policy, written to *.rollout400.json so R5's 100-episode records survive."""
    import time
    import rollout as RO
    sel = selections_b()
    jobs = [(a, s) for a in sel for s in SEEDS]
    mine = [j for k, j in enumerate(jobs) if k % nworkers == worker]
    tasks = [(C.SUITE, t) for t in C.tasks()]
    print(f"[r5b/roll] worker {worker}/{nworkers}: {len(mine)} of {len(jobs)} policies", flush=True)
    t0 = time.time()
    for k, (arm, seed) in enumerate(mine):
        out = os.path.join(CKPT, f"{arm}_s{seed}.rollout400.json")
        if os.path.exists(out):
            continue
        succ, info = RO.run_rollouts(os.path.join(CKPT, f"{arm}_s{seed}.pt"), tasks,
                                     N_ROLLOUTS_B, workers=procs)
        per = {t: float(np.mean(v)) for t, v in succ.items()}
        rec = {"arm": arm, "seed": seed, "per_task": per, "n_tasks": len(per),
               "success_rate": float(np.mean(list(per.values()))) if per else float("nan"),
               "n_episodes": int(sum(len(v) for v in succ.values())),
               "wall_s": info["wall_s"], "n_errors": info["n_errors"]}
        json.dump(rec, open(out, "w"), indent=1)
        print(f"[r5b/roll] {k+1}/{len(mine)} {arm} s{seed} success={rec['success_rate']:.3f} "
              f"({rec['n_episodes']} eps, errs={rec['n_errors']}, "
              f"{(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"[r5b/roll] worker {worker} DONE ({(time.time()-t0)/60:.1f} m)", flush=True)


# --------------------------------------------------------------------------- scoring
def load():
    loss, succ = {}, {}
    for f in sorted(glob.glob(os.path.join(CKPT, "*.rollout400.json"))):
        d = json.load(open(f))
        succ.setdefault(d["arm"], {})[d["seed"]] = d["success_rate"]
    for f in sorted(glob.glob(os.path.join(CKPT, "*.json"))):
        if ".rollout" in f:
            continue
        d = json.load(open(f))
        loss.setdefault(d["arm"], {})[d["seed"]] = d["plain_loss"]
    return loss, succ


def hierarchical_bootstrap(tbl, arm, rnd_arms, n_boot=N_BOOT, seed=0):
    """CI on mean(arm) - mean(RANDOM), resampling replicates AND seeds within arm."""
    rng = np.random.default_rng(seed)
    a_seeds = np.array(list(tbl[arm].values()))
    R_ = [np.array(list(tbl[r].values())) for r in rnd_arms if r in tbl]
    out = np.empty(n_boot)
    for b in range(n_boot):
        a = rng.choice(a_seeds, size=len(a_seeds), replace=True).mean()
        idx = rng.integers(0, len(R_), len(R_))
        r = np.mean([rng.choice(R_[i], size=len(R_[i]), replace=True).mean() for i in idx])
        out[b] = a - r
    return float(out.mean()), np.percentile(out, [100 * ALPHA, 100 * (1 - ALPHA)])


def score():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists -- R5b is SCORED ONCE.")
    loss, succ = load()
    rnd = sorted(a for a in succ if a.startswith("RANDOM"))
    res = {"n_random_replicates": len(rnd), "alpha": ALPHA, "arms": {}}
    for name, tbl, better in (("success", succ, "higher"), ("loss", loss, "lower")):
        rr = sorted(a for a in tbl if a.startswith("RANDOM"))
        base = float(np.mean([np.mean(list(tbl[a].values())) for a in rr]))
        print(f"\n=== {name} ({better} is better) ===")
        print(f"  RANDOM: {len(rr)} replicates x {len(tbl[rr[0]])} seeds, mean={base:.4f}")
        for arm in ("DM-top", "GD-top", "DM-bottom"):
            if arm not in tbl:
                continue
            d, (lo, hi) = hierarchical_bootstrap(tbl, arm, rr)
            v = float(np.mean(list(tbl[arm].values())))
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
            print(f"  {arm:10s} mean={v:.4f}  diff={d:+.4f} "
                  f"CI[{lo:+.4f}, {hi:+.4f}]  {sig}")
            res["arms"].setdefault(arm, {})[name] = {
                "mean": v, "diff_vs_random": d, "ci": [float(lo), float(hi)],
                "significant": bool(lo > 0 or hi < 0), "random_mean": base}
    json.dump(res, open(OUT.replace(".csv", ".json"), "w"), indent=1)
    with open(OUT, "w") as fh:
        fh.write("arm,outcome,mean,random_mean,diff,ci_lo,ci_hi,significant\n")
        for arm, d in res["arms"].items():
            for out_name, v in d.items():
                fh.write(f"{arm},{out_name},{v['mean']:.6f},{v['random_mean']:.6f},"
                         f"{v['diff_vs_random']:.6f},{v['ci'][0]:.6f},{v['ci'][1]:.6f},"
                         f"{v['significant']}\n")
    print(f"\n[r5b] wrote {OUT}")


def main():
    ap = argparse.ArgumentParser()
    for f in ("verify", "train", "rollouts", "score"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    if a.verify:
        bad = verify_prefix()
        sel = selections_b()
        print(f"[r5b] {len(sel)} arms x {len(SEEDS)} seeds = {len(sel)*len(SEEDS)} policies, "
              f"{N_ROLLOUTS_B*10} episodes each = {len(sel)*len(SEEDS)*N_ROLLOUTS_B*10} episodes")
        print(f"[r5b] arms differing from R5's: {bad if bad else 'NONE (prefix identical)'}")
    if a.train:
        train(a.worker, a.nworkers)
    if a.rollouts:
        rollouts(a.worker, a.nworkers, a.procs)
    if a.score:
        score()


if __name__ == "__main__":
    main()
