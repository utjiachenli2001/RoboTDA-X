"""U0 -- measure the retrain cost model against training-set SIZE.

The whole ~6,000-retrain budget for the corpus-size ladder rests on one assumption:
`train_one` runs a FIXED `total_steps` (8000) at a fixed batch size, so per-retrain cost
should be independent of how many demos are in the training set. If that is wrong -- if
cost scales with N -- the top rung dominates and the ladder must be re-sized before any
GPU is committed.

This times the two phases separately, because they have different expected behaviour:

  build+train : Bank() construction is O(N) (loads N .npz files, builds windows, moves to
                GPU); the training loop itself is O(total_steps) and should be flat in N.
  eval        : the held-out bank is fixed, so this should be flat in N by construction.
                Timed anyway -- an assumption asserted is an assumption untested.

Writes JSON to stdout and to --out. No repo state is touched; nothing is cached.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from if_repair import data as D  # noqa: E402

D.add_repo_paths()

import torch  # noqa: E402

import dataset  # noqa: E402
import train as TR  # noqa: E402
from bootstrap import PROC  # noqa: E402
from if_repair import retrain as RT  # noqa: E402


def free_goal_pool():
    """libero_goal demo ids NOT touched by the old campaign (its 15 train + 10 heldout).

    475 of the 500. Returned sorted by (task, demo) so the selection is deterministic.
    """
    used = set()
    for c in dataset.manifest()["clusters"]:
        if c["cluster"] != "C1":
            continue
        for key in ("train_demos", "heldout_demos"):
            for d in c[key]:
                used.add(dataset.did(c["suite"], d["task"], d["demo"]))
    root = os.path.join(PROC, "libero_goal")
    out = []
    for task in sorted(os.listdir(root)):
        for f in sorted(os.listdir(os.path.join(root, task))):
            if not f.endswith(".npz"):
                continue
            i = dataset.did("libero_goal", task, f[:-4])
            if i not in used:
                out.append(i)
    return out


def time_one(demo_ids, cfg, seed=90001):
    """-> (train_seconds, eval_seconds, meta). Synchronised so the numbers mean something."""
    torch.cuda.synchronize()
    t0 = time.time()
    model, meta = RT.train_one(demo_ids, seed, seed, cfg)
    torch.cuda.synchronize()
    t1 = time.time()
    l2, nll = RT.heldout_frame_losses(model)
    torch.cuda.synchronize()
    t2 = time.time()
    del model
    torch.cuda.empty_cache()
    return t1 - t0, t2 - t1, meta, int(len(l2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[27, 55, 110, 258, 470],
                    help="training-set sizes to time (the ladder's actual retained sizes)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None, help="override total_steps (smoke only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
        raise SystemExit("refusing to run: export CUDA_VISIBLE_DEVICES=0 first "
                         "(bootstrap.py pins ALLOWED_GPUS=(4,5,6,7), which is wrong on this box "
                         "and silently falls back to CPU)")
    if not torch.cuda.is_available():
        raise SystemExit("refusing to run: torch.cuda.is_available() is False")

    cfg = TR.load_cfg()
    if a.steps:
        cfg["total_steps"] = a.steps

    pool = free_goal_pool()
    print(f"[u0] free libero_goal pool: {len(pool)} demos", flush=True)
    print(f"[u0] device: {torch.cuda.get_device_name(0)}  total_steps={cfg['total_steps']}",
          flush=True)

    # Warmup. The first retrain in a process pays CUDA context creation, cuBLAS/cuDNN
    # autotuning and allocator growth -- in the smoke run that made a 470-demo retrain look
    # CHEAPER than a 27-demo one, which is the exact wrong conclusion. Burn it first.
    wcfg = dict(cfg)
    wcfg["total_steps"] = 200
    t_warm = time_one(pool[:32], wcfg, seed=1)[0]
    print(f"[u0] warmup (32 demos, 200 steps): {t_warm:.1f}s -- discarded", flush=True)

    rows = []
    for n in a.sizes:
        if n > len(pool):
            print(f"[u0] SKIP n={n}: exceeds free pool ({len(pool)})", flush=True)
            continue
        for r in range(a.repeats):
            # deterministic, contiguous slice -- we are timing cost, not measuring science,
            # so the demo identity does not matter as long as the count does.
            ids = pool[:n]
            tr_s, ev_s, meta, n_eval_frames = time_one(ids, cfg, seed=90001 + r)
            row = {"n_demos": n, "repeat": r, "train_s": round(tr_s, 2),
                   "eval_s": round(ev_s, 2), "total_s": round(tr_s + ev_s, 2),
                   "n_windows": meta["n_windows"], "steps": meta["steps"],
                   "final_loss": meta["final_loss"], "n_eval_frames": n_eval_frames}
            rows.append(row)
            print(f"[u0] n={n:4d} windows={meta['n_windows']:7d} "
                  f"train={tr_s:7.1f}s eval={ev_s:6.1f}s total={tr_s + ev_s:7.1f}s "
                  f"loss={meta['final_loss']:.4f}", flush=True)

    # the number that replaces the ~88 retrains/hour assumption
    for row in rows:
        row["retrains_per_hour_1gpu"] = round(3600.0 / row["total_s"], 1)

    payload = {"device": torch.cuda.get_device_name(0),
               "total_steps": cfg["total_steps"],
               "batch_size": cfg["batch_size"],
               "free_pool_size": len(pool),
               "rows": rows}
    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        payload["scaling"] = {
            "small_n": lo["n_demos"], "small_total_s": lo["total_s"],
            "large_n": hi["n_demos"], "large_total_s": hi["total_s"],
            "size_ratio": round(hi["n_demos"] / lo["n_demos"], 2),
            "cost_ratio": round(hi["total_s"] / lo["total_s"], 3),
        }
    js = json.dumps(payload, indent=2)
    print(js, flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(js + "\n")
        print(f"[u0] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
