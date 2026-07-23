"""P4: closed-loop success reliability curve. EVAL ONLY -- no retraining.

Re-evaluates EXISTING checkpoints on C1 (near-floor) and C2 (mid-range, Gate-0 passer) probe
tasks at 50 rollouts/task, storing PER-EPISODE outcomes so that 10 / 30 / 50-episode estimates
are formed by SUBSAMPLING the same rollouts.

N_ROLLOUTS = 50, NOT the 90 the brief asked for. See phase2/PHASE2_DEFECT.md: every LIBERO probe
task has exactly 50 initial states and rollout.py indexes them `ep % 50`, so episodes 50..89
would be bit-identical replays of 0..39 under this deterministic policy/env. 50 is every distinct
episode the instrument can supply. Pooled over a cluster's 3 probe tasks the ladder is
30 / 90 / 150 DISTINCT episodes per success estimate.

Episode indices 0..49 index the env's initial state exactly as Phase 1 did, so the first 10 are
the SAME episodes Phase-1's Stage-G probe ran: the subsampling ladder is nested and directly
comparable to Phase 1.

Parallelism: rollout.run_rollouts spawns one worker PER TASK, so 6 probe tasks would use only 6
of the 64 cores. Here each task's episodes are split into chunks, giving 6 x 2 = 12 independent
rollout jobs per model so the pool can use more workers.

Results -> phase2/runs/P4/<run>/success50.json. Phase-1 run dirs are never written to.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT, RESULTS  # noqa: E402
import rollout as R  # noqa: E402

N_ROLLOUTS = 50                  # = every distinct init state a LIBERO task has (see PHASE2_DEFECT.md)
CHUNK = 25                       # 2 chunks/task -> 12 jobs over the 6 probe tasks
CLUSTERS = ["C1", "C2"]
OUT = os.path.join(ROOT, "phase2/runs/P4")


def probe_tasks():
    man = json.load(open(os.path.join(RESULTS, "corpus_manifest.json")))
    by_c = {c["cluster"]: c for c in man["clusters"]}
    tl, owner = [], {}
    for c in CLUSTERS:
        for t in by_c[c]["probe_tasks"]:
            tl.append((by_c[c]["suite"], t))
            owner[t] = c
    return tl, owner


def rollout_chunked(ckpt, task_list, n, workers, chunk=CHUNK):
    """Same rollout code path as Phase 1 (R._init_worker / R._rollout_task), but the episode
    range of each task is split into chunks so the worker pool can exceed len(task_list)."""
    import multiprocessing as mp
    import dataset
    mean, std = dataset.norm_stats()
    D = dataset.obj_pad_dim()

    jobs = []
    for (s, t) in task_list:
        for lo in range(0, n, chunk):
            jobs.append((s, t, list(range(lo, min(lo + chunk, n))), R.HORIZON))

    ctx = mp.get_context("spawn")
    parts, errs = {}, []
    t0 = time.time()
    with ctx.Pool(min(workers, len(jobs)), initializer=R._init_worker,
                  initargs=(ckpt, D, mean, std)) as pool:
        # imap (ORDERED) so each result can be paired back to the job that produced it:
        # _rollout_task does NOT echo its episode indices, and imap_unordered would make the
        # chunk->episode-range mapping unrecoverable. Ordering the results costs no parallelism.
        for job, r in zip(jobs, pool.imap(R._rollout_task, jobs)):
            _, t, ep_idx, _ = job
            if r["err"]:
                errs.append(r)
                print(f"[P4] ERROR {r['task']}: {r['err'][:200]}", flush=True)
                continue
            assert r["task"] == t and len(r["success"]) == len(ep_idx), (r["task"], t)
            parts.setdefault(t, []).extend(zip(ep_idx, r["success"]))
    succ = {t: [bool(v) for _, v in sorted(vs)] for t, vs in parts.items()}   # ordered by ep index
    return succ, {"wall_s": time.time() - t0, "n_errors": len(errs), "errors": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()

    models = json.load(open(a.models))
    tl, owner = probe_tasks()
    assert len(tl) == 6, tl

    for m in models:
        od = os.path.join(OUT, m["name"])
        os.makedirs(od, exist_ok=True)
        if os.path.exists(os.path.join(od, "p4.marker")):
            print(f"[P4] SKIP {m['name']}", flush=True)
            continue
        t0 = time.time()
        succ, info = rollout_chunked(m["ckpt"], tl, N_ROLLOUTS, a.workers)
        bad = [t for (_, t) in tl if len(succ.get(t, [])) != N_ROLLOUTS]
        if info["n_errors"] or bad:
            print(f"[P4] FAIL {m['name']}: {info['n_errors']} errors, incomplete={bad}", flush=True)
            json.dump({"errors": info["errors"], "incomplete": bad},
                      open(os.path.join(od, "errors.json"), "w"), indent=1)
            continue
        rec = {"name": m["name"], "ckpt": m["ckpt"], "n_rollouts": N_ROLLOUTS,
               "mask_id": m.get("mask_id"), "seed": m.get("seed"),
               "per_task": succ, "cluster_of_task": owner, "wall_s": time.time() - t0}
        tmp = os.path.join(od, "success50.json.tmp")
        json.dump(rec, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(od, "success50.json"))
        open(os.path.join(od, "p4.marker"), "w").write(
            json.dumps({"wall_s": rec["wall_s"]}) + "\n")
        rates = {c: sum(sum(succ[t]) for t in succ if owner[t] == c) / (3 * N_ROLLOUTS)
                 for c in CLUSTERS}
        print(f"[P4] OK {m['name']} {rec['wall_s']:.0f}s "
              f"C1={rates['C1']:.3f} C2={rates['C2']:.3f}", flush=True)


if __name__ == "__main__":
    main()
