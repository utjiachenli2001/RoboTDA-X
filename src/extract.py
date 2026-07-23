"""Replay-extract per-demo state caches (proprio + raw object-state + actions).

LIBERO hdf5 files do NOT store `object-state` (only proprio + images), so the spec's
observation must be reconstructed by replaying each demo's stored full sim state through the
env and reading the env observation dict. This also guarantees the offline (training) and
online (rollout) featurizations come from the identical code path.

Cache: data/proc/<suite>/<task>/<demo_id>.npz  keys: proprio(T,16), object(T,dobj), actions(T,7)
Resumable: skips demos whose npz already exists. Parallel over tasks (one env per task).
"""
import os
import sys
import json
import argparse
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import DATA, PROC, RESULTS, write_done, is_done
import libero_env as LE
from clusters import hdf5_filename, suite_task_names


def npz_path(suite, task, demo_id):
    return os.path.join(PROC, suite, task, f"{demo_id}.npz")


def extract_task(job):
    """job = (suite, task, demo_ids or None). Returns (suite, task, n_new, err)."""
    suite, task, demo_ids = job
    try:
        h5 = os.path.join(DATA, suite, hdf5_filename(task))
        if not os.path.exists(h5):
            return (suite, task, 0, f"missing hdf5 {h5}")
        os.makedirs(os.path.join(PROC, suite, task), exist_ok=True)
        with h5py.File(h5, "r") as f:
            data = f["data"]
            all_ids = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
            ids = demo_ids if demo_ids is not None else all_ids
            todo = [d for d in ids if not os.path.exists(npz_path(suite, task, d))]
            if not todo:
                return (suite, task, 0, None)
            env = LE.make_env(LE.get_bddl_path(suite, task), horizon=1000, seed=0)
            n = 0
            for d in todo:
                states = data[d]["states"][:]
                actions = data[d]["actions"][:].astype(np.float32)
                env.reset()
                P, O = LE.replay_states_extract(env, states)
                assert P.shape[0] == actions.shape[0] == O.shape[0], \
                    f"len mismatch P{P.shape} A{actions.shape} O{O.shape}"
                tmp = npz_path(suite, task, d) + ".tmp.npz"
                np.savez_compressed(tmp, proprio=P.astype(np.float32),
                                    object=O.astype(np.float32), actions=actions)
                os.replace(tmp, npz_path(suite, task, d))
                n += 1
            try:
                env.close()
            except Exception:
                pass
        return (suite, task, n, None)
    except Exception:
        import traceback
        return (suite, task, 0, traceback.format_exc()[-800:])


def build_jobs(include_full_goal=True):
    man = json.load(open(os.path.join(RESULTS, "corpus_manifest.json")))
    need = {}
    for cm in man["clusters"]:
        suite = cm["suite"]
        for grp in ("train_demos", "heldout_demos"):
            for dd in cm[grp]:
                need.setdefault((suite, dd["task"]), set()).add(dd["demo"])
    if include_full_goal:   # Stage C quantity-sweep reserve: all 50 demos of each goal task
        for t in suite_task_names("libero_goal"):
            need[("libero_goal", t)] = None
    jobs = []
    for (suite, task) in sorted(need.keys()):
        ids = need[(suite, task)]
        jobs.append((suite, task, sorted(ids) if ids is not None else None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(PROC, exist_ok=True)
    if is_done(PROC):
        print("[extract] already done (marker present)")
        return
    jobs = build_jobs()
    print(f"[extract] {len(jobs)} task-jobs, {args.workers} workers", flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    total, errs = 0, []
    with ctx.Pool(args.workers) as pool:
        for i, (suite, task, n, err) in enumerate(pool.imap_unordered(extract_task, jobs)):
            total += n
            if err:
                errs.append((suite, task, err))
                print(f"[extract {i+1}/{len(jobs)}] ERROR {suite}/{task}: {err[:200]}", flush=True)
            else:
                print(f"[extract {i+1}/{len(jobs)}] {suite}/{task[:45]} +{n}", flush=True)
    print(f"[extract] new={total} errors={len(errs)}", flush=True)
    if errs:
        json.dump([{"suite": s, "task": t, "err": e} for s, t, e in errs],
                  open(os.path.join(PROC, "extract_errors.json"), "w"), indent=1)
        print("[extract] FAILED — see extract_errors.json")
        sys.exit(1)
    write_done(PROC, f"extract new={total} jobs={len(jobs)}\n")
    print("[extract] DONE", flush=True)


if __name__ == "__main__":
    main()
