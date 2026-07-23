"""P2 step 1: per-task probe sets + a read-only PROC overlay.

Probe rule (preregistered): for each of the 27 tasks of C1/C2/C5, the 5 HIGHEST-indexed demos
-- demo_45..demo_49. Every LIBERO task has 50 demos. These indices are provably disjoint from
  * the 135-demo training corpus (uses only demo_0/1/2),
  * every cluster's 10-demo held-out set (uses only demo_1/2),
  * the P3 quantity ladders (which fill from the LOW demo indices upward).
Assertions below are machine-checked, not asserted in prose.

PROC overlay: Phase-1 extracted all 50 libero_goal demos (Stage-C Q=490) but only demo_0/1/2
for the other suites. The 85 missing feature files (C2, C5) are extracted HERE, into
phase2/data/proc/, and an overlay directory of symlinks to Phase-1's data/proc is built so
that dataset.PROC can point at ONE root containing both. Nothing outside phase2/ is written.
Normalization statistics stay Phase-1's frozen results/norm_stats.npz (never refit).
"""
import json
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import DATA, PROC, RESULTS, ROOT  # noqa: E402

P2 = os.path.join(ROOT, "phase2")
OVERLAY = os.path.join(P2, "data/proc")          # real new npz + symlinks to Phase-1's
PROBE_DEMOS = [f"demo_{i}" for i in range(45, 50)]
CLUSTERS = ["C1", "C2", "C5"]


def hdf5_for(suite, task):
    d = os.path.join(DATA, suite)
    for f in os.listdir(d):
        if f.startswith(task) and f.endswith(".hdf5"):
            return os.path.join(d, f)
    raise FileNotFoundError(f"{suite}/{task}")


def extract_one(job):
    """Replay stored sim states -> (proprio, object, actions) npz. Same code path as
    Phase-1 src/extract.py (LE.replay_states_extract), so features are identical in kind."""
    suite, task, demos = job
    import libero_env as LE
    try:
        outdir = os.path.join(OVERLAY, suite, task)
        os.makedirs(outdir, exist_ok=True)
        todo = [d for d in demos if not os.path.exists(os.path.join(outdir, f"{d}.npz"))]
        if not todo:
            return (suite, task, 0, None)
        env = LE.make_env(LE.get_bddl_path(suite, task), horizon=1000, seed=0)
        n = 0
        with h5py.File(hdf5_for(suite, task), "r") as f:
            avail = set(f["data"].keys())
            missing = [d for d in todo if d not in avail]
            if missing:
                return (suite, task, 0, f"hdf5 lacks {missing} (has {len(avail)} demos)")
            for d in todo:
                states = f["data"][d]["states"][:]
                actions = f["data"][d]["actions"][:].astype(np.float32)
                env.reset()
                Pp, Oo = LE.replay_states_extract(env, states)
                assert Pp.shape[0] == actions.shape[0] == Oo.shape[0]
                # NB: np.savez_compressed APPENDS ".npz" unless the name already ends in it,
                # so the temp name must end in .npz (Phase-1 extract.py does the same).
                tmp = os.path.join(outdir, f"{d}.tmp.npz")
                np.savez_compressed(tmp, proprio=Pp.astype(np.float32),
                                    object=Oo.astype(np.float32), actions=actions)
                os.replace(tmp, os.path.join(outdir, f"{d}.npz"))
                n += 1
        try:
            env.close()
        except Exception:
            pass
        return (suite, task, n, None)
    except Exception:
        import traceback
        return (suite, task, 0, traceback.format_exc()[-500:])


def main():
    man = json.load(open(os.path.join(RESULTS, "corpus_manifest.json")))
    by_c = {c["cluster"]: c for c in man["clusters"]}

    # ---------------------------------------------------------------- disjointness (checked)
    used = set()
    for c in man["clusters"]:
        for d in c["train_demos"]:
            used.add((c["suite"], d["task"], d["demo"]))
        for d in c["heldout_demos"]:
            used.add((c["suite"], d["task"], d["demo"]))
    used_idx = sorted({int(d.split("_")[1]) for (_, _, d) in used})
    print(f"[check] demo indices used by the 135-corpus + all held-out sets: {used_idx}")
    assert max(used_idx) < 45, "corpus reaches demo_45+; the probe rule would collide!"

    probes, jobs = {}, []
    for cn in CLUSTERS:
        c = by_c[cn]
        for t in c["tasks"]:
            key = f"{cn}/{t}"
            ids = [f"{c['suite']}/{t}/{d}" for d in PROBE_DEMOS]
            for d in PROBE_DEMOS:
                assert (c["suite"], t, d) not in used, f"probe collides with corpus: {t}/{d}"
            probes[key] = {"cluster": cn, "suite": c["suite"], "task": t, "demo_ids": ids}
            jobs.append((c["suite"], t, PROBE_DEMOS))
    assert len(probes) == 27, len(probes)
    print(f"[check] 27 tasks x 5 probe demos = {sum(len(v['demo_ids']) for v in probes.values())}, "
          f"all disjoint from the corpus and every held-out set")

    # ---------------------------------------------------------------------------- extraction
    with Pool(16) as p:
        res = p.map(extract_one, jobs)
    errs = [(s, t, e) for (s, t, n, e) in res if e]
    n_new = sum(n for (_, _, n, e) in res)
    print(f"[extract] {n_new} new feature files written under phase2/data/proc; {len(errs)} errors")
    for s, t, e in errs:
        print(f"  ERROR {s}/{t}: {e}")
    if errs:
        sys.exit(2)

    # ------------------------------------------------------------- symlink overlay -> Phase 1
    n_link = 0
    for suite in sorted(os.listdir(PROC)):
        sp = os.path.join(PROC, suite)
        if not os.path.isdir(sp):
            continue
        for task in sorted(os.listdir(sp)):
            tp = os.path.join(sp, task)
            if not os.path.isdir(tp):
                continue
            od = os.path.join(OVERLAY, suite, task)
            os.makedirs(od, exist_ok=True)
            for f in os.listdir(tp):
                if not f.endswith(".npz"):
                    continue
                dst = os.path.join(od, f)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(tp, f), dst)   # link, never copy or move
                    n_link += 1
    print(f"[overlay] {n_link} symlinks to Phase-1 features (Phase-1 data untouched)")

    # ------------------------------------------------------------------- verify loadability
    import dataset
    dataset.PROC = OVERLAY
    bad = []
    for k, v in probes.items():
        for did in v["demo_ids"]:
            try:
                z = dataset.load_raw(did)          # -> dict(state(T,128), actions(T,7), proprio)
                assert z["state"].shape[0] == z["actions"].shape[0] > 0
                assert z["state"].shape[1] == dataset.state_dim() == 128
            except Exception as e:
                bad.append((did, f"{type(e).__name__}: {e}"[:120]))
    print(f"[verify] {135 - len(bad)}/135 probe demos load through the overlay; {len(bad)} bad")
    for b in bad[:5]:
        print("  ", b)
    if bad:
        sys.exit(2)

    out = {
        "rule": "the 5 highest-indexed demos of each task: demo_45..demo_49",
        "n_tasks": 27, "n_demos_per_task": 5,
        "disjoint_from": ["135-demo training corpus", "all 9 held-out sets",
                          "P3 quantity ladders (fill from low indices)"],
        "max_demo_index_used_by_phase1_corpus": max(used_idx),
        "proc_overlay": OVERLAY,
        "norm_stats": "Phase-1 results/norm_stats.npz (frozen, never refit)",
        "probes": probes,
    }
    p = os.path.join(P2, "results/per_task_probes.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
