"""RQ2 moderators -- TRAJECTORY-SPACE similarity only (spec §7). NO image/DINO features anywhere.

  DTW      : dynamic time warping over the ee trajectory (eef position 3-D + gripper width 1-D),
             z-scaled by the training pool's stats so the gripper channel is commensurate.
  MMD      : maximum mean discrepancy (RBF kernel, median heuristic bandwidth) between two
             demos' object-state frame distributions.
  BDDL     : Jaccard overlap of the bddl object-category sets of the demos' tasks.

Outputs results/similarity.npz (135x135 demo-level matrices) and the cluster-level aggregates
used by the moderator regression and by Stage I's similarity-top-15 condition.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RESULTS
import dataset

OUT = os.path.join(RESULTS, "similarity.npz")


def ee_traj(demo_id):
    """(T,4): eef position (3) + gripper width (1), scaled by train-pool std."""
    import phases
    P = dataset.load_raw(demo_id)["proprio"]
    w = phases.gripper_width(P)[:, None]
    X = np.concatenate([P[:, 0:3], w], 1).astype(np.float64)
    return X


def dtw_dist(A, B, radius=10):
    from fastdtw import fastdtw
    d, _ = fastdtw(A, B, radius=radius, dist=lambda x, y: float(np.linalg.norm(x - y)))
    return float(d) / (len(A) + len(B))     # length-normalized


def mmd_rbf(X, Y, gamma=None, cap=200, rng=None):
    """MMD^2 with an RBF kernel between two frame sets (subsampled to `cap` frames)."""
    rng = rng or np.random.default_rng(0)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    if len(Y) > cap:
        Y = Y[rng.choice(len(Y), cap, replace=False)]
    Z = np.concatenate([X, Y], 0)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    if gamma is None:
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-8)
    n, m = len(X), len(Y)
    K = np.exp(-gamma * d2)
    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def bddl_objects(task):
    meta = json.load(open(os.path.join(RESULTS, "task_meta.json")))
    return set(meta[task]["objects"])


def build(force=False):
    if os.path.exists(OUT) and not force:
        return np.load(OUT, allow_pickle=True)
    ids, by_c = dataset.train_pool()
    N = len(ids)
    meta = json.load(open(os.path.join(RESULTS, "task_meta.json")))

    trajs = [ee_traj(d) for d in ids]
    objs = [dataset.load_raw(d)["state"][:, 16:] for d in ids]   # object-state block
    tasks = [dataset.parse_did(d)[1] for d in ids]

    print(f"[moderators] DTW over {N}x{N} demo pairs...", flush=True)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = dtw_dist(trajs[i], trajs[j])
        if i % 20 == 0:
            print(f"  DTW row {i}/{N}", flush=True)

    print("[moderators] object-state MMD...", flush=True)
    M = np.zeros((N, N))
    rng = np.random.default_rng(0)
    for i in range(N):
        for j in range(i + 1, N):
            M[i, j] = M[j, i] = mmd_rbf(objs[i], objs[j], rng=rng)

    print("[moderators] bddl object overlap...", flush=True)
    B = np.zeros((N, N))
    osets = [set(meta[t]["objects"]) for t in tasks]
    for i in range(N):
        for j in range(N):
            u = osets[i] | osets[j]
            B[i, j] = len(osets[i] & osets[j]) / len(u) if u else 0.0

    np.savez(OUT, demo_ids=np.array(ids), dtw=D, mmd=M, bddl=B)
    print(f"[moderators] wrote {OUT}")
    return np.load(OUT, allow_pickle=True)


def cluster_matrices():
    """Cluster-level means of each demo-level similarity matrix -> (9,9) dicts."""
    z = build()
    ids = [str(x) for x in z["demo_ids"]]
    _, by_c = dataset.train_pool()
    cl = dataset.clusters()
    idx = {c: [ids.index(d) for d in by_c[c]] for c in cl}
    out = {}
    for key in ("dtw", "mmd", "bddl"):
        A = z[key]
        Mx = np.zeros((9, 9))
        for i, ci in enumerate(cl):
            for j, cj in enumerate(cl):
                sub = A[np.ix_(idx[ci], idx[cj])]
                if ci == cj:                       # exclude the diagonal (self-pairs)
                    m = ~np.eye(len(idx[ci]), dtype=bool)
                    Mx[i, j] = float(sub[m].mean())
                else:
                    Mx[i, j] = float(sub.mean())
        out[key] = Mx
    return out, cl


def within_target_redundancy():
    """mean pairwise in-target DTW per cluster (the redundancy regressor)."""
    M, cl = cluster_matrices()
    return {c: float(M["dtw"][i, i]) for i, c in enumerate(cl)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    z = build(force=a.force)
    M, cl = cluster_matrices()
    print("\ncluster-mean DTW (rows=from, cols=to):")
    print("      " + " ".join(f"{c:>6}" for c in cl))
    for i, c in enumerate(cl):
        print(f"{c:>5} " + " ".join(f"{M['dtw'][i,j]:6.3f}" for j in range(9)))
    print("\nwithin-target redundancy (mean pairwise in-target DTW):")
    for c, v in within_target_redundancy().items():
        print(f"  {c}: {v:.4f}")
