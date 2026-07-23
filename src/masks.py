"""Mask construction for the ground-truth corpora (Stage F cluster-grain, Stage G demo-grain).

Stage F (cluster grain), spec §6:
    K=72 masks, each exactly 5 of the 9 clusters (=> exactly 75 demos).
    Balanced: every cluster in exactly 40 masks (72*5/9 = 40).
    Pairwise co-inclusion constrained to [17,23] (design expectation = 72*(5/9)*(4/8) = 20).
    Randomized construction + swap repair, fixed seed. Emits results/mask_manifest.json
    with per-cluster inclusion counts and the 9x9 co-inclusion matrix as an audit.

Stage G (demo grain), spec §6:
    K=24 masks, each exactly 68 of the 135 demos, stratified within cluster:
    per mask, 5 clusters contribute 8 demos and 4 contribute 7 (5*8 + 4*7 = 68).
    Rotate which clusters give 8; balance so every demo appears in 11-13 masks.
    Emits results/demo_mask_manifest.json.

A swap that exchanges cluster x (in mask A, not B) with cluster y (in mask B, not A)
preserves BOTH masks' sizes and BOTH clusters' inclusion counts -- so the balance
constraints are invariant under repair, and only co-inclusion is optimized.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RESULTS
import dataset

CLUSTER_MASK_SEED = 7
DEMO_MASK_SEED = 11
NOISE_CEIL_SEED = 13

K_CLUSTER = 72
CLUSTERS_PER_MASK = 5
INCLUSIONS_PER_CLUSTER = 40      # 72*5/9
COINC_LO, COINC_HI = 17, 23

K_DEMO = 24
DEMOS_PER_MASK = 68
N_NOISE_CEIL_MASKS = 12


# ------------------------------------------------------------------ Stage F
def coinc(M):
    """M: (K,9) binary -> 9x9 co-inclusion counts."""
    return M.T @ M


def build_cluster_masks(seed=CLUSTER_MASK_SEED, K=K_CLUSTER, C=9, per=CLUSTERS_PER_MASK,
                        max_iter=200000):
    rng = np.random.default_rng(seed)
    # --- init: exact column balance by greedy "largest remaining capacity" fill.
    # Feasible by Gale-Ryser: filling each mask with the `per` clusters that still have the
    # most remaining inclusions never strands capacity.
    M = np.zeros((K, C), dtype=np.int64)
    rem = np.full(C, INCLUSIONS_PER_CLUSTER, dtype=int)       # 9 x 40 = 360 = 72 x 5
    for k in range(K):
        order = np.lexsort((rng.random(C), -rem))             # by remaining desc, random ties
        pick = order[:per]
        assert rem[pick].min() > 0, "infeasible fill"
        M[k, pick] = 1
        rem[pick] -= 1
    assert (M.sum(1) == per).all(), f"mask sizes wrong: {set(M.sum(1))}"
    assert (M.sum(0) == INCLUSIONS_PER_CLUSTER).all(), f"cluster balance wrong: {M.sum(0)}"

    # --- swap repair: minimize deviation of off-diagonal co-inclusion from target
    target = K * (per / C) * ((per - 1) / (C - 1))     # = 20.0
    off = ~np.eye(C, dtype=bool)

    def cost(M):
        X = coinc(M).astype(float)
        d = X[off] - target
        return float((d ** 2).sum()), X

    cur, X = cost(M)
    for it in range(max_iter):
        lo, hi = coinc(M)[off].min(), coinc(M)[off].max()
        if lo >= COINC_LO and hi <= COINC_HI:
            break
        a, b = rng.integers(0, K, 2)
        if a == b:
            continue
        xs = np.nonzero(M[a] & ~M[b])[0]     # in a, not in b
        ys = np.nonzero(M[b] & ~M[a])[0]     # in b, not in a
        if len(xs) == 0 or len(ys) == 0:
            continue
        x = xs[rng.integers(len(xs))]
        y = ys[rng.integers(len(ys))]
        M[a, x], M[a, y] = 0, 1              # swap preserves row sums AND column sums
        M[b, y], M[b, x] = 0, 1
        new, _ = cost(M)
        if new <= cur:
            cur = new
        else:                                 # revert
            M[a, x], M[a, y] = 1, 0
            M[b, y], M[b, x] = 1, 0
    X = coinc(M)
    return M, X, it


def cluster_mask_manifest(force=False):
    path = os.path.join(RESULTS, "mask_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    M, X, iters = build_cluster_masks()
    off = ~np.eye(9, dtype=bool)
    lo, hi = int(X[off].min()), int(X[off].max())
    masks = []
    for k in range(K_CLUSTER):
        inc = [cl[j] for j in np.nonzero(M[k])[0]]
        demos = [d for c in inc for d in by_c[c]]
        assert len(demos) == 75, f"mask {k} has {len(demos)} demos"
        masks.append({"mask_id": f"F{k:03d}", "clusters": inc, "n_demos": len(demos),
                      "demos": demos})
    # noise-ceiling replicate subset: 12 masks, stratified so each cluster appears in 6-8
    rng = np.random.default_rng(NOISE_CEIL_SEED)
    best, best_score = None, None
    for _ in range(20000):
        sel = rng.choice(K_CLUSTER, N_NOISE_CEIL_MASKS, replace=False)
        cnt = M[sel].sum(0)
        if cnt.min() >= 6 and cnt.max() <= 8:
            best = sorted(sel.tolist())
            break
    assert best is not None, "could not find a stratified noise-ceiling subset"
    out = {
        "seed": CLUSTER_MASK_SEED, "K": K_CLUSTER, "clusters_per_mask": CLUSTERS_PER_MASK,
        "demos_per_mask": 75, "swap_iters": int(iters),
        "inclusion_counts": {cl[j]: int(M[:, j].sum()) for j in range(9)},
        "coinclusion_matrix": X.tolist(),
        "coinclusion_offdiag_min": lo, "coinclusion_offdiag_max": hi,
        "coinclusion_ok": bool(lo >= COINC_LO and hi <= COINC_HI),
        "noise_ceiling_masks": [f"F{k:03d}" for k in best],
        "noise_ceiling_seed": NOISE_CEIL_SEED,
        "noise_ceiling_cluster_counts": {cl[j]: int(M[best][:, j].sum()) for j in range(9)},
        "masks": masks,
    }
    json.dump(out, open(path, "w"), indent=1)
    return out


# ------------------------------------------------------------------ Stage G
def build_demo_masks(seed=DEMO_MASK_SEED):
    """24 masks x 68 demos, within-cluster stratified 8/8/8/8/8/7/7/7/7 (rotating)."""
    rng = np.random.default_rng(seed)
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    C, K = len(cl), K_DEMO

    # which clusters give 8 in each mask: each mask has exactly 5 eights;
    # over 24 masks each cluster is an "eight" 13 or 14 times (6x13 + 3x14 = 120 = 24*5).
    n8 = np.array([14, 14, 14, 13, 13, 13, 13, 13, 13])
    rng.shuffle(n8)
    E = np.zeros((K, C), dtype=int)
    rem = n8.copy()
    for k in range(K):
        order = np.argsort(-(rem + rng.random(C) * 1e-6))   # greedy by remaining need
        pick = order[:5]
        E[k, pick] = 1
        rem[pick] -= 1
    assert (E.sum(1) == 5).all() and (E.sum(0) == n8).all()

    # per cluster: choose which demos, balancing per-demo inclusion counts
    masks = [[] for _ in range(K)]
    per_demo_counts = {}
    for j, c in enumerate(cl):
        demos = by_c[c]                                  # 15 demos
        cnt = np.zeros(len(demos), dtype=int)
        for k in range(K):
            take = 8 if E[k, j] else 7
            # pick the `take` demos with the smallest inclusion count so far (ties: random, seeded)
            order = np.lexsort((rng.random(len(demos)), cnt))
            sel = order[:take]
            cnt[sel] += 1
            masks[k].extend([demos[i] for i in sel])
        per_demo_counts.update({demos[i]: int(cnt[i]) for i in range(len(demos))})
    for k in range(K):
        assert len(masks[k]) == DEMOS_PER_MASK, f"mask {k}: {len(masks[k])} demos"
    return masks, per_demo_counts, E, cl


def demo_mask_manifest(force=False):
    path = os.path.join(RESULTS, "demo_mask_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    masks, cnts, E, cl = build_demo_masks()
    _, by_c = dataset.train_pool()
    lo, hi = min(cnts.values()), max(cnts.values())
    out = {
        "seed": DEMO_MASK_SEED, "K": K_DEMO, "demos_per_mask": DEMOS_PER_MASK,
        "per_demo_inclusion_min": lo, "per_demo_inclusion_max": hi,
        "per_demo_inclusion_ok": bool(lo >= 11 and hi <= 13),
        "per_demo_counts": cnts,
        "masks": [{"mask_id": f"G{k:03d}", "n_demos": len(m), "demos": m,
                   "in_target_count": {c: sum(1 for d in m if d in set(by_c[c]))
                                       for c in cl}}
                  for k, m in enumerate(masks)],
    }
    json.dump(out, open(path, "w"), indent=1)
    return out


if __name__ == "__main__":
    f = cluster_mask_manifest(force=True)
    X = np.array(f["coinclusion_matrix"])
    print(f"[F] K={f['K']} inclusion counts={set(f['inclusion_counts'].values())} "
          f"coinc off-diag [{f['coinclusion_offdiag_min']},{f['coinclusion_offdiag_max']}] "
          f"ok={f['coinclusion_ok']} (swap iters={f['swap_iters']})")
    print("    noise-ceiling masks:", f["noise_ceiling_masks"])
    print("    nc cluster counts:", f["noise_ceiling_cluster_counts"])
    g = demo_mask_manifest(force=True)
    print(f"[G] K={g['K']} demos/mask={g['demos_per_mask']} "
          f"per-demo inclusion [{g['per_demo_inclusion_min']},{g['per_demo_inclusion_max']}] "
          f"ok={g['per_demo_inclusion_ok']}")
    itc = [m["in_target_count"] for m in g["masks"]]
    print("    in-target counts per mask (C1):", sorted(set(x["C1"] for x in itc)))
