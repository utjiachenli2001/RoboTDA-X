"""P14 -- the FRESH demo-mask corpus. 16 masks no analysis has ever touched.

Phase-3 P7 found C2 and C9 clear half-ceiling at some grains -- but DESCRIPTIVELY, on the 24
Stage-G masks, which had already been used for exploration. P14 confirms or kills that on ground
truth that has never been looked at.

CONSTRUCTION (preregistered): the SAME rule as src/masks.py:build_demo_masks -- 68 of the 135
demos per mask, within-cluster stratified (5 clusters contribute 8 demos, 4 contribute 7:
5*8 + 4*7 = 68), per-demo inclusion balanced -- with K = 16 and a NEW seed, 1104.

  * fixed alpha 68/135: identical to Stage G, so the mask corpus is exchangeable with it and the
    P11 champion (fit on nothing, frozen) applies unchanged.
  * K = 16: each mask has exactly 5 "eights", so over 16 masks there are 80 eights across the
    9 clusters -> eight clusters are an "eight" 9 times and one is 8 times (9*8 + 8 = 80).
  * seed 1104 is NEW: Phase 1 used DEMO_MASK_SEED = 11 for the 24 Stage-G masks; P8a used
    rng(909) to SELECT 12 of those 24. 1104 has never been used in this study.

MANDATORY CHECK (preregistered): ZERO of the 16 new masks may coincide, as a SET of demo ids,
with ANY of the 24 Stage-G masks. If one does, the seed is incremented and the check re-run; the
number of increments is reported. (Coincidence is astronomically unlikely -- C(15,8)^5 * C(15,7)^4
possibilities -- but "unlikely" is not "checked".)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402

MASK_SEED = 1104          # PREREGISTERED
K = 16                    # PREREGISTERED
DEMOS_PER_MASK = 68       # PREREGISTERED (identical to Stage G)


def build(seed=MASK_SEED, K=K):
    """The src/masks.py:build_demo_masks rule, at K=16 with a fresh seed."""
    rng = np.random.default_rng(seed)
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    C = len(cl)

    # how many times each cluster is an "eight": 5 per mask x K masks, spread over C clusters
    total8 = 5 * K
    base, extra = divmod(total8, C)                      # K=16 -> 80 = 9*8 + 8 -> base=8, extra=8
    n8 = np.array([base + 1] * extra + [base] * (C - extra))
    assert n8.sum() == total8, (n8.sum(), total8)
    rng.shuffle(n8)

    E = np.zeros((K, C), dtype=int)                      # E[k,j] = 1 iff cluster j gives 8 in mask k
    rem = n8.copy()
    for k in range(K):
        order = np.argsort(-(rem + rng.random(C) * 1e-6))
        pick = order[:5]
        E[k, pick] = 1
        rem[pick] -= 1
    assert (E.sum(1) == 5).all(), f"each mask must have exactly 5 eights: {set(E.sum(1))}"
    assert (E.sum(0) == n8).all(), "cluster 'eight' counts drifted"

    masks = [[] for _ in range(K)]
    per_demo_counts = {}
    for j, c in enumerate(cl):
        demos = by_c[c]                                  # 15 demos
        cnt = np.zeros(len(demos), dtype=int)
        for k in range(K):
            take = 8 if E[k, j] else 7
            order = np.lexsort((rng.random(len(demos)), cnt))   # least-included first, seeded ties
            sel = order[:take]
            cnt[sel] += 1
            masks[k].extend([demos[i] for i in sel])
        per_demo_counts.update({demos[i]: int(cnt[i]) for i in range(len(demos))})
    for k in range(K):
        assert len(masks[k]) == DEMOS_PER_MASK, f"mask {k}: {len(masks[k])} demos"
    return masks, per_demo_counts, E, cl


def main():
    L.assert_prereg_locked()
    _, by_c = dataset.train_pool()

    stage_g = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    g_sets = {m["mask_id"]: frozenset(m["demos"]) for m in stage_g}
    assert len(g_sets) == 24, f"expected 24 Stage-G masks, got {len(g_sets)}"

    seed, increments, collisions = MASK_SEED, 0, []
    while True:
        masks, cnts, E, cl = build(seed=seed)
        new_sets = [frozenset(m) for m in masks]
        collisions = [{"new_index": i, "stage_g_mask": gid}
                      for i, s in enumerate(new_sets) for gid, gs in g_sets.items() if s == gs]
        if not collisions:
            break
        increments += 1
        seed += 1
        print(f"[P14] COLLISION with Stage G at seed {seed-1}: {collisions}; incrementing")
        if increments > 50:
            raise RuntimeError("mask seed incremented 50x without escaping Stage-G coincidence")

    # also assert the 16 new masks are distinct from EACH OTHER
    assert len(set(frozenset(m) for m in masks)) == len(masks), "duplicate masks within P14"

    lo, hi = min(cnts.values()), max(cnts.values())
    # per-demo inclusion expectation at K=16, alpha=68/135: 16*68/135 = 8.06
    out = {
        "stage": "P14",
        "PREREGISTERED_mask_seed": MASK_SEED,
        "mask_seed_used": seed,
        "seed_increments_due_to_collision": increments,
        "collisions_with_stage_G": collisions,
        "coincidence_check": {
            "n_stage_G_masks_compared": len(g_sets),
            "n_new_masks": len(masks),
            "n_coinciding": len(collisions),
            "PASS": len(collisions) == 0,
            "rule": "a new mask coincides iff its SET of demo ids equals a Stage-G mask's set",
        },
        "K": len(masks),
        "demos_per_mask": DEMOS_PER_MASK,
        "n_demos_total": 135,
        "alpha": 68 / 135,
        "eights_per_cluster": {cl[j]: int(E[:, j].sum()) for j in range(len(cl))},
        "per_demo_inclusion_min": int(lo),
        "per_demo_inclusion_max": int(hi),
        "per_demo_inclusion_expected": 16 * 68 / 135,
        "per_demo_counts": cnts,
        "masks": [{"mask_id": f"H{k:03d}", "n_demos": len(m), "demos": m,
                   "in_target_count": {c: sum(1 for d in m if d in set(by_c[c])) for c in cl}}
                  for k, m in enumerate(masks)],
    }
    p = L.atomic_write_json(os.path.join(P4_RESULTS, "p14_mask_manifest.json"), out)

    print(f"[P14] {len(masks)} fresh masks, seed={seed} (increments={increments}), "
          f"{DEMOS_PER_MASK}/135 demos each")
    print(f"[P14] per-demo inclusion [{lo},{hi}] (expected {16*68/135:.2f})")
    print(f"[P14] eights per cluster: {out['eights_per_cluster']}")
    print(f"[P14] COINCIDENCE CHECK vs the 24 Stage-G masks: "
          f"{'PASS (0 coincide)' if not collisions else 'FAIL'}")
    itc = [m["in_target_count"] for m in out["masks"]]
    print(f"[P14] C2 in-target counts: {sorted(set(x['C2'] for x in itc))}; "
          f"C9: {sorted(set(x['C9'] for x in itc))}")
    print(f"[P14] -> {p}")


if __name__ == "__main__":
    main()
