"""P3 regime boundary: masks + jobs at Q in {15, 50, 150} on C1's Goal suite.

Q ladder: Phase-1 stage_c.goal_demo_ladder() EXACTLY (alphabetical tasks, round-robin, lowest
demo index first, skipping C1's 10 held-out demos), extended to Q=150. Nested by construction:
Q15 subset Q50 subset Q150 subset Q490.

Masks (preregistered): K=12 per Q, alpha=0.6 -> 9 / 30 / 90 demos per mask, balanced so every
demo appears in floor/ceil(K*alpha)=7 or 8 masks. Deterministic, seed 77.

Per Q: 12 masks x 4 seeds (601-604) = 48 retrains, plus 5 full-Q models (seeds 611-615) as
that scale's attribution ensemble. 3 Q x (48 + 5) = 159 retrains.

Eval per model: heldout losses (C1's 10 held-out demos) THEN cluster_tasks C1 = the full
libero_goal suite, 10 tasks x 20 rollouts = 200 episodes.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT  # noqa: E402
import stage_c as SC  # noqa: E402

QS = [15, 50, 150]
K = 12
ALPHA = 0.6
MASK_SEED = 77                    # preregistered
SEEDS = [601, 602, 603, 604]      # mask retrains
FULL_SEEDS = [611, 612, 613, 614, 615]   # full-Q attribution ensemble
N_ROLLOUTS = 20
WORKERS = 12
RUNS_P2 = os.path.join(ROOT, "phase2/runs")


def ladder():
    """Phase-1's exact rule (stage_c.goal_demo_ladder), extended to Q=150.

    stage_c hardcodes Q in {15,50,490}, so the rule is re-expressed here and then VERIFIED to
    reproduce Phase-1's ladder bit-for-bit at Q=15/50/490. If Phase-1's selection ever differed
    from this rule the assertion fires and P3 stops -- the ladder is not merely claimed nested,
    it is checked against the artifact-producing code.
    """
    import dataset
    from clusters import suite_task_names

    _, by_c = dataset.train_pool()
    _, ho_c = dataset.heldout_pool()
    base = list(by_c["C1"])
    held = set(ho_c["C1"])
    tasks = sorted(suite_task_names("libero_goal"))
    taken, extra = set(base), []
    for rank in range(50):
        for t in tasks:
            d = dataset.did("libero_goal", t, f"demo_{rank}")
            if d in taken or d in held:
                continue
            extra.append(d)
            taken.add(d)
    lad = {q: base + extra[:q - len(base)] for q in (15, 50, 150)}
    lad[490] = base + extra

    ref = SC.goal_demo_ladder()                     # Phase-1's own function
    for q in (15, 50, 490):
        assert lad[q] == ref[q], f"Q={q} ladder differs from Phase-1's stage_c ladder!"
    for q in QS:
        assert len(lad[q]) == q, f"Q={q}: got {len(lad[q])}"
        assert not (set(lad[q]) & held), f"Q={q} leaks C1 held-out demos"
    assert set(lad[15]) <= set(lad[50]) <= set(lad[150]) <= set(lad[490]), "ladder not nested"
    print("[ladder] reproduces Phase-1 stage_c ladder exactly at Q=15/50/490; nested through 150")
    return {q: lad[q] for q in QS}


def balanced_masks(demos, k=K, alpha=ALPHA, seed=MASK_SEED):
    """K masks of exactly m = round(alpha*Q) demos, every demo in 7 or 8 masks (exact balance).

    Greedy: fill each mask with the demos of largest remaining capacity; ties broken by a
    seeded permutation, so the result is deterministic given (demos, k, alpha, seed).
    """
    rng = np.random.default_rng(seed)
    n = len(demos)
    m = int(round(alpha * n))
    total = k * m
    base, extra = divmod(total, n)          # every demo gets `base`; `extra` demos get one more
    cap = np.full(n, base, dtype=int)
    cap[rng.permutation(n)[:extra]] += 1
    assert cap.sum() == total

    rem = cap.copy()
    masks = []
    for j in range(k):
        jitter = rng.permutation(n)                    # deterministic tie-break
        order = sorted(range(n), key=lambda i: (-rem[i], jitter[i]))
        pick = order[:m]
        assert all(rem[i] > 0 for i in pick), "capacity exhausted -- balance infeasible"
        rem[pick] -= 1
        masks.append(sorted(pick))
    assert rem.sum() == 0, f"leftover capacity {rem.sum()}"

    incl = np.zeros(n, int)
    for mk in masks:
        incl[mk] += 1
    assert incl.min() >= base and incl.max() <= base + 1, (incl.min(), incl.max())
    return [{"mask_id": f"Q{n}M{j:02d}", "demos": [demos[i] for i in mk]}
            for j, mk in enumerate(masks)], {"m": m, "incl_min": int(incl.min()),
                                             "incl_max": int(incl.max()), "target": total / n}


def build():
    lad = ladder()
    man, jobs = {"seed": MASK_SEED, "K": K, "alpha": ALPHA, "Q": {}}, []
    for q in QS:
        masks, stats = balanced_masks(lad[q])
        man["Q"][str(q)] = {"n_demos": q, "demos_per_mask": stats["m"],
                            "per_demo_inclusion_min": stats["incl_min"],
                            "per_demo_inclusion_max": stats["incl_max"],
                            "per_demo_inclusion_target": stats["target"],
                            "pool": lad[q], "masks": masks}
        for mk in masks:
            for s in SEEDS:
                jobs.append({"run_dir": os.path.join(RUNS_P2, "P3", f"{mk['mask_id']}_s{s}"),
                             "demos": mk["demos"], "seed": s, "n_rollouts": N_ROLLOUTS,
                             "eval": "cluster_tasks", "target": "C1", "workers": WORKERS,
                             "Q": q, "mask_id": mk["mask_id"], "kind": "mask"})
        for s in FULL_SEEDS:                                   # full-Q attribution ensemble
            jobs.append({"run_dir": os.path.join(RUNS_P2, "P3", f"Q{q}full_s{s}"),
                         "demos": lad[q], "seed": s, "n_rollouts": N_ROLLOUTS,
                         "eval": "cluster_tasks", "target": "C1", "workers": WORKERS,
                         "Q": q, "mask_id": f"Q{q}full", "kind": "full"})
    return man, jobs


if __name__ == "__main__":
    man, jobs = build()
    assert len(jobs) == 3 * (K * len(SEEDS) + len(FULL_SEEDS)) == 159, len(jobs)
    json.dump(man, open(os.path.join(ROOT, "phase2/results/p3_mask_manifest.json"), "w"), indent=1)
    json.dump(jobs, open(os.path.join(ROOT, "phase2/results/p3_jobs.json"), "w"), indent=1)
    for q in QS:
        d = man["Q"][str(q)]
        print(f"Q={q:3d}: {K} masks x {d['demos_per_mask']} demos "
              f"(alpha={d['demos_per_mask']/q:.3f}); per-demo inclusion "
              f"{d['per_demo_inclusion_min']}-{d['per_demo_inclusion_max']} "
              f"(target {d['per_demo_inclusion_target']:.1f})")
    print(f"{len(jobs)} jobs ({K*len(SEEDS)*3} mask retrains + {len(FULL_SEEDS)*3} full-Q)")
