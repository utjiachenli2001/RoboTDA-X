"""Build the P8a (48 factorial retrains) and P9 (10 ensemble retrains) job lists, then run them.

P8a -- 12 of the 24 Stage-G masks, chosen by default_rng(909).choice(24, 12, replace=False) over
       the SORTED mask ids (preregistered, frozen). Each mask gets the full 2x2:
           init_seed in {701, 702}  x  order_seed in {801, 802}
       -> 48 retrains. Full probe battery on each.

P9  -- 10 additional FULL-CORPUS (135-demo) models, seeds 211-220, identical config to Phase-1's
       Stage E (seeds 201-210). Full probe battery. -> E = 20 total ensemble members.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402

MASK_SEED = 909
INIT_SEEDS = [701, 702]
ORDER_SEEDS = [801, 802]
P9_SEEDS = list(range(211, 221))
N_ROLLOUTS = 10


def p8a_masks():
    man = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    ids = sorted(m["mask_id"] for m in man)
    assert len(ids) == 24, f"expected 24 Stage-G masks, got {len(ids)}"
    pick = np.random.default_rng(MASK_SEED).choice(24, size=12, replace=False)
    chosen = sorted(ids[i] for i in pick)
    by_id = {m["mask_id"]: m for m in man}
    return [by_id[c] for c in chosen]


def p8a_jobs():
    jobs = []
    for m in p8a_masks():
        for i in INIT_SEEDS:
            for o in ORDER_SEEDS:
                jobs.append({
                    "run_dir": os.path.join(P3_RUNS, "P8a", f"{m['mask_id']}_i{i}_o{o}"),
                    "demos": m["demos"], "mask_id": m["mask_id"],
                    "init_seed": i, "order_seed": o,
                    "trainer": "factorial", "eval": "probe",
                    "n_rollouts": N_ROLLOUTS, "workers": 12,
                })
    return jobs


def p9_jobs():
    ids, _ = dataset.train_pool()
    assert len(ids) == 135
    return [{"run_dir": os.path.join(P3_RUNS, "P9", f"ens_s{s}"),
             "demos": ids, "seed": s, "trainer": "stock", "eval": "probe",
             "n_rollouts": N_ROLLOUTS, "workers": 12} for s in P9_SEEDS]


def main():
    j8, j9 = p8a_jobs(), p9_jobs()
    masks = sorted({j["mask_id"] for j in j8})
    L.atomic_write_json(os.path.join(P3_RESULTS, "p8a_jobs.json"), j8)
    L.atomic_write_json(os.path.join(P3_RESULTS, "p9_jobs.json"), j9)
    L.atomic_write_json(os.path.join(P3_RESULTS, "p8a_mask_selection.json"), {
        "rule": f"default_rng({MASK_SEED}).choice(24, size=12, replace=False) over sorted mask ids",
        "seed": MASK_SEED, "n_masks": len(masks), "masks": masks,
        "init_seeds": INIT_SEEDS, "order_seeds": ORDER_SEEDS,
        "design": "2 init x 2 order, fully crossed, per mask -> 48 retrains"})
    print(f"[P8a] 12 masks: {masks}")
    print(f"[P8a] {len(j8)} jobs -> phase3/results/p8a_jobs.json")
    print(f"[P9 ] {len(j9)} jobs (seeds {P9_SEEDS[0]}-{P9_SEEDS[-1]}, 135 demos each)")

    if "--run" in sys.argv:
        import orch3
        # P9 first: it is short (10 jobs) and unblocks the attribution work.
        orch3.run_jobs(j9, "P9")
        orch3.run_jobs(j8, "P8a")


if __name__ == "__main__":
    main()
