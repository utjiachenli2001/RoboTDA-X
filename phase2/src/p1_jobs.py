"""P1: 4 new seeds (403-406) on the EXISTING 24 Stage-G demo masks -> 96 retrains.

Masks are READ from results/demo_mask_manifest.json and NOT resampled.
Probe battery identical to Phase-1 Stage G (N_ROLLOUTS=10 on each cluster's 3 probe tasks,
plus the held-out L2 / transport / interaction losses) -- so the new outcomes merge
row-for-row with results/stage_G_outcomes.parquet.
"""
import json
import os
import sys

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import RESULTS, ROOT  # noqa: E402

P1_SEEDS = [403, 404, 405, 406]          # preregistered
N_ROLLOUTS = 10                          # identical to Phase-1 Stage G
WORKERS = 12
RUNS_P2 = os.path.join(ROOT, "phase2/runs")


def jobs():
    man = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))
    assert man["K"] == 24 and man["demos_per_mask"] == 68, "Stage-G mask manifest changed!"
    out = []
    for m in man["masks"]:
        for s in P1_SEEDS:
            out.append({
                "run_dir": os.path.join(RUNS_P2, "stage_G6", f"{m['mask_id']}_s{s}"),
                "demos": m["demos"], "seed": s, "n_rollouts": N_ROLLOUTS,
                "eval": "probe", "workers": WORKERS, "mask_id": m["mask_id"],
            })
    return out


if __name__ == "__main__":
    j = jobs()
    assert len(j) == 96, len(j)
    assert all(len(x["demos"]) == 68 for x in j)
    p = os.path.join(ROOT, "phase2/results/p1_jobs.json")
    json.dump(j, open(p, "w"), indent=1)
    print(f"wrote {len(j)} jobs -> {p}")
    print(f"masks={len({x['mask_id'] for x in j})} seeds={sorted({x['seed'] for x in j})}")
