"""Phase-5 job list: P15 (diffusion S=10). P16 and P17 are zero-retrain and have no jobs here.

P15 -- the SAME 24 Stage-G masks x 2 NEW diffusion seeds (609, 610) -> S = 10 with 601-608. The
       frozen config (phase3/results/p10_config_frozen.json) is untouched -- train_diffusion.py
       reads it itself; Phase 5 passes no hyperparameters. Eval is the PREREGISTERED economy
       'p15econ' (C1-only rollouts + all-9 held-out L2). 48 retrains.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5lib as L
from p5lib import P5_RESULTS, P5_RUNS, RESULTS

P15_SEEDS = [609, 610]                      # PREREGISTERED
N_ROLLOUTS = 10                             # C1's 3 probe tasks x 10 = 30 episodes / model


def stage_g_masks():
    man = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    assert len(man) == 24, f"expected 24 Stage-G masks, got {len(man)}"
    return man


def p15_jobs(seeds=None):
    seeds = seeds or P15_SEEDS
    return [{"run_dir": os.path.join(P5_RUNS, "P15", f"{m['mask_id']}_s{s}"),
             "demos": m["demos"], "mask_id": m["mask_id"], "seed": s, "trainer": "diffusion",
             "eval": "p15econ", "n_rollouts": N_ROLLOUTS, "workers": 12}
            for m in stage_g_masks() for s in seeds]


def main():
    L.assert_prereg_locked()
    j = p15_jobs()
    L.atomic_write_json(os.path.join(P5_RESULTS, "p15_jobs.json"), j)
    print(f"[P15] {len(j)} jobs (24 masks x diffusion seeds {P15_SEEDS}) "
          f"-> C1-only rollouts: {len(j) * 3 * N_ROLLOUTS} episodes; held-out L2 for all 9 targets")


if __name__ == "__main__":
    main()
