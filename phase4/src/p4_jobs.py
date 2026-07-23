"""Phase-4 job lists: P12 (BC S=10), P13 (diffusion S=8), P14 (fresh heterogeneity corpus).

P12 -- the EXISTING 24 Stage-G demo masks x 4 NEW BC seeds (407-410) -> S = 10 with 401-406.
       Identical config (configs/policy.yaml), identical trainer (src/train.py), identical probe
       battery (27 tasks x 10 rollouts => 30 episodes per cluster target). 96 retrains.

P13 -- the SAME 24 masks x 2 NEW diffusion seeds (607, 608) -> S = 8 with 601-606. The frozen
       config (phase3/results/p10_config_frozen.json) is untouched -- train_diffusion.py reads it
       itself; Phase 4 passes no hyperparameters. Probe battery as P10b. 48 retrains.

P14 -- the 16 FRESH masks (phase4/results/p14_mask_manifest.json) x 4 BC seeds (411-414).
       Probes RESTRICTED to C2's and C9's clusters (--clusters C2,C9): 6 tasks x 10 rollouts
       = 60 episodes per model, plus held-out L2 for C2 and C9. 64 retrains.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, P4_RUNS, RESULTS

P12_SEEDS = [407, 408, 409, 410]           # PREREGISTERED (fallback: [407, 408] -> S=8)
P13_SEEDS = [607, 608]                     # PREREGISTERED
P14_SEEDS = [411, 412, 413, 414]           # PREREGISTERED
P14_CLUSTERS = ["C2", "C9"]                # PREREGISTERED (probe restriction)
N_ROLLOUTS = 10


def stage_g_masks():
    man = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    assert len(man) == 24, f"expected 24 Stage-G masks, got {len(man)}"
    return man


def p12_jobs(seeds=None):
    seeds = seeds or P12_SEEDS
    return [{"run_dir": os.path.join(P4_RUNS, "P12", f"{m['mask_id']}_s{s}"),
             "demos": m["demos"], "mask_id": m["mask_id"], "seed": s, "trainer": "stock",
             "eval": "probe", "n_rollouts": N_ROLLOUTS, "workers": 12}
            for m in stage_g_masks() for s in seeds]


def p13_jobs(seeds=None):
    seeds = seeds or P13_SEEDS
    return [{"run_dir": os.path.join(P4_RUNS, "P13", f"{m['mask_id']}_s{s}"),
             "demos": m["demos"], "mask_id": m["mask_id"], "seed": s, "trainer": "diffusion",
             "eval": "probe", "n_rollouts": N_ROLLOUTS, "workers": 12}
            for m in stage_g_masks() for s in seeds]


def p14_jobs(seeds=None):
    seeds = seeds or P14_SEEDS
    p = os.path.join(P4_RESULTS, "p14_mask_manifest.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing -- run p14_masks.py first (it emits the manifest "
                                f"AND runs the Stage-G coincidence check).")
    man = json.load(open(p))
    assert man["coincidence_check"]["PASS"], "P14 mask corpus failed the Stage-G coincidence check"
    return [{"run_dir": os.path.join(P4_RUNS, "P14", f"{m['mask_id']}_s{s}"),
             "demos": m["demos"], "mask_id": m["mask_id"], "seed": s, "trainer": "stock",
             "eval": "probe", "clusters": P14_CLUSTERS, "n_rollouts": N_ROLLOUTS, "workers": 12}
            for m in man["masks"] for s in seeds]


def main():
    L.assert_prereg_locked()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "p12"):
        j = p12_jobs()
        L.atomic_write_json(os.path.join(P4_RESULTS, "p12_jobs.json"), j)
        print(f"[P12] {len(j)} jobs (24 masks x seeds {P12_SEEDS}) "
              f"-> {len(j) * 27 * N_ROLLOUTS} episodes")
    if which in ("all", "p13"):
        j = p13_jobs()
        L.atomic_write_json(os.path.join(P4_RESULTS, "p13_jobs.json"), j)
        print(f"[P13] {len(j)} jobs (24 masks x diffusion seeds {P13_SEEDS}) "
              f"-> {len(j) * 27 * N_ROLLOUTS} episodes")
    if which in ("all", "p14"):
        j = p14_jobs()
        L.atomic_write_json(os.path.join(P4_RESULTS, "p14_jobs.json"), j)
        print(f"[P14] {len(j)} jobs (16 fresh masks x seeds {P14_SEEDS}, clusters "
              f"{P14_CLUSTERS}) -> {len(j) * 6 * N_ROLLOUTS} episodes")


if __name__ == "__main__":
    main()
