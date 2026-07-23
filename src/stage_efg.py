"""STAGES E / F / G -- the ground-truth retraining corpora (spec §6). The main compute.

E  TRAK ensemble        : 10 models on all 135 demos, seeds 201-210. Dual use: TRAK
                          ensembling and the rank-stability jackknife.
F  cluster-grain corpus : 72 balanced masks (5 of 9 clusters = 75 demos) x seeds 301,302
                          = 144, plus the 12 noise-ceiling replicate masks x seeds 303,304
                          = 24  ->  168 retrains.
G  demo-grain corpus    : 24 masks (68 of 135 demos, within-cluster stratified) x seeds
                          401,402 = 48 retrains.

Every model gets the full probe battery: 27 probe tasks x 10 rollouts + held-out
plain/transport/interaction losses on all 9 clusters.
These retrains are ATTRIBUTION-AGNOSTIC ground truth: they run regardless of Gate 1.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS, RESULTS
import dataset
import masks as MK
import orchestrator as O

E_SEEDS = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210]
F_SEEDS = [301, 302]
F_NC_SEEDS = [303, 304]
G_SEEDS = [401, 402]
N_ROLLOUTS = 10
WORKERS = 12


def jobs_E():
    ids, _ = dataset.train_pool()
    return [{"run_dir": os.path.join(RUNS, "stage_E", f"ens_s{s}"), "demos": ids, "seed": s,
             "n_rollouts": N_ROLLOUTS, "eval": "probe", "workers": WORKERS} for s in E_SEEDS]


def jobs_F():
    man = MK.cluster_mask_manifest()
    nc = set(man["noise_ceiling_masks"])
    jobs = []
    for m in man["masks"]:
        seeds = list(F_SEEDS) + (list(F_NC_SEEDS) if m["mask_id"] in nc else [])
        for s in seeds:
            jobs.append({"run_dir": os.path.join(RUNS, "stage_F", f"{m['mask_id']}_s{s}"),
                         "demos": m["demos"], "seed": s, "n_rollouts": N_ROLLOUTS,
                         "eval": "probe", "workers": WORKERS,
                         "mask_id": m["mask_id"], "clusters_in": m["clusters"]})
    return jobs


def jobs_G():
    man = MK.demo_mask_manifest()
    return [{"run_dir": os.path.join(RUNS, "stage_G", f"{m['mask_id']}_s{s}"),
             "demos": m["demos"], "seed": s, "n_rollouts": N_ROLLOUTS,
             "eval": "probe", "workers": WORKERS, "mask_id": m["mask_id"]}
            for m in man["masks"] for s in G_SEEDS]


def collect(stage, jobs):
    """Gather every run's outcomes.json into one table (artifact-only; missing = did not run)."""
    rows, missing = [], []
    for j in jobs:
        p = os.path.join(j["run_dir"], "outcomes.json")
        if not os.path.exists(p):
            missing.append(os.path.basename(j["run_dir"]))
            continue
        o = json.load(open(p))["outcomes"]
        for c, v in o.items():
            rows.append({"stage": stage, "run": os.path.basename(j["run_dir"]),
                         "mask_id": j.get("mask_id", "full135"), "seed": j["seed"],
                         "target": c, **{k: v[k] for k in
                                         ("success_rate", "n_episodes", "plain_loss",
                                          "transport_loss", "interaction_loss")}})
    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, f"{stage}_outcomes.parquet")
    df.to_parquet(out, index=False)
    print(f"[{stage}] collected {len(rows)} rows from {len(jobs)-len(missing)}/{len(jobs)} runs "
          f"-> {out}")
    if missing:
        print(f"[{stage}] MISSING {len(missing)} runs: {missing[:8]}"
              f"{' ...' if len(missing) > 8 else ''}")
        json.dump(missing, open(os.path.join(RESULTS, f"{stage}_missing.json"), "w"), indent=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["E", "F", "G"])
    ap.add_argument("--collect_only", action="store_true")
    a = ap.parse_args()
    jobs = {"E": jobs_E, "F": jobs_F, "G": jobs_G}[a.stage]()
    print(f"[stage_{a.stage}] {len(jobs)} jobs")
    if not a.collect_only:
        O.run_jobs(jobs, f"stage_{a.stage}")
    collect(f"stage_{a.stage}", jobs)


if __name__ == "__main__":
    main()
