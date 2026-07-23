"""P10 job lists: P10a (Gate-0 sanity, 12), P10b (demo-grain LDS, 96), P10ens (attribution, 5).

P10a -- C1 and C5, {target-only (15 demos), co-train (135)} x 3 seeds (611,612,613), PAIRED.
        Descriptive: does the co-training phenomenon exist for this policy class at all?
        Evaluated with cluster_eval on the target cluster (like Phase-1 Gate 0).

P10b -- the EXISTING 24 Stage-G demo masks (NOT resampled) x 4 seeds (601-604) = 96 retrains.
        Full probe battery. This is the replication of the CORE null.

P10ens- 5 full-corpus (135-demo) diffusion models, seeds 621-625, as the attribution ensemble.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402

P10A_SEEDS = [611, 612, 613]
P10B_SEEDS = [601, 602, 603, 604]
ENS_SEEDS = [621, 622, 623, 624, 625]
N_ROLLOUTS = 10


def p10a_jobs():
    _, by_c = dataset.train_pool()
    full, _ = dataset.train_pool()
    jobs = []
    for tgt in ("C1", "C5"):
        for arm, demos in (("target", by_c[tgt]), ("cotrain", full)):
            for s in P10A_SEEDS:
                jobs.append({
                    "run_dir": os.path.join(P3_RUNS, "P10a", f"{tgt}_{arm}_s{s}"),
                    "demos": demos, "seed": s, "trainer": "diffusion",
                    "eval": "cluster_tasks", "target": tgt,
                    "n_rollouts": 20, "workers": 12,
                })
    return jobs


def p10b_jobs(seeds=None):
    man = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    seeds = seeds or P10B_SEEDS
    return [{"run_dir": os.path.join(P3_RUNS, "P10b", f"{m['mask_id']}_s{s}"),
             "demos": m["demos"], "mask_id": m["mask_id"], "seed": s, "trainer": "diffusion",
             "eval": "probe", "n_rollouts": N_ROLLOUTS, "workers": 12}
            for m in man for s in seeds]


def p10ens_jobs():
    ids, _ = dataset.train_pool()
    return [{"run_dir": os.path.join(P3_RUNS, "P10ens", f"dpens_s{s}"),
             "demos": ids, "seed": s, "trainer": "diffusion", "eval": "probe",
             "n_rollouts": N_ROLLOUTS, "workers": 12} for s in ENS_SEEDS]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ja, jb, je = p10a_jobs(), p10b_jobs(), p10ens_jobs()
    L.atomic_write_json(os.path.join(P3_RESULTS, "p10a_jobs.json"), ja)
    L.atomic_write_json(os.path.join(P3_RESULTS, "p10b_jobs.json"), jb)
    L.atomic_write_json(os.path.join(P3_RESULTS, "p10ens_jobs.json"), je)
    print(f"[P10a] {len(ja)} jobs (C1/C5 x target/cotrain x seeds {P10A_SEEDS})")
    print(f"[P10b] {len(jb)} jobs (24 masks x seeds {P10B_SEEDS})")
    print(f"[P10ens] {len(je)} jobs (135-demo, seeds {ENS_SEEDS})")

    if "--run" in sys.argv:
        import orch3
        if which in ("all", "ens"):
            orch3.run_jobs(je, "P10ens")
        if which in ("all", "a"):
            orch3.run_jobs(ja, "P10a")
        if which in ("all", "b"):
            orch3.run_jobs(jb, "P10b")


if __name__ == "__main__":
    main()
