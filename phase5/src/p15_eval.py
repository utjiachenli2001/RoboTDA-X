"""P15 PREREGISTERED evaluation economy for the diffusion S=10 replication.

The P13/P15 verdict outcome is the held-out executed-action L2 (plain_loss), which
evaluate_diffusion.heldout_losses computes with a FORWARD PASS -- no closed-loop simulation. So a
new diffusion model can produce everything the verdict needs for ALL 9 targets without rolling out
a single episode. This script therefore:

  (a) computes held-out L2 (plain_loss) AND phase-masked L2 (transport/interaction) for ALL 9
      targets via heldout_losses -- NO simulation. THIS is what the verdict uses.
  (b) runs closed-loop success rollouts for C1's 3 probe tasks ONLY (a DESCRIPTIVE success record),
      cutting eval cost ~9x vs the full 27-task probe battery.

Bit-commensurability with the existing S<=8 ground truth (phase4/results/p13_outcomes_S8.parquet):
this script passes NO n_ddim override, exactly as rollout_diffusion.probe_battery did for the
existing runs (n_ddim=None -> DP.N_DDIM_STEPS). The outcomes.json schema matches probe_battery's,
so p15_analyze's ingest reads new and old rows identically. Non-C1 clusters carry success_rate=None
and n_episodes=0 (no episodes were simulated for them, by design).

P6.1 FIX: on a rollout error we raise WITHOUT writing outcomes.json, so a partially-failed run can
never leave a complete-looking artifact behind.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join("/mnt/sdb/ljc/RoboTDA-X", "src"))
import bootstrap  # noqa: F401,E402
from bootstrap import write_done  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5lib as L  # noqa: E402

sys.path.insert(0, L.P3_SRC)
import rollout_diffusion as RD  # noqa: E402  (frozen closed-loop rollout)

LOSS_KEYS = ("plain_loss", "transport_loss", "interaction_loss",
             "denoise_loss", "denoise_transport", "denoise_interaction")
ROLLOUT_CLUSTER = "C1"        # PREREGISTERED: success rollouts for C1's probe tasks ONLY


def econ_eval(run_dir, ckpt="final.pt", n_rollouts=10, workers=12):
    import dataset
    import evaluate_diffusion as EVD

    if L.is_marked(run_dir, "probe"):
        print(f"[p15eval] SKIP (probe.marker): {run_dir}")
        return L.read_artifact(run_dir, "outcomes.json")

    t0 = time.time()
    probes = dataset.probe_tasks()
    suite_of = dataset.suite_of_cluster()
    clusters = dataset.clusters()
    ckpt_path = os.path.join(run_dir, ckpt)

    # ---- (b) closed-loop success rollouts for C1's 3 probe tasks ONLY (descriptive)
    c1_tasks = [(suite_of[ROLLOUT_CLUSTER], t) for t in probes[ROLLOUT_CLUSTER]]
    succ, info = RD.run_rollouts(ckpt_path, c1_tasks, n_rollouts, workers)   # n_ddim=None
    if info["n_errors"]:
        L.atomic_write_json(os.path.join(run_dir, "rollout_errors.json"), info["errors"])
        raise RuntimeError(f"{info['n_errors']} rollout task errors in {run_dir} -- "
                           f"outcomes.json deliberately NOT written")

    # ---- (a) held-out L2 for ALL 9 targets -- forward pass, NO simulation, n_ddim=None
    model = EVD.load_model(ckpt_path, device="cuda")
    losses = EVD.heldout_losses(model, device="cuda")

    out = {}
    for c in clusters:
        if c == ROLLOUT_CLUSTER:
            s = [x for t in probes[c] for x in succ.get(t, [])]
            out[c] = {
                "success_rate": float(np.mean(s)) if s else None,
                "n_episodes": len(s),
                "per_task_success": {t: float(np.mean(succ[t])) if t in succ else None
                                     for t in probes[c]},
                **{k: losses[c][k] for k in LOSS_KEYS},
            }
        else:
            out[c] = {
                "success_rate": None,                 # PREREGISTERED economy: not simulated
                "n_episodes": 0,
                "per_task_success": {},
                **{k: losses[c][k] for k in LOSS_KEYS},
            }

    meta = {"ckpt": ckpt, "policy": "diffusion", "eval": "p15econ",
            "rollout_cluster": ROLLOUT_CLUSTER, "n_rollouts": n_rollouts,
            "n_ddim_steps": __import__("diffusion_policy").N_DDIM_STEPS,
            "rollout_wall_s": info["wall_s"], "n_errors": 0,
            "total_wall_s": time.time() - t0}
    L.atomic_write_json(os.path.join(run_dir, "outcomes.json"),
                        {"outcomes": out, "meta": meta})
    write_done(run_dir, json.dumps(meta) + "\n", name="probe")
    return {"outcomes": out, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    r = econ_eval(a.run_dir, a.ckpt, a.n_rollouts, a.workers)
    o = r["outcomes"]
    print(f"[p15eval] {a.run_dir} wall={r['meta']['total_wall_s']:.0f}s "
          f"C1 succ={o['C1']['success_rate']} C1 L2={o['C1']['plain_loss']:.3f} "
          f"C5 L2={o['C5']['plain_loss']:.3f}")


if __name__ == "__main__":
    main()
