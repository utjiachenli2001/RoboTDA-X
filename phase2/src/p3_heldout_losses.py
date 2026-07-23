"""Losses-only eval pass (no rollouts): held-out L2 / transport / interaction per cluster.

P3's success outcome comes from rollout.py --cluster_tasks C1 (the full 10-task Goal suite at
20 rollouts). That path does not write held-out losses, so this cheap forward-pass stage adds
them. Chained BEFORE the rollout stage, so the run's clustereval.marker implies both exist.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import is_done, write_done  # noqa: E402
import evaluate as EV  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    a = ap.parse_args()

    if is_done(a.run_dir, "heldout"):
        print(f"[heldout] SKIP (heldout.marker): {a.run_dir}")
        sys.exit(0)

    model = EV.load_model(os.path.join(a.run_dir, a.ckpt), device="cuda")
    losses = EV.heldout_losses(model, device="cuda")
    p = os.path.join(a.run_dir, "heldout_losses.json")
    json.dump({"ckpt": a.ckpt, "losses": losses}, open(p, "w"), indent=1)
    write_done(a.run_dir, json.dumps({"ckpt": a.ckpt, "C1_plain_loss": losses["C1"]["plain_loss"]}) + "\n",
               name="heldout")
    print(f"[heldout] {a.run_dir}: C1 plain_loss={losses['C1']['plain_loss']:.4f}")
