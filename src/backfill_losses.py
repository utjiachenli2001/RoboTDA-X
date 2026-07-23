"""Backfill the L2 held-out functional into already-completed runs' outcomes.json.

The rollout/success half of those runs is untouched and still valid; only the loss half is
recomputed, from each run's saved final.pt (forward passes only -- NO retraining). Needed
because the original GMM-NLL functional proved to be an unstable measurement (see policy.l2).

Existing keys are OVERWRITTEN with the L2 values and the old NLL numbers are preserved under
*_nll, so nothing is silently lost.
"""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS
import evaluate as EV


def backfill(run_dir):
    op = os.path.join(run_dir, "outcomes.json")
    ck = os.path.join(run_dir, "final.pt")
    if not (os.path.exists(op) and os.path.exists(ck)):
        return False
    d = json.load(open(op))
    if d.get("meta", {}).get("loss_functional") == "l2":
        return False                                  # already backfilled
    model = EV.load_model(ck, device="cuda")
    losses = EV.heldout_losses(model, device="cuda")
    for c, v in d["outcomes"].items():
        if c in losses:
            v.update({k: losses[c][k] for k in
                      ("plain_loss", "transport_loss", "interaction_loss",
                       "plain_loss_nll", "transport_loss_nll", "interaction_loss_nll",
                       "median_nll")})
    d.setdefault("meta", {})["loss_functional"] = "l2"
    tmp = op + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, op)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--globs", nargs="+",
                    default=[os.path.join(RUNS, "stage_D", "*"),
                             os.path.join(RUNS, "stage_E", "*")])
    a = ap.parse_args()
    dirs = sorted({d for g in a.globs for d in glob.glob(g) if os.path.isdir(d)})
    n = 0
    for d in dirs:
        if backfill(d):
            n += 1
            print(f"[backfill] {os.path.basename(d)}", flush=True)
    print(f"[backfill] updated {n}/{len(dirs)} runs with the L2 functional")


if __name__ == "__main__":
    main()
