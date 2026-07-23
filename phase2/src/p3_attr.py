"""P3 attribution: per-demo scores AT EACH SCALE Q, from that Q's own 5 full-Q models.

Phase-1's attribution.compute already takes (ensemble_dirs, train_ids, targets), so this is a
direct reuse -- same exact estimators (TracIn / TRAK N x N dual / Woodbury IF), same ridge, same
train gradient (GMM NLL) and same test functional (L2 on C1's 10 held-out demos).

Target = [("C1", "plain")] = exactly P3's held-out L2 outcome.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT  # noqa: E402
import attribution as AT  # noqa: E402

P2 = os.path.join(ROOT, "phase2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, required=True, choices=[15, 50, 150])
    a = ap.parse_args()

    man = json.load(open(f"{P2}/results/p3_mask_manifest.json"))
    pool = man["Q"][str(a.Q)]["pool"]
    assert len(pool) == a.Q

    ens = [os.path.join(P2, "runs/P3", f"Q{a.Q}full_s{s}") for s in (611, 612, 613, 614, 615)]
    miss = [e for e in ens if not os.path.exists(os.path.join(e, "final.pt"))]
    assert not miss, f"missing full-Q checkpoints: {miss}"

    out = f"{P2}/results/p3_influence_Q{a.Q}.parquet"
    AT.compute(ens, out, variant="base", ridge_rel=1e-2,
               train_ids=pool, targets=[("C1", "plain")])
    print(f"[p3attr] Q={a.Q}: wrote {out}")
