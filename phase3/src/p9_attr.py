"""P9 -- per-member attribution for the 10 NEW full-corpus models (seeds 211-220) -> E = 20.

Phase-1's Stage E gave 10 ensemble members (ens_s201..210) and their per-member per-demo scores
already exist in results/influence_table_per_member.parquet. P9 adds 10 more members so the
ATTRIBUTION side of the study can be priced the way Phase-2 P4 priced the ground-truth side.

Same estimators, same functional ('plain'), same FROZEN default ridge (ridge_rel = 1e-2) as
Phase 1 -- because the question here is "how many ENSEMBLE MEMBERS does a stable ranking need",
and changing the ridge at the same time would confound it. (The ridge question is P6.5's.)

The P6.2 probe-leak guard runs on every member before its gradients are used.
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import attribution as AT  # noqa: E402

RIDGE = 1e-2          # the FROZEN Phase-1 default -- deliberately not re-tuned here
OUT = os.path.join(P3_RESULTS, "p9_influence_new_members.parquet")


def main():
    runs = sorted(glob.glob(os.path.join(P3_RUNS, "P9", "ens_s*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "final.pt"))]
    print(f"[P9] {len(runs)} new ensemble members: {[os.path.basename(r) for r in runs]}")
    assert len(runs) == 10, f"expected 10 new members, got {len(runs)}"

    # P6.2 GUARD
    heldout = set(dataset.heldout_pool()[0])
    L.assert_no_probe_leak(runs, heldout, context="P9 attribution, test side = held-out pool")
    print("[P9] probe-leak guard PASSED")

    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    N = len(train_ids)
    clusters = dataset.clusters()
    targets = [(c, "plain") for c in clusters]

    rows = []
    t0 = time.time()
    for run in runs:
        member = os.path.basename(run)
        meta = __import__("json").load(open(os.path.join(run, "train_meta.json")))
        cfg = meta["cfg"]

        # ---- TracIn over this member's 5 checkpoints
        tracin = {}
        for cp in [os.path.join(run, c) for c in meta["ckpts"]]:
            model = AT.load_ckpt_model(cp)
            step = torch.load(cp, map_location="cpu", weights_only=False)["step"]
            eta = AT.lr_at_step(cfg, step)
            tg, order = AT.build_targets(model, hbank, "base", targets)
            TG = torch.stack([tg[k] for k in order])
            for d in train_ids:
                gd = AT.demo_gradient(model, tbank, slices[d])
                dots = (TG @ gd).cpu().numpy()
                for j, (c, f) in enumerate(order):
                    tracin[(c, d)] = tracin.get((c, d), 0.0) + eta * float(dots[j])
            del model, TG, tg
            torch.cuda.empty_cache()
        for (c, d), v in tracin.items():
            rows.append(("TracIn", "plain", c, d, member, v))

        # ---- TRAK + IF (exact dual forms) at the final checkpoint
        model = AT.load_ckpt_model(os.path.join(run, "final.pt"))
        tg, order = AT.build_targets(model, hbank, "base", targets)
        TG = torch.stack([tg[k] for k in order])
        PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device="cuda")
        for i, d in enumerate(train_ids):
            PHI[i] = AT.demo_gradient(model, tbank, slices[d])
        G = (PHI @ PHI.T).double()
        K = (PHI @ TG.T).double()
        lam = RIDGE * float(torch.diagonal(G).mean())
        I = torch.eye(N, device="cuda", dtype=torch.float64)
        trak = torch.linalg.solve(G + lam * I, K)
        inner = torch.linalg.solve(lam * N * I + G, K)
        iff = (K - G @ inner) / lam
        for j, (c, f) in enumerate(order):
            for i, d in enumerate(train_ids):
                rows.append(("TRAK", "plain", c, d, member, float(trak[i, j])))
                rows.append(("IF", "plain", c, d, member, float(iff[i, j])))
        del model, PHI, G, K, TG, tg, trak, iff, inner, I
        torch.cuda.empty_cache()
        print(f"[P9] {member} done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows, columns=["attributor", "functional", "target", "demo_id", "member",
                                     "score"])
    df.to_parquet(OUT, index=False)
    print(f"[P9] wrote {OUT}: {len(df)} rows, {df.member.nunique()} members "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
