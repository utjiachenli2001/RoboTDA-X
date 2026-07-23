"""P2 step 2: per-TASK attribution.

Phase 1 attributed the 135 training demos toward 9 CLUSTER-level held-out functionals.
P2 attributes them toward 27 TASK-level functionals: for each task, the plain L2 loss on that
task's 5 fresh probe demos (demo_45..49, disjoint from the corpus -- see per_task_probes.json).

Estimators, models, and math are Phase 1's, unchanged (src/attribution.py):
  TracIn  sum_c eta_c <g_i(theta_c), g_test(theta_c)>  over each member's 5 checkpoints
  TRAK    (G + lam I)^-1 k_test          (exact N x N dual, N=135)
  IF      Woodbury empirical-Fisher inverse
Train gradient = GMM NLL (the training objective). Test functional = L2 (Phase-1 convention).

Ensembles:
  --ens stageE : the 10 Stage-E members            (PRIMARY, preregistered)
  --ens stageB : the 15 Stage-B co-train members   (SECONDARY, reported separately)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT, RUNS  # noqa: E402
import dataset  # noqa: E402

# every demo (training AND probe) is resolved through the Phase-2 overlay: it contains
# symlinks to Phase-1's features plus the newly extracted probe demos. Norm stats stay frozen.
dataset.PROC = os.path.join(ROOT, "phase2/data/proc")

import attribution as AT  # noqa: E402  (imports dataset; picks up the patched PROC)

RIDGE_REL = 1e-2          # identical to Phase 1


def probe_targets(model, pbank, task_rows):
    """Plain-L2 target gradient for each of the 27 tasks (mask=None => plain mean)."""
    out, order = {}, []
    for k in sorted(task_rows):
        g = AT.target_gradient(model, pbank, task_rows[k], None)
        out[k] = g
        order.append(k)
    return out, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", choices=["stageE", "stageB"], default="stageE")
    a = ap.parse_args()

    probes = json.load(open(os.path.join(ROOT, "phase2/results/per_task_probes.json")))["probes"]
    assert len(probes) == 27

    if a.ens == "stageE":
        ens = [os.path.join(RUNS, "stage_E", d) for d in sorted(os.listdir(os.path.join(RUNS, "stage_E")))]
    else:
        ens = [os.path.join(RUNS, "stage_B", d)
               for d in sorted(os.listdir(os.path.join(RUNS, "stage_B"))) if "cotrain" in d]
    print(f"[p2attr] ensemble={a.ens}: {len(ens)} members", flush=True)

    # ------------------------------------------------------------------ banks
    train_ids, _ = dataset.train_pool()
    N = len(train_ids)
    assert N == 135, N
    tbank = dataset.Bank(train_ids)
    slices = tbank.demo_slices()
    rows_of = {d: slices[d] for d in train_ids}
    nwin = {d: int(len(rows_of[d])) for d in train_ids}

    probe_ids, owner_task = [], []
    for k, v in sorted(probes.items()):
        for d in v["demo_ids"]:
            probe_ids.append(d)
            owner_task.append(k)
    pbank = dataset.Bank(probe_ids)
    pslices = pbank.demo_slices()
    task_rows = {}
    for k, v in sorted(probes.items()):
        task_rows[k] = np.concatenate([pslices[d] for d in v["demo_ids"]])
    print(f"[p2attr] probe bank: {len(probe_ids)} demos, {pbank.n} windows, "
          f"{len(task_rows)} task functionals", flush=True)

    cluster_of = dataset.cluster_of_task()
    t0, rows = time.time(), []

    for run in ens:
        member = os.path.basename(run)
        meta = json.load(open(os.path.join(run, "train_meta.json")))
        cfg = meta["cfg"]
        ckpts = [os.path.join(run, c) for c in meta["ckpts"]]

        # ---------------- TracIn over this member's 5 checkpoints
        tracin = {}
        for cp in ckpts:
            model = AT.load_ckpt_model(cp)
            step = torch.load(cp, map_location="cpu", weights_only=False)["step"]
            eta = AT.lr_at_step(cfg, step)
            tg, order = probe_targets(model, pbank, task_rows)
            TG = torch.stack([tg[k] for k in order])
            for d in train_ids:
                gd = AT.demo_gradient(model, tbank, rows_of[d])
                dots = (TG @ gd).cpu().numpy()
                for j, k in enumerate(order):
                    tracin[(k, d)] = tracin.get((k, d), 0.0) + eta * float(dots[j])
            del model, TG, tg
            torch.cuda.empty_cache()
        for (k, d), v in tracin.items():
            rows.append(("TracIn", k, d, member, v))
        print(f"[p2attr] TracIn {member} done ({time.time()-t0:.0f}s)", flush=True)

        # ---------------- TRAK + IF on the final checkpoint (exact dual forms)
        model = AT.load_ckpt_model(os.path.join(run, "final.pt"))
        tg, order = probe_targets(model, pbank, task_rows)
        TG = torch.stack([tg[k] for k in order])                  # (27, p)
        PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device="cuda")
        for i, d in enumerate(train_ids):
            PHI[i] = AT.demo_gradient(model, tbank, rows_of[d])
        G = (PHI @ PHI.T).double()
        K = (PHI @ TG.T).double()
        lam = RIDGE_REL * float(torch.diagonal(G).mean())
        I = torch.eye(N, device="cuda", dtype=torch.float64)
        trak = torch.linalg.solve(G + lam * I, K)
        inner = torch.linalg.solve(lam * N * I + G, K)
        iff = (K - G @ inner) / lam
        for j, k in enumerate(order):
            for i, d in enumerate(train_ids):
                rows.append(("TRAK", k, d, member, float(trak[i, j])))
                rows.append(("IF", k, d, member, float(iff[i, j])))
        del model, PHI, G, K, TG, tg, trak, iff, inner, I
        torch.cuda.empty_cache()
        print(f"[p2attr] TRAK+IF {member} done ({time.time()-t0:.0f}s)", flush=True)

    pm = pd.DataFrame(rows, columns=["attributor", "task", "demo_id", "member", "score"])
    df = (pm.groupby(["attributor", "task", "demo_id"], as_index=False)
            .agg(score=("score", "mean"), n_members=("score", "size")))
    df["n_windows"] = df["demo_id"].map(nwin)
    df["cluster_of_demo"] = df["demo_id"].map(lambda d: cluster_of[dataset.parse_did(d)[1]])
    df["cluster_of_task"] = df["task"].map(lambda t: t.split("/")[0])
    df["is_insider"] = df["cluster_of_demo"] == df["cluster_of_task"]
    df["functional"] = "plain"
    df["ensemble"] = a.ens

    out = os.path.join(ROOT, f"phase2/results/per_task_influence_{a.ens}.parquet")
    df.to_parquet(out, index=False)
    pm.to_parquet(out.replace(".parquet", "_per_member.parquet"), index=False)
    print(f"[p2attr] wrote {out}: {len(df)} rows "
          f"(3 attributors x 27 tasks x 135 demos = {3*27*135}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
