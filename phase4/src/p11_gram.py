"""P11 attribution input -- extend the Gram cache to ensemble members 211-220 (E=10 -> E=20).

WHY THIS IS NEEDED. P11's CHAMPION is TracIn, and its per-member scores already exist for all 20
members (results/influence_table_per_member.parquet for 201-210; phase3/results/
p9_influence_new_members.parquet for 211-220, functional 'plain', all 9 targets, 135 demos). So
the champion costs ZERO GPU.

P11's preregistered SECONDARY estimator -- the exact lambda -> infinity limit, i.e. the RAW
gradient dot product K_m[i,j] = <g_demo_i, g_target_j> -- is NOT recoverable from the archived
IF/TRAK scores (both are (G + cI)^-1 K solves; inverting them back needs G, which was never
stored for these members). phase3/results/p6_gram_cache.npz holds G and K for members 201-210
only. This script computes G and K for members 211-220: ONE gradient pass, 0 retrains, ~0.5 GPU-h.

GUARDS
  * p4lib.assert_no_probe_leak() on every member BEFORE any gradient is taken (P6.2).
  * SELF-VALIDATION: from the new (G, K) we recompute TRAK and IF at the FROZEN default ridge and
    require them to reproduce the ARCHIVED per-member scores in p9_influence_new_members.parquet
    -- which were produced by a different script (phase3/src/p9_attr.py) on a different day. If
    the Gram cache did not reproduce them, this cache would be a different estimator wearing the
    same name. Spearman must be 1.000000 and the max relative difference must be tiny.

Nothing here is a verdict. This script produces an INPUT.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, P3_RESULTS, P3_RUNS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import attribution as AT  # noqa: E402
from lds import spearman  # noqa: E402

RIDGE = 1e-2                      # the FROZEN Phase-1 default -- NOT retuned (P11 is not a sweep)
OUT = os.path.join(P4_RESULTS, "p11_gram_cache_new_members.npz")
VALID = os.path.join(P4_RESULTS, "p11_gram_selfvalidation.json")


def main():
    L.assert_prereg_locked()
    if os.path.exists(OUT) and "--force" not in sys.argv:
        print(f"[P11-gram] cache exists -> {OUT} (use --force to rebuild)")
        return

    runs = sorted(glob.glob(os.path.join(P3_RUNS, "P9", "ens_s*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "final.pt"))]
    assert len(runs) == 10, f"expected 10 new members (211-220), got {len(runs)}"
    print(f"[P11-gram] members: {[os.path.basename(r) for r in runs]}")

    # ---- P6.2 GUARD, before any gradient is taken
    heldout = set(dataset.heldout_pool()[0])
    g = L.assert_no_probe_leak(runs, heldout,
                               context="P11 Gram cache, test side = held-out pool")
    print(f"[P11-gram] probe-leak guard PASSED: {g}")

    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    N = len(train_ids)
    clusters = dataset.clusters()
    targets = [(c, "plain") for c in clusters]     # 'plain' == the held-out L2 outcome

    Gs, Ks, members = [], [], []
    t0 = time.time()
    for run in runs:
        m = os.path.basename(run)
        model = AT.load_ckpt_model(os.path.join(run, "final.pt"))
        tg, order = AT.build_targets(model, hbank, "base", targets)
        TG = torch.stack([tg[k] for k in order])                       # (T, p)
        PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device="cuda")
        for i, d in enumerate(train_ids):
            PHI[i] = AT.demo_gradient(model, tbank, slices[d])
        G = (PHI @ PHI.T).double().cpu().numpy()                       # (N,N)
        K = (PHI @ TG.T).double().cpu().numpy()                        # (N,T)
        Gs.append(G)
        Ks.append(K)
        members.append(m)
        del model, PHI, TG, tg
        torch.cuda.empty_cache()
        print(f"[P11-gram] {m} done ({time.time()-t0:.0f}s)", flush=True)

    G = np.stack(Gs)
    K = np.stack(Ks)
    np.savez(OUT, G=G, K=K, members=np.array(members),
             train_ids=np.array(train_ids), targets=np.array([c for c, _ in targets]))
    print(f"[P11-gram] wrote {OUT}: G{G.shape} K{K.shape}")

    # ---------------------------------------------------------------- SELF-VALIDATION
    arch = pd.read_parquet(os.path.join(P3_RESULTS, "p9_influence_new_members.parquet"))
    checks = []
    for mi, m in enumerate(members):
        Gm = torch.tensor(G[mi], dtype=torch.float64)
        Km = torch.tensor(K[mi], dtype=torch.float64)
        lam = RIDGE * float(torch.diagonal(Gm).mean())
        I = torch.eye(N, dtype=torch.float64)
        trak = torch.linalg.solve(Gm + lam * I, Km).numpy()
        inner = torch.linalg.solve(lam * N * I + Gm, Km)
        iff = ((Km - Gm @ inner) / lam).numpy()
        for attr, S in (("TRAK", trak), ("IF", iff)):
            for j, c in enumerate(clusters):
                a = arch[(arch.attributor == attr) & (arch.target == c) & (arch.member == m)]
                a = a.set_index("demo_id")["score"].reindex(train_ids).values
                b = S[:, j]
                rho = spearman(a, b)
                den = np.maximum(np.abs(a), 1e-30)
                mrd = float(np.max(np.abs(a - b) / den))
                checks.append({"member": m, "attributor": attr, "target": c,
                               "spearman_vs_archived": rho, "max_rel_diff": mrd,
                               "PASS": bool(np.isfinite(rho) and rho > 0.999999 and mrd < 1e-3)})

    n_pass = sum(c["PASS"] for c in checks)
    worst = max(checks, key=lambda c: c["max_rel_diff"])
    out = {
        "stage": "P11 Gram cache self-validation",
        "claim": ("TRAK and IF recomputed from the NEW Gram cache at the frozen default ridge "
                  "reproduce the ARCHIVED per-member scores from phase3/src/p9_attr.py -- a "
                  "different script, a different run. If they did not, this cache would be a "
                  "different estimator wearing the same name."),
        "ridge_rel": RIDGE, "n_checks": len(checks), "n_pass": n_pass,
        "ALL_PASS": n_pass == len(checks),
        "worst_max_rel_diff": worst["max_rel_diff"],
        "worst_cell": {k: worst[k] for k in ("member", "attributor", "target")},
        "min_spearman": min(c["spearman_vs_archived"] for c in checks),
        "probe_leak_guard": g,
        "checks": checks,
    }
    L.atomic_write_json(VALID, out)
    print(f"[P11-gram] SELF-VALIDATION: {n_pass}/{len(checks)} pass, "
          f"min Spearman={out['min_spearman']:.6f}, worst max-rel-diff={worst['max_rel_diff']:.2e}")
    if n_pass != len(checks):
        raise SystemExit("[P11-gram] SELF-VALIDATION FAILED -- refusing to emit an attribution "
                         "input that does not reproduce the archived scores.")


if __name__ == "__main__":
    main()
