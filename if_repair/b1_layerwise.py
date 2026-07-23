"""B1 -- layerwise / last-layer influence.

The cached Gram collapses all 19.2M parameters into one inner product, so it cannot say
WHERE the influence signal lives. Phase A found k* ~ 1: with N=135 demos the full-parameter
Gram carries about one eigendirection distinguishable from noise. The direct implication is
that Phi is far too high-dimensional for 135 demos to constrain -- so restricting Phi to a
small block (the action head is 39.5K params, 0.2% of the model) is the natural test.

Efficiency: Phi is computed ONCE per member at full width and then sliced per group, so all
11 groups cost one gradient pass, not eleven.

Ensemble caveat (BLOCKERS #6): these members are REGENERATED, not the originals -- training
is deterministic within an environment but not across torch/CUDA/GPU versions. So every
baseline in this file is recomputed on the SAME regenerated ensemble; nothing here is
compared against the cached-Gram numbers.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import gradients as GR  # noqa: E402
from if_repair import spectral as SP  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import attribution as AT  # noqa: E402
from p6_lambda_sweep import demo_grain_lds, ALPHA  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def member_grams(run_dir, device="cuda"):
    """One gradient pass -> {group: (G, K)} for every parameter group."""
    model = AT.load_ckpt_model(os.path.join(run_dir, "final.pt"))
    groups = GR.param_groups(model)
    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    clusters = dataset.clusters()
    tg, order = AT.build_targets(model, hbank, "base", [(c, "plain") for c in clusters])
    TG = torch.stack([tg[k] for k in order])

    # flat index ranges per parameter name, in named_parameters order
    spans, o = {}, 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        spans[n] = (o, o + p.numel())
        o += p.numel()

    N = len(train_ids)
    PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device=device)
    t0 = time.time()
    for i, d in enumerate(train_ids):
        PHI[i] = AT.demo_gradient(model, tbank, slices[d])
    out = {}
    for gname, names in groups.items():
        if gname == "ALL":
            idx = None
        else:
            idx = torch.as_tensor(
                np.concatenate([np.arange(*spans[n]) for n in names]),
                device=device, dtype=torch.long)
        P = PHI if idx is None else PHI[:, idx]
        T = TG if idx is None else TG[:, idx]
        out[gname] = ((P @ P.T).double().cpu().numpy(), (P @ T.T).double().cpu().numpy())
        del P, T
    print(f"[b1] {os.path.basename(run_dir)}: {len(out)} groups ({time.time()-t0:.0f}s)",
          flush=True)
    del model, PHI, TG
    torch.cuda.empty_cache()
    return out, train_ids, [c for c, _ in order]


def build_ensemble(members, device="cuda"):
    """-> {group: Z-dict with G,K stacked over members}."""
    per, tids, tgts = {}, None, None
    for m in members:
        g, tids, tgts = member_grams(os.path.join(GR.REGEN, m), device=device)
        for k, (G, K) in g.items():
            per.setdefault(k, {"G": [], "K": [], "members": []})
            per[k]["G"].append(G); per[k]["K"].append(K); per[k]["members"].append(m)
    return {k: {"G": np.stack(v["G"]), "K": np.stack(v["K"]),
                "members": np.array(v["members"]),
                "train_ids": np.array(tids), "targets": np.array(tgts)}
            for k, v in per.items()}


def evaluate_groups(ens, tier="bc_s10", targets=("C1", "C5"), ks=(0, 1, 2, 5, 10, 20)):
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    rows = []
    for gname, Z in ens.items():
        N = Z["K"].shape[1]
        variants = {"GradDot_dmean": scores_graddot(Z, normalize_per_member=True)}
        tids = list(Z["train_ids"])
        for k in list(ks) + [N]:
            S = SP.truncated_if(Z, k, normalize="dmean")
            variants[f"trunc_k{k}"] = {
                tgts: {tids[i]: float(S[i, j]) for i in range(len(tids))}
                for j, tgts in enumerate(list(Z["targets"]))}
        for vname, sc in variants.items():
            for t in targets:
                rho, p, n, _, _ = demo_grain_lds(sc[t], gm, obs[t])
                c = float(ceil[t])
                rows.append({"group": gname, "estimator": vname, "target": t,
                             "n_params_frac": None, "lds": float(rho), "ceiling": c,
                             "ratio": float(rho) / c, "p": float(p), "n": n,
                             "passed": bool(np.isfinite(rho) and rho >= 0.5 * c
                                            and p < ALPHA)})
    return pd.DataFrame(rows)


def main():
    members = sorted(os.path.basename(d) for d in
                     __import__("glob").glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    print(f"[b1] ensemble = {members}")
    ens = build_ensemble(members)
    df = evaluate_groups(ens)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "b1_layerwise.csv"), index=False)
    print("=" * 96)
    print(f"B1 -- LAYERWISE INFLUENCE (regenerated E={len(members)} ensemble), demo grain n=24")
    print("=" * 96)
    for t in ("C1", "C5"):
        print(f"\n--- {t}: ratio-to-ceiling by group x estimator ---")
        piv = df[df.target == t].pivot_table(index="group", columns="estimator",
                                             values="ratio")
        cols = [c for c in ["GradDot_dmean", "trunc_k0", "trunc_k1", "trunc_k2",
                            "trunc_k5", "trunc_k10", "trunc_k20", "trunc_k135"]
                if c in piv.columns]
        print(piv[cols].round(3).to_string())
    best = df.loc[df.groupby("target").ratio.idxmax()]
    print("\nbest group per target:")
    print(best[["target", "group", "estimator", "lds", "ratio", "p",
                "passed"]].to_string(index=False))
    # spectrum per group
    print("\nk* (parallel analysis, 100 perms) by group:")
    for gname in ["ALL", "head", "embed", "last_block", "block_00"]:
        if gname in ens:
            ns = SP.spectrum_null(ens[gname], n_perm=100, seed=0)
            print(f"  {gname:14s} k*_median={ns['k_star_median']:3d} "
                  f"(min {ns['k_star_min']}, max {ns['k_star_max']})")


if __name__ == "__main__":
    main()
