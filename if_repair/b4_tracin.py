"""B4 -- TracIn: checkpoint DENSITY, learning-rate weighting, and parameter GRAIN.

TracIn integrates the influence along the training trajectory instead of reading it off the
final weights:

    score_i = sum_c eta_c <g_i(theta_c), g_test(theta_c)>

The repo's own attribution.py uses all 5 evenly spaced checkpoints with the LR in force at each
one. Nothing has ever varied that choice, yet it is the only estimator here with a knob that is
neither a preconditioner nor a normalisation -- and TracIn is the DIFFUSION winner (0.479 vs
GradDot 0.414) while losing on BC, so the trajectory term is exactly where the policy-class
flip could live.

Three questions, one cache:
  density  -- 1, 2, 3, 5 checkpoints. Does averaging over the trajectory help, or is the final
              checkpoint all there is? (1 checkpoint + no LR weight == GradDot, so the density
              sweep contains the baseline as an endpoint and the comparison is exact.)
  weighting-- eta_c = LR at that step (the repo's choice) vs uniform. The LR schedule is cosine,
              so LR weighting downweights the late checkpoints by ~10x; that is a strong prior
              that early training carries the signal, and it has never been tested.
  grain    -- ALL parameters vs the action head. B1 found the head (0.2% of parameters) carries
              essentially all the C1 signal at the final checkpoint; if that holds along the
              trajectory the two knobs are independent.

Per (member, checkpoint) we store (G, K) for every parameter group -- 25 gradient passes total.
Every density/weighting/grain combination is then a reweighted sum of cached matrices, so the
whole sweep costs one pass, not one pass per cell.
"""
from __future__ import annotations

import glob
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
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "runs", "b4_tracin_cache.npz")
GROUPS = ("ALL", "head", "embed", "block_00", "last_block")


def build_cache(groups=GROUPS, device="cuda", force=False):
    """{(member, ckpt): {group: (G, K)}} + eta, step. One Phi pass per (member, checkpoint)."""
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return json.loads(str(z["index"])), z
    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    clusters = dataset.clusters()
    tgt = [(c, "plain") for c in clusters]

    store, index = {}, {"members": members, "groups": list(groups), "ckpts": [], "eta": [],
                        "step": []}
    t0 = time.time()
    for mi, m in enumerate(members):
        run = os.path.join(GR.REGEN, m)
        meta = json.load(open(os.path.join(run, "train_meta.json")))
        cfg = meta["cfg"]
        cks = list(meta["ckpts"])
        if mi == 0:
            index["ckpts"] = cks
        for ci, ck in enumerate(cks):
            path = os.path.join(run, ck)
            step = int(torch.load(path, map_location="cpu", weights_only=False)["step"])
            eta = float(AT.lr_at_step(cfg, step))
            model = AT.load_ckpt_model(path)
            gmap = GR.param_groups(model)
            spans, o = {}, 0
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                spans[n] = (o, o + p.numel())
                o += p.numel()
            tg, order = AT.build_targets(model, hbank, "base", tgt)
            TG = torch.stack([tg[k] for k in order])
            PHI = torch.empty((len(train_ids), TG.shape[1]), dtype=torch.float32,
                              device=device)
            for i, d in enumerate(train_ids):
                PHI[i] = AT.demo_gradient(model, tbank, slices[d])
            for g in groups:
                if g == "ALL":
                    P, T = PHI, TG
                else:
                    idx = torch.as_tensor(
                        np.concatenate([np.arange(*spans[n]) for n in gmap[g]]),
                        device=device, dtype=torch.long)
                    P, T = PHI[:, idx], TG[:, idx]
                store[f"{g}|{mi}|{ci}|G"] = (P @ P.T).double().cpu().numpy()
                store[f"{g}|{mi}|{ci}|K"] = (P @ T.T).double().cpu().numpy()
            if mi == 0:
                index["eta"].append(eta)
                index["step"].append(step)
            else:
                index["eta"][ci] = eta          # identical schedule across members
            del model, PHI, TG, tg
            torch.cuda.empty_cache()
            print(f"[b4] {m} {ck} step={step} eta={eta:.3e} ({time.time()-t0:.0f}s)",
                  flush=True)
    index["train_ids"] = list(train_ids)
    index["targets"] = [c for c, _ in [(c, "plain") for c in clusters]]
    np.savez_compressed(CACHE, index=json.dumps(index), **store)
    return index, np.load(CACHE, allow_pickle=True)


def tracin_Z(index, z, group, ckpt_idx, lr_weight=True):
    """Build a Z-dict whose K is the TracIn sum over the selected checkpoints.

    G is taken at the LAST selected checkpoint: it only sets the per-member normalisation
    (`dmean`) and the preconditioner for the truncated variants, and mixing Grams from
    different points on the trajectory would not be a Gram of anything.
    """
    M = len(index["members"])
    Gs, Ks = [], []
    for mi in range(M):
        acc = None
        for ci in ckpt_idx:
            w = index["eta"][ci] if lr_weight else 1.0
            Kc = z[f"{group}|{mi}|{ci}|K"] * w
            acc = Kc if acc is None else acc + Kc
        Gs.append(z[f"{group}|{mi}|{ckpt_idx[-1]}|G"])
        Ks.append(acc)
    return {"G": np.stack(Gs), "K": np.stack(Ks),
            "members": np.array(index["members"]),
            "train_ids": np.array(index["train_ids"]),
            "targets": np.array(index["targets"])}


def sweep(index, z, tier="bc_s10", targets=("C1", "C5"), groups=GROUPS):
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    nck = len(index["ckpts"])
    # density: the last d checkpoints (d=1 is the final weights = GradDot's vantage point)
    plans = {f"last{d}": list(range(nck - d, nck)) for d in (1, 2, 3, nck)}
    plans["evenly2"] = [0, nck - 1]
    plans["evenly3"] = [0, nck // 2, nck - 1]
    rows = []
    for group in groups:
        for pname, ck in plans.items():
            for lrw in (True, False):
                if len(ck) == 1 and lrw:
                    continue                 # a single checkpoint's eta is a global scale
                Z = tracin_Z(index, z, group, ck, lr_weight=lrw)
                tids = list(Z["train_ids"])
                variants = {"TracIn": scores_graddot(Z, normalize_per_member=True)}
                for k in (1, 5):
                    S = SP.truncated_if(Z, k, normalize="dmean")
                    variants[f"TracIn_trunc_k{k}"] = {
                        tg: {tids[i]: float(S[i, j]) for i in range(len(tids))}
                        for j, tg in enumerate(list(Z["targets"]))}
                for vname, sc in variants.items():
                    for t in targets:
                        rho, p, n, _, _ = demo_grain_lds(sc[t], gm, obs[t])
                        c = float(ceil[t])
                        rows.append({"group": group, "density": pname,
                                     "n_ckpt": len(ck), "lr_weighted": lrw,
                                     "estimator": vname, "target": t,
                                     "lds": float(rho), "ceiling": c,
                                     "ratio": float(rho) / c, "p": float(p), "n": n,
                                     "passed": bool(np.isfinite(rho) and rho >= 0.5 * c
                                                    and p < ALPHA)})
    return pd.DataFrame(rows)


def single_checkpoints(index, z, tier="bc_s10", targets=("C1", "C5"), groups=("ALL", "head")):
    """Score each checkpoint ALONE, to ask whether TracIn's gain is integration or location.

    If one checkpoint matches the multi-checkpoint sum, the density knob is not doing the work:
    the finding is that some point in the middle of training attributes better than the end.
    The spread ACROSS single checkpoints also bounds how much of any "gain" is measurement
    noise at n = 24 masks.
    """
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    nck = len(index["ckpts"])
    rows = []
    for group in groups:
        plans = [(f"single_ckpt_{i}", [i], False) for i in range(nck)]
        plans += [("last5_lr", list(range(nck)), True),
                  ("last5_unweighted", list(range(nck)), False)]
        for label, ck, lrw in plans:
            Z = tracin_Z(index, z, group, ck, lr_weight=lrw)
            sc = scores_graddot(Z, normalize_per_member=True)
            for t in targets:
                rho, p, n, _, _ = demo_grain_lds(sc[t], gm, obs[t])
                c = float(ceil[t])
                rows.append({"group": group, "cell": label,
                             "step": index["step"][ck[0]] if len(ck) == 1 else None,
                             "target": t, "lds": float(rho), "ratio": float(rho) / c,
                             "p": float(p)})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--decompose", action="store_true",
                    help="score each checkpoint alone instead of running the full sweep")
    a = ap.parse_args()
    index, z = build_cache(force=a.force)
    if a.decompose:
        sc = single_checkpoints(index, z)
        os.makedirs(RESULTS, exist_ok=True)
        sc.to_csv(os.path.join(RESULTS, "b4_single_ckpt.csv"), index=False)
        print("=" * 90)
        print("B4 -- is the TracIn gain INTEGRATION, or one better checkpoint?")
        print("=" * 90)
        print(sc.pivot_table(index=["group", "cell"], columns="target", values="ratio",
                             sort=False).round(3).to_string())
        return
    df = sweep(index, z)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b4_tracin.csv"), index=False)

    print("=" * 100)
    print("B4 -- TracIn density x LR weighting x grain (regenerated E=5), demo grain n=24")
    print("=" * 100)
    print(f"checkpoint steps {index['step']}  eta {[f'{e:.2e}' for e in index['eta']]}")
    for t in ("C1", "C5"):
        sub = df[(df.target == t) & (df.estimator == "TracIn")]
        print(f"\n--- {t}: ratio-to-ceiling, plain TracIn ---")
        print(sub.pivot_table(index=["density", "lr_weighted"], columns="group",
                              values="ratio").round(3).to_string())
    best = df.loc[df.groupby("target").ratio.idxmax()]
    print("\nbest cell per target:")
    print(best[["target", "group", "density", "lr_weighted", "estimator", "lds", "ratio",
                "p", "passed"]].to_string(index=False))
    print("\nidentity check -- last1/no-LR must equal GradDot_dmean on the same ensemble:")
    for g in ("ALL", "head"):
        r = df[(df.group == g) & (df.density == "last1") & (~df.lr_weighted)
               & (df.estimator == "TracIn")]
        print(f"  {g:10s} " + "  ".join(f"{x.target}={x.ratio:.3f}" for x in r.itertuples()))


if __name__ == "__main__":
    main()
