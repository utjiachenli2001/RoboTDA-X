"""B9 -- why `embed` and `block_00`? Structured subspaces of MATCHED size.

B6 established that k* tracks WHICH subspace Phi is restricted to, not its dimension, and left
the obvious question open: what property of a subspace makes its Gram carry demo-to-demo
structure? Two candidate axes, confounded in B1's five hand-picked groups:

  DEPTH     early layers (block_00) had k* = 6, the last block k* = 1. Is "early" the predictor?
  SIDE      `embed` (input projections) had the highest k* = 9. Is "input-side" the predictor?

B9 decouples them on the SAME cached full-width Phi (one column-slice per subspace, zero GPU
beyond the pass B1 already did). Three structured families, each a set of subspaces that share a
role but differ in position:

  per-block      block_00 .. block_05 in full -- the depth axis at fixed width (~3.15M each).
  within-block   for each block, attention (attn.in_proj + out_proj) vs MLP -- the two
                 sublayers, matched-ish, to see if the structure lives in mixing or in the FFN.
  side           input-side (state_proj + lang_proj + pos) vs output-side (head) vs the first
                 vs last LayerNorm -- the SIDE axis at small width.
  random-matched a random subset the SAME size as each structured one, as the null: if a
                 structured subspace has k* > its size-matched random control, its structure is
                 not merely a dimension effect.

For each subspace we report params, k* (parallel analysis), and best-k LDS ratio on C1/C5, so
the question "does k* predict where inverting helps" is answered per subspace rather than in
aggregate.
"""
from __future__ import annotations

import glob
import json
import os
import re
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
CACHE = os.path.join(HERE, "runs", "b9_structured.npz")
KS = (0, 1, 2, 3, 5, 10, 20)


def structured_groups(model):
    """{name: ([param names], role, depth)} -- structured subspaces on the BC transformer."""
    named = [n for n, p in model.named_parameters() if p.requires_grad]
    groups = {}

    def add(name, pred, role, depth):
        sel = [n for n in named if pred(n)]
        if sel:
            groups[name] = (sel, role, depth)

    # depth axis: each block in full
    for b in range(6):
        add(f"block_{b:02d}", lambda n, b=b: n.startswith(f"blocks.{b}."), "block", b)
    # within-block: attention vs MLP, per block
    for b in range(6):
        add(f"blk{b:02d}_attn", lambda n, b=b: n.startswith(f"blocks.{b}.attn"),
            "attn", b)
        add(f"blk{b:02d}_mlp", lambda n, b=b: n.startswith(f"blocks.{b}.mlp"), "mlp", b)
    # side axis
    add("input_proj", lambda n: n.startswith(("state_proj", "lang_proj", "pos")), "input", -1)
    add("state_proj_only", lambda n: n.startswith("state_proj"), "input", -1)
    add("lang_proj_only", lambda n: n.startswith("lang_proj"), "input", -1)
    add("output_head", lambda n: n.startswith("head"), "output", 99)
    add("first_ln", lambda n: n.startswith("blocks.0.ln1"), "norm", 0)
    add("last_ln", lambda n: n == "ln_f.weight" or n.startswith("ln_f"), "norm", 99)
    return groups


def build_cache(force=False, device="cuda", n_rand_per=2, seed=0):
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        idx = json.loads(str(z["index"]))
        ens = {k: {"G": z[f"{i}_G"], "K": z[f"{i}_K"], "members": z["members"],
                   "train_ids": z["train_ids"], "targets": z["targets"]}
               for i, k in enumerate(idx["keys"])}
        return idx, ens
    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    clusters = dataset.clusters()

    meta = {}          # name -> (params, role, depth)
    per = {}
    t0 = time.time()
    for mi, m in enumerate(members):
        model = AT.load_ckpt_model(os.path.join(GR.REGEN, m, "final.pt"))
        groups = structured_groups(model)
        spans, o = {}, 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                spans[n] = (o, o + p.numel()); o += p.numel()
        ptot = o
        tg, order = AT.build_targets(model, hbank, "base", [(c, "plain") for c in clusters])
        TG = torch.stack([tg[k] for k in order])
        PHI = torch.empty((len(train_ids), ptot), dtype=torch.float32, device=device)
        for i, d in enumerate(train_ids):
            PHI[i] = AT.demo_gradient(model, tbank, slices[d])

        # structured subspaces
        plan = {}
        for name, (names, role, depth) in groups.items():
            idx = np.concatenate([np.arange(*spans[n]) for n in names])
            plan[name] = (idx, role, depth, len(idx))
        # size-matched random controls (same across members: seed by size)
        for name, (idx, role, depth, sz) in list(plan.items()):
            for r in range(n_rand_per):
                g = np.random.default_rng(seed * 100003 + sz + r)
                plan[f"rand_{name}_{r}"] = (g.choice(ptot, size=sz, replace=False),
                                            "random", depth, sz)
        for name, (idx, role, depth, sz) in plan.items():
            ii = torch.as_tensor(idx, device=device, dtype=torch.long)
            P, T = PHI[:, ii], TG[:, ii]
            per.setdefault(name, {"G": [], "K": []})
            per[name]["G"].append((P @ P.T).double().cpu().numpy())
            per[name]["K"].append((P @ T.T).double().cpu().numpy())
            meta[name] = (int(sz), role, int(depth))
            del P, T
        print(f"[b9] {m}: {len(plan)} subspaces ({time.time()-t0:.0f}s)", flush=True)
        del model, PHI, TG, tg
        torch.cuda.empty_cache()

    keys = list(per)
    store = {}
    for i, k in enumerate(keys):
        store[f"{i}_G"] = np.stack(per[k]["G"]); store[f"{i}_K"] = np.stack(per[k]["K"])
    idx = {"keys": keys, "meta": meta}
    np.savez_compressed(CACHE, index=json.dumps(idx), members=np.array(members),
                        train_ids=np.array(train_ids), targets=np.array([c for c, _ in order]),
                        **store)
    ens = {k: {"G": np.stack(per[k]["G"]), "K": np.stack(per[k]["K"]),
               "members": np.array(members), "train_ids": np.array(train_ids),
               "targets": np.array([c for c, _ in order])} for k in keys}
    return idx, ens


def evaluate(idx, ens, tier="bc_s10", targets=("C1", "C5"), n_perm=100):
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    meta = idx["meta"]
    rows = []
    for name, Z in ens.items():
        ns = SP.spectrum_null(Z, n_perm=n_perm, seed=0)
        tids = list(Z["train_ids"])
        base = scores_graddot(Z, normalize_per_member=True)
        rec = {"subspace": name, "params": meta[name][0], "role": meta[name][1],
               "depth": meta[name][2], "k_star": ns["k_star_median"]}
        for t in targets:
            rho0 = demo_grain_lds(base[t], gm, obs[t])[0]
            c = float(ceil[t])
            best = rho0
            for k in KS:
                if k == 0:
                    continue
                S = SP.truncated_if(Z, k, normalize="dmean")
                sc = {tids[i]: float(S[i, list(Z["targets"]).index(t)])
                      for i in range(len(tids))}
                r = demo_grain_lds(sc, gm, obs[t])[0]
                if np.isfinite(r) and r > best:
                    best = r
            rec[f"{t}_ratio_k0"] = rho0 / c
            rec[f"{t}_ratio_best"] = best / c
            rec[f"{t}_gain"] = (best - rho0) / c
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    idx, ens = build_cache(force=a.force)
    df = evaluate(idx, ens)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b9_structured.csv"), index=False)

    print("=" * 108)
    print("B9 -- structured subspaces of Phi (regenerated E=5): does k* track ROLE, DEPTH, or size?")
    print("=" * 108)
    struct = df[df.role != "random"].copy()
    rand = df[df.role == "random"].copy()

    print("\n--- k* by role (structured only) ---")
    print(struct.groupby("role").agg(n=("subspace", "size"), params_med=("params", "median"),
                                     k_star_med=("k_star", "median"),
                                     k_star_max=("k_star", "max")).round(0).to_string())

    print("\n--- depth axis: the six blocks in full (matched ~3.15M each) ---")
    blk = struct[struct.role == "block"].sort_values("depth")
    print(blk[["subspace", "params", "k_star", "C1_ratio_best", "C5_ratio_best",
               "C1_gain", "C5_gain"]].round(3).to_string(index=False))

    print("\n--- structured vs size-matched RANDOM control (k*) ---")
    for _, r in struct.sort_values("k_star", ascending=False).head(12).iterrows():
        ctrl = rand[rand.subspace.str.startswith(f"rand_{r.subspace}_")]
        cm = ctrl.k_star.mean() if len(ctrl) else np.nan
        flag = "  <-- structure beyond size" if r.k_star > cm + 0.5 else ""
        print(f"  {r.subspace:18s} n={r.params:>9d} role={r.role:7s} "
              f"k*={r.k_star:2d}  random-matched k*={cm:.1f}{flag}")

    print("\n--- does k* predict where inverting helps? (structured subspaces) ---")
    for t in ("C1", "C5"):
        hi = struct[struct.k_star >= 3]
        lo = struct[struct.k_star <= 1]
        print(f"  {t}: mean gain-from-inverting  k*>=3: {hi[f'{t}_gain'].mean():+.3f}  "
              f"k*<=1: {lo[f'{t}_gain'].mean():+.3f}")


if __name__ == "__main__":
    main()
