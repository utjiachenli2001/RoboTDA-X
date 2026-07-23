"""Phase B -- recompute per-demo gradients from checkpoints (the part the cache collapsed).

The published repo ships NO weights (BLOCKERS #3), so Phase B was blocked. But training is
bit-deterministic (phase3/results/p8b_determinism*.json), every member's seed + demo list +
cfg are recorded in runs/stage_E/ens_s*/{train_meta,demos}.json, and a member trains in ~94 s
on the H200. So the checkpoints are REGENERABLE.

Whether the regenerated weights are the ORIGINAL ones is not assumed -- it is tested:
rebuild (G, K) for a regenerated member and compare against that member's slice of
phase3/results/p6_gram_cache.npz. If they agree, every Phase-B number is directly comparable
to the paper's cached-Gram results. If they do not, Phase B still runs but only against a
self-consistent new ensemble, and that limitation is reported.

Layer selection (B1) is done by name-prefix filter on named_parameters, so a per-block Gram
is the same code path with a different mask.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import attribution as AT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REGEN = os.path.join(HERE, "runs", "regen")


def param_groups(model):
    """-> {group_name: [param_name, ...]} for layer-block ablation (B1).

    Names look like `blocks.0.attn.in_proj_weight`, `state_proj.weight`, `pos`, `head.*`.
    """
    import re
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    groups = {"ALL": names}
    for n in names:
        m = re.match(r"^blocks\.(\d+)\.", n)
        if m:
            groups.setdefault(f"block_{int(m.group(1)):02d}", []).append(n)
        elif n.startswith(("pos", "state_proj", "lang_proj", "embed")):
            groups.setdefault("embed", []).append(n)
        else:
            groups.setdefault("head", []).append(n)
    # useful composites
    blk = sorted(k for k in groups if k.startswith("block_"))
    if blk:
        groups["last_block"] = groups[blk[-1]]
        groups["head_plus_last_block"] = groups.get("head", []) + groups[blk[-1]]
    return groups


def build_gram(run_dir, ckpt="final.pt", keep=None, device="cuda", verbose=True):
    """-> (G, K, train_ids, targets). `keep`: list of param names to restrict Phi to."""
    model = AT.load_ckpt_model(os.path.join(run_dir, ckpt))
    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    clusters = dataset.clusters()
    targets = [(c, "plain") for c in clusters]

    keepset = None if keep is None else set(keep)
    idx = None
    if keepset is not None:
        offs, sel, o = [], [], 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            k = p.numel()
            if n in keepset:
                sel.append((o, o + k))
            o += k
        idx = np.concatenate([np.arange(a, b) for a, b in sel]) if sel else np.array([], int)
        idx = torch.as_tensor(idx, device=device, dtype=torch.long)

    tg, order = AT.build_targets(model, hbank, "base", targets)
    TG = torch.stack([tg[k] for k in order])
    if idx is not None:
        TG = TG[:, idx]
    N = len(train_ids)
    PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device=device)
    t0 = time.time()
    for i, d in enumerate(train_ids):
        g = AT.demo_gradient(model, tbank, slices[d])
        PHI[i] = g[idx] if idx is not None else g
    G = (PHI @ PHI.T).double().cpu().numpy()
    K = (PHI @ TG.T).double().cpu().numpy()
    if verbose:
        print(f"[gram] {os.path.basename(run_dir)}/{ckpt} p={TG.shape[1]} "
              f"({time.time()-t0:.0f}s)", flush=True)
    del model, PHI, TG
    torch.cuda.empty_cache()
    return G, K, train_ids, [c for c, _ in order]


def compare_to_cache(G, K, member="ens_s201"):
    """Is the regenerated member bit-comparable to the cached Gram slice?"""
    Z = np.load(D.P6, allow_pickle=True)
    m = list(Z["members"]).index(member)
    Gc, Kc = Z["G"][m], Z["K"][m]
    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
    from scipy.stats import spearmanr
    return {
        "member": member,
        "G_rel_fro": rel(G, Gc), "K_rel_fro": rel(K, Kc),
        "K_spearman_per_target": [float(spearmanr(K[:, j], Kc[:, j]).statistic)
                                  for j in range(K.shape[1])],
        "diag_ratio_median": float(np.median(np.diag(G) / np.diag(Gc))),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="ens_s201")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    run = os.path.join(REGEN, a.member)
    G, K, tids, tgts = build_gram(run)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if a.verify:
        r = compare_to_cache(G, K, a.member)
        print(json.dumps(r, indent=1))
        with open(os.path.join(HERE, "results", f"regen_verify_{a.member}.json"), "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
