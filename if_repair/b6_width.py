"""B6 -- the Phi-width curve: turn the 5-group k* observation into a controlled curve.

B1 observed that k* (the number of Gram eigendirections above a permutation null) rises from 1
at full width to 6-9 when Phi is restricted to `block_00` or `embed`, and that only in that
regime does inverting beat not inverting. That is the mechanism claim of the whole project, and
it currently rests on five hand-picked parameter groups of different sizes AND different roles.
Width and role are confounded: `embed` is both the smallest group and the only input-side one.

This decouples them. Phi is restricted to RANDOM parameter subsets of increasing size, several
independent draws per size. A random subset has no architectural role at all, so if k* still
falls as the subset grows, the driver is dimension -- p/N -- and not which layer was chosen.

Cheap: Phi is computed once per member at full width (10.4 GB on the GPU) and every subset is a
column slice of it.
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
CACHE = os.path.join(HERE, "runs", "b6_width_cache.npz")

WIDTHS = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000)
N_DRAWS = 3
KS = (0, 1, 2, 3, 5, 10, 20, 50, 135)


def build_cache(force=False, device="cuda", seed=0):
    """-> {(width, draw): {G,K}} over the regenerated ensemble, plus the full-width cell."""
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        idx = json.loads(str(z["index"]))
        return idx, {tuple(k): {"G": z[f"{i}_G"], "K": z[f"{i}_K"],
                                "members": z["members"], "train_ids": z["train_ids"],
                                "targets": z["targets"]}
                     for i, k in enumerate(idx["keys"])}
    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    clusters = dataset.clusters()

    per, ptot, tgts = {}, None, None
    t0 = time.time()
    for m in members:
        model = AT.load_ckpt_model(os.path.join(GR.REGEN, m, "final.pt"))
        tg, order = AT.build_targets(model, hbank, "base", [(c, "plain") for c in clusters])
        TG = torch.stack([tg[k] for k in order])
        tgts = [c for c, _ in order]
        ptot = TG.shape[1]
        PHI = torch.empty((len(train_ids), ptot), dtype=torch.float32, device=device)
        for i, d in enumerate(train_ids):
            PHI[i] = AT.demo_gradient(model, tbank, slices[d])
        # the SAME random subsets for every member: a subset must name the same coordinates
        # across members or the ensemble average is over different estimators.
        rng = np.random.default_rng(seed)
        cells = [(ptot, 0)] + [(w, d) for w in WIDTHS for d in range(N_DRAWS)]
        for (w, dr) in cells:
            if w >= ptot:
                P, T = PHI, TG
            else:
                g = np.random.default_rng(seed * 1000 + w + dr)
                sel = torch.as_tensor(g.choice(ptot, size=w, replace=False),
                                      device=device, dtype=torch.long)
                P, T = PHI[:, sel], TG[:, sel]
            per.setdefault((w, dr), {"G": [], "K": []})
            per[(w, dr)]["G"].append((P @ P.T).double().cpu().numpy())
            per[(w, dr)]["K"].append((P @ T.T).double().cpu().numpy())
            if w < ptot:
                del P, T
        del model, PHI, TG, tg
        torch.cuda.empty_cache()
        print(f"[b6] {m}: {len(cells)} widths ({time.time()-t0:.0f}s)", flush=True)
    keys = list(per)
    store = {}
    for i, k in enumerate(keys):
        store[f"{i}_G"] = np.stack(per[k]["G"])
        store[f"{i}_K"] = np.stack(per[k]["K"])
    index = {"keys": [list(k) for k in keys], "p_total": int(ptot)}
    np.savez_compressed(CACHE, index=json.dumps(index), members=np.array(members),
                        train_ids=np.array(train_ids), targets=np.array(tgts), **store)
    return index, {k: {"G": np.stack(per[k]["G"]), "K": np.stack(per[k]["K"]),
                       "members": np.array(members), "train_ids": np.array(train_ids),
                       "targets": np.array(tgts)} for k in keys}


def curve(index, ens, tier="bc_s10", targets=("C1", "C5"), n_perm=100):
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    rows = []
    for (w, dr), Z in sorted(ens.items()):
        ns = SP.spectrum_null(Z, n_perm=n_perm, seed=0)
        tids = list(Z["train_ids"])
        base = scores_graddot(Z, normalize_per_member=True)
        for t in targets:
            rho0, p0, n0, _, _ = demo_grain_lds(base[t], gm, obs[t])
            c = float(ceil[t])
            best = (rho0, 0)
            for k in KS:
                if k == 0:
                    continue
                S = SP.truncated_if(Z, k, normalize="dmean")
                sc = {tids[i]: float(S[i, list(Z["targets"]).index(t)])
                      for i in range(len(tids))}
                rho, _, _, _, _ = demo_grain_lds(sc, gm, obs[t])
                if np.isfinite(rho) and rho > best[0]:
                    best = (rho, k)
            rows.append({"width": int(w), "draw": int(dr),
                         "frac_of_model": w / index["p_total"],
                         "k_star": ns["k_star_median"], "k_star_min": ns["k_star_min"],
                         "k_star_max": ns["k_star_max"], "target": t,
                         "lds_k0": float(rho0), "ratio_k0": float(rho0) / c,
                         "lds_best": float(best[0]), "ratio_best": float(best[0]) / c,
                         "best_k": int(best[1]), "ceiling": c,
                         "gain_from_inverting": float(best[0] - rho0)})
    return pd.DataFrame(rows)


def kstar_compare(index, ens, n_perm=100, seed=0):
    """The decisive table: architectural groups vs RANDOM subsets, same code path.

    B1 read k* = 6-9 off `block_00` and `embed` and concluded that Phi was too wide for 135
    demos. That conclusion needs random subsets to be legitimate, because width and role are
    confounded in any group chosen by architecture. This puts both on one axis.
    """
    from if_repair import b1_layerwise as B1
    import glob as _glob
    members = sorted(os.path.basename(d) for d in
                     _glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    arch = B1.build_ensemble(members)
    rows = []
    for g in ("ALL", "head", "embed", "block_00", "last_block"):
        if g not in arch:
            continue
        ns = SP.spectrum_null(arch[g], n_perm=n_perm, seed=seed)
        rows.append({"kind": "architectural", "group": g, "draw": 0,
                     "k_star": ns["k_star_median"],
                     "per_member": str(ns["k_star_per_member"])})
    for (w, dr), Z in sorted(ens.items()):
        ns = SP.spectrum_null(Z, n_perm=n_perm, seed=seed)
        rows.append({"kind": "random", "group": f"random_{w}", "draw": dr,
                     "n_params": int(w), "k_star": ns["k_star_median"],
                     "per_member": str(ns["k_star_per_member"])})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kstar_compare", action="store_true",
                    help="emit the architectural-vs-random k* table (BLOCKERS #10)")
    a = ap.parse_args()
    index, ens = build_cache(force=a.force)
    if a.kstar_compare:
        kc = kstar_compare(index, ens)
        os.makedirs(RESULTS, exist_ok=True)
        kc.to_csv(os.path.join(RESULTS, "b6_kstar_compare.csv"), index=False)
        print("=" * 90)
        print("k* : ARCHITECTURAL groups vs RANDOM subsets of Phi (same spectrum_null settings)")
        print("=" * 90)
        print(kc.to_string(index=False))
        print("\nThe test: block_00 and last_block have IDENTICAL dimension (3,152,384).")
        print("If k* were a function of p/N they could not differ.")
        return
    df = curve(index, ens)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b6_width.csv"), index=False)
    print("=" * 100)
    print("B6 -- k* and the value of inverting, vs RANDOM Phi width (regenerated E=5)")
    print("=" * 100)
    agg = (df.groupby(["width", "target"])
             .agg(frac=("frac_of_model", "first"), k_star=("k_star", "mean"),
                  k_star_sd=("k_star", "std"), ratio_k0=("ratio_k0", "mean"),
                  ratio_best=("ratio_best", "mean"), best_k=("best_k", "median"),
                  gain=("gain_from_inverting", "mean"))
             .reset_index())
    for t in ("C1", "C5"):
        print(f"\n--- {t} ---")
        s = agg[agg.target == t].sort_values("width")
        print(s[["width", "frac", "k_star", "k_star_sd", "ratio_k0", "ratio_best",
                 "best_k", "gain"]].round(4).to_string(index=False))
    print("\nSpearman(width, k*) over all cells: "
          f"{df.groupby('width').k_star.mean().reset_index().corr(method='spearman').iloc[0,1]:+.3f}")
    k = df.groupby("width").agg(k_star=("k_star", "mean"),
                                gain=("gain_from_inverting", "mean"))
    print("Spearman(k*, gain from inverting) across widths: "
          f"{k.corr(method='spearman').iloc[0,1]:+.3f}")


if __name__ == "__main__":
    main()
