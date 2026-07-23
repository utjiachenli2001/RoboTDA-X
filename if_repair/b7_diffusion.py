"""B7 -- the same TracIn knobs on the DIFFUSION policy: a direct shot at win condition 2.

Win condition 2 (one estimator passing C1 in BOTH policy classes) has never been reached, and
the two arms disagree about which estimator is best: TracIn wins on diffusion (0.479 vs GradDot
0.414) and loses on BC. B4 has just found that TracIn's trajectory term DOES help on BC once
the density and the parameter grain are swept -- head-restricted Phi over all 5 checkpoints,
LR-weighted, C1 ratio 0.602 vs GradDot's 0.506.

That makes the unification question concrete and cheap: does the SAME configuration -- not
merely the same estimator name -- win on both classes? A knob that has to be re-tuned per policy
class is not a unification.

What this needs that the repo did not have: diffusion CHECKPOINTS. p17's cache holds only the
final-weights Gram, and TracIn needs the trajectory. Regenerating the 5 members costs 331 s
each (if_repair/regen_dp.sh), the same trick that unblocked B1.

Everything diffusion-specific is imported from the repo rather than reimplemented -- the
epsilon-prediction gradients and, critically, the FROZEN (t, eps) bank from p10_bank. Drawing
fresh noise per side would make the inner product measure the RNG instead of the data, which is
the one thing a diffusion TDA must not do.

Ensemble caveat, as everywhere in Phase B: these are REGENERATED members. `verify` compares the
rebuilt Gram against p17's cached slice, and every baseline is recomputed on the regenerated
ensemble.
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
from if_repair import spectral as SP  # noqa: E402

D.add_repo_paths()
sys.path.insert(0, os.path.join(D.ROOT, "phase3", "src"))
import dataset  # noqa: E402
import p10_attr as PA  # noqa: E402
import evaluate_diffusion as EVD  # noqa: E402
from diffusion_data import ChunkBank, heldout_chunk_bank  # noqa: E402
from p10_bank import load_bank  # noqa: E402
from p6_lambda_sweep import demo_grain_lds, ALPHA  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REGEN_DP = os.path.join(HERE, "runs", "regen_dp")
CACHE = os.path.join(HERE, "runs", "b7_diffusion_cache.npz")
TIER = "diff_s10"


# The architectures differ, so a BC group name cannot simply be reused. These are the closest
# ROLE analogues, which is what a cross-arm claim needs -- "the action head carries the signal"
# is a claim about the output layer, not about a string in a parameter name.
#   BC head (39.5K, 0.2%)        <-> act_out    (2.7K,  0.01%)
#   BC embed (268K, 1.4%)        <-> embed      (197K,  1.1%)  = state_proj + lang_proj
#   BC block_00 (3.15M, 16.4%)   <-> obs_blocks_00 (1.77M, 9.7%)
#   BC last_block (3.15M, 16.4%) <-> den_blocks_05 (1.77M, 9.7%)
DP_GROUPS = ("ALL", "act_out", "embed", "obs_blocks_00", "den_blocks_05")


def dp_param_groups(model):
    """{group: [param names]} for the diffusion net, by top-level module and block index."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    groups = {"ALL": names}
    for n in names:
        m = re.match(r"^([A-Za-z_0-9]+)\.(\d+)\.", n)
        groups.setdefault(n.split(".")[0], []).append(n)
        if m:
            groups.setdefault(f"{m.group(1)}_{int(m.group(2)):02d}", []).append(n)
    groups["embed"] = groups.get("state_proj", []) + groups.get("lang_proj", [])
    return {k: v for k, v in groups.items() if v}


def build_cache(force=False, device="cuda", groups=None):
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return json.loads(str(z["index"])), z
    runs = sorted(glob.glob(os.path.join(REGEN_DP, "dpens_s*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "final.pt"))]
    if len(runs) < 2:
        raise RuntimeError(f"need the regenerated diffusion ensemble; found {len(runs)}")
    tb, eb = load_bank(device)
    train_ids, _ = dataset.train_pool()
    cfg = json.load(open(os.path.join(runs[0], "train_meta.json")))["cfg"]
    tbank = ChunkBank(train_ids, H=cfg["h_chunk"])
    hbank = heldout_chunk_bank()
    slices = tbank.demo_slices()

    store = {"index": None}
    index = {"members": [os.path.basename(r) for r in runs], "ckpts": [], "eta": [],
             "step": [], "train_ids": list(train_ids), "targets": dataset.clusters()}
    t0 = time.time()
    gsel = None
    for mi, run in enumerate(runs):
        meta = json.load(open(os.path.join(run, "train_meta.json")))
        cks = list(meta["ckpts"])
        if mi == 0:
            index["ckpts"] = cks
        for ci, ck in enumerate(cks):
            path = os.path.join(run, ck)
            model = EVD.load_model(path)
            step = int(torch.load(path, map_location="cpu", weights_only=False)["step"])
            eta = float(PA._lr_at(meta["cfg"], step))
            if gsel is None:
                allg = dp_param_groups(model)
                gsel = [g for g in (groups or DP_GROUPS) if g in allg]
                index["groups"] = list(gsel)
                spans, o = {}, 0
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        spans[n] = (o, o + p.numel())
                        o += p.numel()
                index["p_total"] = o
                gnames = {g: allg[g] for g in gsel}
            tg, order = PA.build_targets(model, hbank, tb, eb, "denoise",
                                         cfg["n_ddim_steps"])
            TG = torch.stack([tg[c] for c in order])
            PHI = torch.empty((len(train_ids), TG.shape[1]), dtype=torch.float32,
                              device=device)
            for i, d in enumerate(train_ids):
                PHI[i] = PA.demo_gradient(model, tbank, slices[d], tb, eb)
            for g in gsel:
                if g == "ALL":
                    P, T = PHI, TG
                else:
                    idx = torch.as_tensor(
                        np.concatenate([np.arange(*spans[n]) for n in gnames[g]]),
                        device=device, dtype=torch.long)
                    P, T = PHI[:, idx], TG[:, idx]
                store[f"{g}|{mi}|{ci}|G"] = (P @ P.T).double().cpu().numpy()
                store[f"{g}|{mi}|{ci}|K"] = (P @ T.T).double().cpu().numpy()
            if mi == 0:
                index["eta"].append(eta)
                index["step"].append(step)
            del model, PHI, TG, tg
            torch.cuda.empty_cache()
            print(f"[b7] {os.path.basename(run)} {ck} step={step} eta={eta:.3e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    store.pop("index")
    np.savez_compressed(CACHE, index=json.dumps(index), **store)
    return index, np.load(CACHE, allow_pickle=True)


def verify_against_p17(index, z):
    """Is the regenerated diffusion Gram comparable to p17's cached slice? (BLOCKERS #6)"""
    from scipy.stats import spearmanr
    Zc = D.gram_diffusion()
    last = len(index["ckpts"]) - 1
    out = []
    for mi, m in enumerate(index["members"]):
        if m not in list(Zc["members"]):
            continue
        j = list(Zc["members"]).index(m)
        G = z[f"ALL|{mi}|{last}|G"]
        K = z[f"ALL|{mi}|{last}|K"]
        Gc, Kc = Zc["G"][j], Zc["K"][j]
        out.append({
            "member": m,
            "G_rel_fro": float(np.linalg.norm(G - Gc) / max(np.linalg.norm(Gc), 1e-300)),
            "K_rel_fro": float(np.linalg.norm(K - Kc) / max(np.linalg.norm(Kc), 1e-300)),
            "diag_ratio_median": float(np.median(np.diag(G) / np.diag(Gc))),
            "K_spearman_mean": float(np.mean([spearmanr(K[:, c], Kc[:, c]).statistic
                                              for c in range(K.shape[1])])),
        })
    return pd.DataFrame(out)


def tracin_Z(index, z, group, ckpt_idx, lr_weight=True):
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
    return {"G": np.stack(Gs), "K": np.stack(Ks), "members": np.array(index["members"]),
            "train_ids": np.array(index["train_ids"]),
            "targets": np.array(index["targets"])}


def sweep(index, z, targets=("C1", "C5")):
    gm, obs, ceil = D.demo_masks(), D.outcomes(TIER), D.ceilings(TIER)
    nck = len(index["ckpts"])
    plans = {f"last{d}": list(range(nck - d, nck)) for d in (1, 2, 3, nck)}
    plans["evenly3"] = [0, nck // 2, nck - 1]
    rows = []
    for group in index["groups"]:
        for pname, ck in plans.items():
            for lrw in (True, False):
                if len(ck) == 1 and lrw:
                    continue
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
                        rows.append({"group": group, "density": pname, "n_ckpt": len(ck),
                                     "lr_weighted": lrw, "estimator": vname, "target": t,
                                     "lds": float(rho), "ceiling": c,
                                     "ratio": float(rho) / c, "p": float(p), "n": n,
                                     "passed": bool(np.isfinite(rho) and rho >= 0.5 * c
                                                    and p < ALPHA)})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    index, z = build_cache(force=a.force)
    ver = verify_against_p17(index, z)
    df = sweep(index, z)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b7_diffusion.csv"), index=False)
    ver.to_csv(os.path.join(RESULTS, "b7_regen_verify.csv"), index=False)

    print("=" * 100)
    print(f"B7 -- TracIn on the DIFFUSION arm (regenerated E={len(index['members'])}), "
          f"tier {TIER} (median aggregator), demo grain")
    print("=" * 100)
    print("\nregeneration check vs the archived p17 cache:")
    print(ver.round(4).to_string(index=False))
    for t in ("C1", "C5"):
        sub = df[(df.target == t) & (df.estimator == "TracIn")]
        print(f"\n--- {t}: ratio-to-ceiling, plain TracIn ---")
        print(sub.pivot_table(index=["density", "lr_weighted"], columns="group",
                              values="ratio").round(3).to_string())
    best = df.loc[df.groupby("target").ratio.idxmax()]
    print("\nbest cell per target:")
    print(best[["target", "group", "density", "lr_weighted", "estimator", "lds", "ceiling",
                "ratio", "p", "passed"]].to_string(index=False))
    print("\n--- WIN CONDITION 2: does the BC-winning CONFIGURATION win here too? ---")
    print("BC C1 winner: TracIn, density=last5, LR-weighted, Phi = the output head.")
    print("Transferred by ROLE (head -> act_out) and, as a control, on ALL:")
    for t in ("C1", "C5"):
        for grp in ("act_out", "ALL"):
            for dens, lrw in (("last5", True), ("last1", False)):
                r = df[(df.target == t) & (df.group == grp) & (df.density == dens)
                       & (df.lr_weighted == lrw) & (df.estimator == "TracIn")]
                if len(r):
                    r = r.iloc[0]
                    tag = "  <- BC winner's config" if (dens, lrw) == ("last5", True) else \
                          "  (= GradDot control)"
                    print(f"  {t} {grp:14s} {dens:6s} lr={str(lrw):5s} "
                          f"ratio={r.ratio:+.3f} p={r.p:.4f} pass={r.passed}{tag}")


if __name__ == "__main__":
    main()
