"""B2 -- score estimators against REDESIGNED target functionals (after the ceiling gate).

`functionals.py` defines the weightings and gates them. This module does the other half: it
rebuilds the test-side gradient TG for each weighting, so the estimator differentiates exactly
the functional the outcome measures, and then scores it.

Both sides must move together. Differentiating the plain functional while scoring against a
re-weighted outcome (or the reverse) measures a mismatch, not an estimator -- and this corpus
produces large negative LDS when the two disagree (B1: block_01/C7 = -0.142).

Ensemble: the REGENERATED E=5 (BLOCKERS #6). Every baseline here is recomputed on it, so the
comparison across weightings is internally valid; absolute numbers are not comparable to the
E=20 cached-Gram rows.
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
from if_repair import functionals as F  # noqa: E402
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
CACHE = os.path.join(HERE, "runs", "b2_gram.npz")


def member_grams(run_dir, weightings, device="cuda"):
    """One Phi pass -> {weighting: (G, K)} over all 9 clusters."""
    model = AT.load_ckpt_model(os.path.join(run_dir, "final.pt"))
    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    clusters = dataset.clusters()
    _, by_c = dataset.heldout_pool()
    id2k = {d: k for k, d in enumerate(hbank.ids)}
    rows_of = {c: np.concatenate([np.nonzero(hbank.owner == id2k[d])[0] for d in by_c[c]])
               for c in clusters}

    t0 = time.time()
    TG = {}
    for wname in weightings:
        w = F.weights(wname)
        cols = []
        for c in clusters:
            g = AT.target_gradient(model, hbank, rows_of[c], w)
            if g is None:
                raise RuntimeError(f"{wname}/{c}: zero total weight on this cluster")
            cols.append(g)
        TG[wname] = torch.stack(cols)                      # (9, p)

    N = len(train_ids)
    p = next(iter(TG.values())).shape[1]
    PHI = torch.empty((N, p), dtype=torch.float32, device=device)
    for i, d in enumerate(train_ids):
        PHI[i] = AT.demo_gradient(model, tbank, slices[d])
    G = (PHI @ PHI.T).double().cpu().numpy()
    out = {wname: (G, (PHI @ T.T).double().cpu().numpy()) for wname, T in TG.items()}
    print(f"[b2] {os.path.basename(run_dir)}: {len(weightings)} functionals "
          f"({time.time()-t0:.0f}s)", flush=True)
    del model, PHI, TG
    torch.cuda.empty_cache()
    return out, train_ids, clusters


def build_ensemble(weightings, device="cuda", force=False):
    """-> {weighting: Z-dict}. Cached: the Phi pass is the expensive part."""
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return json.loads(str(z["index"])), {k: {"G": z[f"{k}_G"], "K": z[f"{k}_K"],
                                                 "members": z["members"],
                                                 "train_ids": z["train_ids"],
                                                 "targets": z["targets"]}
                                             for k in json.loads(str(z["index"]))["weightings"]}
    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    per, tids, tgts = {}, None, None
    for m in members:
        g, tids, tgts = member_grams(os.path.join(GR.REGEN, m), weightings, device=device)
        for k, (G, K) in g.items():
            per.setdefault(k, {"G": [], "K": []})
            per[k]["G"].append(G); per[k]["K"].append(K)
    ens = {k: {"G": np.stack(v["G"]), "K": np.stack(v["K"]), "members": np.array(members),
               "train_ids": np.array(tids), "targets": np.array(tgts)}
           for k, v in per.items()}
    index = {"weightings": list(weightings), "members": members}
    np.savez_compressed(CACHE, index=json.dumps(index), members=np.array(members),
                        train_ids=np.array(tids), targets=np.array(tgts),
                        **{f"{k}_G": v["G"] for k, v in ens.items()},
                        **{f"{k}_K": v["K"] for k, v in ens.items()})
    return index, ens


# --------------------------------------------------------------------------- scoring
def score_table(ens, obs_source="archived", campaign="A", targets=("C1", "C5"),
                ks=(0, 1, 2, 5, 10)):
    """Every (weighting, estimator, target) scored against the MATCHING outcome + ceiling."""
    gm = D.demo_masks()
    rows = []
    for wname, Z in ens.items():
        if obs_source == "archived":
            if wname not in ("plain", "transport", "interaction"):
                continue
            raw = F.archived_outcomes(wname)
        else:
            raw = F.campaign_outcomes(campaign, wname, targets)
        obs = {t: F.seed_mean(raw[t]) for t in raw}
        ceil = {t: F.split_half_ceiling(raw[t])["ceiling"] for t in raw}
        tids = list(Z["train_ids"])
        variants = {"GradDot_dmean": scores_graddot(Z, normalize_per_member=True)}
        for k in ks:
            S = SP.truncated_if(Z, k, normalize="dmean")
            variants[f"trunc_k{k}"] = {
                tg: {tids[i]: float(S[i, j]) for i in range(len(tids))}
                for j, tg in enumerate(list(Z["targets"]))}
        for vname, sc in variants.items():
            for t in targets:
                if t not in obs or t not in sc:
                    continue
                rho, p, n, _, _ = demo_grain_lds(sc[t], gm, obs[t])
                c = float(ceil[t])
                rows.append({"weighting": wname, "estimator": vname, "target": t,
                             "obs_source": obs_source, "lds": float(rho), "ceiling": c,
                             "ratio": float(rho) / c if c else np.nan,
                             "p": float(p), "n": int(n),
                             "passed": bool(np.isfinite(rho) and rho >= 0.5 * c
                                            and p < ALPHA)})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weightings", default=",".join(F.WEIGHTINGS))
    ap.add_argument("--obs", default="archived", choices=["archived", "campaign"])
    ap.add_argument("--campaign", default="A")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--targets", default="C1,C5")
    a = ap.parse_args()
    wl = tuple(a.weightings.split(","))
    targets = tuple(a.targets.split(","))

    if a.obs == "campaign":
        gate = F.ceiling_table("campaign", weightings=wl, targets=targets, campaign=a.campaign)
    else:
        gate = F.ceiling_table("archived", weightings=wl, targets=targets)
    os.makedirs(RESULTS, exist_ok=True)
    gate.to_csv(os.path.join(RESULTS, f"b2_ceilings_{a.obs}.csv"), index=False)
    print("=" * 90)
    print(f"B2 STEP 3 -- CEILING GATE (>= {F.GATE}) for each target functional, source={a.obs}")
    print("=" * 90)
    print(gate.pivot_table(index="target", columns="weighting", values="ceiling")
          .round(4).to_string())
    ok = sorted(gate[gate.gate_pass].weighting.unique())
    bad = sorted(set(gate.weighting.unique()) - set(ok))
    print(f"\npass gate: {ok}")
    if bad:
        print(f"FAIL gate (not scored): {bad}")

    _, ens = build_ensemble(wl, force=a.force)
    ens = {k: v for k, v in ens.items() if k in ok}
    df = score_table(ens, obs_source=a.obs, campaign=a.campaign, targets=targets)
    df.to_csv(os.path.join(RESULTS, f"b2_scores_{a.obs}.csv"), index=False)
    print("\n" + "=" * 90)
    print(f"B2 STEP 4 -- ratio-to-ceiling by functional (regenerated E=5), source={a.obs}")
    print("=" * 90)
    for t in targets:
        sub = df[df.target == t]
        if sub.empty:
            continue
        print(f"\n--- {t} ---")
        print(sub.pivot_table(index="weighting", columns="estimator",
                             values="ratio").round(3).to_string())
    if not df.empty:
        best = df.loc[df.groupby("target").ratio.idxmax()]
        print("\nbest functional per target:")
        print(best[["target", "weighting", "estimator", "lds", "ceiling", "ratio", "p",
                    "passed"]].to_string(index=False))


if __name__ == "__main__":
    main()
