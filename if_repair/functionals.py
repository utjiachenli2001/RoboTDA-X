"""B2 -- target-functional redesign: frame weightings, their OUTCOMES, and their CEILINGS.

The study's target functional is the uniform mean held-out L2 over a cluster's 10 held-out
demos. Both sides of the LDS use it: the estimator differentiates it (attribution.target_gradient)
and the outcome measures it (p12 `neg_plain_loss`). Redesigning the functional therefore means
redesigning BOTH sides, and the honest order of operations is fixed:

    1. define a frame weighting w over the held-out bank,
    2. compute the OUTCOME  -sum_f w_f l2_f / sum_f w_f  for every (mask, seed) retrain,
    3. measure that outcome's own split-half CEILING,
    4. only if the ceiling clears the gate, score estimators against it.

Skipping step 3 is how a study ends up reporting a ratio against a proxy that nothing could
have predicted; the sibling study scored -0.48 that way. GATE = 0.4 (HANDOFF.md).

The weights themselves are computed from the REGENERATED E=5 ensemble, never from mask
outcomes -- a weighting fitted on outcomes would re-introduce the circularity of BLOCKERS #7.

Weightings
  plain        w = 1                                   the study's functional (control)
  transport    w = transport phase mask                archived outcome exists
  interaction  w = interaction phase mask              archived outcome exists
  ens_var      w = Var over ensemble members of the predicted action (trace)
               -- "where the policy class is undecided". This is the natural notion of a
               frame whose behaviour the training data has not pinned down, so it is where
               adding or removing a demo should have the most room to matter.
  fail_div     w = ensemble-mean per-frame L2 error
               -- "where the policy is already wrong". Divergence states.
  *_q75        the top-quartile INDICATOR of the same quantity: a hard mask, matching the
               form of transport/interaction rather than a soft reweighting.

NOT ATTEMPTED: rollout-visited states. It needs LIBERO/robosuite to roll the policy out, and
neither is installed in this environment (BLOCKERS #10); it would also require re-rolling
every mask retrain, which is far outside the GPU budget.
"""
from __future__ import annotations

import functools
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
from lds import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CAMPAIGNS = os.path.join(HERE, "runs", "campaigns")
WEIGHT_CACHE = os.path.join(HERE, "runs", "heldout_weights.npz")

GATE = 0.4
SOFT_WEIGHTINGS = ("ens_var", "fail_div")
WEIGHTINGS = ("plain", "transport", "interaction",
              "ens_var", "ens_var_q75", "fail_div", "fail_div_q75")


# --------------------------------------------------------------------- ensemble weights
def build_weight_cache(device="cuda", force=False):
    """Predicted-action variance and mean L2 error per held-out frame, from the regen E=5.

    One forward pass per member over the 14461 held-out frames -- seconds of GPU.
    """
    if os.path.exists(WEIGHT_CACHE) and not force:
        return dict(np.load(WEIGHT_CACHE))
    import torch
    from if_repair import gradients as GR
    import evaluate as EV
    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    bank = EV.heldout_bank("base")
    acts, l2s = [], []
    for m in members:
        model = EV.load_model(os.path.join(GR.REGEN, m, "final.pt"), device=device)
        A, Q = [], []
        with torch.no_grad():
            for i in range(0, bank.n, 1024):
                s = torch.from_numpy(bank.S[i:i + 1024]).to(device)
                l = torch.from_numpy(bank.L[i:i + 1024]).to(device)
                a = torch.from_numpy(bank.A[i:i + 1024]).to(device)
                A.append(model.mean_action(s, l).float().cpu().numpy())
                Q.append(model.l2(s, l, a).float().cpu().numpy())
        acts.append(np.concatenate(A, 0))
        l2s.append(np.concatenate(Q, 0))
        del model
        torch.cuda.empty_cache()
    Aall = np.stack(acts)                                    # (M, F, 7)
    ens_var = Aall.var(axis=0).sum(axis=1).astype(np.float64)  # trace of the action covariance
    fail_div = np.stack(l2s).mean(axis=0).astype(np.float64)
    os.makedirs(os.path.dirname(WEIGHT_CACHE), exist_ok=True)
    np.savez_compressed(WEIGHT_CACHE, ens_var=ens_var, fail_div=fail_div,
                        members=np.array(members))
    return {"ens_var": ens_var, "fail_div": fail_div, "members": np.array(members)}


@functools.lru_cache(maxsize=None)
def frame_meta():
    """cluster / transport / interaction per held-out frame. Fixed for every run."""
    from if_repair import retrain as RT
    f = RT.frame_index()
    return {"cluster_of_row": f["cluster_of_row"],
            "transport": f["transport"].astype(np.float64),
            "interaction": f["interaction"].astype(np.float64)}


@functools.lru_cache(maxsize=None)
def weights(name: str):
    """-> (F,) non-negative frame weights, mean-normalised to 1."""
    fm = frame_meta()
    if name == "plain":
        w = np.ones(len(fm["transport"]))
    elif name in ("transport", "interaction"):
        w = fm[name].copy()
    elif name in SOFT_WEIGHTINGS:
        w = np.asarray(build_weight_cache()[name], float).copy()
    elif name.endswith("_q75") and name[:-4] in SOFT_WEIGHTINGS:
        v = np.asarray(build_weight_cache()[name[:-4]], float)
        w = (v >= np.percentile(v, 75)).astype(np.float64)
    else:
        raise KeyError(name)
    s = w.mean()
    return w / s if s > 0 else w


# --------------------------------------------------------------------- outcomes
def _campaign_files(campaign):
    return sorted(glob.glob(os.path.join(CAMPAIGNS, campaign, "*.npz")))


@functools.lru_cache(maxsize=None)
def campaign_frames(campaign):
    """-> (run_ids, mask_ids, seed_init, seed_order, L2 (R,F))."""
    files = _campaign_files(campaign)
    if not files:
        raise RuntimeError(f"campaign {campaign}: no runs under {CAMPAIGNS}/{campaign}")
    rid, mid, si, so, L = [], [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        rid.append(m["run_id"]); mid.append(m["mask_id"])
        si.append(m["seed_init"]); so.append(m["seed_order"])
        L.append(z["l2"])
    return (np.array(rid), np.array(mid), np.array(si), np.array(so),
            np.stack(L).astype(np.float64))


def campaign_outcomes(campaign, weighting, targets=None):
    """-> {target: {mask_id: {seed_key: neg weighted loss}}}. seed_key = (init, order)."""
    rid, mid, si, so, L = campaign_frames(campaign)
    fm = frame_meta()
    w = weights(weighting)
    targets = targets or D.ALL_TARGETS
    out = {}
    for t in targets:
        rows = fm["cluster_of_row"] == t
        wv = w[rows]
        den = wv.sum()
        if den <= 0:
            continue
        vals = -(L[:, rows] @ wv) / den
        d = {}
        for k in range(len(rid)):
            d.setdefault(str(mid[k]), {})[(int(si[k]), int(so[k]))] = float(vals[k])
        out[t] = d
    return out


def archived_outcomes(functional="plain"):
    """{target: {mask_id: {seed: neg loss}}} from p12_outcomes_S10, at the archived depth."""
    df = pd.read_parquet(D.TIERS["bc_s10"]["outcomes"])
    df = df[df.seed.isin(D.TIERS["bc_s10"]["seeds"])]
    col = f"neg_{functional}_loss"
    out = {}
    for t, sub in df.groupby("target"):
        d = {}
        for _, r in sub.iterrows():
            d.setdefault(str(r.mask_id), {})[int(r.seed)] = float(r[col])
        out[str(t)] = d
    return out


def seed_mean(obs_by_mask_seed, keys=None):
    """{mask: {seed: v}} -> {mask: mean over seeds}, the form demo_grain_lds wants."""
    out = {}
    for m, d in obs_by_mask_seed.items():
        vs = [v for k, v in d.items() if keys is None or k in keys]
        if vs:
            out[m] = float(np.mean(vs))
    return out


# --------------------------------------------------------------------- ceilings
def split_half_ceiling(obs_by_mask_seed, max_splits=200):
    """Reliability of the seed-MEAN outcome, by exhaustive disjoint half-splits + Spearman-Brown.

    This is the archived recipe (p12_ceilings: `ceiling_5v5_splithalf_uncorrected` averaged
    over all 126 disjoint 5v5 splits of 10 seeds, then 2r/(1+r) for the 10-seed mean).
    tests/test_retrain.py reproduces the archived C1 number with it.
    """
    masks = sorted(obs_by_mask_seed)
    seeds = sorted({s for m in masks for s in obs_by_mask_seed[m]},
                   key=lambda x: (str(type(x)), str(x)))
    masks = [m for m in masks if all(s in obs_by_mask_seed[m] for s in seeds)]
    S = len(seeds)
    if S < 2 or len(masks) < 4:
        return {"ceiling": np.nan, "half": np.nan, "n_masks": len(masks), "n_seeds": S,
                "n_splits": 0}
    h = S // 2
    combos = list(itertools.combinations(range(S), h))
    seen, splits = set(), []
    for c in combos:
        rest = tuple(sorted(set(range(S)) - set(c)))
        if len(rest) != h:
            continue                      # odd S: drop the leftover seed rather than reuse it
        key = frozenset([c, rest])
        if key in seen:
            continue
        seen.add(key)
        splits.append((c, rest))
    splits = splits[:max_splits]
    rs = []
    for a, b in splits:
        va = np.array([np.mean([obs_by_mask_seed[m][seeds[i]] for i in a]) for m in masks])
        vb = np.array([np.mean([obs_by_mask_seed[m][seeds[i]] for i in b]) for m in masks])
        r = spearman(va, vb)
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return {"ceiling": np.nan, "half": np.nan, "n_masks": len(masks), "n_seeds": S,
                "n_splits": 0}
    half = float(np.mean(rs))
    k = S / h                              # Spearman-Brown factor to the full seed depth
    ceil = k * half / (1 + (k - 1) * half) if (1 + (k - 1) * half) != 0 else np.nan
    return {"ceiling": float(ceil), "half": half, "n_masks": len(masks), "n_seeds": S,
            "n_splits": len(rs)}


def ceiling_table(source, weightings=WEIGHTINGS, targets=None, campaign="A"):
    """Step 3 of the protocol: every functional's own ceiling, BEFORE any estimator sees it."""
    targets = list(targets or D.ALL_TARGETS)
    rows = []
    for wname in weightings:
        if source == "archived":
            if wname not in ("plain", "transport", "interaction"):
                continue
            obs = archived_outcomes(wname)
        else:
            obs = campaign_outcomes(campaign, wname, targets)
        for t in targets:
            if t not in obs:
                continue
            c = split_half_ceiling(obs[t])
            rows.append({"source": source, "weighting": wname, "target": t,
                         "ceiling": c["ceiling"], "half_split": c["half"],
                         "n_masks": c["n_masks"], "n_seeds": c["n_seeds"],
                         "n_splits": c["n_splits"],
                         "gate_pass": bool(np.isfinite(c["ceiling"]) and c["ceiling"] >= GATE)})
    return pd.DataFrame(rows)
