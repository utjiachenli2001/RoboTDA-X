"""B3 -- KFAC / EK-FAC: the one genuinely different H^-1, and a falsifiable prediction.

Everything tried so far preconditions with the 135x135 demo Gram: truncation, damping,
shrinkage, layer restriction. All of them estimate curvature from 135 samples, and Phase A's
k* ~ 1 says that is hopeless -- the Gram carries about one direction distinguishable from
random demo pairing.

KFAC estimates curvature somewhere else entirely. For a linear layer with input activation a
and output pre-activation gradient g, it factors the Fisher as

    F_l  ~  A_l (x) S_l,      A_l = E[a a^T]  (in x in),   S_l = E[g g^T]  (out x out)

and those expectations are taken over ~92k training FRAMES, not 135 demos. For the action head
(75 x 512) that is 92k samples for a 512x512 and a 75x75 matrix -- an entirely different
sample-size regime from 135 samples for a 39,499-dimensional covariance.

THE PREDICTION (from B1's mechanism result). If attribution fails here because p/N makes
curvature unestimable, KFAC should help exactly where the Gram already had structure to find
(k* > 1: block_00, embed) and where its own factors are well-determined, and NOT become a
uniform win. If instead the field's "the solve is wrong" story were right, KFAC should improve
things everywhere. The two accounts differ, so this is a test rather than a sweep.

Cost. score_i = g_i^T F^-1 g_test is computed by preconditioning the NINE TEST gradients, not
the 135 demo gradients: F^-1 g_test is per layer S^-1 G_test A^-1, and then K = Phi @ (F^-1 TG)^T
reuses the same Phi pass every other estimator here uses. So the whole damping sweep is nearly
free once A and S are accumulated.

Coverage. KFAC applies to nn.Linear modules (state_proj, lang_proj, the MLPs, attn.out_proj,
head). nn.MultiheadAttention's fused in_proj_weight is not a Linear module and its
pre-activation gradient is not exposed by a module hook, so those parameters -- and the
LayerNorms, biases and the positional embedding -- are left UNPRECONDITIONED (identity block).
That is a block-diagonal preconditioner that is KFAC where KFAC is defined, which is the
honest version; `frac_preconditioned` is reported with every row.
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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import gradients as GR  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import attribution as AT  # noqa: E402
from p6_lambda_sweep import demo_grain_lds, ALPHA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "runs", "b3_kfac_cache.npz")

LAMBDAS = (1e-4, 1e-2, 1.0, 1e2, 1e4)
# A group both RESTRICTS Phi (as B1 does) and selects which Linear layers inside it get
# preconditioned. Restricting only the preconditioner would be meaningless: preconditioning the
# head's 0.2% of coordinates while scoring on all 19.2M leaves 99.8% of the score untouched, so
# the cell would report GradDot with a rounding error. Phi and the preconditioner must be
# restricted together, and then the control is B1's GradDot on the SAME restricted Phi
# (method="none" rows below).
GROUP_PREFIX = {
    "ALL": None,                              # every Linear module, full-width Phi
    "head": ("head",),
    "embed": ("state_proj", "lang_proj"),
    "block_00": ("blocks.0.",),
    "last_block": ("blocks.5.",),
}


# ------------------------------------------------------------------ factor accumulation
def linear_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def accumulate_factors(model, bank, device="cuda", chunk=512, max_frames=None):
    """-> {layer_name: (A (in,in), S (out,out), n)} empirical-Fisher Kronecker factors."""
    lin = linear_modules(model)
    A = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float64, device=device)
         for n, m in lin.items()}
    S = {n: torch.zeros(m.out_features, m.out_features, dtype=torch.float64, device=device)
         for n, m in lin.items()}
    nseen = {n: 0 for n in lin}
    handles, caught = [], {}

    def fwd_hook(name):
        def h(mod, inp, out):
            a = inp[0].detach()
            a = a.reshape(-1, a.shape[-1])
            caught[name] = a
            out.register_hook(lambda g, nm=name, aa=a: _acc(nm, aa, g))
        return h

    def _acc(name, a, g):
        g = g.detach().reshape(-1, g.shape[-1])
        A[name] += (a.double().T @ a.double())
        S[name] += (g.double().T @ g.double())
        nseen[name] += a.shape[0]

    for n, m in lin.items():
        handles.append(m.register_forward_hook(fwd_hook(n)))

    N = bank.n if max_frames is None else min(bank.n, max_frames)
    for i in range(0, N, chunk):
        s = torch.from_numpy(bank.S[i:i + chunk]).to(device)
        l = torch.from_numpy(bank.L[i:i + chunk]).to(device)
        a = torch.from_numpy(bank.A[i:i + chunk]).to(device)
        model.zero_grad(set_to_none=True)
        model.nll(s, l, a).sum().backward()
    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    return {n: (A[n] / max(nseen[n], 1), S[n] / max(nseen[n], 1), nseen[n]) for n in lin}


def param_spans(model):
    spans, o = {}, 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        spans[n] = (o, o + p.numel())
        o += p.numel()
    return spans, o


# ------------------------------------------------------------------ preconditioning
def eig_cache(factors, spans, prefixes=None):
    """Eigendecompose each covered layer's (A, S) ONCE; the damping sweep then costs nothing.

    Both the KFAC inverse and the EK-FAC basis need exactly these eigenpairs, and eigh of a
    2048x2048 factor is the dominant cost if it is redone per lambda.
    """
    out = {}
    for name, (A, S, n) in factors.items():
        wname = f"{name}.weight"
        if wname not in spans:
            continue
        if prefixes is not None and not any(name.startswith(p) for p in prefixes):
            continue
        wA, UA = torch.linalg.eigh(A)
        wS, US = torch.linalg.eigh(S)
        tA = float(torch.diagonal(A).sum()) / A.shape[0]
        tS = float(torch.diagonal(S).sum()) / S.shape[0]
        out[name] = {"wA": torch.clamp(wA, min=0.0), "UA": UA,
                     "wS": torch.clamp(wS, min=0.0), "US": US,
                     "tA": tA, "tS": tS, "span": spans[wname],
                     "shape": (S.shape[0], A.shape[0])}
    return out


def precondition_targets(TG, eigs, ptot, lam_rel, ekfac_scale=None, eps=1e-12):
    """F^-1 applied to each test gradient, block-diagonally over the covered Linear layers.

    Uncovered parameters keep their raw gradient (identity block), so a group restriction is a
    statement about WHICH curvature is modelled, not about which gradient coordinates exist.
    """
    out = TG.clone()
    covered = 0
    for name, e in eigs.items():
        a, b = e["span"]
        covered += b - a
        # pi-adjusted factored Tikhonov damping (Martens & Grosse 2015, sec. 6.3)
        tA, tS = max(e["tA"], 1e-30), max(e["tS"], 1e-30)
        lam = lam_rel * float(np.sqrt(tA * tS))
        pi = float(np.sqrt(tA / tS))
        UA, US, wA, wS = e["UA"], e["US"], e["wA"], e["wS"]
        Gm = out[:, a:b].reshape(out.shape[0], *e["shape"]).double()
        P = US.T @ Gm @ UA                                      # Kronecker eigenbasis
        if ekfac_scale is not None and name in ekfac_scale:
            # EK-FAC: divide by the MEASURED second moment in that basis. Uses the
            # eigenvectors of A and S but not their eigenvalues -- so the damping has to be
            # relative to the scale of THAT quantity, not to the Kronecker factors'.
            L2 = ekfac_scale[name]
            P = P / (L2 + lam_rel * float(L2.mean()))
        else:
            # KFAC: the Kronecker eigenvalues are the outer product wS (x) wA
            denom = torch.clamp(wS[:, None] + lam / pi, min=eps) * \
                torch.clamp(wA[None, :] + lam * pi, min=eps)
            P = P / denom.unsqueeze(0)
        out[:, a:b] = (US @ P @ UA.T).reshape(out.shape[0], -1).float()
    return out, covered


def ekfac_second_moment(eigs, PHI):
    """Eigenbasis of (A, S) plus the MEASURED second moment of demo gradients in that basis.

    The eigenvalue correction is estimated from the 135 demo gradients -- i.e. from the same
    135 samples that make the Gram unusable. That is deliberate: KFAC's factors come from ~92k
    frames and EK-FAC's eigenvalues from 135 demos, so contrasting them isolates whether the
    win (if any) comes from the factored STRUCTURE or from more curvature detail.
    """
    out = {}
    for name, e in eigs.items():
        a, b = e["span"]
        Gd = PHI[:, a:b].reshape(PHI.shape[0], *e["shape"]).double()
        P = e["US"].T @ Gd @ e["UA"]
        out[name] = (P ** 2).mean(dim=0)                        # (out, in)
    return out


# ------------------------------------------------------------------ driver
def build_member(run_dir, lambdas=LAMBDAS, groups=GROUP_PREFIX, device="cuda"):
    """-> {(group, lam, method): (G, K)} for one member. One Phi pass, many preconditioners."""
    model = AT.load_ckpt_model(os.path.join(run_dir, "final.pt"))
    spans, ptot = param_spans(model)
    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    clusters = dataset.clusters()

    t0 = time.time()
    factors = accumulate_factors(model, tbank, device=device)
    print(f"[b3] {os.path.basename(run_dir)}: factors for {len(factors)} Linear layers "
          f"({time.time()-t0:.0f}s)", flush=True)

    tg, order = AT.build_targets(model, hbank, "base", [(c, "plain") for c in clusters])
    TG = torch.stack([tg[k] for k in order])
    PHI = torch.empty((len(train_ids), TG.shape[1]), dtype=torch.float32, device=device)
    for i, d in enumerate(train_ids):
        PHI[i] = AT.demo_gradient(model, tbank, slices[d])

    b1_groups = GR.param_groups(model)
    out = {}
    for gname, pref in groups.items():
        eigs = eig_cache(factors, spans, prefixes=pref)
        L2 = ekfac_second_moment(eigs, PHI)
        if gname == "ALL":
            idx, PG, npar = None, PHI, ptot
        else:
            idx = torch.as_tensor(
                np.concatenate([np.arange(*spans[n]) for n in b1_groups[gname]]),
                device=device, dtype=torch.long)
            PG, npar = PHI[:, idx], int(idx.numel())
        Gg = (PG @ PG.T).double().cpu().numpy()
        covered = sum(b - a for a, b in (e["span"] for e in eigs.values()))
        for method in ("none", "kfac", "ekfac"):
            for lam in (lambdas if method != "none" else (0.0,)):
                if method == "none":
                    TGp = TG
                else:
                    TGp, _ = precondition_targets(
                        TG, eigs, ptot, lam, ekfac_scale=L2 if method == "ekfac" else None)
                TGg = TGp if idx is None else TGp[:, idx]
                out[(gname, lam, method)] = (
                    Gg, (PG @ TGg.T).double().cpu().numpy(), covered / max(npar, 1))
                del TGg
                if method != "none":
                    del TGp
        del eigs, L2, PG
    print(f"[b3] {os.path.basename(run_dir)}: {len(out)} preconditioners "
          f"({time.time()-t0:.0f}s)", flush=True)
    del model, PHI, TG, factors
    torch.cuda.empty_cache()
    return out, train_ids, [c for c, _ in order]


def build_ensemble(force=False, device="cuda"):
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        idx = json.loads(str(z["index"]))
        ens = {tuple(k): {"G": z[f"{i}_G"], "K": z[f"{i}_K"], "frac": float(z[f"{i}_frac"]),
                          "members": z["members"], "train_ids": z["train_ids"],
                          "targets": z["targets"]}
               for i, k in enumerate(idx["keys"])}
        return ens
    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    per, tids, tgts = {}, None, None
    for m in members:
        o, tids, tgts = build_member(os.path.join(GR.REGEN, m), device=device)
        for k, (G, K, fr) in o.items():
            per.setdefault(k, {"G": [], "K": [], "frac": fr})
            per[k]["G"].append(G); per[k]["K"].append(K)
    keys = list(per)
    store = {}
    for i, k in enumerate(keys):
        store[f"{i}_G"] = np.stack(per[k]["G"])
        store[f"{i}_K"] = np.stack(per[k]["K"])
        store[f"{i}_frac"] = np.array(per[k]["frac"])
    np.savez_compressed(CACHE, index=json.dumps({"keys": [list(k) for k in keys]}),
                        members=np.array(members), train_ids=np.array(tids),
                        targets=np.array(tgts), **store)
    return {k: {"G": np.stack(per[k]["G"]), "K": np.stack(per[k]["K"]),
                "frac": per[k]["frac"], "members": np.array(members),
                "train_ids": np.array(tids), "targets": np.array(tgts)} for k in keys}


def score(ens, tier="bc_s10", targets=("C1", "C5")):
    from p6_lambda_extend import scores_graddot
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    rows = []
    for (gname, lam, method), Z in ens.items():
        sc = scores_graddot(Z, normalize_per_member=True)
        for t in targets:
            rho, p, n, _, _ = demo_grain_lds(sc[t], gm, obs[t])
            c = float(ceil[t])
            rows.append({"group": gname, "lam_rel": float(lam), "method": method,
                         "frac_preconditioned": float(Z["frac"]), "target": t,
                         "lds": float(rho), "ceiling": c, "ratio": float(rho) / c,
                         "p": float(p), "n": n,
                         "passed": bool(np.isfinite(rho) and rho >= 0.5 * c and p < ALPHA)})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ens = build_ensemble(force=a.force)
    df = score(ens)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b3_kfac.csv"), index=False)
    print("=" * 100)
    print("B3 -- KFAC / EK-FAC (regenerated E=5), demo grain n=24, ratio-to-ceiling")
    print("=" * 100)
    for t in ("C1", "C5"):
        print(f"\n--- {t}: ratio-to-ceiling. Phi RESTRICTED to the group; "
              f"method=none is B1's GradDot control on the same Phi ---")
        sub = df[df.target == t].copy()
        sub["row"] = sub.method + "_" + sub.lam_rel.map(lambda x: f"{x:.0e}")
        sub.loc[sub.method == "none", "row"] = "none (GradDot)"
        print(sub.pivot_table(index="row", columns="group", values="ratio",
                              sort=False).round(3).to_string())
    print("\nfraction of each group's parameters actually preconditioned by KFAC:")
    print(df.groupby("group").frac_preconditioned.first().round(4).to_string())
    print("\n--- THE PREDICTION: does KFAC beat the identity where k* > 1 "
          "(block_00, embed) and not where k* = 1 (ALL, head, last_block)? ---")
    for t in ("C1", "C5"):
        base = df[(df.target == t) & (df.method == "none")].set_index("group").ratio
        best = (df[(df.target == t) & (df.method != "none")]
                .loc[lambda d: d.groupby("group").ratio.idxmax()].set_index("group"))
        for g in ["ALL", "head", "last_block", "block_00", "embed"]:
            if g in base.index and g in best.index:
                b = best.loc[g]
                print(f"  {t} {g:11s} k*={'>1' if g in ('block_00','embed') else ' 1'}  "
                      f"identity {base[g]:+.3f} -> best {b.method}/{b.lam_rel:.0e} "
                      f"{b.ratio:+.3f}  delta {b.ratio-base[g]:+.3f}")
    best = df.loc[df.groupby("target").ratio.idxmax()]
    print("\nbest cell per target:")
    print(best[["target", "group", "method", "lam_rel", "lds", "ceiling", "ratio", "p",
                "passed"]].to_string(index=False))


if __name__ == "__main__":
    main()
