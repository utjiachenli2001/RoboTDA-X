"""W2 -- exact surrogate leave-one-out on the frozen-trunk head.

FINDINGS establishes the action head reproduces the full model to 4 decimals (paired sd 0.005):
0.2% of the parameters carry the signal. So freeze the trunk and treat attribution as a linear
problem in the 512-D feature space phi(s) = ln_f(trunk(s))[:, -1]. Fit a ridge head that predicts
the executed action (the surrogate for what the outcome measures -- held-out L2 to the true
action), then compute the EXACT leave-one-demo-out effect on each target cluster's held-out L2 by
downdating the 512x512 normal equations. No linearization, no ridge on a 135x135 Gram, no
approximation: the counterfactual "retrain the head without demo d" is available in closed form.

    W    = (Phi^T Phi + lam I)^-1 Phi^T A            (ridge head, A = true actions)
    W_-d = (Phi^T Phi - Phi_d^T Phi_d + lam I)^-1 (Phi^T A - Phi_d^T A_d)   (exact demo removal)
    score_d,t = heldout_L2_t(W_-d) - heldout_L2_t(W)   (removal raising loss => positive influence)

DIAGNOSTIC VALUE, stated up front (HANDOFF): if this EXACT LOO on the frozen-trunk surrogate also
fails to beat GradDot, then linearization error is exonerated and the failure is feature noise /
trunk plasticity -- a real conclusion regardless of the sign of the result. Baseline is GradDot on
the SAME regenerated E=5 head-space ensemble (BLOCKERS #6), never the cached E=20 number.
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
from if_repair import gradients as GR  # noqa: E402
from if_repair import retrain as RT  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)     # relative to mean eigenvalue of Phi^T Phi
DEV_TARGETS = ("C1", "C5", "C2")


@torch.no_grad()
def features(model, S, L, device="cuda", batch=4096):
    """phi(s) = ln_f(trunk(state_proj(S)+lang_proj(L)+pos))[:, -1]  -> (N, 512), replicating
    BCTransformer.forward up to the head (drop is identity in eval)."""
    out = np.empty((S.shape[0], model.state_proj.out_features), dtype=np.float32)
    for i in range(0, S.shape[0], batch):
        s = torch.from_numpy(S[i:i + batch]).to(device)
        l = torch.from_numpy(L[i:i + batch]).to(device)
        x = model.state_proj(s) + model.lang_proj(l).unsqueeze(1) + model.pos
        for b in model.blocks:
            x = b(x, model.causal)
        h = model.ln_f(x)[:, -1]
        out[i:i + batch] = h.float().cpu().numpy()
    return out


def member_scores(member, device="cuda", targets=None):
    """-> {target: {demo_id: score}} for one regenerated member, per lambda. Also returns the
    per-lambda held-out-L2 of the full surrogate for diagnostics. `targets` defaults to the dev
    triple; pass a wider list (e.g. all 9 clusters) to use the surrogate as a full attribution
    view (P6 multi-view)."""
    targets = list(targets) if targets is not None else list(DEV_TARGETS)
    model = EV.load_model(os.path.join(GR.REGEN, member, "final.pt"), device=device)

    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    slices = tbank.demo_slices()
    Phi = features(model, tbank.S, tbank.L, device)            # (Ntr, 512)
    A = tbank.A.astype(np.float64)                             # (Ntr, 7)

    hbank = EV.heldout_bank("base")
    PhiH = features(model, hbank.S, hbank.L, device).astype(np.float64)   # (Nh, 512)
    AH = hbank.A.astype(np.float64)
    fidx = RT.frame_index()
    cluster_of_row = fidx["cluster_of_row"]
    del model
    torch.cuda.empty_cache()

    Phi = Phi.astype(np.float64)
    d = Phi.shape[1]
    PtP = Phi.T @ Phi                                          # (512,512)
    PtA = Phi.T @ A                                            # (512,7)
    mean_eig = np.trace(PtP) / d
    # held-out rows per requested target
    rows_t = {t: np.nonzero(cluster_of_row == t)[0] for t in targets}

    def heldout_l2(W):
        return {t: float(np.mean(np.sum((PhiH[rows_t[t]] @ W - AH[rows_t[t]]) ** 2, axis=1)))
                for t in targets}

    out = {}     # lam -> {target: {demo: score}}
    diag = {}
    for lr in LAMBDAS:
        lam = lr * mean_eig
        M = PtP + lam * np.eye(d)
        Minv_PtA = np.linalg.solve(M, PtA)
        base = heldout_l2(Minv_PtA)
        sc = {t: {} for t in targets}
        for demo in train_ids:
            r = slices[demo]
            Pd = Phi[r]                                        # (nd, 512)
            Md = M - Pd.T @ Pd
            bd = PtA - Pd.T @ A[r]
            Wd = np.linalg.solve(Md, bd)
            l2d = heldout_l2(Wd)
            for t in targets:
                sc[t][demo] = l2d[t] - base[t]                # removal raises loss => positive
        out[lr] = sc
        diag[lr] = base
    return out, diag


def graddot_regen_head():
    """GradDot_dmean on the regenerated E=5 head-space ensemble (the honest same-ensemble bar)."""
    from if_repair import b1_layerwise as B1
    from p6_lambda_extend import scores_graddot
    members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(x, "final.pt")))
    ens = B1.build_ensemble(members)
    return scores_graddot(ens["head"], normalize_per_member=True), members


def paired_bootstrap(px, pg, out, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    px, pg, out = map(lambda a: np.asarray(a, float), (px, pg, out))
    n = len(out)
    d0 = spearman(px, out) - spearman(pg, out)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rx, rg = spearman(px[i], out[i]), spearman(pg[i], out[i])
        if np.isfinite(rx) and np.isfinite(rg):
            diffs.append(rx - rg)
    diffs = np.array(diffs)
    return d0, (float((diffs <= 0).mean()) if len(diffs) else np.nan)


def mask_pred(scores, masks):
    return np.array([sum(scores.get(dd, 0.0) for dd in m["demos"]) for m in masks])


def main():
    t0 = time.time()
    members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(x, "final.pt")))
    print(f"[W2] {len(members)} regen members: {members}", flush=True)

    # average surrogate scores over members, per lambda
    agg = None
    for mi, m in enumerate(members):
        sc, diag = member_scores(m)
        if agg is None:
            agg = {lr: {t: {d: 0.0 for d in sc[lr][t]} for t in DEV_TARGETS} for lr in LAMBDAS}
        for lr in LAMBDAS:
            for t in DEV_TARGETS:
                for d, v in sc[lr][t].items():
                    agg[lr][t][d] += v / len(members)
        print(f"[W2] member {m} done ({time.time()-t0:.0f}s)", flush=True)
    surrogate = {lr: {t: agg[lr][t] for t in DEV_TARGETS} for lr in LAMBDAS}

    graddot, _ = graddot_regen_head()

    # ---- stage 1: LDS on the archived G-series (bc_s10 outcomes + ceilings), all lambdas
    gm = D.demo_masks()
    obs, ceil = D.outcomes("bc_s10"), D.ceilings("bc_s10")
    from p6_lambda_sweep import demo_grain_lds
    rows = []
    for lr in LAMBDAS:
        for t in DEV_TARGETS:
            rho, p, n, _, _ = demo_grain_lds(surrogate[lr][t], gm, obs[t])
            rows.append({"lambda_rel": lr, "target": t, "estimator": "surrogate_LOO",
                         "lds": rho, "ceiling": ceil[t], "ratio": rho / ceil[t], "p": p})
    for t in DEV_TARGETS:
        rho, p, n, _, _ = demo_grain_lds(graddot[t], gm, obs[t])
        rows.append({"lambda_rel": np.nan, "target": t, "estimator": "GradDot_head_regenE5",
                     "lds": rho, "ceiling": ceil[t], "ratio": rho / ceil[t], "p": p})
    df1 = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df1.to_csv(os.path.join(RESULTS, "b12_headloo_gseries.csv"), index=False)
    print("\n" + "=" * 92)
    print("W2 -- surrogate-LOO vs GradDot(head, regenE5), G-series (archived p12 outcomes)")
    print("=" * 92)
    print(df1.round(4).to_string(index=False))

    # pick best lambda by G-series C5 (dev), then confirm paired across G/H/I
    best_lr = max(LAMBDAS, key=lambda lr: df1[(df1.lambda_rel == lr) &
                                              (df1.target == "C5")].lds.iloc[0])
    print(f"\n[W2] best lambda_rel by G-series C5 = {best_lr}")
    xs = surrogate[best_lr]

    draws = [("G", [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in D.demo_masks()], "A"),
             ("H", [{"mask_id": m["mask_id"], "demos": m["demos"]}
                    for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0]], "B"),
             ("I", [{"mask_id": m["mask_id"], "demos": m["demos"]}
                    for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0]], "I")]
    prows = []
    pool = {t: {"px": [], "pg": [], "o": []} for t in DEV_TARGETS}
    for name, masks, camp in draws:
        for t in DEV_TARGETS:
            raw = F.campaign_outcomes(camp, "plain", targets=(t,))[t]
            obs_t = F.seed_mean(raw)
            out = np.array([obs_t.get(m["mask_id"], np.nan) for m in masks])
            ok = np.isfinite(out)
            px, pg = mask_pred(xs[t], masks)[ok], mask_pred(graddot[t], masks)[ok]
            o = out[ok]
            d0, pp = paired_bootstrap(px, pg, o)
            prows.append({"draw": name, "target": t, "surrogate_lds": spearman(px, o),
                          "graddot_lds": spearman(pg, o), "paired_delta_rho": d0, "paired_p": pp})
            pool[t]["px"] += list(px); pool[t]["pg"] += list(pg); pool[t]["o"] += list(o)
    for t in DEV_TARGETS:
        px, pg, o = (np.array(pool[t][k]) for k in ("px", "pg", "o"))
        d0, pp = paired_bootstrap(px, pg, o)
        prows.append({"draw": "POOLED_GHI", "target": t, "surrogate_lds": spearman(px, o),
                      "graddot_lds": spearman(pg, o), "paired_delta_rho": d0, "paired_p": pp})
    df2 = pd.DataFrame(prows)
    df2.to_csv(os.path.join(RESULTS, "b12_headloo_paired_ghi.csv"), index=False)
    print("\n" + "=" * 92)
    print(f"W2 -- surrogate-LOO (lambda_rel={best_lr}) PAIRED vs GradDot(head,regenE5) on G/H/I")
    print("=" * 92)
    print(df2.round(4).to_string(index=False))
    print("\nDIAGNOSTIC verdict (pooled):")
    for t in DEV_TARGETS:
        r = df2[(df2.draw == "POOLED_GHI") & (df2.target == t)].iloc[0]
        v = ("beats GradDot" if r.paired_delta_rho > 0 and r.paired_p < 0.05
             else "does NOT beat GradDot -> linearization error exonerated on " + t)
        print(f"  {t}: pooled paired Delta_rho = {r.paired_delta_rho:+.4f} (p={r.paired_p:.4f}) -> {v}")


if __name__ == "__main__":
    main()
