"""Per-demo attribution: TracIn, TRAK, and Influence Functions (spec §7).

UNIT OF ATTRIBUTION = one DEMO (masks add/remove whole demos), not one window.

  g_d      = sum over demo d's windows of grad NLL         (the demo's contribution to the
             training objective; SUM, not mean, because removing d removes all its windows,
             so a longer demo really does contribute more gradient. Demo window counts are
             stored alongside the scores so any length confound is visible.)
  g_{t,f}  = grad of the target functional: the plain / transport-masked / interaction-masked
             held-out loss of target cluster t over its 10 held-out demos (evaluate.py).

Estimators (9 targets x 3 functionals = 27 test gradients):

  TracIn   score_i = sum_c  eta_c * <g_i(theta_c), g_test(theta_c)>
           over the 5 evenly spaced checkpoints of each of the E=10 ensemble runs; eta_c is
           the LR actually in force at that checkpoint's step (from the frozen schedule).

  TRAK     score = (G + lambda I)^{-1} k_test,  averaged over the E=10 ensemble members,
           where G_ij = <g_i, g_j> and (k_test)_i = <g_i, g_test>.

  IF       score_i = g_i^T (F + lambda I)^{-1} g_test  with F the empirical Fisher
           (1/N) sum_j g_j g_j^T, computed EXACTLY by Woodbury:
             g_i^T (F+lam I)^{-1} g_test = (1/lam) [ G_i,test - G[i,:] (lam*N I + G)^{-1} k_test ]

DELIBERATE DEVIATION FROM THE SPEC (which asks for dattri's TRAK / TracIn / EK-FAC IF):
dattri 0.3.0 is installed, but its attributors are per-SAMPLE (a dataloader item) whereas our
unit is a demo with a ragged number of windows, and both of its approximations are unnecessary
here:
  * TRAK projects gradients to k dims with a JL sketch. With N=135 << p=19.2M the k x k Gram
    (Phi^T Phi) is SINGULAR for any useful k, and the projection only injects sketching noise.
    The N x N dual (kernel) form used here is the exact same estimator without the sketch.
  * EK-FAC exists to APPROXIMATE a Fisher inverse that is intractable when N and p are both
    large. At N=135 the exact empirical-Fisher inverse is available in closed form (Woodbury).
We therefore compute the EXACT estimators. This is strictly more accurate, and it gives
attribution its best possible shot -- so a Gate-1 failure cannot be blamed on sketching or
factorization error.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS, RESULTS
import dataset
import evaluate as EV
import policy as P

FUNCTIONALS = ("plain", "transport", "interaction")
RIDGE = 1e-2          # lambda, relative to the mean diagonal of G (set adaptively below)


# ------------------------------------------------------------------ gradient machinery
def flat_grad(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()])


def demo_gradient(model, bank, rows, chunk=512):
    """SUM of per-window gradients over one demo (accumulated in chunks)."""
    model.zero_grad(set_to_none=True)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        s = torch.from_numpy(bank.S[r]).cuda()
        l = torch.from_numpy(bank.L[r]).cuda()
        a = torch.from_numpy(bank.A[r]).cuda()
        model.nll(s, l, a).sum().backward()      # SUM: accumulate raw gradient contribution
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()]).detach()


def target_gradient(model, hbank, rows, mask, chunk=512):
    """Gradient of the masked/plain MEAN held-out L2 functional over the target's frames.

    loss = sum_f m_f * l2_f / sum_f m_f    (mask=None -> plain mean)

    The TEST functional is L2 (bounded, stable); the TRAIN gradient below is still the
    gradient of the training objective (GMM NLL), which is the correct pairing for TracIn /
    TRAK / influence functions -- the training loss and the test functional need not be the
    same, and the mean GMM NLL is unusable as a test functional (see policy.l2).
    """
    w = np.ones(len(rows)) if mask is None else mask[rows].astype(np.float64)
    Z = w.sum()
    if Z <= 0:
        return None
    model.zero_grad(set_to_none=True)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        s = torch.from_numpy(hbank.S[r]).cuda()
        l = torch.from_numpy(hbank.L[r]).cuda()
        a = torch.from_numpy(hbank.A[r]).cuda()
        wt = torch.from_numpy(w[i:i + chunk]).float().cuda()
        ((model.l2(s, l, a) * wt).sum() / Z).backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()]).detach()


def build_targets(model, hbank, variant="base", targets=None):
    """targets: list of (cluster, functional); default = all 9 x 3. -> ({(c,f): grad}, order)."""
    _, by_c = dataset.heldout_pool()
    id2k = {d: k for k, d in enumerate(hbank.ids)}
    tr = hbank.masks[variant]["transport"]
    it = hbank.masks[variant]["interaction"]
    targets = targets or [(c, f) for c in dataset.clusters() for f in FUNCTIONALS]
    out, order = {}, []
    for (c, f) in targets:
        rows = np.concatenate([np.nonzero(hbank.owner == id2k[d])[0] for d in by_c[c]])
        m = None if f == "plain" else (tr if f == "transport" else it)
        out[(c, f)] = target_gradient(model, hbank, rows, m)
        order.append((c, f))
    return out, order


def load_ckpt_model(path):
    return EV.load_model(path, device="cuda")


def lr_at_step(cfg, step):
    """The LR actually in force at a step under the frozen cosine schedule (for TracIn)."""
    total = int(cfg["total_steps"])
    warm = max(1, int(cfg["warmup_frac"] * total))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + np.cos(np.pi * p))


# ------------------------------------------------------------------ main computation
def compute(ensemble_dirs, out_parquet, variant="base", ridge_rel=1e-2,
            train_ids=None, targets=None):
    t0 = time.time()
    if train_ids is None:
        train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank(variant)
    slices = tbank.demo_slices()
    rows_of = {d: slices[d] for d in train_ids}
    N = len(train_ids)
    nwin = {d: int(len(rows_of[d])) for d in train_ids}

    # PER-MEMBER scores are kept (not just the ensemble average) so that the headline
    # statistics can carry jackknife/half-split CIs over the E=10 ensemble (spec §7).
    rows = []          # (attributor, functional, target, demo_id, member, score)

    for run in ensemble_dirs:
        member = os.path.basename(run)
        meta = json.load(open(os.path.join(run, "train_meta.json")))
        cfg = meta["cfg"]
        ckpts = [os.path.join(run, c) for c in meta["ckpts"]]
        final = os.path.join(run, "final.pt")

        # ---------- TracIn: sum over this member's 5 checkpoints
        tracin = {}
        for ci, cp in enumerate(ckpts):
            model = load_ckpt_model(cp)
            step = torch.load(cp, map_location="cpu", weights_only=False)["step"]
            eta = lr_at_step(cfg, step)
            tg, order = build_targets(model, hbank, variant, targets)
            TG = torch.stack([tg[k] for k in order])            # (T, p)
            for d in train_ids:
                gd = demo_gradient(model, tbank, rows_of[d])
                dots = (TG @ gd).cpu().numpy()
                for j, (c, f) in enumerate(order):
                    tracin[(c, f, d)] = tracin.get((c, f, d), 0.0) + eta * float(dots[j])
            del model, TG, tg
            torch.cuda.empty_cache()
        print(f"[attr] TracIn {member}: {len(ckpts)} ckpts ({time.time()-t0:.0f}s)", flush=True)
        for (c, f, d), v in tracin.items():
            rows.append(("TracIn", f, c, d, member, v))

        # ---------- TRAK + IF: final checkpoint of this member (exact dual forms)
        model = load_ckpt_model(final)
        tg, order = build_targets(model, hbank, variant, targets)
        TG = torch.stack([tg[k] for k in order])                 # (T, p)
        PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device="cuda")
        for i, d in enumerate(train_ids):
            PHI[i] = demo_gradient(model, tbank, rows_of[d])
        G = (PHI @ PHI.T).double()                               # (N,N)
        K = (PHI @ TG.T).double()                                # (N,T)
        lam = ridge_rel * float(torch.diagonal(G).mean())
        I = torch.eye(N, device="cuda", dtype=torch.float64)

        trak = torch.linalg.solve(G + lam * I, K)                # (N,T)
        # IF via Woodbury on the empirical Fisher F = (1/N) sum_j g_j g_j^T
        inner = torch.linalg.solve(lam * N * I + G, K)           # (N,T)
        iff = (K - G @ inner) / lam                              # (N,T)

        for j, (c, f) in enumerate(order):
            for i, d in enumerate(train_ids):
                rows.append(("TRAK", f, c, d, member, float(trak[i, j])))
                rows.append(("IF", f, c, d, member, float(iff[i, j])))
        del model, PHI, G, K, TG, tg, trak, iff, inner, I
        torch.cuda.empty_cache()
        print(f"[attr] TRAK+IF {member} done ({time.time()-t0:.0f}s)", flush=True)

    import pandas as pd
    per_member = pd.DataFrame(rows, columns=["attributor", "functional", "target", "demo_id",
                                             "member", "score"])
    pm_path = out_parquet.replace(".parquet", "_per_member.parquet")
    per_member.to_parquet(pm_path, index=False)

    # ensemble aggregate = MEAN over members (for all three attributors)
    df = (per_member.groupby(["attributor", "functional", "target", "demo_id"], as_index=False)
                    .agg(score=("score", "mean"), n_members=("score", "size")))
    df["n_windows"] = df["demo_id"].map(nwin)
    df["cluster_of_demo"] = df["demo_id"].map(
        lambda d: dataset.cluster_of_task()[dataset.parse_did(d)[1]])
    df.to_parquet(out_parquet, index=False)
    print(f"[attr] wrote {out_parquet}: {len(df)} rows "
          f"({len(ensemble_dirs)} ensemble members; per-member scores -> {pm_path}), "
          f"{time.time()-t0:.0f}s", flush=True)
    return df


def tracin_scores(run_dir, train_ids, targets, variant="base"):
    """TracIn only, for ONE run's checkpoints, over an arbitrary training set.

    Used by Stage C, where the Q=490 co-train model has 610 training demos: storing all their
    gradients (the TRAK/IF dual form) would need ~47 GB, but TracIn only needs a running dot
    product, so it costs nothing extra.
    -> {(cluster, functional): {demo_id: score}}
    """
    meta = json.load(open(os.path.join(run_dir, "train_meta.json")))
    cfg = meta["cfg"]
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank(variant)
    rows_of = tbank.demo_slices()
    acc = {t: {} for t in targets}
    for cp in [os.path.join(run_dir, c) for c in meta["ckpts"]]:
        model = load_ckpt_model(cp)
        step = torch.load(cp, map_location="cpu", weights_only=False)["step"]
        eta = lr_at_step(cfg, step)
        tg, order = build_targets(model, hbank, variant, targets)
        TG = torch.stack([tg[k] for k in order])
        for d in train_ids:
            gd = demo_gradient(model, tbank, rows_of[d])
            dots = (TG @ gd).cpu().numpy()
            for j, k in enumerate(order):
                acc[k][d] = acc[k].get(d, 0.0) + eta * float(dots[j])
        del model, TG, tg
        torch.cuda.empty_cache()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble_glob", default=os.path.join(RUNS, "stage_E", "ens_s*"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "influence_table.parquet"))
    ap.add_argument("--variant", default="base")
    ap.add_argument("--max_members", type=int, default=10)
    a = ap.parse_args()
    import glob
    dirs = sorted(glob.glob(a.ensemble_glob))[:a.max_members]
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "final.pt"))]
    if not dirs:
        raise SystemExit(f"no ensemble runs at {a.ensemble_glob}")
    print(f"[attr] {len(dirs)} ensemble members: {[os.path.basename(d) for d in dirs]}")
    compute(dirs, a.out, a.variant)


if __name__ == "__main__":
    main()
