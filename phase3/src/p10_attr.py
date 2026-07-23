"""P10 attribution for the DIFFUSION policy: TracIn + TRAK-dual on the DENOISING loss.

The estimators are the same exact dual forms Phase 1 used (no sketching, no EK-FAC), so a failure
here cannot be blamed on an approximation. What changes is the LOSS whose gradients are taken:
the DDPM epsilon-prediction MSE instead of the GMM NLL.

THE FIXED (t, eps) BANK -- the one thing a diffusion TDA must get right.
A diffusion training loss is an EXPECTATION over a random timestep t and a random noise eps. If
the train-side and the test-side gradients were each taken at fresh random (t, eps), their inner
product would be dominated by that sampling noise: TracIn/TRAK would be measuring the RNG rather
than the data. So a SINGLE bank of 32 (t, eps) pairs -- stratified over t, drawn once with
rng(1031), frozen to disk with a SHA-256 -- is shared by BOTH sides (p10_bank.py). This is the
MOTIVE-style variance reduction the preregistration commits to, and it also removes the
timestep-induced bias: every demo and every target is scored at exactly the same noise levels.

PREREGISTERED ESTIMATORS: TracIn and TRAK-dual. IF is ALSO computed (it is free once the Gram G
and the cross-term K exist) but is reported DESCRIPTIVELY and is NOT part of the preregistered
"any attributor" criterion, which ranges over {TracIn, TRAK} only.

TEST-SIDE FUNCTIONAL: the denoising loss on the target cluster's 10 held-out demos, with the same
bank (PRIMARY). The L2-on-executed-DDIM-action test side is attempted as the preregistered
SECONDARY; if it is intractable it is reported as "did not run", never silently dropped.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402

import diffusion_policy as DP  # noqa: E402
from diffusion_data import ChunkBank, heldout_chunk_bank  # noqa: E402
from p10_bank import load_bank  # noqa: E402
import evaluate_diffusion as EVD  # noqa: E402

RIDGE = 1e-2
CHUNK = 256


def demo_gradient(model, bank, rows, tb, eb):
    """SUM of per-window gradients of the BANK denoising loss over one demo."""
    model.zero_grad(set_to_none=True)
    for i in range(0, len(rows), CHUNK):
        r = rows[i:i + CHUNK]
        s = torch.from_numpy(bank.S[r]).cuda()
        l = torch.from_numpy(bank.L[r]).cuda()
        a = torch.from_numpy(bank.AC[r]).cuda()
        model.bank_loss(s, l, a, tb, eb).sum().backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()]).detach()


def target_gradient_denoise(model, hbank, rows, tb, eb):
    """Gradient of the MEAN bank denoising loss over the target's held-out frames (PRIMARY)."""
    model.zero_grad(set_to_none=True)
    Z = float(len(rows))
    for i in range(0, len(rows), CHUNK):
        r = rows[i:i + CHUNK]
        s = torch.from_numpy(hbank.S[r]).cuda()
        l = torch.from_numpy(hbank.L[r]).cuda()
        a = torch.from_numpy(hbank.AC[r]).cuda()
        (model.bank_loss(s, l, a, tb, eb).sum() / Z).backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()]).detach()


def target_gradient_l2(model, hbank, rows, n_steps, chunk=64):
    """Gradient of the MEAN L2 on the EXECUTED DDIM action (preregistered SECONDARY).

    Requires backprop THROUGH the DDIM sampler (n_steps denoiser calls), so it is far heavier
    than the denoising-loss test side. Attempted; if it OOMs, the caller reports 'did not run'.
    """
    model.zero_grad(set_to_none=True)
    Z = float(len(rows))
    s_ = model.schedule("cuda")
    g = torch.Generator(device="cpu").manual_seed(DP.DDIM_INIT_SEED)
    x0_init = torch.randn(1, model.H, DP.ACTION_DIM, generator=g).cuda()
    ts = torch.linspace(model.T, 1, n_steps).round().long().tolist()
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        s = torch.from_numpy(hbank.S[r]).cuda()
        l = torch.from_numpy(hbank.L[r]).cuda()
        a1 = torch.from_numpy(hbank.A1[r]).cuda()
        B = s.shape[0]
        x = x0_init.expand(B, -1, -1).contiguous()
        cond = model.encode_obs(s, l)                       # differentiable
        for k, tc in enumerate(ts):
            t = torch.full((B,), tc, device="cuda", dtype=torch.long)
            eps = model.eps_pred(cond, x, t)
            ab_t = s_.alpha_bar[tc]
            x0h = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1, 1)
            tn = ts[k + 1] if k + 1 < len(ts) else 0
            ab_n = s_.alpha_bar[tn]
            x = ab_n.sqrt() * x0h + (1 - ab_n).sqrt() * eps
        loss = ((x[:, 0].clamp(-1, 1) - a1) ** 2).sum(-1).sum() / Z
        loss.backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()]).detach()


def build_targets(model, hbank, tb, eb, kind="denoise", n_steps=10):
    _, by_c = dataset.heldout_pool()
    id2k = {d: k for k, d in enumerate(hbank.ids)}
    out, order = {}, []
    for c in dataset.clusters():
        rows = np.concatenate([np.nonzero(hbank.owner == id2k[d])[0] for d in by_c[c]])
        if kind == "denoise":
            out[c] = target_gradient_denoise(model, hbank, rows, tb, eb)
        else:
            out[c] = target_gradient_l2(model, hbank, rows, n_steps)
        order.append(c)
    return out, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens_glob", default=os.path.join(P3_RUNS, "P10ens", "dpens_s*"))
    ap.add_argument("--out", default=os.path.join(P3_RESULTS, "p10_influence.parquet"))
    ap.add_argument("--test_side", default="denoise", choices=["denoise", "l2"])
    a = ap.parse_args()

    runs = sorted(glob.glob(a.ens_glob))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "final.pt"))]
    print(f"[P10attr] {len(runs)} diffusion ensemble members: "
          f"{[os.path.basename(r) for r in runs]}")
    assert len(runs) >= 5, f"need >= 5 ensemble members, got {len(runs)}"

    # P6.2 GUARD
    heldout = set(dataset.heldout_pool()[0])
    L.assert_no_probe_leak(runs, heldout,
                           context=f"P10 diffusion attribution, test side = {a.test_side}")
    print("[P10attr] probe-leak guard PASSED")

    tb, eb = load_bank("cuda")
    print(f"[P10attr] fixed (t,eps) bank: K={tb.shape[0]}, t={tb[:8].tolist()}...")

    train_ids, _ = dataset.train_pool()
    cfg = json.load(open(os.path.join(runs[0], "train_meta.json")))["cfg"]
    tbank = ChunkBank(train_ids, H=cfg["h_chunk"])
    hbank = heldout_chunk_bank()
    slices = tbank.demo_slices()
    N = len(train_ids)
    n_steps = cfg["n_ddim_steps"]

    rows = []
    t0 = time.time()
    for run in runs:
        member = os.path.basename(run)
        meta = json.load(open(os.path.join(run, "train_meta.json")))

        # ---- TracIn over this member's checkpoints
        tracin = {}
        for cp in [os.path.join(run, c) for c in meta["ckpts"]]:
            model = EVD.load_model(cp)
            step = torch.load(cp, map_location="cpu", weights_only=False)["step"]
            eta = _lr_at(meta["cfg"], step)
            tg, order = build_targets(model, hbank, tb, eb, a.test_side, n_steps)
            TG = torch.stack([tg[c] for c in order])
            for d in train_ids:
                gd = demo_gradient(model, tbank, slices[d], tb, eb)
                dots = (TG @ gd).cpu().numpy()
                for j, c in enumerate(order):
                    tracin[(c, d)] = tracin.get((c, d), 0.0) + eta * float(dots[j])
            del model, TG, tg
            torch.cuda.empty_cache()
        for (c, d), v in tracin.items():
            rows.append(("TracIn", c, d, member, v))
        print(f"[P10attr] TracIn {member} ({time.time()-t0:.0f}s)", flush=True)

        # ---- TRAK-dual (+ IF, descriptive) at the final checkpoint
        model = EVD.load_model(os.path.join(run, "final.pt"))
        tg, order = build_targets(model, hbank, tb, eb, a.test_side, n_steps)
        TG = torch.stack([tg[c] for c in order])
        PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device="cuda")
        for i, d in enumerate(train_ids):
            PHI[i] = demo_gradient(model, tbank, slices[d], tb, eb)
        G = (PHI @ PHI.T).double()
        K = (PHI @ TG.T).double()
        lam = RIDGE * float(torch.diagonal(G).mean())
        I = torch.eye(N, device="cuda", dtype=torch.float64)
        trak = torch.linalg.solve(G + lam * I, K)
        inner = torch.linalg.solve(lam * N * I + G, K)
        iff = (K - G @ inner) / lam
        for j, c in enumerate(order):
            for i, d in enumerate(train_ids):
                rows.append(("TRAK", c, d, member, float(trak[i, j])))
                rows.append(("IF", c, d, member, float(iff[i, j])))
        del model, PHI, G, K, TG, tg, trak, iff, inner, I
        torch.cuda.empty_cache()
        print(f"[P10attr] TRAK+IF {member} ({time.time()-t0:.0f}s)", flush=True)

    pm = pd.DataFrame(rows, columns=["attributor", "target", "demo_id", "member", "score"])
    pmp = a.out.replace(".parquet", "_per_member.parquet")
    pm.to_parquet(pmp, index=False)
    df = (pm.groupby(["attributor", "target", "demo_id"], as_index=False)
            .agg(score=("score", "mean"), n_members=("score", "size")))
    df["cluster_of_demo"] = df["demo_id"].map(
        lambda d: dataset.cluster_of_task()[dataset.parse_did(d)[1]])
    df["test_side"] = a.test_side
    df.to_parquet(a.out, index=False)
    print(f"[P10attr] wrote {a.out}: {len(df)} rows ({len(runs)} members, "
          f"test_side={a.test_side}, {time.time()-t0:.0f}s)")


def _lr_at(cfg, step):
    total = int(cfg["total_steps"])
    warm = max(1, int(cfg["warmup_frac"] * total))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + np.cos(np.pi * p))


if __name__ == "__main__":
    main()
