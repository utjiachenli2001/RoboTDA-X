"""W5 -- mid-training GradDot: settle the intermediate-checkpoint thread from pass-3 B4.

B4 decomposed the 5-checkpoint TracIn sum and found the best single checkpoint (step 6400, C1
0.598) beats the final weights (0.506) -- but adjacent checkpoints swing by up to 0.42, four
times the claimed gain. Either there is a real "attribute from the middle of training" effect or
the LDS is far noisier than n=24 suggests. A DENSER checkpoint grid on one member separates them:

  Step 1  retrain ONE member saving a ckpt every 400 steps (~20 ckpts); GradDot_dmean per ckpt on
          C1+C5, head Phi. If the curve is spiky (adjacent swings >~ 0.2) -> ckpt noise, close it.
  Step 2  (only on a smooth plateau) all 5 members, GradDot@tau*, tau* chosen on C1 dev only.

Weak prior (B4 already hinted spiky). This is step 1.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import gradients as GR  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import policy as P  # noqa: E402
import train as TR  # noqa: E402
import attribution as AT  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402
from p6_lambda_sweep import demo_grain_lds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CKPTDIR = os.path.join(HERE, "runs", "ckptgrid")
EVERY = 400
TARGETS = ("C1", "C5")


def train_with_ckpts(demo_ids, seed, cfg, device="cuda"):
    """retrain.train_one, but snapshot a full ckpt dict every EVERY steps."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    bank = dataset.Bank(demo_ids)
    S = torch.from_numpy(bank.S).to(device)
    A = torch.from_numpy(bank.A).to(device)
    L = torch.from_numpy(bank.L).to(device)
    N = bank.n
    model = P.build(dataset.state_dim(), cfg).to(device)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])
    bs = min(cfg["batch_size"], N)
    steps_per_epoch = max(1, N // bs)
    total_steps = int(cfg["total_steps"])
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_at(s):
        if s < warmup:
            return cfg["lr"] * (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + np.cos(np.pi * p))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else torch.float32
    os.makedirs(CKPTDIR, exist_ok=True)
    saved = []
    step = 0
    model.train()
    while step < total_steps:
        perm = torch.randperm(N, generator=gen).to(device)
        for b in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = perm[b * bs:(b + 1) * bs]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=(cfg["amp"] == "bf16")):
                loss = model.nll(S[idx], L[idx], A[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            step += 1
            if step % EVERY == 0 or step == total_steps:
                path = os.path.join(CKPTDIR, f"ckpt_step{step:05d}.pt")
                torch.save({"state_dim": dataset.state_dim(), "cfg": cfg,
                            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "step": step}, path)
                saved.append((step, path))
    del S, A, L, model
    torch.cuda.empty_cache()
    return saved


def graddot_lds_for_ckpt(path, head_names):
    G, K, tids, tgts = GR.build_gram(os.path.dirname(path), ckpt=os.path.basename(path),
                                     keep=head_names, verbose=False)
    Z = {"G": G[None], "K": K[None], "train_ids": tids, "targets": tgts}
    sc = scores_graddot(Z, normalize_per_member=True)
    obs, ceil = D.outcomes("bc_s10"), D.ceilings("bc_s10")
    out = {}
    for t in TARGETS:
        rho, p, n, _, _ = demo_grain_lds(sc[t], D.demo_masks(), obs[t])
        out[t] = (rho, rho / ceil[t])
    return out


def main():
    t0 = time.time()
    cfg = TR.load_cfg()
    train_ids, _ = dataset.train_pool()
    print(f"[W5] retraining one member with a ckpt every {EVERY} steps "
          f"(total {cfg['total_steps']})", flush=True)
    saved = train_with_ckpts(train_ids, seed=401, cfg=cfg)
    print(f"[W5] {len(saved)} ckpts saved ({time.time()-t0:.0f}s). Building head-Phi GradDot...",
          flush=True)

    # head param names from a throwaway model
    m = P.build(dataset.state_dim(), cfg)
    head_names = GR.param_groups(m)["head"]
    del m

    rows = []
    for step, path in saved:
        r = graddot_lds_for_ckpt(path, head_names)
        rows.append({"step": step, "C1_lds": r["C1"][0], "C1_ratio": r["C1"][1],
                     "C5_lds": r["C5"][0], "C5_ratio": r["C5"][1]})
        print(f"  step {step:5d}: C1 {r['C1'][0]:.3f}  C5 {r['C5'][0]:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b15_ckptgrid.csv"), index=False)

    print("\n" + "=" * 72)
    print("W5 -- GradDot_dmean (head Phi) vs training step, one member")
    print("=" * 72)
    print(df.round(4).to_string(index=False))
    for t in TARGETS:
        v = df[f"{t}_lds"].values
        swings = np.abs(np.diff(v))
        print(f"\n{t}: final-step lds={v[-1]:.3f}, max lds={v.max():.3f} at step "
              f"{int(df.step.values[v.argmax()])}, max adjacent swing={swings.max():.3f}, "
              f"mean swing={swings.mean():.3f}")
    maxswing = max(np.abs(np.diff(df.C1_lds.values)).max(),
                   np.abs(np.diff(df.C5_lds.values)).max())
    print(f"\nVERDICT: max adjacent swing = {maxswing:.3f}  -> "
          f"{'SPIKY (ckpt noise, close the thread)' if maxswing >= 0.2 else 'smooth (proceed to step 2)'}")
    print(f"[W5] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
