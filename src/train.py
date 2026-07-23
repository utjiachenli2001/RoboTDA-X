"""Trainer: one fixed hyperparameter set for ALL runs in the study (configs/policy.yaml).

Contract (spec §3, §0):
  * one directory per run; `done.marker` written atomically on completion; existing marker => skip
  * --seed controls init AND batching
  * saves N_CKPT evenly spaced checkpoints (needed by TracIn) + final
  * trains with the PLAIN action loss (GMM NLL). The transport/interaction-masked losses are
    evaluation/attribution functionals only -- never a training objective.

Usage:
  python train.py --run_dir runs/<name> --demos <path/to/demos.json> --seed 101 [--epochs N]
  demos.json = {"demos": ["<suite>/<task>/<demo_k>", ...]}
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
from bootstrap import CONFIGS, is_done, write_done
import dataset
import policy as P

DEFAULT_CFG = {
    "d_model": 512, "n_layer": 6, "n_head": 8, "n_modes": 5, "dropout": 0.1,
    "batch_size": 256, "lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.95],
    "warmup_frac": 0.05, "lr_min": 1e-5, "grad_clip": 1.0,
    "total_steps": 8000, "n_ckpt": 5, "amp": "bf16",
}
# NOTE (frozen protocol decision): every run in the study gets the SAME total number of
# gradient steps, not the same number of epochs. Gate 0 (15 vs 135 demos) and Stage C
# (Q = 15/50/500) compare datasets of different size; at fixed epochs the larger dataset
# silently receives proportionally more optimization, which would confound "more data helps"
# with "more gradient steps". Fixing the step budget equalizes optimization and isolates the
# data effect. Inside Stages F/G every retrain has identical data size (75 / 68 demos), so
# the two conventions coincide there.
CFG_PATH = os.path.join(CONFIGS, "policy.yaml")


def load_cfg():
    """Frozen hyperparameters. configs/policy.yaml wins if present."""
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CFG_PATH):
        import yaml
        cfg.update(yaml.safe_load(open(CFG_PATH)) or {})
    return cfg


def train(run_dir, demo_ids, seed, cfg, device="cuda", log_every=50):
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    bank = dataset.Bank(demo_ids)
    S = torch.from_numpy(bank.S).to(device)          # (N,CTX,128)
    A = torch.from_numpy(bank.A).to(device)          # (N,7)
    L = torch.from_numpy(bank.L).to(device)          # (N,384)
    N = bank.n

    model = P.build(dataset.state_dim(), cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])
    bs = min(cfg["batch_size"], N)
    steps_per_epoch = max(1, N // bs)
    total_steps = int(cfg["total_steps"])
    n_epochs = int(np.ceil(total_steps / steps_per_epoch))
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_at(s):
        if s < warmup:
            return cfg["lr"] * (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + np.cos(np.pi * p))

    ckpt_at = sorted(set(int(round(total_steps * (i + 1) / cfg["n_ckpt"]))
                         for i in range(cfg["n_ckpt"])))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else torch.float32

    logf = open(os.path.join(run_dir, "train_log.jsonl"), "w")
    step, saved, ep = 0, [], 0
    recent = []
    model.train()
    while step < total_steps:
        perm = torch.randperm(N, generator=gen).to(device)
        ep_loss, nb = 0.0, 0
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
            ep_loss += float(loss.detach())
            nb += 1
            if step in ckpt_at:
                p = os.path.join(run_dir, f"ckpt_{len(saved)}.pt")
                torch.save({"model": model.state_dict(), "step": step, "epoch": ep,
                            "cfg": cfg, "state_dim": dataset.state_dim()}, p)
                saved.append(os.path.basename(p))
        if nb:
            recent.append(ep_loss / nb)
            recent = recent[-10:]
        if ep % log_every == 0 or step >= total_steps:
            rec = {"epoch": ep, "step": step, "loss": ep_loss / max(1, nb),
                   "lr": lr_at(step), "elapsed": time.time() - t0}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"[train] ep {ep} step {step}/{total_steps} loss {rec['loss']:.4f} "
                  f"({rec['elapsed']:.0f}s)", flush=True)
        ep += 1
    logf.close()

    # final == last evenly spaced checkpoint; also written as final.pt for convenience
    torch.save({"model": model.state_dict(), "step": step, "epoch": ep,
                "cfg": cfg, "state_dim": dataset.state_dim()},
               os.path.join(run_dir, "final.pt"))
    meta = {"run_dir": run_dir, "seed": seed, "n_demos": len(demo_ids), "n_windows": N,
            "steps": step, "epochs_run": ep, "steps_per_epoch": steps_per_epoch,
            "ckpts": saved, "wall_s": time.time() - t0, "cfg": cfg, "demos": list(demo_ids),
            "final_loss": float(np.mean(recent)) if recent else float("nan"),
            "params_M": P.n_params(model) / 1e6}
    json.dump(meta, open(os.path.join(run_dir, "train_meta.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--demos", required=True, help="path to demos.json")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if is_done(a.run_dir, "train"):
        print(f"[train] SKIP (train.marker): {a.run_dir}")
        return
    cfg = load_cfg()
    if a.steps:
        cfg["total_steps"] = a.steps
    demos = json.load(open(a.demos))["demos"]
    meta = train(a.run_dir, demos, a.seed, cfg, device=a.device)
    write_done(a.run_dir, json.dumps({"stage": "train", "wall_s": meta["wall_s"],
                                      "final_loss": meta["final_loss"]}) + "\n", name="train")
    print(f"[train] DONE {a.run_dir} wall={meta['wall_s']:.0f}s "
          f"windows={meta['n_windows']} loss={meta['final_loss']:.4f}")


if __name__ == "__main__":
    main()
