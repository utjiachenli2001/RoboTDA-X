"""P10 diffusion-policy trainer. Mirrors src/train.py exactly except for the objective.

Everything the study holds fixed is held fixed here too: one directory per run, atomic
train.marker, --seed controls init AND batching, N_CKPT evenly spaced checkpoints (TracIn needs
them), the same total gradient-step budget (so the two policy classes get the same optimization),
and the frozen Phase-1 norm_stats.

The ONLY change is the loss: DDPM epsilon-prediction MSE on the H-step action chunk, instead of
the GMM NLL on the single next action.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join("/mnt/sdb/ljc/RoboTDA-X", "src"))
import bootstrap  # noqa: F401,E402
from bootstrap import is_done, write_done  # noqa: E402
import dataset  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusion_policy as DP  # noqa: E402
from diffusion_data import ChunkBank  # noqa: E402

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                        "p10_config_frozen.json")

DEFAULT_CFG = {
    "d_model": 384, "n_obs_layer": 4, "n_den_layer": 6, "n_head": 6, "dropout": 0.1,
    "h_chunk": 8, "n_train_steps": 100, "n_ddim_steps": 10,
    "batch_size": 256, "lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.95],
    "warmup_frac": 0.05, "lr_min": 1e-5, "grad_clip": 1.0,
    "total_steps": 8000, "n_ckpt": 5, "amp": "bf16",
}


def load_cfg(override=None):
    """The FROZEN diffusion config. p10_config_frozen.json wins once calibration has written it."""
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CFG_PATH):
        frozen = json.load(open(CFG_PATH))
        cfg.update(frozen.get("cfg", frozen))
    if override:
        cfg.update(override)
    return cfg


def train(run_dir, demo_ids, seed, cfg, device="cuda", log_every=50):
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    bank = ChunkBank(demo_ids, H=cfg["h_chunk"])
    S = torch.from_numpy(bank.S).to(device)
    AC = torch.from_numpy(bank.AC).to(device)
    L = torch.from_numpy(bank.L).to(device)
    N = bank.n

    model = DP.build(dataset.state_dim(), cfg).to(device)
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

    ckpt_at = sorted(set(int(round(total_steps * (i + 1) / cfg["n_ckpt"]))
                         for i in range(cfg["n_ckpt"])))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gcuda = torch.Generator(device=device).manual_seed(seed)     # the (t, eps) draws
    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else torch.float32

    logf = open(os.path.join(run_dir, "train_log.jsonl"), "w")
    step, saved, ep, recent = 0, [], 0, []
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
                loss = model.ddpm_loss(S[idx], L[idx], AC[idx], generator=gcuda).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            step += 1
            ep_loss += float(loss.detach())
            nb += 1
            if step in ckpt_at:
                p = os.path.join(run_dir, f"ckpt_{len(saved)}.pt")
                torch.save({"model": model.state_dict(), "step": step, "epoch": ep, "cfg": cfg,
                            "state_dim": dataset.state_dim(), "policy": "diffusion"}, p)
                saved.append(os.path.basename(p))
        if nb:
            recent.append(ep_loss / nb)
            recent = recent[-10:]
        if ep % log_every == 0 or step >= total_steps:
            rec = {"epoch": ep, "step": step, "loss": ep_loss / max(1, nb), "lr": lr_at(step),
                   "elapsed": time.time() - t0}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"[trainDP] ep {ep} step {step}/{total_steps} loss {rec['loss']:.5f} "
                  f"({rec['elapsed']:.0f}s)", flush=True)
        ep += 1
    logf.close()

    torch.save({"model": model.state_dict(), "step": step, "epoch": ep, "cfg": cfg,
                "state_dim": dataset.state_dim(), "policy": "diffusion"},
               os.path.join(run_dir, "final.pt"))
    meta = {"run_dir": run_dir, "seed": seed, "policy": "diffusion", "n_demos": len(demo_ids),
            "n_windows": N, "steps": step, "epochs_run": ep, "steps_per_epoch": steps_per_epoch,
            "ckpts": saved, "wall_s": time.time() - t0, "cfg": cfg, "demos": list(demo_ids),
            "final_loss": float(np.mean(recent)) if recent else float("nan"),
            "params_M": DP.n_params(model) / 1e6,
            "cudnn_deterministic": True, "torch_version": torch.__version__}
    json.dump(meta, open(os.path.join(run_dir, "train_meta.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg_json", default=None, help="JSON dict of cfg overrides (calibration)")
    ap.add_argument("--cfg_file", default=None,
                    help="path to a JSON file of cfg overrides. PREFERRED over --cfg_json: a JSON "
                         "string on a command line that is itself inside `bash -c '...'` has its "
                         "quotes eaten by the outer shell.")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if is_done(a.run_dir, "train"):
        print(f"[trainDP] SKIP (train.marker): {a.run_dir}")
        return
    ov = None
    if a.cfg_file:
        ov = json.load(open(a.cfg_file))
    elif a.cfg_json:
        ov = json.loads(a.cfg_json)
    cfg = load_cfg(ov)
    if a.steps:
        cfg["total_steps"] = a.steps
    demos = json.load(open(a.demos))["demos"]
    meta = train(a.run_dir, demos, a.seed, cfg, device=a.device)
    write_done(a.run_dir, json.dumps({"stage": "train", "wall_s": meta["wall_s"],
                                      "final_loss": meta["final_loss"],
                                      "params_M": meta["params_M"]}) + "\n", name="train")
    print(f"[trainDP] DONE {a.run_dir} wall={meta['wall_s']:.0f}s "
          f"params={meta['params_M']:.1f}M loss={meta['final_loss']:.5f}")


if __name__ == "__main__":
    main()
