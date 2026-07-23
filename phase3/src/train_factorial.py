"""P8a trainer: src/train.py with the INIT RNG and the DATA-ORDER RNG split into two seeds.

Phase 1's trainer already uses two distinct RNG sources -- it just drives both from one seed:

    src/train.py:56   torch.manual_seed(seed)     <-- parameter init AND the dropout mask stream
    src/train.py:57   np.random.seed(seed)
    src/train.py:82   gen = torch.Generator(device="cpu").manual_seed(seed)   <-- batch permutation

This file splits them: --init_seed drives the first pair, --order_seed drives the Generator.
Every other line -- the config, the step budget, the checkpoint schedule, the LR schedule, the
order of RNG consumption -- is preserved EXACTLY, so that

    train_factorial(init_seed=s, order_seed=s)  ==  train(seed=s)      bit-for-bit.

That equality is the instrument check (verified by p8_bitcheck.py before the stage launches).
It is what licenses calling the two factors separable; without it the decomposition is fiction.

DISCLOSED CONFOUND (preregistered, not discovered later): torch.manual_seed also seeds the CUDA
dropout mask stream. So the INIT factor is really "parameter init + dropout masks", not init
alone. This is stated in advance and reported as such.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join("/mnt/sdb/ljc/RoboTDA-X", "src"))
import bootstrap  # noqa: F401,E402
from bootstrap import is_done, write_done  # noqa: E402
import dataset  # noqa: E402
import policy as P  # noqa: E402
from train import load_cfg  # noqa: E402  (the FROZEN Phase-1 config -- never re-tuned)


def train_factorial(run_dir, demo_ids, init_seed, order_seed, cfg, device="cuda",
                    log_every=50, deterministic=True):
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    # cuDNN determinism (preregistered; recorded in train_meta.json).
    # NB the bit-identity check against src/train.py is what actually proves the factorization;
    # these flags are belt-and-braces. They are applied identically to EVERY P8a run, so they
    # cannot bias the init-vs-order contrast.
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---- FACTOR "INIT": parameter initialization + the dropout mask stream
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    bank = dataset.Bank(demo_ids)
    S = torch.from_numpy(bank.S).to(device)
    A = torch.from_numpy(bank.A).to(device)
    L = torch.from_numpy(bank.L).to(device)
    N = bank.n

    model = P.build(dataset.state_dim(), cfg).to(device)      # consumes the INIT stream
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

    # ---- FACTOR "ORDER": the batch permutation Generator (and nothing else)
    gen = torch.Generator(device="cpu").manual_seed(order_seed)

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
            print(f"[trainF] ep {ep} step {step}/{total_steps} loss {rec['loss']:.4f} "
                  f"({rec['elapsed']:.0f}s)", flush=True)
        ep += 1
    logf.close()

    torch.save({"model": model.state_dict(), "step": step, "epoch": ep,
                "cfg": cfg, "state_dim": dataset.state_dim()},
               os.path.join(run_dir, "final.pt"))
    meta = {"run_dir": run_dir, "init_seed": init_seed, "order_seed": order_seed,
            "seed": init_seed,               # for compatibility with Phase-1 readers
            "n_demos": len(demo_ids), "n_windows": N, "steps": step, "epochs_run": ep,
            "steps_per_epoch": steps_per_epoch, "ckpts": saved, "wall_s": time.time() - t0,
            "cfg": cfg, "demos": list(demo_ids),
            "final_loss": float(np.mean(recent)) if recent else float("nan"),
            "params_M": P.n_params(model) / 1e6,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "torch_version": torch.__version__,
            "factor_definitions": {
                "INIT": "torch.manual_seed + np.random.seed -> parameter init AND dropout stream",
                "ORDER": "torch.Generator(cpu) -> the per-epoch batch permutation only"}}
    json.dump(meta, open(os.path.join(run_dir, "train_meta.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--init_seed", type=int, required=True)
    ap.add_argument("--order_seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_deterministic", action="store_true")
    a = ap.parse_args()

    if is_done(a.run_dir, "train"):
        print(f"[trainF] SKIP (train.marker): {a.run_dir}")
        return
    cfg = load_cfg()
    if a.steps:
        cfg["total_steps"] = a.steps
    demos = json.load(open(a.demos))["demos"]
    meta = train_factorial(a.run_dir, demos, a.init_seed, a.order_seed, cfg, device=a.device,
                           deterministic=not a.no_deterministic)
    write_done(a.run_dir, json.dumps({"stage": "train", "wall_s": meta["wall_s"],
                                      "init_seed": a.init_seed, "order_seed": a.order_seed,
                                      "final_loss": meta["final_loss"]}) + "\n", name="train")
    print(f"[trainF] DONE {a.run_dir} init={a.init_seed} order={a.order_seed} "
          f"wall={meta['wall_s']:.0f}s loss={meta['final_loss']:.4f}")


if __name__ == "__main__":
    main()
