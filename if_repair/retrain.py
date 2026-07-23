"""Pass 3 -- retrain campaigns that produce NEW ESTIMANDS, not more seeds of the old one.

The prime directive for this project is "different methods, not more data". These retrains
obey it: none of them deepens an ensemble or adds seeds to an existing outcome. Each produces
something the archive structurally cannot contain.

  A (`archived` masks, seeds 401-410)
      Stores the PER-FRAME held-out loss of every retrain. The archive keeps only three
      pre-aggregated functionals (plain / transport / interaction), so any RE-WEIGHTED target
      functional -- B2's whole point -- has no matching OUTCOME and therefore no split-half
      ceiling, and cannot be honestly scored. With per-frame losses on disk, an arbitrary
      frame weighting w gives outcome_w(mask, seed) = -sum_f w_f l2_f / sum_f w_f for free,
      forever, at zero further GPU cost.

  C (`crn`) init x order seed grid
      src/train.py derives BOTH the initialization and the batch order from one --seed, so the
      claim "init is ~72% of outcome variance" cannot be checked with archived runs. Splitting
      the seed makes the decomposition measurable.

  B (`fresh` masks, seed 4711)
      A fresh mask draw from the repo's own generator. The 24 archived masks have been used
      for every number in this project and the confirmatory family has been consumed once;
      only new masks can add evidence, not a re-run.

No repo file is edited. The training loop below is a faithful re-implementation of
src/train.py:train() with the seed split made explicit; tests/test_retrain.py asserts that the
split path with seed_init == seed_order reproduces the legacy path exactly. Weights are NOT
kept (240 x 77 MB would be 18 GB and nothing downstream reads them) -- each run is evaluated
in-process and only its per-frame losses survive.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import policy as P  # noqa: E402
import train as TR  # noqa: E402
import masks as MK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
CAMPAIGNS = os.path.join(RUNS, "campaigns")

# Campaign A matches the archived protocol exactly: same 24 masks, same 10 seeds, same
# aggregator. That is deliberate -- it is what makes the regenerated outcome table comparable
# in CONSTRUCTION to p12_outcomes_S10, so a difference between them is attributable to the
# environment (BLOCKERS #6) rather than to a protocol change.
A_SEEDS = tuple(range(401, 411))
# Campaign C: 3 inits x 3 orders on a subset of masks. A full grid is what separates the two
# variance components; the mask subset keeps it affordable.
C_INITS = (401, 402, 403)
C_ORDERS = (401, 402, 403)
C_N_MASKS = 8
# Campaign B: fresh masks. 6 seeds -- the archived 6-seed SB ceiling (0.933 on C1) is within
# 0.02 of the 10-seed one, so depth 6 costs little reliability and buys 24 more masks' worth of
# GPU elsewhere. This family tests exactly TWO preregistered hypotheses.
B_SEEDS = (4401, 4402, 4403, 4404, 4405, 4406)
FRESH_MASK_SEED = 4711


# --------------------------------------------------------------------------- training
def train_one(demo_ids, seed_init, seed_order, cfg, device="cuda"):
    """src/train.py:train(), with the single --seed split into (init, order).

    Legacy equivalence: src/train.py calls torch.manual_seed(seed) once, then builds the model
    (consuming the CPU generator) and draws batch permutations from an explicitly seeded CPU
    Generator. Dropout draws from the CUDA generator, which manual_seed also seeded. Nothing in
    the training loop touches the global CPU generator. So re-seeding between build and loop is
    a no-op when seed_init == seed_order, and otherwise cleanly separates the two streams.
    """
    t0 = time.time()
    torch.manual_seed(seed_init)
    np.random.seed(seed_init)
    bank = dataset.Bank(demo_ids)
    S = torch.from_numpy(bank.S).to(device)
    A = torch.from_numpy(bank.A).to(device)
    L = torch.from_numpy(bank.L).to(device)
    N = bank.n

    model = P.build(dataset.state_dim(), cfg).to(device)      # init <- seed_init

    torch.manual_seed(seed_order)                             # dropout <- seed_order
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

    gen = torch.Generator(device="cpu").manual_seed(seed_order)   # batch order <- seed_order
    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else torch.float32
    step, ep, recent = 0, 0, []
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
        if nb:
            recent = (recent + [ep_loss / nb])[-10:]
        ep += 1
    model.eval()
    del S, A, L
    torch.cuda.empty_cache()
    return model, {"seed_init": seed_init, "seed_order": seed_order, "n_demos": len(demo_ids),
                   "n_windows": int(N), "steps": step, "epochs_run": ep,
                   "final_loss": float(np.mean(recent)) if recent else float("nan"),
                   "wall_s": time.time() - t0}


# --------------------------------------------------------------------------- evaluation
def heldout_frame_losses(model, device="cuda"):
    """Per-FRAME (l2, nll) over the fixed 14461-frame held-out bank. The new artifact."""
    bank = EV.heldout_bank("base")
    return EV.per_frame(model, bank, device=device)


def frame_index():
    """Row -> (cluster, demo) map for the held-out bank. Identical for every run."""
    bank = EV.heldout_bank("base")
    _, by_c = dataset.heldout_pool()
    id2k = {d: k for k, d in enumerate(bank.ids)}
    cluster_of_row = np.empty(bank.n, dtype=object)
    for c, dd in by_c.items():
        for d in dd:
            cluster_of_row[bank.owner == id2k[d]] = c
    return {"cluster_of_row": cluster_of_row,
            "transport": bank.masks["base"]["transport"],
            "interaction": bank.masks["base"]["interaction"],
            "owner": bank.owner, "ids": list(bank.ids)}


def aggregate_outcomes(l2, nll, fidx):
    """The three archived functionals, recomputed from per-frame losses (a self-check)."""
    out = {}
    for c in dataset.clusters():
        rows = fidx["cluster_of_row"] == c
        q, t, i = l2[rows], fidx["transport"][rows], fidx["interaction"][rows]
        n = nll[rows]
        out[c] = {"plain_loss": float(q.mean()),
                  "transport_loss": float((q * t).sum() / max(t.sum(), 1)),
                  "interaction_loss": float((q * i).sum() / max(i.sum(), 1)),
                  "plain_loss_nll": float(n.mean()),
                  "n_frames": int(rows.sum())}
    return out


# --------------------------------------------------------------------------- job plans
def fresh_demo_masks(seed=FRESH_MASK_SEED):
    """The repo's own Stage-G generator at a different seed -> a fresh, disjoint mask draw."""
    masks, cnts, _, _ = MK.build_demo_masks(seed=seed)
    return [{"mask_id": f"H{k:03d}", "n_demos": len(m), "demos": m}
            for k, m in enumerate(masks)], cnts


def jobs(campaign):
    if campaign == "A":
        man = MK.demo_mask_manifest()
        return [{"run_id": f"A_{m['mask_id']}_i{s}_o{s}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": s, "seed_order": s}
                for m in man["masks"] for s in A_SEEDS]
    if campaign == "C":
        man = MK.demo_mask_manifest()
        sel = man["masks"][:C_N_MASKS]
        return [{"run_id": f"C_{m['mask_id']}_i{i}_o{o}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": i, "seed_order": o}
                for m in sel for i in C_INITS for o in C_ORDERS]
    if campaign == "B":
        ms, _ = fresh_demo_masks()
        return [{"run_id": f"B_{m['mask_id']}_i{s}_o{s}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": s, "seed_order": s}
                for m in ms for s in B_SEEDS]
    raise KeyError(campaign)


# --------------------------------------------------------------------------- driver
def run_job(job, cfg, fidx, outdir, device="cuda"):
    out_npz = os.path.join(outdir, job["run_id"] + ".npz")
    if os.path.exists(out_npz):
        return "skip"
    model, meta = train_one(job["demos"], job["seed_init"], job["seed_order"], cfg,
                            device=device)
    l2, nll = heldout_frame_losses(model, device=device)
    del model
    torch.cuda.empty_cache()
    agg = aggregate_outcomes(l2, nll, fidx)
    tmp = out_npz + ".tmp.npz"
    np.savez_compressed(tmp, l2=l2.astype(np.float32), nll=nll.astype(np.float32),
                        meta=json.dumps({**meta, "run_id": job["run_id"],
                                         "mask_id": job["mask_id"],
                                         "outcomes": agg}))
    os.replace(tmp, out_npz)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, choices=["A", "B", "C"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    # Extra workers can be added mid-flight from the other end of the job list: run_job skips a
    # job whose output already exists, so a reverse worker and a forward worker converge without
    # coordination. Duplicated effort is bounded by the number of workers at the crossover.
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()

    cfg = TR.load_cfg()
    if a.steps:
        cfg["total_steps"] = a.steps
    outdir = os.path.join(CAMPAIGNS, a.campaign)
    os.makedirs(outdir, exist_ok=True)
    J = jobs(a.campaign)
    if a.limit:
        J = J[:a.limit]
    mine = [j for k, j in enumerate(J) if k % a.nworkers == a.worker]
    if a.reverse:
        mine = mine[::-1]
    fidx = frame_index()
    print(f"[retrain {a.campaign}] worker {a.worker}/{a.nworkers}: {len(mine)} of {len(J)} jobs,"
          f" steps={cfg['total_steps']}", flush=True)
    t0 = time.time()
    for k, j in enumerate(mine):
        r = run_job(j, cfg, fidx, outdir)
        if r == "skip":
            continue
        print(f"[retrain {a.campaign}] {k+1}/{len(mine)} {j['run_id']} "
              f"loss={r['final_loss']:.4f} wall={r['wall_s']:.0f}s "
              f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    print(f"[retrain {a.campaign}] worker {a.worker} DONE "
          f"({(time.time()-t0)/3600:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
