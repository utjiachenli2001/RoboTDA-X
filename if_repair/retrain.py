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
# Campaign B: fresh masks, 10 seeds to MATCH campaign A and the archived protocol.
#
# This started at 6, on the reasoning that the archived 6-seed SB ceiling (0.933 on C1) is within
# 0.02 of the 10-seed one so the reliability cost is negligible. That reasoning was wrong, and
# measurably so: holding the masks, the estimator and the outcome table fixed and varying ONLY
# the number of seeds averaged into the outcome, GradDot_ALL on C1 scores 0.362 at depth 6 and
# 0.475 at depth 10 -- and on C5 it moves the other way (0.414 -> 0.374). Dividing by the ceiling
# does not make the ratio invariant to depth. Comparing a depth-6 confirmatory number against a
# depth-10 dev number therefore confounds the hypothesis with the protocol, so seeds 4407-4410
# were added and the family recomputed at matched depth. Both tables are reported.
B_SEEDS = (4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410)
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
# Third mask draw (campaign I). Confirms the paired B8 results (KFAC-embed, datamodel on C5)
# out of sample: by the time B8 ran, the G- and H-series were both consumed dev data, so the
# 100%-of-draws paired result rides on masks the estimators have effectively seen. A genuinely
# fresh draw is the only clean confirmation. Seed 9973 is disjoint from 11 (G) and 4711 (H);
# tests assert the three draws share no mask.
FRESH_MASK_SEED_I = 9973

# Fourth mask draw (campaign J) -- the pass-4 out-of-sample confirmation draw. Seed 20260723
# (the pass-4 date) is disjoint from 11 (G), 4711 (H) and 9973 (I); tests/test_jseries.py asserts
# all four draws are pairwise disjoint. Frozen here BEFORE campaign J is launched.
FRESH_MASK_SEED_J = 20260723

# Fifth mask draw (campaign K) -- the pass-5 confirmation draw for the unified-leverage-family
# generality claim (P1). Seed 20260724 is disjoint from G/H/I/J; tests/test_jseries.py asserts it.
FRESH_MASK_SEED_K = 20260724

# Sixth mask draw (campaign L) -- the pass-6 capstone: a SECOND fresh draw for the C5 leverage win
# (RelatIF, the exact surrogate-LOO, and their ensemble), so the campaign-J C5 result is replicated
# out of sample on an independent draw. Seed 20260725, disjoint from G/H/I/J/K.
FRESH_MASK_SEED_L = 20260725

# Seventh draw (campaign M) -- the pass-7 resolving campaign. W0.2 (p7_design.py) measured the
# exchange rate between masks and seeds on this corpus: the paired sd is FLAT in depth (0.179 at
# 2 seeds vs 0.183 at 10, n=24) and falls as 1/sqrt(n) in masks, so a retrain spent on a new mask
# is worth about five spent on a new seed. M therefore buys masks: 144 of them at depth 2 (288
# retrains), which W0.2 predicts gives a paired sd ~0.045 and a 95% CI of width ~0.18 -- the
# resolution six passes of 24-mask draws never reached, for less GPU than any one of them.
#
# build_demo_masks is hardcoded to K_DEMO = 24 (its per-demo inclusion balance depends on it:
# n8 sums to 24*5), and changing the generator would invalidate every existing draw. So M is SIX
# sub-draws at six fresh seeds, concatenated. Each sub-draw is a valid 24-mask design from the
# UNMODIFIED generator, and W0.1 measured the between-draw variance component at zero on every
# set tested, so pooling them is exactly the operation the pooled analysis is built on. The first
# digit of the mask id records the sub-draw, so the mask bootstrap can stratify within it.
FRESH_MASK_SEED_M = (20260726, 20260727, 20260728, 20260729, 20260730, 20260731)
M_DEPTH = 2

# Campaign N (pass 8) -- CLUSTER grain. Masks come from p8_masks.manifest(): the complete
# enumeration of |S| in {4,5,6} minus Stage F's signatures, 278 masks. Depth 5, and the job list
# is ordered SEED-MAJOR so that every prefix is a complete balanced design -- the campaign is
# time-boxed and its prespecified stopping rule reads the largest fully-completed depth off the
# run directory. Mask-major ordering would leave a truncated run with some masks at full depth
# and others missing entirely, which is not a design.
N_DEPTH = 5


def fresh_demo_masks(seed=FRESH_MASK_SEED, prefix="H"):
    """The repo's own Stage-G generator at a different seed -> a fresh, disjoint mask draw."""
    masks, cnts, _, _ = MK.build_demo_masks(seed=seed)
    return [{"mask_id": f"{prefix}{k:03d}", "n_demos": len(m), "demos": m}
            for k, m in enumerate(masks)], cnts


def fresh_demo_masks_pooled(seeds=FRESH_MASK_SEED_M, prefix="M"):
    """-> 24*len(seeds) masks from len(seeds) independent sub-draws of the same generator.

    Mask ids are f"{prefix}{subdraw}{k:02d}" (M000..M023, M100..M123, ...), so the id is unique
    across sub-draws AND records which sub-draw it came from.
    """
    out = []
    for d, s in enumerate(seeds):
        masks, _, _, _ = MK.build_demo_masks(seed=s)
        out += [{"mask_id": f"{prefix}{d}{k:02d}", "n_demos": len(m), "demos": m,
                 "subdraw": f"{prefix}{d}", "seed": s} for k, m in enumerate(masks)]
    return out


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
    if campaign == "I":
        ms, _ = fresh_demo_masks(seed=FRESH_MASK_SEED_I, prefix="I")
        return [{"run_id": f"I_{m['mask_id']}_i{s}_o{s}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": s, "seed_order": s}
                for m in ms for s in B_SEEDS]      # same 10-seed depth as A and B
    if campaign == "J":
        ms, _ = fresh_demo_masks(seed=FRESH_MASK_SEED_J, prefix="J")
        return [{"run_id": f"J_{m['mask_id']}_i{s}_o{s}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": s, "seed_order": s}
                for m in ms for s in B_SEEDS]      # same 10-seed depth as A, B and I
    if campaign == "K":
        ms, _ = fresh_demo_masks(seed=FRESH_MASK_SEED_K, prefix="K")
        return [{"run_id": f"K_{m['mask_id']}_i{s}_o{s}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": s, "seed_order": s}
                for m in ms for s in B_SEEDS]      # same 10-seed depth as the others
    if campaign == "L":
        ms, _ = fresh_demo_masks(seed=FRESH_MASK_SEED_L, prefix="L")
        return [{"run_id": f"L_{m['mask_id']}_i{s}_o{s}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": s, "seed_order": s}
                for m in ms for s in B_SEEDS]      # same 10-seed depth as the others
    if campaign == "M":
        ms = fresh_demo_masks_pooled()
        return [{"run_id": f"M_{m['mask_id']}_i{s}_o{s}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": s, "seed_order": s}
                for m in ms for s in B_SEEDS[:M_DEPTH]]   # depth 2, by the W0.2 allocation result
    if campaign == "N":
        from if_repair import p8_masks as P8M
        ms = P8M.manifest()["masks"]
        return [{"run_id": f"N_{m['mask_id']}_i{sd}_o{sd}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": sd, "seed_order": sd}
                for sd in B_SEEDS[:N_DEPTH] for m in ms]      # SEED-MAJOR: see N_DEPTH note
    if campaign == "O":
        # PASS 9 -- sub-cluster grain at a FIXED 75-demo training set, so |S| variation does not
        # exist rather than being controlled after the fact. `p9_stratum_control` found campaign
        # N's pooled primary is substantially a training-set-SIZE effect: GradDot is a fixed
        # estimator yet still scores Kendall 0.353 pooled on outcomes shuffled within stratum,
        # against a real 0.475. Fixing the retained size removes that channel by construction.
        #
        # SEED-MAJOR, like campaign N: every prefix of the job list is a complete balanced design,
        # so the preregistered stopping rule can analyse the largest complete depth whenever the
        # box stops, without the analysis depending on which masks happened to finish.
        from if_repair import p9_masks as P9M
        ms = P9M.all_masks()
        return [{"run_id": f"O_{m['mask_id']}_i{sd}_o{sd}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": sd, "seed_order": sd}
                for sd in B_SEEDS[:P9M.DEPTH] for m in ms]

    if campaign == "P":
        # PASS 10 -- the k=15 CENSUS. The 33 C5-conditional 5of9 cluster masks that campaign N does
        # not hold, which completes the C(8,4)=70 conditional population on one outcome pipeline.
        #
        # DESCRIPTIVE ONLY. These 33 are the Stage F DISCOVERY DRAW for the cluster-grain hypothesis
        # (p8_cluster_grain's W1 scan selected it on them), so no alpha is available here -- see
        # BLOCKERS #28 and #31. Campaign N's 37 were the only unselected-upon masks of this kind that
        # will ever exist on this corpus, and pass 9 spent them. Depth 4 to match N's on-disk depth.
        from if_repair import p10_k15_census as P10C
        ms = P10C.manifest()["masks"]
        return [{"run_id": f"P_{m['mask_id']}_i{sd}_o{sd}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": sd, "seed_order": sd}
                for sd in B_SEEDS[:P10C.SEED_SLOTS] for m in ms]

    if campaign == "R":
        # PASS 10 -- the SECOND independent partition at k=3 and k=5. Identical to campaign O in
        # every respect except the partition seed, which is the whole experiment: pass 9's rungs rest
        # on one committed partition and therefore carry partition-sampling variance the k=15 rung
        # structurally cannot. p9_prereg named this check; the curve came out close (0.356 vs 0.365).
        # Verified at build time to share ZERO groups with pass 9's partition. Seed-major.
        from if_repair import p10_masks2 as P10M
        ms = P10M.all_masks()
        return [{"run_id": f"R_{m['mask_id']}_i{sd}_o{sd}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": sd, "seed_order": sd}
                for sd in B_SEEDS[:P10M.DEPTH] for m in ms]

    if campaign == "Q":
        # PASS 11 -- two ADDITIONAL seed slots on campaign O's identical 800 masks, taking that
        # campaign from depth 2 to depth 4. The masks are unchanged, so the depth-2 and depth-4 reads
        # differ ONLY in depth and the comparison is not confounded with a fresh draw.
        #
        # Why it is worth running now, having been deferred twice as a footnote: BLOCKERS #42 shows
        # the ratio is INFLATED at low depth, so pass 9/10's NEGATIVE results at depth 2 were already
        # conservative and a stricter denominator cannot rescue them. But #50's POSITIVE result -- the
        # datamodel clearing the bar, including out of partition -- is also at depth 2, and a positive
        # claim deserves the unbiased ceiling. This asks whether it survives one.
        #
        # Campaign O's own scoring stays frozen; this writes a separate result file.
        from if_repair import p9_masks as P9M
        ms = P9M.all_masks()
        return [{"run_id": f"Q_{m['mask_id']}_i{sd}_o{sd}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": sd, "seed_order": sd}
                for sd in B_SEEDS[2:4] for m in ms]

    if campaign == "S":
        # PASS 13 -- two ADDITIONAL seed slots on campaign R's identical 800 masks, depth 2 -> 4.
        #
        # #51 re-read the WITHIN-campaign datamodel at an unbiased ceiling and it survived, but the
        # cross-partition TRANSFER arm -- #50, the project's headline positive -- could not be
        # re-read, because the scoring side is campaign R and campaign R had only depth 2. Only the
        # fit side could improve. This buys the missing half: with R at depth 4, the transfer claim
        # can be stated at the same unbiased denominator as everything else.
        #
        # Campaign R's own preregistered scoring stays frozen (confirm_rseries.csv untouched); this
        # feeds a separate descriptive read, exactly as campaign Q did for campaign O.
        from if_repair import p10_masks2 as P10M
        ms = P10M.all_masks()
        return [{"run_id": f"S_{m['mask_id']}_i{sd}_o{sd}", "mask_id": m["mask_id"],
                 "demos": m["demos"], "seed_init": sd, "seed_order": sd}
                for sd in B_SEEDS[2:4] for m in ms]

    if campaign == "D":
        # W2 duels. Each duel is a pair of 68-demo masks differing in exactly ONE demo, trained
        # at MATCHED seed slots so the shared mask x init interaction differences out. The
        # manifest is frozen and committed by p7_duels.py --stage select before any training.
        # Pilot duels come first in the list, so `--limit 48` runs exactly the pilot arm.
        with open(os.path.join(HERE, "results", "p7_duel_manifest.json")) as fh:
            man = json.load(fh)
        out = []
        for d in sorted(man["duels"], key=lambda x: (x["arm"] != "pilot", x["duel_id"])):
            depth = man["pilot_depth"] if d["arm"] == "pilot" else man["full_depth"]
            for side in ("a", "b"):
                for s in man["duel_seeds"][:depth]:
                    out.append({"run_id": f"D_{d['duel_id']}{side}_i{s}_o{s}",
                                "mask_id": f"{d['duel_id']}{side}",
                                "demos": d[f"mask_{side}"], "seed_init": s, "seed_order": s})
        return out
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
    ap.add_argument("--campaign", required=True,
                    choices=["A", "B", "C", "I", "J", "K", "L", "M", "D", "N", "O", "P", "R", "Q", "S"])
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
