"""PASS 18 -- campaign U masks: the FIXED-RETAINED selection ladder. Zero GPU.

Campaign T (50% removal) is dead: its variance pilot measured a ceiling of ZERO at pool 370,
because training on 185 of 370 demos puts the model past the point where which demos it received
still matters. Campaign U instead holds the RETAINED count fixed at 25 demonstrations and grows the
candidate pool, so every retrain sits at an operating point the four-pool conditioned gate measured
healthy (r = 0.872 / 0.866 / 0.805 / 0.920).

WHY THE CONSTRUCTION IS i.i.d. AND NOT COMPLEMENTARY PAIRS. The complement of "retain 25 of 370" is
"retain 345", which is not a campaign-U mask -- there are no complementary pairs in this design, and
74 groups do not divide into 5-group rounds anyway (74 = 14x5 + 4). Campaign T's exact inclusion
balance is therefore unobtainable here. Masks are drawn i.i.d. uniformly over 5-group subsets and
inclusion balance is STATISTICAL, with the realised spread reported rather than asserted to be
zero. The gain is that masks become exchangeable, which is what makes the mask bootstrap valid --
the property complementary pairing destroyed.

THE CONDITIONING RULE (prereg 2.2). Five groups of five need not cover all ten tasks: 31.2% of
unconditioned 5-group masks miss at least one task entirely, and a missing task moves 10% of the
eval bank hard. Task coverage moves the outcome AND every estimator's summed prediction, which is
BLOCKERS #41 -- the nuisance axis that credits both sides. Every mask must therefore contain at
least one demonstration of each of the 10 tasks, enforced by rejection sampling, with the rejection
rate reported.

CROSS-POOL SIGNATURE COLLISIONS ARE SHARED, NOT DROPPED. Pools nest and every mask is a 25-demo
set, so the same training set is reachable from more than one pool. Campaign T could assert
cross-rung disjointness by size; campaign U cannot and must not -- a demo set at a given seed is the
same model, so `jobs()` dedupes by (signature, seed) and the pools share that retrain.

ALLOCATION. 184 / 368 / 736 / 1361 masks per partition -- 18.4 per coefficient at every pool. The
constant is not cosmetic: at a fixed mask count, per-coefficient information would rise along the
ladder and bias the datamodel arm, invisibly to the permutation null. The x1.84 scale is set by the
preregistered power requirement (MDE <= 0.0502 under BOTH partition assumptions).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p18_corpus as C  # noqa: E402
from if_repair.p18_masks import signature  # noqa: E402

D.add_repo_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# --- frozen before any campaign-U retrain exists ---------------------------------------
MASK_SEED = 20260812
RETAINED_DEMOS = 25
RETAINED_GROUPS = RETAINED_DEMOS // C.GROUP_SIZE          # 5
DEPTH = 2
SEEDS = (4401, 4402)                                      # depth-2 slots, as campaigns O/R used
RESERVE_PAIRS = ((4403, 4404), (4405, 4406))              # prereg 3.4 disposition
MASKS_PER_COEFFICIENT = 18.4                              # x1.84 power allocation

N_MASKS = {50: 184, 100: 368, 200: 736, 370: 1361}


def _n_masks(pool):
    return N_MASKS[pool]


def build(pool, partition="A", n_masks=None, seed=MASK_SEED):
    """-> (masks, stats). i.i.d. uniform 5-group subsets, task-coverage conditioned."""
    gs = C.groups(pool, partition)
    n_masks = n_masks or _n_masks(pool)
    rng = np.random.default_rng([seed, pool, {"A": 0, "B": 1}[partition]])
    out, seen, drawn, rejected = [], set(), 0, 0
    while len(out) < n_masks:
        drawn += 1
        sel = rng.permutation(len(gs))[:RETAINED_GROUPS]
        gids = sorted(gs[i]["group_id"] for i in sel)
        demos = C.mask_demos(gids, pool, partition)
        if len({d.split("/")[1] for d in demos}) != C.N_TASKS:
            rejected += 1
            continue                                        # prereg 2.2 coverage rule
        sig = signature(demos)
        if sig in seen:
            rejected += 1
            continue                                        # duplicate training set
        seen.add(sig)
        out.append({"mask_id": f"U{pool}{partition}_{len(out):05d}", "pool": pool,
                    "partition": partition, "groups": gids, "n_groups": len(gids),
                    "n_demos": len(demos), "demos": demos, "sig": sig})
    return out, {"drawn": drawn, "rejected": rejected,
                 "rejection_rate": round(rejected / drawn, 4) if drawn else 0.0}


def jobs(seeds=SEEDS):
    """Retrain job list, deduped by (signature, seed), SEED-MAJOR then pool-major.

    Seed-major keeps every prefix a complete balanced design, so a time-boxed run is analysable
    at the largest completed depth (campaign N's rule). Depth 2 means a stop before the second
    seed completes yields no primary result -- the ceiling is undefined at odd depth (#39) -- and
    the prereg accepts that in advance.
    """
    out, seen = [], set()
    for sd in seeds:
        for pool in C.RUNGS:
            for p in C.PARTITIONS:
                for m in build(pool, p)[0]:
                    key = (m["sig"], sd)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"run_id": f"U{pool}_{m['sig']}_i{sd}_o{sd}",
                                "mask_id": m["mask_id"], "pool": pool, "partition": p,
                                "sig": m["sig"], "demos": m["demos"],
                                "seed_init": sd, "seed_order": sd})
    return out


# --------------------------------------------------------------------------- audit
def inclusion(pool, partition="A"):
    cnt = {g["group_id"]: 0 for g in C.groups(pool, partition)}
    for m in build(pool, partition)[0]:
        for g in m["groups"]:
            cnt[g] += 1
    return cnt


def audit(pool, partition="A"):
    ms, st = build(pool, partition)
    inc = inclusion(pool, partition)
    G = len(C.groups(pool, partition))
    p = RETAINED_GROUPS / G
    n = len(ms)
    return {
        "pool": pool, "partition": partition, "n_masks": n, "n_groups": G,
        "retained_demos": sorted({m["n_demos"] for m in ms}),
        "unique_signatures": len({m["sig"] for m in ms}),
        "all_cover_10_tasks": all(
            len({d.split("/")[1] for d in m["demos"]}) == C.N_TASKS for m in ms),
        "rejection_rate": st["rejection_rate"],
        "masks_per_coefficient": round(n / G, 1),
        "inclusion_mean": round(float(np.mean(list(inc.values()))), 1),
        "inclusion_sd": round(float(np.std(list(inc.values()), ddof=1)), 2),
        "inclusion_expected_sd": round(float(np.sqrt(n * p * (1 - p))), 2),
        "info_per_coefficient": round(n * p * (1 - p), 1),
    }


def cross_pool_shared_signatures():
    per = {}
    for pool in C.RUNGS:
        per[pool] = {m["sig"] for p in C.PARTITIONS for m in build(pool, p)[0]}
    out = {}
    for a in C.RUNGS:
        for b in C.RUNGS:
            if a < b and (per[a] & per[b]):
                out[f"{a}x{b}"] = len(per[a] & per[b])
    return out


def manifest(path=None, force=False):
    path = path or os.path.join(RESULTS, "p18_campaign_u_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    J = jobs()
    out = {"pass": 18, "campaign": "U", "mask_seed": MASK_SEED, "depth": DEPTH,
           "seeds": list(SEEDS), "reserve_pairs": [list(x) for x in RESERVE_PAIRS],
           "retained_demos": RETAINED_DEMOS, "n_masks": N_MASKS,
           "audit": {f"{n}{p}": audit(n, p) for n in C.RUNGS for p in C.PARTITIONS},
           "cross_pool_shared_signatures": cross_pool_shared_signatures(),
           "n_jobs": len(J),
           "masks": {f"{n}{p}": build(n, p)[0] for n in C.RUNGS for p in C.PARTITIONS}}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(f"[p18/U] campaign U -- seed={MASK_SEED} retained={RETAINED_DEMOS} depth={DEPTH} "
          f"seeds={SEEDS} reserves={RESERVE_PAIRS}")
    tot = 0
    for pool in C.RUNGS:
        for p in C.PARTITIONS:
            au = audit(pool, p)
            tot += au["n_masks"]
            print(f"  pool={pool:3d} {p}  masks={au['n_masks']:5d}  keeps {au['retained_demos']} "
                  f"demos  cover10={au['all_cover_10_tasks']}  reject={au['rejection_rate']:.3f}  "
                  f"m/coef={au['masks_per_coefficient']:5.1f}  info={au['info_per_coefficient']:.1f}  "
                  f"incl sd={au['inclusion_sd']} (exp {au['inclusion_expected_sd']})")
    sh = cross_pool_shared_signatures()
    print(f"[p18/U] cross-pool shared training sets: {sh if sh else 'none'} (shared, not dropped)")
    J = jobs()
    print(f"[p18/U] {tot} masks -> {len(J)} retrains (naive {tot * DEPTH}); "
          f"at 564.7/h on h100-2 that is {len(J) / 564.7:.1f} h")
    man = manifest(force=a.force)
    print(f"[p18/U] manifest: {man['n_jobs']} jobs committed")


if __name__ == "__main__":
    main()
