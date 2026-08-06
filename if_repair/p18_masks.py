"""PASS 18 -- campaign T masks: the corpus-size ladder. Zero GPU.

Four rungs (N = 50/100/200/370), two independent partitions each, masks at exactly 50% retained.

BALANCE BY CONSTRUCTION, NOT BY REPAIR -- two mechanisms, one per regime.

  rung 50 is ENUMERATED. Ten groups, a mask keeps five, so the whole mask space is C(10,5) = 252
  and campaign T takes all of it. This is a combinatorial cap of the same kind as BLOCKERS #38
  and #45 -- no GPU budget enlarges it -- but here it is a gift rather than a wall: complete
  enumeration puts every group in exactly C(9,4) = 126 of the 252 masks, which is exact inclusion
  balance for free, and it is closed under complementation, so the complementary-pair structure
  the other rungs have to construct is already present.

  rungs 100/200/370 use COMPLEMENTARY PAIRS, as campaign O did: permute the groups, take the
  first half as one mask and the second half as its partner. Every round places each group in
  exactly one of the two masks, so after R rounds every group has appeared in exactly R of the 2R
  masks -- spread zero, with no swap-repair loop.

WHY EXACTLY 50% RETAINED, AND WHY THAT IS NOT THE PLAN'S 55%. Complementary pairing at a fixed
retained size forces the two sides to be the same size, which forces 50%. Campaign O could sit at
55.6% only because its conditioning rule FORCED the target cluster's groups into every mask, which
broke the symmetry. This ladder's outcome target is a held-out bank outside the training pool
entirely, so there is nothing to force, and 50% is the fixed point. BLOCKERS #41 is satisfied
either way and more cleanly here: at a given rung every mask trains on exactly N/2 demos, so
training-set size does not vary WITHIN a rung at all.

EQUAL MASK COUNTS ACROSS RUNGS, AND THE ONE THING THAT COSTS. 400 masks at every rung the cap
allows means every rung's ratio is estimated at the same precision, which is what a TREND estimate
wants. It does mean masks-per-coefficient falls along the ladder (25.2 / 20 / 10 / 5.4), since the
coefficient count grows with N. KTD4 worried about that regime drifting. It was measured:
BLOCKERS #55 held the grain fixed and varied masks-per-coefficient over a 6x range, and the slopes
were -0.010 and -0.001 with CIs containing zero -- the over-determination effect is unsupported.
Reported per rung anyway, so a reader can see it did not move rather than take it on trust.

CROSS-PARTITION SIGNATURE COLLISIONS ARE SHARED, NOT DROPPED. At rung 50 both partitions enumerate
their whole space, so a 25-demo training set can occasionally be expressible as five partition-A
groups AND five partition-B groups. Dropping such a mask would break the complete enumeration and
with it the exact balance; running it twice would burn GPU to train the identical model twice, since
the same demo set at the same seed IS the same model. So a mask's run identity is a hash of its DEMO
SET, and the two partitions simply share that retrain. Zero design cost, and the saving is free.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p18_corpus as C  # noqa: E402

D.add_repo_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Frozen before any campaign-T retrain exists.
MASK_SEED = 20260810
DEPTH = 2                    # even (BLOCKERS #39), allocation-optimal (BLOCKERS #29)
SEEDS = (4401, 4402)         # the first DEPTH slots of retrain.B_SEEDS, so depth is comparable
RETAINED_FRACTION = 0.5

# Masks per rung PER PARTITION. Rung 50 is the complete space; the rest are equal.
N_MASKS = {50: 252, 100: 400, 200: 400, 370: 400}


def signature(demos):
    """Stable short id for a TRAINING SET. Two masks with the same signature are the same run."""
    h = hashlib.sha1("\n".join(sorted(demos)).encode()).hexdigest()
    return h[:12]


def _keep_count(n):
    g = n // C.GROUP_SIZE
    if g % 2:
        raise ValueError(f"rung {n} has an odd group count ({g}); complementary pairs need even")
    return g // 2


def build(n, partition="A", n_masks=None, seed=MASK_SEED):
    gs = C.groups(n, partition)
    gids = [g["group_id"] for g in gs]
    keep = _keep_count(n)
    n_masks = n_masks or N_MASKS[n]
    full = math.comb(len(gids), keep)

    out = []
    if n_masks >= full:
        # complete enumeration -- exact balance and closure under complementation for free
        for sel in itertools.combinations(range(len(gids)), keep):
            out.append(sorted(gids[i] for i in sel))
        mode = "enumerated"
    else:
        if n_masks % 2:
            raise ValueError("n_masks must be even -- masks are built in complementary pairs")
        rng = np.random.default_rng([seed, n, {"A": 0, "B": 1}[partition]])
        seen = set()
        while len(out) < n_masks:
            perm = rng.permutation(len(gids))
            for side in (0, 1):
                sel = perm[:keep] if side == 0 else perm[keep:]
                m = sorted(gids[i] for i in sel)
                key = tuple(m)
                if key in seen:          # a repeat would break exact balance for its round
                    continue
                seen.add(key)
                out.append(m)
        mode = "complementary_pairs"

    masks = []
    for k, gg in enumerate(out):
        demos = C.mask_demos(gg, n, partition)
        if len(demos) != n // 2:
            raise AssertionError(f"rung {n}{partition} mask keeps {len(demos)}, expected {n // 2}")
        masks.append({"mask_id": f"T{n}{partition}_{k:04d}", "rung": n, "partition": partition,
                      "groups": gg, "n_groups": len(gg), "n_demos": len(demos),
                      "demos": demos, "sig": signature(demos)})
    return masks, mode


def all_masks():
    """Every campaign-T mask, all rungs and both partitions."""
    out = []
    for n in C.RUNGS:
        for p in C.PARTITIONS:
            out += build(n, p)[0]
    return out


def jobs():
    """The retrain job list, DEDUPED BY (signature, seed).

    Ordered SEED-MAJOR and then rung-major so that every prefix of the list is a complete
    balanced design -- campaign N's rule (see retrain.N_DEPTH). A time-boxed run can then be
    analysed at the largest fully-completed depth without the analysis depending on which masks
    happened to finish.
    """
    out, seen = [], set()
    for sd in SEEDS:
        for n in C.RUNGS:
            for p in C.PARTITIONS:
                for m in build(n, p)[0]:
                    key = (m["sig"], sd)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"run_id": f"T{n}_{m['sig']}_i{sd}_o{sd}", "mask_id": m["mask_id"],
                                "rung": n, "partition": p, "sig": m["sig"],
                                "demos": m["demos"], "seed_init": sd, "seed_order": sd})
    return out


# --------------------------------------------------------------------------- audit
def balance(n, partition="A"):
    """{group_id: how many masks retain it}. The spread must be 0."""
    cnt = {g["group_id"]: 0 for g in C.groups(n, partition)}
    for m in build(n, partition)[0]:
        for g in m["groups"]:
            cnt[g] += 1
    return cnt


def audit(n, partition="A"):
    ms, mode = build(n, partition)
    b = balance(n, partition)
    sigs = [m["sig"] for m in ms]
    return {
        "rung": n, "partition": partition, "mode": mode, "n_masks": len(ms),
        "retained_demos": sorted({m["n_demos"] for m in ms}),
        "retained_groups": sorted({m["n_groups"] for m in ms}),
        "unique_signatures": len(set(sigs)),
        "balance_spread": max(b.values()) - min(b.values()),
        "balance_value": sorted(set(b.values())),
        "masks_per_coefficient": round(len(ms) / len(C.groups(n, partition)), 1),
    }


def cross_partition_shared_signatures(n):
    a = {m["sig"] for m in build(n, "A")[0]}
    b = {m["sig"] for m in build(n, "B")[0]}
    return sorted(a & b)


def cross_rung_shared_signatures():
    """Must be empty: rungs retain 25/50/100/185 demos, so a collision is impossible by size.
    Asserted rather than assumed -- "unlikely" is not a control (p9_masks)."""
    per = {}
    for n in C.RUNGS:
        per[n] = {m["sig"] for p in C.PARTITIONS for m in build(n, p)[0]}
    bad = {}
    for a, b in itertools.combinations(C.RUNGS, 2):
        sh = per[a] & per[b]
        if sh:
            bad[f"{a}x{b}"] = sorted(sh)[:5]
    return bad


def manifest(path=None, force=False):
    path = path or os.path.join(RESULTS, "p18_mask_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    J = jobs()
    out = {
        "pass": 18, "campaign": "T", "mask_seed": MASK_SEED, "depth": DEPTH,
        "seeds": list(SEEDS), "retained_fraction": RETAINED_FRACTION,
        "n_masks": {str(k): v for k, v in N_MASKS.items()},
        "audit": {f"{n}{p}": audit(n, p) for n in C.RUNGS for p in C.PARTITIONS},
        "cross_partition_shared_signatures": {str(n): cross_partition_shared_signatures(n)
                                              for n in C.RUNGS},
        "cross_rung_shared_signatures": cross_rung_shared_signatures(),
        "n_jobs": len(J),
        "n_masks_total": sum(N_MASKS[n] for n in C.RUNGS) * len(C.PARTITIONS),
        "masks": {f"{n}{p}": build(n, p)[0] for n in C.RUNGS for p in C.PARTITIONS},
    }
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(f"[p18/masks] campaign T -- seed={MASK_SEED} depth={DEPTH} seeds={SEEDS}")
    total = 0
    for n in C.RUNGS:
        for p in C.PARTITIONS:
            au = audit(n, p)
            total += au["n_masks"]
            print(f"  rung N={n:3d} {p}  masks={au['n_masks']:3d} [{au['mode']:19s}] "
                  f"keeps {au['retained_demos']} demos / {au['retained_groups']} groups  "
                  f"uniq_sig={au['unique_signatures']:3d} balance_spread={au['balance_spread']} "
                  f"m/coef={au['masks_per_coefficient']}")
        sh = cross_partition_shared_signatures(n)
        print(f"    rung {n}: A|B shared training sets = {len(sh)} (shared retrains, not dropped)")
    bad = cross_rung_shared_signatures()
    print(f"[p18/masks] cross-rung signature collisions: {bad if bad else 'none'}")
    J = jobs()
    print(f"[p18/masks] {total} masks -> {len(J)} retrains after dedupe "
          f"(naive {total * DEPTH}); at 85/h that is {len(J) / 85:.1f} h")
    man = manifest(force=a.force)
    print(f"[p18/masks] manifest: {man['n_jobs']} jobs committed")


if __name__ == "__main__":
    main()
