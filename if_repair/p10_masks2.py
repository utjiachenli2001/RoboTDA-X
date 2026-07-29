"""PASS 10 -- campaign R: a SECOND, INDEPENDENT partition at k=3 and k=5. Zero GPU.

WHY THIS IS THE 18 HOURS.

Pass 9's k=3 and k=5 rungs rest on ONE committed partition of the corpus into groups
(`p9_grain.GROUP_SEED = 20260728`). Each cluster's 15 demos were shuffled once and chunked, so a
"group of 3" is one particular triple among many possible ones. That means those two rungs carry
partition-sampling variance that the k=15 rung structurally cannot -- a cluster has no composition
freedom. Pass 9 wrote that caveat down, and `p9_prereg.md` named this exact check as "the cheap
robustness check if the curve comes out close". It came out close: 0.356 vs 0.365.

The direction of this threat is UNKNOWN. A second partition could leave both rungs where they are,
or move them, and either outcome is informative. The alternative use of the same 18 hours -- a
depth-4 re-read of campaign O -- has a direction that is already MEASURED (BLOCKERS #42: the same
5of9 masks give ratio 0.666 at depth 2 and 0.496 at depth 4) and answers an allocation question
BLOCKERS #29 already half-answered ("the mean ratio-to-ceiling is flat across allocations"). Spending
the budget to confirm a known direction while an unknown-direction threat sits deferred is the wrong
allocation, so the partition draw wins.

WHAT IS HELD IDENTICAL TO CAMPAIGN O. Everything except the partition seed: 400 masks per grain, a
fixed 75 retained demos, all of C5's groups retained, complementary-pair construction for exact group
balance, exact demo-set-signature disjointness from every consumed campaign, depth 2 in seed slots
{4401, 4402}. Only `GROUP_SEED` changes. That is what makes the comparison a partition comparison
rather than a design comparison.

NOTE ON DISJOINTNESS. Campaign O's masks and campaign R's are drawn from DIFFERENT partitions, so
their group vocabularies differ -- but a mask is ultimately a demo list, and two different partitions
can produce the same 75-demo set. The signature check is therefore over demo sets and spans campaigns
A-P, not over group ids.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p9_grain as G  # noqa: E402
from if_repair import p9_masks as P9M  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# A DIFFERENT partition of the same corpus. This single constant is the whole experiment.
GROUP_SEED2 = 20260730
MASK_SEED2 = 20260731

TARGET = P9M.TARGET
RETAINED_DEMOS = P9M.RETAINED_DEMOS
GRAINS = P9M.GRAINS
N_MASKS = dict(P9M.N_MASKS)
DEPTH = P9M.DEPTH


def _plan(k, target=TARGET):
    gs = G.groups(k, seed=GROUP_SEED2)
    n_keep = RETAINED_DEMOS // k
    forced = [g["group_id"] for g in gs if g["cluster"] == target]
    free = [g["group_id"] for g in gs if g["cluster"] != target]
    n_free = n_keep - len(forced)
    if n_free <= 0 or n_free >= len(free):
        raise ValueError(f"k={k}: degenerate design")
    return gs, n_keep, forced, free, n_free


def _mask_demos(group_ids, k):
    idx = {g["group_id"]: g for g in G.groups(k, seed=GROUP_SEED2)}
    demos = [d for g in group_ids for d in idx[g]["demos"]]
    assert len(demos) == len(set(demos)), "groups overlap -- partition is broken"
    return sorted(demos)


def consumed_signatures():
    """Demo-set signatures spent by every prior campaign, INCLUDING campaign O's sub-cluster masks.

    Two different partitions can yield the same 75-demo set, so this must be over demo sets rather
    than over group ids.
    """
    sigs = set(P9M.consumed_signatures())
    for m in P9M.all_masks():
        sigs.add(frozenset(m["demos"]))
    try:
        from if_repair import p10_k15_census as P10C
        for m in P10C.manifest()["masks"]:
            sigs.add(frozenset(m["demos"]))
    except Exception:
        pass
    return sigs


def build(k, n_masks=None, seed=MASK_SEED2, target=TARGET):
    gs, n_keep, forced, free, n_free = _plan(k, target)
    n_masks = n_masks or N_MASKS[k]
    if n_masks % 2:
        raise ValueError("n_masks must be even -- complementary pairs")
    rng = np.random.default_rng([seed, k])
    consumed = consumed_signatures()
    whole = {c: {g["group_id"] for g in gs if g["cluster"] == c} for c in dataset.clusters()}

    out, collisions = [], 0
    for _ in range(n_masks // 2):
        perm = [free[i] for i in rng.permutation(len(free))]
        for side in (0, 1):
            sel = perm[:n_free] if side == 0 else perm[len(free) - n_free:]
            gids = sorted(forced + list(sel))
            demos = _mask_demos(gids, k)
            if len(demos) != RETAINED_DEMOS:
                raise AssertionError(f"k={k} mask has {len(demos)} demos")
            if frozenset(demos) in consumed:
                collisions += 1
                continue
            kept = set(gids)
            out.append({"mask_id": f"R{k}_{len(out):04d}", "k": k, "grain": f"g{k}",
                        "groups": gids, "n_groups": len(gids), "n_demos": len(demos),
                        "demos": demos, "stratum": f"k{k}_n{RETAINED_DEMOS}",
                        "n_whole_clusters_tiled": sum(1 for gg in whole.values() if gg <= kept)})
    return out, {"n_masks": len(out), "n_keep": n_keep, "n_forced": len(forced),
                 "n_free_chosen": n_free, "n_free_pool": len(free),
                 "signature_collisions_dropped": collisions}


def _assert_ok(masks, k, audit, target=TARGET):
    gs, n_keep, forced, free, n_free = _plan(k, target)
    sigs = [tuple(sorted(m["demos"])) for m in masks]
    assert len(sigs) == len(set(sigs)), f"k={k}: duplicate signatures"
    consumed = consumed_signatures()
    assert not any(frozenset(s) in consumed for s in sigs), f"k={k}: collides with a spent mask"
    for m in masks:
        assert m["n_demos"] == RETAINED_DEMOS
        assert set(forced) <= set(m["groups"]), f"{m['mask_id']} drops a {target} group"
        assert m["n_groups"] == n_keep
        assert m["n_whole_clusters_tiled"] < RETAINED_DEMOS // 15
    counts = {g: 0 for g in free}
    for m in masks:
        for g in m["groups"]:
            if g in counts:
                counts[g] += 1
    spread = max(counts.values()) - min(counts.values())
    assert spread <= 1, f"k={k}: inclusion not balanced, spread={spread}"
    audit["inclusion_per_free_group"] = sorted(set(counts.values()))
    audit["inclusion_spread"] = spread
    # the partition must actually DIFFER from pass 9's, or the campaign measures nothing
    a = {tuple(g["demos"]) for g in G.groups(k, seed=GROUP_SEED2)}
    b = {tuple(g["demos"]) for g in G.groups(k)}
    audit["groups_shared_with_pass9_partition"] = len(a & b)
    assert len(a & b) < len(a), f"k={k}: partition is identical to pass 9's"


def manifest(path=None, force=False):
    path = path or os.path.join(RESULTS, "p10_mask_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    runs = os.path.join(HERE, "runs", "campaigns", "R")
    if os.path.isdir(runs) and os.listdir(runs):
        raise SystemExit("campaign R already has runs -- the manifest is frozen")
    grains, audits = {}, {}
    for k in GRAINS:
        ms, au = build(k)
        _assert_ok(ms, k, au)
        grains[str(k)], audits[str(k)] = ms, au
    out = {"pass": 10, "campaign": "R", "grain": "sub-cluster", "target": TARGET,
           "retained_demos": RETAINED_DEMOS, "depth": DEPTH,
           "group_seed": GROUP_SEED2, "mask_seed": MASK_SEED2,
           "pass9_group_seed": G.GROUP_SEED, "grains": list(GRAINS),
           "purpose": "second independent partition -- the robustness check p9_prereg named",
           "audit": audits, "masks": grains}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def all_masks(man=None):
    man = man or manifest()
    return [m for k in man["grains"] for m in man["masks"][str(k)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    man = manifest(force=a.force)
    print(f"[p10/masks2] campaign R -- SECOND partition (group_seed {man['group_seed']} vs pass 9's "
          f"{man['pass9_group_seed']}), fixed {man['retained_demos']} demos")
    tot = 0
    for k in man["grains"]:
        au = man["audit"][str(k)]
        n = len(man["masks"][str(k)])
        tot += n
        print(f"  k={k:2d}  masks={n:4d}  keep {au['n_keep']} ({au['n_forced']} forced + "
              f"{au['n_free_chosen']} of {au['n_free_pool']})  spread={au['inclusion_spread']}  "
              f"groups shared with pass 9: {au['groups_shared_with_pass9_partition']}"
              f"/{au['n_free_pool'] + au['n_forced']}  dropped={au['signature_collisions_dropped']}")
    print(f"  total {tot} masks x depth {man['depth']} = {tot * man['depth']} retrains "
          f"~ {tot * man['depth'] / 88:.1f} h wall")


if __name__ == "__main__":
    main()
