"""PASS 9 -- campaign O masks: SUB-CLUSTER grain at a FIXED training-set size. Zero GPU.

WHY THIS DESIGN AND NOT THE ONE PASS 9 STARTED WITH.

`p9_stratum_control.py` found that campaign N's committed primary is substantially a training-set
SIZE effect. GradDot is a fixed estimator, yet on outcomes shuffled within stratum it still scores
Kendall 0.353 pooled against a real 0.475 -- because |S| in {4,5,6} sets the training set to
60/75/90 demos, that moves the outcome directly, and every estimator's mask prediction is a SUM
over kept demos and so grows with the count too. Correlate the two pooled and both sides earn
credit for arithmetic. Within stratum, where the size is constant, the permutation null collapses
to ~0.000 and the half-ceiling bar is NOT cleared in any stratum (ratios 0.411 / 0.496 / 0.317).

The obvious repair -- report within stratum -- runs straight into BLOCKERS #38: the cluster mask
axis is CAPPED at C(9,4)+C(9,5)+C(9,6) = 336 subsets and campaign N consumed 278 of them, so the
per-stratum n is stuck at 56/37/56 and the ratio CIs are [0.09, 0.78], [0.12, 0.94], [-0.02, 0.72].
Nothing can be added at those sizes. The cluster grain cannot answer its own question.

SUB-CLUSTER GRAIN IS THE ONLY PLACE THE QUESTION IS ANSWERABLE, and the reason is combinatorial
rather than conceptual. At k=3 a mask that keeps 75 demos keeps 25 of 45 groups: C(45,25) ~ 3e12.
At k=5 it keeps 15 of 27: C(27,15) ~ 1.7e7. So hundreds of masks can be drawn at EXACTLY 75
retained demos, and |S| variation is not controlled after the fact -- it does not exist. That turns
the confound from a covariate into a non-event, and it is the whole reason campaign O is worth GPU.

THE ESTIMAND. Holding the training set at 75 demos, how does the measurability of attribution
depend on the GRAIN at which demos are removed? Three rungs:

    k = 3   25 of 45 groups     campaign O, fresh
    k = 5   15 of 27 groups     campaign O, fresh
    k = 15   5 of  9 clusters   campaign N's 5of9 stratum, ALREADY ON DISK, zero GPU

k=1 (demo grain) is deliberately NOT a rung. Demo-grain masks retain 7-8 demos of every cluster and
never remove one whole (`src/masks.py` stratification), so they are a different removal geometry
rather than the same estimand at smaller k. It is shown as a labelled reference and carries no alpha.

CONDITIONING, preregistered. ALL of the target cluster's groups are retained -- 5 of 5 at k=3, 3 of
3 at k=5, the whole cluster at k=15. This is the sub-cluster reading of campaign N's
"target-in-mask" and it is fixed here rather than left to the implementer, because the alternative
(">= 1 group retained") is a different estimand with a different n.

BALANCE BY CONSTRUCTION, NOT BY REPAIR. Masks are built in COMPLEMENTARY PAIRS: permute the
non-target groups, take the first half as one mask's selection and the second half as its partner's.
Every round therefore puts each non-target group in exactly one of the two masks, so after R rounds
every group has appeared in exactly R masks -- exact balance with no swap-repair loop, and uniform
co-inclusion in expectation from the permutation. Pass 8 got exact balance from complete
enumeration; the sub-cluster space cannot be enumerated, and complementary pairing is the next best
thing.

DISJOINTNESS. A sub-cluster mask whose retained groups happen to be exactly the full group sets of
five clusters IS a campaign-N |S|=5 signature -- an accidental re-run of a mask already consumed.
Vanishingly unlikely under this construction, and asserted anyway, because "unlikely" is not a
control.
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

D.add_repo_paths()
import dataset  # noqa: E402
import masks as MK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

MASK_SEED = 20260729
TARGET = "C5"
RETAINED_DEMOS = 75          # matches campaign N's 5of9 stratum exactly
GRAINS = (3, 5)
N_MASKS = {3: 400, 5: 400}   # depth 2 -> 1600 retrains ~ 18 h wall at the measured 88/h
DEPTH = 2                    # even (BLOCKERS #39), allocation-optimal (BLOCKERS #29)


def _plan(k, target=TARGET):
    """How many groups a 75-demo mask keeps, and which are forced by the conditioning rule."""
    gs = G.groups(k)
    n_keep = RETAINED_DEMOS // k
    forced = [g["group_id"] for g in gs if g["cluster"] == target]
    free = [g["group_id"] for g in gs if g["cluster"] != target]
    n_free = n_keep - len(forced)
    if n_free <= 0 or n_free >= len(free):
        raise ValueError(f"k={k}: degenerate design, n_free={n_free} of {len(free)}")
    return gs, n_keep, forced, free, n_free


def consumed_signatures():
    """Demo-set signatures already spent, so campaign O cannot re-run one.

    Cluster masks (Stage F + campaign N) are the ones that can actually collide: a sub-cluster mask
    is only reachable as a cluster mask if its groups tile whole clusters.
    """
    sigs = set()
    for m in MK.cluster_mask_manifest()["masks"]:
        sigs.add(frozenset(m["demos"]))
    try:
        from if_repair import p8_masks as P8M
        for m in P8M.manifest()["masks"]:
            sigs.add(frozenset(m["demos"]))
    except Exception:
        pass
    return sigs


def build(k, n_masks=None, seed=MASK_SEED, target=TARGET):
    gs, n_keep, forced, free, n_free = _plan(k, target)
    n_masks = n_masks or N_MASKS[k]
    if n_masks % 2:
        raise ValueError("n_masks must be even -- masks are built in complementary pairs")
    rng = np.random.default_rng([seed, k])
    consumed = consumed_signatures()
    _, by_c = dataset.train_pool()
    whole_cluster_groups = {c: {g["group_id"] for g in gs if g["cluster"] == c}
                            for c in dataset.clusters()}

    out, collisions = [], 0
    for r in range(n_masks // 2):
        perm = [free[i] for i in rng.permutation(len(free))]
        for side in (0, 1):
            sel = perm[:n_free] if side == 0 else perm[len(free) - n_free:]
            gids = sorted(forced + list(sel))
            demos = G.mask_demos(gids, k)
            if len(demos) != RETAINED_DEMOS:
                raise AssertionError(f"k={k} mask has {len(demos)} demos, expected "
                                     f"{RETAINED_DEMOS}")
            if frozenset(demos) in consumed:
                collisions += 1
                continue
            kept = set(gids)
            tiles = [c for c, gg in whole_cluster_groups.items() if gg <= kept]
            out.append({
                "mask_id": f"O{k}_{len(out):04d}", "k": k, "grain": f"g{k}",
                "groups": gids, "n_groups": len(gids), "n_demos": len(demos), "demos": demos,
                "stratum": f"k{k}_n{RETAINED_DEMOS}",
                "n_whole_clusters_tiled": len(tiles),
            })
    return out, {"n_masks": len(out), "n_keep": n_keep, "n_forced": len(forced),
                 "n_free_chosen": n_free, "n_free_pool": len(free),
                 "signature_collisions_dropped": collisions}


def _assert_ok(masks, k, audit, target=TARGET):
    gs, n_keep, forced, free, n_free = _plan(k, target)
    sigs = [frozenset(m["demos"]) for m in masks]
    assert len(sigs) == len({tuple(sorted(s)) for s in sigs}), f"k={k}: duplicate mask signatures"
    consumed = consumed_signatures()
    assert not any(s in consumed for s in sigs), f"k={k}: collides with a consumed cluster mask"
    for m in masks:
        assert m["n_demos"] == RETAINED_DEMOS
        assert set(forced) <= set(m["groups"]), f"{m['mask_id']} drops a {target} group"
        assert m["n_groups"] == n_keep
    counts = {g: 0 for g in free}
    for m in masks:
        for g in m["groups"]:
            if g in counts:
                counts[g] += 1
    spread = max(counts.values()) - min(counts.values())
    assert spread <= 1, f"k={k}: group inclusion not balanced, spread={spread}"
    audit["inclusion_per_free_group"] = sorted(set(counts.values()))
    audit["inclusion_spread"] = spread
    # every mask keeps exactly one training-set size -- the entire point of the design
    assert len({m["n_demos"] for m in masks}) == 1
    audit["retained_demos"] = RETAINED_DEMOS
    audit["max_whole_clusters_tiled"] = max(m["n_whole_clusters_tiled"] for m in masks)


def manifest(path=None, force=False):
    path = path or os.path.join(RESULTS, "p9_mask_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    runs = os.path.join(HERE, "runs", "campaigns", "O")
    if os.path.isdir(runs) and os.listdir(runs):
        raise SystemExit("campaign O already has runs -- the manifest is frozen, refusing to "
                         "rebuild it (see p9_prereg.md)")
    grains, audits = {}, {}
    for k in GRAINS:
        ms, au = build(k)
        _assert_ok(ms, k, au)
        grains[str(k)], audits[str(k)] = ms, au
    out = {"pass": 9, "campaign": "O", "grain": "sub-cluster", "target": TARGET,
           "retained_demos": RETAINED_DEMOS, "depth": DEPTH, "mask_seed": MASK_SEED,
           "group_seed": G.GROUP_SEED, "grains": list(GRAINS),
           "conditioning": f"ALL of {TARGET}'s groups retained",
           "construction": "complementary-pair permutation blocks -> exact group balance",
           "audit": audits, "masks": {k: v for k, v in grains.items()}}
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
    print(f"[p9/masks] campaign O -- sub-cluster grain, FIXED {man['retained_demos']} retained "
          f"demos, conditioning: {man['conditioning']}")
    tot = 0
    for k in man["grains"]:
        au = man["audit"][str(k)]
        n = len(man["masks"][str(k)])
        tot += n
        print(f"  k={k:2d}  masks={n:4d}  keep {au['n_keep']} groups "
              f"({au['n_forced']} forced + {au['n_free_chosen']} of {au['n_free_pool']})  "
              f"balance spread={au['inclusion_spread']} "
              f"incl/group={au['inclusion_per_free_group']}  "
              f"dropped_collisions={au['signature_collisions_dropped']}")
    print(f"  total {tot} masks x depth {man['depth']} = {tot * man['depth']} retrains "
          f"~ {tot * man['depth'] / 88:.1f} h wall at the measured 88/h")
    print(f"  k=15 comparison rung: campaign N's 5of9 stratum (75 demos), already on disk")


if __name__ == "__main__":
    main()
