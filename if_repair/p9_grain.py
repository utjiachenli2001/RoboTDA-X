"""PASS 9 -- the GRAIN LADDER. Zero GPU.

Pass 8 moved the unit of attribution from one demo to a cluster of fifteen and the answer changed
sign in both directions: plain GradDot cleared the absolute half-ceiling bar for the first time in
eight passes, and every self-influence correction from passes 4-7 reversed. What pass 8 did NOT
learn is where in between the transition happens. It measured k = 1 and k = 15 and nothing else, so
"the unit was too small" is true but not yet actionable by anyone else -- a project designing its
own attribution benchmark cannot use "somewhere between 1 and 15".

This module builds the rungs in between.

WHY DIVISORS OF 15. The ladder is k in {1, 3, 5, 15}. Every rung divides the cluster size exactly,
so a cluster splits into 15/k equal groups with no ragged remainder. A grain of, say, 8 would leave
a 7-demo tail group and confound k with "the odd group at the end"; on a ladder whose entire purpose
is to isolate group SIZE, that is not a cost worth paying for a more evenly spaced x-axis. The
divisor structure also makes the ladder nest: the top rung IS the cluster, which is exactly the
estimand campaign N already measured, and the bottom rung IS the single demo, which is what
campaigns A-M measured. Pass 9 therefore buys two points and inherits two.

WHY WITHIN-CLUSTER. A group of 3 drawn inside one cluster and a group of 3 drawn across three
clusters are different objects, and pooling them would let composition move while k moves. The
primary ladder holds composition fixed -- every group lives inside exactly one cluster -- so k is the
only thing varying along it. The across-cluster arm is built too (`across=True`) but carries no
alpha; it answers a different question and is reported as descriptive.

WHY THE GROUPS ARE SHUFFLED BEFORE CHUNKING. `dataset.train_pool()` returns each cluster's demos in
manifest order, which is task-major: chunking it directly would make a group of 3 mean "three demos
of the same task" rather than "a third of a cluster". That would make small groups systematically
more homogeneous than large ones, and homogeneity -- not size -- would drive any trend the ladder
found. Shuffling within the cluster before chunking makes every group a miniature of its cluster in
expectation, which is the object the ladder is supposed to be about. The permutation is seeded per
cluster from a committed constant, so the partition is reproducible and is fixed before any mask is
drawn against it.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Frozen before any campaign-O mask exists. Changing it repartitions the corpus and invalidates
# every mask manifest built against it.
GROUP_SEED = 20260728

CLUSTER_SIZE = 15
# The rungs pass 9 BUYS. 1 and 15 are inherited from campaigns A-M and N respectively.
GRAINS = (3, 5)
LADDER = (1, 3, 5, 15)


def _cluster_size():
    _, by_c = dataset.train_pool()
    sizes = {len(v) for v in by_c.values()}
    assert len(sizes) == 1, f"clusters are not equal-sized: {sorted(sizes)}"
    return sizes.pop()


@functools.lru_cache(maxsize=None)
def groups(k, seed=GROUP_SEED, across=False):
    """Partition the 135-demo train pool into groups of exactly k demos.

    Returns a tuple of dicts: {group_id, cluster, k, demos}. `cluster` is None on the
    across-cluster arm, where a group deliberately spans clusters.
    """
    n = _cluster_size()
    cl = dataset.clusters()
    if not across and n % k:
        raise ValueError(f"k={k} does not divide the cluster size {n}; the ladder uses "
                         f"divisors of {n} so groups are equal and unragged")
    _, by_c = dataset.train_pool()

    if across:
        # One shared stream: the pool is permuted as a whole, so a group may span clusters.
        ids = [d for c in cl for d in by_c[c]]
        if len(ids) % k:
            raise ValueError(f"k={k} does not divide the pool size {len(ids)}")
        rng = np.random.default_rng([seed, 0xACC0])
        shuf = [ids[i] for i in rng.permutation(len(ids))]
        out = [{"group_id": f"x{k}_{gi:03d}", "cluster": None, "k": k,
                "demos": sorted(shuf[gi * k:(gi + 1) * k])}
               for gi in range(len(ids) // k)]
        return tuple(out)

    out = []
    for ci, c in enumerate(cl):
        demos = list(by_c[c])
        rng = np.random.default_rng([seed, ci])
        shuf = [demos[i] for i in rng.permutation(len(demos))]
        for gi in range(n // k):
            out.append({"group_id": f"g{k}_{c}_{gi}", "cluster": c, "k": k,
                        "demos": sorted(shuf[gi * k:(gi + 1) * k])})
    return tuple(out)


@functools.lru_cache(maxsize=None)
def index(k, across=False):
    """{group_id: group} for one rung."""
    return {g["group_id"]: g for g in groups(k, across=across)}


def n_groups(k, across=False):
    return len(groups(k, across=across))


def mask_demos(group_ids, k, across=False):
    """The demo list a retrain sees for a mask expressed as a set of group ids."""
    idx = index(k, across=across)
    missing = [g for g in group_ids if g not in idx]
    if missing:
        raise KeyError(f"unknown group ids at k={k}: {missing[:5]}")
    demos = [d for g in group_ids for d in idx[g]["demos"]]
    assert len(demos) == len(set(demos)), "groups overlap -- partition is broken"
    return sorted(demos)


def audit(k, across=False):
    gs = groups(k, across=across)
    ids, _ = dataset.train_pool()
    seen = [d for g in gs for d in g["demos"]]
    per_cluster = {}
    for g in gs:
        per_cluster[g["cluster"]] = per_cluster.get(g["cluster"], 0) + 1
    return {
        "k": k, "across": across, "n_groups": len(gs),
        "sizes": sorted({len(g["demos"]) for g in gs}),
        "covers_pool": sorted(seen) == sorted(ids),
        "disjoint": len(seen) == len(set(seen)),
        "groups_per_cluster": sorted(set(per_cluster.values())),
    }


def manifest(path=None, force=False):
    """The committed record of the partition every campaign-O mask is expressed in."""
    path = path or os.path.join(RESULTS, "p9_grain_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    out = {"pass": 9, "group_seed": GROUP_SEED, "ladder": list(LADDER), "bought": list(GRAINS),
           "grains": {}}
    for k in GRAINS:
        out["grains"][str(k)] = {"audit": audit(k), "groups": [dict(g) for g in groups(k)]}
    out["across"] = {str(k): {"audit": audit(k, across=True),
                              "groups": [dict(g) for g in groups(k, across=True)]}
                     for k in GRAINS}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    man = manifest(force=a.force)
    print(f"[p9/grain] seed={man['group_seed']} ladder={man['ladder']} bought={man['bought']}")
    for k in LADDER:
        au = audit(k) if k != 1 else {"k": 1, "n_groups": len(dataset.train_pool()[0]),
                                      "sizes": [1], "covers_pool": True, "disjoint": True,
                                      "groups_per_cluster": [CLUSTER_SIZE]}
        tag = "inherited" if k in (1, CLUSTER_SIZE) else "campaign O"
        print(f"  k={k:2d} groups={au['n_groups']:3d} sizes={au['sizes']} "
              f"per-cluster={au['groups_per_cluster']} covers={au['covers_pool']} "
              f"disjoint={au['disjoint']}  [{tag}]")
    for k in GRAINS:
        au = audit(k, across=True)
        print(f"  k={k:2d} groups={au['n_groups']:3d} ACROSS-cluster (descriptive arm)")


if __name__ == "__main__":
    main()
