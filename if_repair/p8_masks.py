"""PASS 8 -- fresh CLUSTER masks for campaign N. Zero GPU.

Stage F drew 72 of the 126 possible 5-of-9 cluster masks by randomized construction plus swap
repair (`src/masks.build_cluster_masks`). That machinery exists because the demo-grain space is
astronomically large and must be sampled. The cluster-grain space is not:

    C(9,4) = 126        C(9,5) = 126        C(9,6) = 84        total 336

At that size the space can be ENUMERATED, which is strictly better than sampling it. A complete
enumeration of a stratum is exactly balanced by construction -- every cluster appears in
C(8, per-1) of the masks -- and its co-inclusion matrix is exactly uniform, so the swap-repair
loop has nothing left to fix. Disjointness from Stage F becomes exact set difference rather than
a probabilistic argument.

The 5-of-9 complement inherits exact balance for free: the full stratum puts every cluster in
C(8,4) = 70 masks and Stage F's 72 were balanced at 40 each, so the 54 that remain are balanced
at exactly 30 each. That is asserted below rather than assumed.

WHY MULTIPLE STRATA. 54 fresh 5-of-9 masks is not a powered confirmation -- p8_design's
allocation curve puts the paired sd at the 5-of-9 cap around 0.07-0.09, barely better than Stage
F itself. Widening to |S| in {4,5,6} raises the fresh supply to 264. The cost is that |S| changes
the training-set size (60/75/90 demos), which moves the outcome directly and would swamp the
attribution ordering if pooled naively. |S| is therefore recorded on every mask and used as a
STRATUM, exactly as `src/analysis.demo_grain_lds` already partials out the in-target demo count.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import masks as MK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
STRATA = (4, 5, 6)


def _sig(clusters):
    return tuple(sorted(clusters))


def stage_f_signatures():
    """The 72 cluster subsets Stage F already consumed -- the exclusion set."""
    man = MK.cluster_mask_manifest()
    return {_sig(m["clusters"]) for m in man["masks"]}


def build(strata=STRATA, exclude=None):
    """Complete enumeration of each stratum minus the exclusion set."""
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    exclude = set() if exclude is None else set(exclude)
    out, audit = [], {}
    for per in strata:
        allsubs = [_sig(c) for c in itertools.combinations(cl, per)]
        assert len(allsubs) == math.comb(len(cl), per)
        fresh = [s for s in allsubs if s not in exclude]
        counts = {c: sum(1 for s in fresh if c in s) for c in cl}
        for k, s in enumerate(sorted(fresh)):
            demos = [d for c in s for d in by_c[c]]
            out.append({"mask_id": f"N{per}_{k:03d}", "clusters": list(s), "n_clusters": per,
                        "n_demos": len(demos), "demos": demos, "stratum": f"{per}of9"})
        audit[f"{per}of9"] = {
            "space": len(allsubs), "excluded": len(allsubs) - len(fresh), "fresh": len(fresh),
            "inclusion_counts": counts,
            "balanced": len(set(counts.values())) == 1,
            "inclusion_per_cluster": sorted(set(counts.values())),
        }
    return out, audit


def manifest(force=False, path=None):
    path = path or os.path.join(RESULTS, "p8_mask_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    excl = stage_f_signatures()
    masks, audit = build(exclude=excl)
    _assert_ok(masks, audit, excl)
    out = {
        "pass": 8, "campaign": "N", "grain": "cluster", "strata": list(STRATA),
        "construction": "complete enumeration of each stratum minus Stage F's 72 signatures",
        "n_masks": len(masks), "n_excluded_stage_f": len(excl),
        "audit": audit, "masks": masks,
    }
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def _assert_ok(masks, audit, excl):
    cl = dataset.clusters()
    sigs = [_sig(m["clusters"]) for m in masks]
    assert len(sigs) == len(set(sigs)), "duplicate mask signatures"
    overlap = set(sigs) & set(excl)
    assert not overlap, f"NOT disjoint from Stage F: {sorted(overlap)[:5]}"
    for m in masks:
        assert len(m["clusters"]) == m["n_clusters"]
        assert m["n_demos"] == 15 * m["n_clusters"], f"{m['mask_id']} has {m['n_demos']} demos"
    # 4of9 and 6of9 are complete enumerations -> exactly balanced at C(8,per-1)
    for per in (4, 6):
        a = audit[f"{per}of9"]
        assert a["balanced"], f"{per}of9 not balanced: {a['inclusion_per_cluster']}"
        assert a["inclusion_per_cluster"] == [math.comb(len(cl) - 1, per - 1)], a
    # 5of9: the complement of Stage F inside the complete 126.
    #
    # Stage F nominally has 72 masks, but its randomized construction + swap repair drew from a
    # space of only 126 and REPEATED subsets: the 72 masks carry just 58 distinct signatures. A
    # repeated cluster subset is not a second mask -- both copies see identical training data, so
    # the repeat buys seed depth, not design coverage. Stage F's effective mask count is 58, and
    # every conditional-n reported against it is correspondingly optimistic.
    a5 = audit["5of9"]
    n_distinct_f = 126 - a5["fresh"]
    assert n_distinct_f == len(excl), f"exclusion bookkeeping: {n_distinct_f} vs {len(excl)}"
    assert a5["fresh"] == 126 - len(excl), f"unexpected fresh 5of9 count {a5['fresh']}"
    a5["stage_f_masks_nominal"] = 72
    a5["stage_f_signatures_distinct"] = len(excl)
    a5["stage_f_repeated_masks"] = 72 - len(excl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    man = manifest(force=a.force)
    print(f"[p8/masks] {man['n_masks']} fresh cluster masks, disjoint from Stage F's "
          f"{man['n_excluded_stage_f']}")
    for k, v in man["audit"].items():
        print(f"  {k:6s} space={v['space']:4d} excluded={v['excluded']:3d} fresh={v['fresh']:4d} "
              f"balanced={v['balanced']} inclusion/cluster={v['inclusion_per_cluster']}")
    tot = sum(m["n_demos"] for m in man["masks"]) / len(man["masks"])
    print(f"  mean demos/mask = {tot:.1f}")


if __name__ == "__main__":
    main()
