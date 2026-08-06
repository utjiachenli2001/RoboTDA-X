"""PASS 18 -- the `libero_goal` corpus for the CORPUS-SIZE LADDER. Zero GPU.

THE QUESTION. Gradient-based attribution clears a usefulness bar at no unit size on the 135-demo
corpus (WHAT_STANDS §1); a design-based datamodel does clear it. The one unresolved question
(WHAT_STANDS §3) is whether the gradient failure reflects the CORPUS SIZE or the APPROACH. The only
clean discriminator is a size ladder at a fixed task distribution, and this module builds it.

WHY `libero_goal` ALONE, AND NOT A GROWN VERSION OF THE OLD CORPUS. The unused pool is 71%
`libero_goal`, so growing the 135 would make a bigger corpus that is also a DIFFERENTLY SHAPED one,
and any trend could be task mix rather than size. `libero_goal` is exactly 10 tasks x 50 demos, so
every rung can hold the task distribution exactly fixed -- N/10 demos of every task, all ten tasks
present at every rung. Size becomes the only moving part.

WHAT IS ACTUALLY ON DISK, AND WHY THE TOP RUNG IS 370 AND NOT 500.
  500  demos in `data/proc/libero_goal` (10 tasks x 50, verified by file count)
  -25  consumed by the old campaign as cluster C1 -- 15 train + 10 heldout. NOT uniform across
       tasks: five tasks gave up 3 and five gave up 2, so the free pool is 47 or 48 per task,
       never 50. Excluded here to keep this corpus unselected-upon (BLOCKERS #28, #31).
  =475 free, i.e. min 47 per task
  -100 reserved as this ladder's held-out EVALUATION BANK, 10 per task
  =375 ladder pool, i.e. min 37 per task
So the largest rung with an exactly equal task split is 370. A rung of 500 was never reachable:
it would need 50 free demos of every task and there are at most 48.

WHY A FRESH 100-DEMO EVAL BANK RATHER THAN REUSING C1's TEN. The outcome is a held-out loss, and
its reliability `r` is what the ceiling measures -- BLOCKERS #42 makes every ratio a function of
that denominator. A 10x larger target bank buys a materially less noisy outcome at 0.1s of eval
cost per retrain (measured in `results/p18_costmodel.json`), which is free next to a 93s retrain.
The 30 demos of ladder span this costs are the cheaper side of that trade.

WHY THE RUNGS NEST. Rung 50 is a subset of rung 100, which is a subset of rung 200, which is a
subset of rung 370. If each rung were an independent draw, a rung-to-rung difference would carry
corpus-sampling variance on top of the effect, and R4 asks for a CI on the TREND. Nesting removes
that channel by construction. The price, stated here and again in the prereg: the ladder is a
statement conditional on ONE nested corpus draw, and does not generalise over draws. With one draw
per rung a non-nested design could not generalise over draws either, and would be noisier.

WHY GROUPS SPAN TASKS RATHER THAN SITTING INSIDE ONE. The attribution unit is a group of 5. If a
group were 5 demos of the same task, then at rung 50 -- where each task has exactly 5 demos -- a
group WOULD BE a whole task, while at rung 370 it would be a seventh of one. The grain's
composition would change along the ladder and confound size with homogeneity, which is precisely
the mistake `p9_grain` documents avoiding within clusters. So each group holds 5 demos of 5
DISTINCT tasks, and every task contributes to exactly N/10 groups at every rung. A group is a
miniature of the corpus at every rung.

THE BOTTOM RUNG IS COMBINATORIALLY CAPPED, AND THAT IS FINE. At rung 50 there are 10 groups and a
mask keeps 5, so the mask space is C(10,5) = 252 -- the ENTIRE space. This is not a budget choice
and no amount of GPU enlarges it (compare BLOCKERS #38 and #45, where a cap like this closed a
question). Campaign T therefore ENUMERATES rung 50 completely, which is strictly better than
sampling it: complete enumeration gives exact inclusion balance by construction, the same property
pass 8 had and pass 9 had to build complementary pairing to approximate.

NO CONDITIONING RULE IS NEEDED. Campaign O had to preregister "all of the target cluster's groups
are retained", because its outcome target lived INSIDE the training pool. This ladder's outcome
target is a held-out bank disjoint from every training demo, so no mask ever has to retain
anything, masks are pure complementary pairs, and the retained fraction is exactly 50% at every
rung. One fewer preregistered degree of freedom than campaign O had.
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
from bootstrap import PROC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SUITE = "libero_goal"
N_TASKS = 10
OLD_CLUSTER = "C1"          # the cluster the old campaign drew its 25 libero_goal demos into

# --- frozen constants. Changing any of these repartitions the corpus and invalidates every
# --- campaign-T mask manifest built against it. They are fixed BEFORE any mask exists.
EVAL_SEED = 20260806        # draws the 100-demo held-out evaluation bank
POOL_SEED = 20260807        # orders the ladder pool, and so fixes which demos each rung holds
GROUP_SEED = 20260808       # partition A grouping
GROUP_SEED_B = 20260809     # partition B grouping (must share zero groups with A)

EVAL_PER_TASK = 10
GROUP_SIZE = 5
RUNGS = (50, 100, 200, 370)
PARTITIONS = ("A", "B")


# --------------------------------------------------------------------------- the pools
@functools.lru_cache(maxsize=1)
def used_by_old_campaign():
    """The 25 `libero_goal` demo ids the old campaign consumed: 15 train + 10 heldout as C1.

    Excluded from everything below. BLOCKERS #28/#31: a corpus that has already been selected
    upon cannot carry alpha for a fresh hypothesis.
    """
    out = set()
    for c in dataset.manifest()["clusters"]:
        if c["cluster"] != OLD_CLUSTER:
            continue
        assert c["suite"] == SUITE, f"{OLD_CLUSTER} is not {SUITE}: {c['suite']}"
        for key in ("train_demos", "heldout_demos"):
            for d in c[key]:
                out.add(dataset.did(c["suite"], d["task"], d["demo"]))
    return frozenset(out)


@functools.lru_cache(maxsize=1)
def tasks():
    return tuple(sorted(os.listdir(os.path.join(PROC, SUITE))))


@functools.lru_cache(maxsize=1)
def free_by_task():
    """{task: (demo ids not touched by the old campaign)}, deterministically ordered."""
    used = used_by_old_campaign()
    out = {}
    for t in tasks():
        ids = []
        for f in sorted(os.listdir(os.path.join(PROC, SUITE, t))):
            if not f.endswith(".npz"):
                continue
            i = dataset.did(SUITE, t, f[:-4])
            if i not in used:
                ids.append(i)
        out[t] = tuple(ids)
    return out


@functools.lru_cache(maxsize=1)
def eval_bank():
    """The ladder's held-out evaluation bank: EVAL_PER_TASK demos of every task.

    Drawn once, from a frozen seed, BEFORE the ladder pool is ordered -- so the ladder pool is
    "what is left", never "what the bank did not want".
    """
    fb = free_by_task()
    out = []
    for ti, t in enumerate(tasks()):
        ids = list(fb[t])
        rng = np.random.default_rng([EVAL_SEED, ti])
        pick = rng.permutation(len(ids))[:EVAL_PER_TASK]
        out += [ids[i] for i in sorted(pick)]
    assert len(out) == N_TASKS * EVAL_PER_TASK
    return tuple(sorted(out))


@functools.lru_cache(maxsize=1)
def ladder_by_task():
    """{task: ordered demo ids available to the ladder}. The ORDER is the nesting.

    Rung N takes the first N/10 of every task, so rung 50 is a subset of rung 100 is a subset of
    rung 200 is a subset of rung 370. The permutation is seeded per task from a frozen constant.
    """
    fb, ev = free_by_task(), set(eval_bank())
    out = {}
    for ti, t in enumerate(tasks()):
        ids = [i for i in fb[t] if i not in ev]
        rng = np.random.default_rng([POOL_SEED, ti])
        out[t] = tuple(ids[i] for i in rng.permutation(len(ids)))
    return out


def per_task_capacity():
    return min(len(v) for v in ladder_by_task().values())


def rung_by_task(n):
    """{task: the n/N_TASKS demos of that task at rung n}."""
    if n % N_TASKS:
        raise ValueError(f"rung {n} is not divisible by {N_TASKS} tasks -- the task "
                         f"distribution could not be held exactly fixed")
    m = n // N_TASKS
    cap = per_task_capacity()
    if m > cap:
        raise ValueError(f"rung {n} needs {m} demos of every task; only {cap} are available "
                         f"(500 on disk - 25 spent by the old campaign - "
                         f"{EVAL_PER_TASK} per task held out)")
    lb = ladder_by_task()
    return {t: lb[t][:m] for t in tasks()}


def rung_demos(n):
    """The sorted demo ids at rung n."""
    return tuple(sorted(d for v in rung_by_task(n).values() for d in v))


# --------------------------------------------------------------------------- the grain
def groups(n, partition="A", group_size=GROUP_SIZE):
    """Partition rung n into groups of `group_size` demos, each from DISTINCT tasks.

    Construction: lay the rung out as a (task x m) matrix, where column j holds the j-th demo of
    every task. Column j is then split by a fresh permutation of the ten tasks into
    N_TASKS/group_size groups. Every task therefore appears exactly once per column and so in
    exactly m groups overall -- exact task balance at group level, by construction rather than by
    repair -- and the task composition of a group varies from column to column.
    """
    if N_TASKS % group_size:
        raise ValueError(f"group_size {group_size} does not divide {N_TASKS} tasks; a group "
                         f"could not hold distinct tasks in equal number")
    per_col = N_TASKS // group_size
    seed = {"A": GROUP_SEED, "B": GROUP_SEED_B}[partition]
    rbt = rung_by_task(n)
    tl = tasks()
    m = n // N_TASKS
    out = []
    for j in range(m):
        rng = np.random.default_rng([seed, n, j])
        perm = rng.permutation(N_TASKS)
        for h in range(per_col):
            sel = perm[h * group_size:(h + 1) * group_size]
            demos = sorted(rbt[tl[t]][j] for t in sel)
            out.append({"group_id": f"T{n}{partition}_{j:02d}{h}", "partition": partition,
                        "rung": n, "demos": demos})
    assert len(out) == n // group_size
    return tuple(out)


def index(n, partition="A"):
    return {g["group_id"]: g for g in groups(n, partition)}


def mask_demos(group_ids, n, partition="A"):
    idx = index(n, partition)
    missing = [g for g in group_ids if g not in idx]
    if missing:
        raise KeyError(f"unknown group ids at rung {n} partition {partition}: {missing[:5]}")
    demos = [d for g in group_ids for d in idx[g]["demos"]]
    assert len(demos) == len(set(demos)), "groups overlap -- the partition is broken"
    return sorted(demos)


# --------------------------------------------------------------------------- audit
def audit(n, partition="A"):
    gs = groups(n, partition)
    seen = [d for g in gs for d in g["demos"]]
    task_of = {}
    for g in gs:
        for d in g["demos"]:
            task_of.setdefault(d.split("/")[1], []).append(g["group_id"])
    per_group_tasks = {len({d.split("/")[1] for d in g["demos"]}) for g in gs}
    return {
        "rung": n, "partition": partition, "n_groups": len(gs),
        "group_sizes": sorted({len(g["demos"]) for g in gs}),
        "covers_rung": sorted(seen) == list(rung_demos(n)),
        "disjoint": len(seen) == len(set(seen)),
        "distinct_tasks_per_group": sorted(per_group_tasks),
        "groups_per_task": sorted({len(v) for v in task_of.values()}),
    }


def cross_partition_shared_groups(n):
    """How many groups partitions A and B have in common. Must be 0 (KTD5)."""
    a = {frozenset(g["demos"]) for g in groups(n, "A")}
    b = {frozenset(g["demos"]) for g in groups(n, "B")}
    return len(a & b)


def manifest(path=None, force=False):
    path = path or os.path.join(RESULTS, "p18_corpus_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    out = {
        "pass": 18, "suite": SUITE, "rungs": list(RUNGS), "group_size": GROUP_SIZE,
        "seeds": {"eval": EVAL_SEED, "pool": POOL_SEED,
                  "group_A": GROUP_SEED, "group_B": GROUP_SEED_B},
        "excluded_old_campaign": sorted(used_by_old_campaign()),
        "eval_bank": list(eval_bank()),
        "per_task_capacity": per_task_capacity(),
        "rung_demos": {str(n): list(rung_demos(n)) for n in RUNGS},
        "groups": {f"{n}{p}": [dict(g) for g in groups(n, p)]
                   for n in RUNGS for p in PARTITIONS},
        "audit": {f"{n}{p}": audit(n, p) for n in RUNGS for p in PARTITIONS},
        "cross_partition_shared_groups": {str(n): cross_partition_shared_groups(n)
                                          for n in RUNGS},
    }
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fb = free_by_task()
    print(f"[p18/corpus] {SUITE}: {sum(len(v) for v in fb.values())} free demos "
          f"({min(map(len, fb.values()))}-{max(map(len, fb.values()))} per task) "
          f"after excluding the old campaign's {len(used_by_old_campaign())}")
    print(f"[p18/corpus] eval bank: {len(eval_bank())} demos ({EVAL_PER_TASK}/task); "
          f"ladder capacity: {per_task_capacity()}/task -> max rung "
          f"{per_task_capacity() * N_TASKS}")
    for n in RUNGS:
        au = audit(n)
        print(f"  rung N={n:3d}  {n // N_TASKS:2d}/task  groups={au['n_groups']:3d} "
              f"sizes={au['group_sizes']} tasks/group={au['distinct_tasks_per_group']} "
              f"groups/task={au['groups_per_task']} covers={au['covers_rung']} "
              f"disjoint={au['disjoint']}  A|B shared groups="
              f"{cross_partition_shared_groups(n)}")
    nest = all(set(rung_demos(RUNGS[i])) <= set(rung_demos(RUNGS[i + 1]))
               for i in range(len(RUNGS) - 1))
    print(f"[p18/corpus] rungs nest: {nest}")
    man = manifest(force=a.force)
    print(f"[p18/corpus] manifest: {len(man['groups'])} group sets committed")


if __name__ == "__main__":
    main()
