"""Pass 18 -- invariants for the corpus-size ladder's corpus and grain.

None of these would raise anywhere downstream if they broke. A rung with an uneven task split, a
ladder that quietly reuses the old campaign's demos, groups that are homogeneous at one rung and
mixed at another, or two "independent" partitions that share groups would all show up only as a
size trend with the wrong shape -- and the whole experiment is that trend's shape. So they are
asserted here, before any GPU is committed.

Zero GPU, zero training. Everything below reads the manifest or recomputes the partition.
"""
import os

import pytest

from if_repair import p18_corpus as C

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ the pools
def test_the_old_campaigns_demos_are_excluded_everywhere():
    """BLOCKERS #28/#31: a corpus already selected upon cannot carry alpha for a fresh
    hypothesis. The 25 libero_goal demos the old campaign used must appear in NOTHING here."""
    used = C.used_by_old_campaign()
    assert len(used) == 25, f"expected 25 consumed demos, found {len(used)}"
    assert not (used & set(C.eval_bank())), "eval bank reuses old-campaign demos"
    for n in C.RUNGS:
        assert not (used & set(C.rung_demos(n))), f"rung {n} reuses old-campaign demos"


def test_free_pool_is_475_and_never_50_per_task():
    """The 25 consumed demos are NOT uniform across tasks -- five tasks gave up 3 and five gave
    up 2. This is why a rung of 500, or even 480, was never reachable."""
    fb = C.free_by_task()
    assert len(fb) == C.N_TASKS
    assert sum(len(v) for v in fb.values()) == 475
    assert set(map(len, fb.values())) == {47, 48}


def test_eval_bank_is_disjoint_from_every_rung():
    """The outcome is a held-out loss. A demo that is both trained on and evaluated against
    would make the outcome partly a training-loss and the whole ladder uninterpretable."""
    ev = set(C.eval_bank())
    assert len(ev) == C.N_TASKS * C.EVAL_PER_TASK
    for n in C.RUNGS:
        assert not (ev & set(C.rung_demos(n))), f"rung {n} trains on eval-bank demos"


def test_eval_bank_is_balanced_across_tasks():
    per = {}
    for d in C.eval_bank():
        per[d.split("/")[1]] = per.get(d.split("/")[1], 0) + 1
    assert set(per.values()) == {C.EVAL_PER_TASK}
    assert len(per) == C.N_TASKS


# ------------------------------------------------------------------ the ladder
@pytest.mark.parametrize("n", C.RUNGS)
def test_every_rung_holds_the_task_distribution_exactly_fixed(n):
    """KTD1. If the task mix moved with N, a trend could be composition rather than size --
    which is the exact confound that ruled out growing the old 135-demo corpus."""
    per = {}
    for d in C.rung_demos(n):
        per[d.split("/")[1]] = per.get(d.split("/")[1], 0) + 1
    assert len(per) == C.N_TASKS, f"rung {n} is missing tasks: {C.N_TASKS - len(per)}"
    assert set(per.values()) == {n // C.N_TASKS}, f"rung {n} task split is uneven: {per}"


@pytest.mark.parametrize("n", C.RUNGS)
def test_rung_size_is_exact(n):
    assert len(C.rung_demos(n)) == n
    assert len(set(C.rung_demos(n))) == n


def test_the_rungs_nest():
    """Nesting removes corpus-draw variance from the rung-to-rung trend, which is what R4
    asks for a CI on."""
    for lo, hi in zip(C.RUNGS, C.RUNGS[1:]):
        assert set(C.rung_demos(lo)) <= set(C.rung_demos(hi)), f"rung {lo} is not inside {hi}"


def test_the_top_rung_is_the_largest_reachable_one():
    """370 is not a round number chosen for taste: it is 10 x the per-task capacity left after
    the old campaign's 25 and the 100-demo eval bank. Guard against it being raised by hand."""
    assert C.per_task_capacity() == 37
    assert max(C.RUNGS) == C.per_task_capacity() * C.N_TASKS


def test_a_rung_beyond_capacity_is_refused_rather_than_truncated():
    with pytest.raises(ValueError, match="only 37 are available"):
        C.rung_demos(500)


def test_a_rung_that_cannot_split_evenly_is_refused():
    with pytest.raises(ValueError, match="not divisible"):
        C.rung_demos(55)


# ------------------------------------------------------------------ the grain
@pytest.mark.parametrize("n", C.RUNGS)
@pytest.mark.parametrize("p", C.PARTITIONS)
def test_groups_partition_the_rung(n, p):
    au = C.audit(n, p)
    assert au["covers_rung"], f"rung {n}{p} groups do not cover the rung"
    assert au["disjoint"], f"rung {n}{p} groups overlap"
    assert au["group_sizes"] == [C.GROUP_SIZE]
    assert au["n_groups"] == n // C.GROUP_SIZE


@pytest.mark.parametrize("n", C.RUNGS)
@pytest.mark.parametrize("p", C.PARTITIONS)
def test_a_group_spans_five_distinct_tasks_at_every_rung(n, p):
    """The grain's COMPOSITION must not move along the ladder. Within-task groups would make a
    group a whole task at rung 50 and a seventh of one at rung 370, so homogeneity -- not size --
    could drive the trend."""
    au = C.audit(n, p)
    assert au["distinct_tasks_per_group"] == [C.GROUP_SIZE]


@pytest.mark.parametrize("n", C.RUNGS)
@pytest.mark.parametrize("p", C.PARTITIONS)
def test_every_task_contributes_to_the_same_number_of_groups(n, p):
    """Exact task balance at group level, by construction rather than by repair."""
    au = C.audit(n, p)
    assert au["groups_per_task"] == [n // C.N_TASKS]


@pytest.mark.parametrize("n", C.RUNGS)
def test_the_two_partitions_share_zero_groups(n):
    """KTD5 / BLOCKERS #54. Cross-partition agreement is only meaningful if the partitions are
    genuinely independent; a shared group would agree with itself for free."""
    assert C.cross_partition_shared_groups(n) == 0


@pytest.mark.parametrize("n", C.RUNGS)
def test_mask_demos_round_trips_through_group_ids(n):
    """Every group id resolves, and taking ALL of them returns exactly the rung."""
    for p in C.PARTITIONS:
        gids = [g["group_id"] for g in C.groups(n, p)]
        assert C.mask_demos(gids, n, p) == list(C.rung_demos(n))
    with pytest.raises(KeyError):
        C.mask_demos(["not_a_group"], n, "A")


# ------------------------------------------------------------------ determinism
def test_the_partition_is_deterministic():
    """The manifest is a committed record. If the partition moved between calls, every mask
    built against it would silently mean something different."""
    a1 = [g["demos"] for g in C.groups(200, "A")]
    a2 = [g["demos"] for g in C.groups(200, "A")]
    assert a1 == a2
    # ...and partition B is a genuinely different draw, not the same one relabelled
    b1 = [g["demos"] for g in C.groups(200, "B")]
    assert a1 != b1


def test_manifest_matches_the_live_construction():
    man = C.manifest()
    assert man["rungs"] == list(C.RUNGS)
    assert man["per_task_capacity"] == C.per_task_capacity()
    assert man["eval_bank"] == list(C.eval_bank())
    for n in C.RUNGS:
        assert man["rung_demos"][str(n)] == list(C.rung_demos(n))
        assert man["cross_partition_shared_groups"][str(n)] == 0
        for p in C.PARTITIONS:
            live = [dict(g) for g in C.groups(n, p)]
            assert man["groups"][f"{n}{p}"] == live, f"manifest drifted at rung {n}{p}"
