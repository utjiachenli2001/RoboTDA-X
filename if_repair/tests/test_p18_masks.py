"""Pass 18 -- invariants for the campaign-T mask family.

Every one of these is a failure that would produce a plausible curve rather than an error. An
unbalanced family, a mask that trains on its own evaluation bank, a rung whose masks vary in
training-set size, or a job list that is not seed-major would each yield a ladder that looks
readable and means something other than what it says.

Zero GPU.
"""
import itertools
import math

import pytest

from if_repair import p18_corpus as C
from if_repair import p18_masks as M


# ------------------------------------------------------------------ size, the #41 rule
@pytest.mark.parametrize("n", C.RUNGS)
@pytest.mark.parametrize("p", C.PARTITIONS)
def test_every_mask_at_a_rung_keeps_exactly_the_same_training_set_size(n, p):
    """BLOCKERS #41. A nuisance axis that moves the outcome AND every estimator's prediction
    credits both sides for arithmetic. Campaign O removed it by fixing the retained size; here
    complementary pairing at 50% makes the variation not exist at all."""
    au = M.audit(n, p)
    assert au["retained_demos"] == [n // 2], f"rung {n}{p} has mixed training-set sizes"
    assert au["retained_groups"] == [n // (2 * C.GROUP_SIZE)]


@pytest.mark.parametrize("n", C.RUNGS)
@pytest.mark.parametrize("p", C.PARTITIONS)
def test_inclusion_balance_is_exact(n, p):
    """Spread zero, by construction rather than by a swap-repair loop."""
    au = M.audit(n, p)
    assert au["balance_spread"] == 0, f"rung {n}{p} balance spread {au['balance_spread']}"
    assert len(au["balance_value"]) == 1


# ------------------------------------------------------------------ the bottom rung's cap
def test_rung_50_is_the_complete_mask_space():
    """C(10,5) = 252 is the ENTIRE space at rung 50. A combinatorial cap of the same kind as
    BLOCKERS #38/#45 -- if this ever reports 'complementary_pairs' the cap has been mis-stated
    and the rung is being sampled from a space it could have enumerated."""
    assert M.N_MASKS[50] == math.comb(10, 5) == 252
    for p in C.PARTITIONS:
        au = M.audit(50, p)
        assert au["mode"] == "enumerated"
        assert au["n_masks"] == 252


@pytest.mark.parametrize("p", C.PARTITIONS)
def test_rung_50_is_closed_under_complementation(p):
    """The property the other rungs must construct, which enumeration gives free."""
    ms, _ = M.build(50, p)
    all_g = {g["group_id"] for g in C.groups(50, p)}
    sel = {frozenset(m["groups"]) for m in ms}
    for s in sel:
        assert frozenset(all_g - s) in sel, "enumeration is not complementation-closed"


# ------------------------------------------------------------------ disjointness
@pytest.mark.parametrize("n", C.RUNGS)
@pytest.mark.parametrize("p", C.PARTITIONS)
def test_no_mask_signature_repeats_within_a_rung(n, p):
    au = M.audit(n, p)
    assert au["unique_signatures"] == au["n_masks"], f"rung {n}{p} repeats a training set"


def test_no_signature_is_shared_across_rungs():
    """Impossible by size (rungs retain 25/50/100/185), asserted anyway -- 'unlikely' is not a
    control (p9_masks)."""
    assert M.cross_rung_shared_signatures() == {}


@pytest.mark.parametrize("n", C.RUNGS)
def test_cross_partition_collisions_are_shared_not_duplicated(n):
    """A training set reachable from both partitions is the SAME model at the same seed. It must
    map to one retrain, not two -- and must not be silently dropped, which would break the
    balance the family was constructed to have."""
    shared = M.cross_partition_shared_signatures(n)
    J = [j for j in M.jobs() if j["rung"] == n and j["seed_init"] == M.SEEDS[0]]
    n_masks = sum(M.audit(n, p)["n_masks"] for p in C.PARTITIONS)
    assert len(J) == n_masks - len(shared), (
        f"rung {n}: {n_masks} masks, {len(shared)} shared -> expected "
        f"{n_masks - len(shared)} retrains, got {len(J)}")


# ------------------------------------------------------------------ the corpus boundary
@pytest.mark.parametrize("n", C.RUNGS)
def test_masks_never_train_on_the_evaluation_bank(n):
    """The outcome would become partly a training loss and the ladder would be uninterpretable."""
    ev = set(C.eval_bank())
    for p in C.PARTITIONS:
        for m in M.build(n, p)[0]:
            assert not (ev & set(m["demos"])), f"rung {n}{p} mask {m['mask_id']} trains on eval"


@pytest.mark.parametrize("n", C.RUNGS)
def test_masks_never_train_on_the_old_campaigns_demos(n):
    used = C.used_by_old_campaign()
    for p in C.PARTITIONS:
        for m in M.build(n, p)[0]:
            assert not (used & set(m["demos"]))


@pytest.mark.parametrize("n", C.RUNGS)
def test_masks_only_use_their_own_rung(n):
    rung = set(C.rung_demos(n))
    for p in C.PARTITIONS:
        for m in M.build(n, p)[0]:
            assert set(m["demos"]) <= rung, f"rung {n}{p} mask escapes its rung"


# ------------------------------------------------------------------ the job list
def test_depth_is_even():
    """BLOCKERS #39 -- the split-half ceiling returns NaN at odd depth."""
    assert M.DEPTH % 2 == 0
    assert len(M.SEEDS) == M.DEPTH


def test_seed_depth_is_identical_at_every_rung():
    """BLOCKERS #42 -- the ratio is depth-dependent, so a ladder with varying depth would
    measure depth rather than size. There is one global SEEDS tuple; this guards against a
    per-rung override being added later."""
    J = M.jobs()
    for n in C.RUNGS:
        seeds = {j["seed_init"] for j in J if j["rung"] == n}
        assert seeds == set(M.SEEDS), f"rung {n} has seed depth {sorted(seeds)}"


def test_job_list_is_seed_major_so_every_prefix_is_a_complete_design():
    """Campaign N's rule (retrain.N_DEPTH). A time-boxed run must be analysable at the largest
    completed depth; mask-major ordering would leave some masks at full depth and others absent,
    which is not a design."""
    J = M.jobs()
    half = len(J) // 2
    assert {j["seed_init"] for j in J[:half]} == {M.SEEDS[0]}
    assert {j["seed_init"] for j in J[half:]} == {M.SEEDS[1]}
    # and the first half covers every distinct training set exactly once
    sigs = [j["sig"] for j in J[:half]]
    assert len(sigs) == len(set(sigs))


def test_jobs_are_deduped_by_signature_and_seed():
    J = M.jobs()
    keys = [(j["sig"], j["seed_init"]) for j in J]
    assert len(keys) == len(set(keys))


def test_run_ids_are_unique():
    J = M.jobs()
    ids = [j["run_id"] for j in J]
    assert len(ids) == len(set(ids))


def test_signature_depends_only_on_the_demo_set():
    a = M.signature(["x/y/1", "x/y/2"])
    assert a == M.signature(["x/y/2", "x/y/1"])
    assert a != M.signature(["x/y/1", "x/y/3"])


def test_manifest_matches_the_live_construction():
    man = M.manifest()
    assert man["campaign"] == "T"
    assert man["depth"] == M.DEPTH
    assert man["n_jobs"] == len(M.jobs())
    assert man["cross_rung_shared_signatures"] == {}
    for n in C.RUNGS:
        for p in C.PARTITIONS:
            assert man["audit"][f"{n}{p}"] == M.audit(n, p), f"manifest drifted at {n}{p}"
