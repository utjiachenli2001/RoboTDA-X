"""Pass 18 -- invariants for campaign U, the fixed-retained selection ladder.

Campaign U's failure modes are all silent. A mask that misses a task re-opens BLOCKERS #41 (task
coverage moves the outcome AND every estimator's summed prediction). A retained count that drifts
off 25 leaves the operating point the gate certified. A masks-per-coefficient that is not constant
biases the datamodel arm in a way the permutation null provably cannot see. None of these would
raise anywhere; each would just produce a readable curve meaning something other than it says.

Zero GPU.
"""
import pytest

from if_repair import p18_corpus as C
from if_repair import p18_campaign_u as U


@pytest.mark.parametrize("pool", C.RUNGS)
@pytest.mark.parametrize("part", C.PARTITIONS)
def test_every_mask_retains_exactly_25_demos(pool, part):
    """The whole design: every retrain sits at the operating point the §7 gate certified."""
    au = U.audit(pool, part)
    assert au["retained_demos"] == [U.RETAINED_DEMOS]


@pytest.mark.parametrize("pool", C.RUNGS)
@pytest.mark.parametrize("part", C.PARTITIONS)
def test_every_mask_covers_all_ten_tasks(pool, part):
    """Prereg §2.2. 31.2% of unconditioned 5-group masks miss a task; a missing task moves 10%
    of the eval bank and moves every estimator's prediction with it -- BLOCKERS #41 exactly."""
    assert U.audit(pool, part)["all_cover_10_tasks"]


@pytest.mark.parametrize("pool", C.RUNGS)
@pytest.mark.parametrize("part", C.PARTITIONS)
def test_masks_per_coefficient_is_constant_across_pools(pool, part):
    """Not cosmetic: at a fixed mask count, per-coefficient information rises 1.87x along the
    ladder and biases H2's slope, invisibly to the permutation null (§3.8's stated limit)."""
    assert U.audit(pool, part)["masks_per_coefficient"] == pytest.approx(18.4, abs=0.15)


@pytest.mark.parametrize("pool", C.RUNGS)
@pytest.mark.parametrize("part", C.PARTITIONS)
def test_no_duplicate_training_sets_within_a_cell(pool, part):
    au = U.audit(pool, part)
    assert au["unique_signatures"] == au["n_masks"]


@pytest.mark.parametrize("pool", C.RUNGS)
@pytest.mark.parametrize("part", C.PARTITIONS)
def test_masks_stay_inside_their_pool_and_off_the_eval_bank(pool, part):
    pool_ids, ev = set(C.rung_demos(pool)), set(C.eval_bank())
    used = C.used_by_old_campaign()
    for m in U.build(pool, part)[0][:200]:
        d = set(m["demos"])
        assert d <= pool_ids, "mask escapes its pool"
        assert not (d & ev), "mask trains on the evaluation bank"
        assert not (d & used), "mask reuses the old campaign's demos"


@pytest.mark.parametrize("pool", C.RUNGS)
@pytest.mark.parametrize("part", C.PARTITIONS)
def test_inclusion_balance_is_statistical_not_wild(pool, part):
    """Campaign T had EXACT balance; complementary pairs do not exist here, so balance is
    statistical. It must still track the binomial expectation -- a large excess would mean the
    coverage rejection is distorting the draw."""
    au = U.audit(pool, part)
    assert au["inclusion_sd"] <= 2.0 * au["inclusion_expected_sd"]


def test_depth_is_even_and_reserves_are_pairs():
    """#39: the split-half ceiling is NaN at odd depth. §3.4 replaces BOTH seeds, so reserves
    must come in pairs."""
    assert U.DEPTH % 2 == 0 and len(U.SEEDS) == U.DEPTH
    assert all(len(p) == 2 for p in U.RESERVE_PAIRS)
    assert not (set(U.SEEDS) & {s for p in U.RESERVE_PAIRS for s in p})


def test_job_list_is_seed_major_so_every_prefix_is_a_complete_design():
    J = U.jobs()
    half = len(J) // 2
    assert {j["seed_init"] for j in J[:half]} == {U.SEEDS[0]}
    assert {j["seed_init"] for j in J[half:]} == {U.SEEDS[1]}


def test_jobs_are_deduped_by_signature_and_seed():
    """Pools nest, so the same 25-demo set is reachable from more than one pool. It is the same
    model at the same seed and must be retrained once -- shared, not dropped."""
    J = U.jobs()
    keys = [(j["sig"], j["seed_init"]) for j in J]
    assert len(keys) == len(set(keys))
    assert len({j["run_id"] for j in J}) == len(J)


def test_cross_pool_collisions_are_accounted_not_asserted_away():
    """Campaign T asserted cross-rung disjointness by size. Campaign U cannot: this returns a
    count, and jobs() must absorb it rather than the manifest forbidding it."""
    sh = U.cross_pool_shared_signatures()
    assert isinstance(sh, dict)
    total_masks = sum(U.N_MASKS[p] for p in C.RUNGS) * len(C.PARTITIONS)
    assert len(U.jobs()) == (total_masks - sum(sh.values())) * U.DEPTH


def test_manifest_matches_live_construction():
    man = U.manifest()
    assert man["campaign"] == "U" and man["retained_demos"] == U.RETAINED_DEMOS
    assert man["n_jobs"] == len(U.jobs())
    for pool in C.RUNGS:
        for part in C.PARTITIONS:
            assert man["audit"][f"{pool}{part}"] == U.audit(pool, part)
