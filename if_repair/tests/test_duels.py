"""W2 duel design -- structural guarantees, asserted before any duel is trained.

The duel's entire validity rests on two masks differing in EXACTLY one demo (so the shared
mask x init interaction differences out) while both remain valid draws from the same stratified
design family (so the comparison is not confounded by the masks being unusual). Both are
mechanical properties that a test can pin, and neither is visible in the outcome.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from if_repair import retrain as RT
from if_repair import p7_duels as PD

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))
import dataset  # noqa: E402

STRATIFICATION = sorted([8, 8, 8, 8, 8, 7, 7, 7, 7])


def _man():
    with open(PD.MANIFEST) as fh:
        return json.load(fh)


def test_each_duel_differs_in_exactly_one_demo():
    for d in _man()["duels"]:
        a, b = set(d["mask_a"]), set(d["mask_b"])
        assert len(a) == len(b) == 68
        assert a - b == {d["a"]}, f"{d['duel_id']}: mask_a's unique demo is not a"
        assert b - a == {d["b"]}, f"{d['duel_id']}: mask_b's unique demo is not b"
        assert len(a & b) == 67


def test_both_masks_keep_the_design_stratification():
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    owner = {dd: c for c in cl for dd in by_c[c]}
    for d in _man()["duels"]:
        for side in ("mask_a", "mask_b"):
            counts = sorted(Counter(owner[x] for x in d[side]).values())
            assert counts == STRATIFICATION, f"{d['duel_id']}/{side}: {counts}"


def test_the_swapped_demos_are_in_the_same_cluster():
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    owner = {dd: c for c in cl for dd in by_c[c]}
    for d in _man()["duels"]:
        assert owner[d["a"]] == owner[d["b"]] == d["cluster"]


def test_estimators_actually_disagree_on_every_duel():
    """A duel where they agree carries no information about which is right."""
    for d in _man()["duels"]:
        assert d["relatif_prefers"] != d["graddot_prefers"], d["duel_id"]
        assert {d["relatif_prefers"], d["graddot_prefers"]} == {d["a"], d["b"]}
        assert min(abs(d["gap_relatif"]), abs(d["gap_graddot"])) >= PD.MIN_RANK_GAP


def test_duels_are_demo_disjoint_so_the_sign_test_is_valid():
    seen = set()
    for d in _man()["duels"]:
        assert d["a"] not in seen and d["b"] not in seen, f"{d['duel_id']} reuses a demo"
        seen.update((d["a"], d["b"]))


def test_duel_jobs_use_matched_seed_slots():
    man = _man()
    jobs = RT.jobs("D")
    assert len(jobs) == len(man["duels"]) * 2 * man["pilot_depth"]
    assert all(j["seed_init"] == j["seed_order"] for j in jobs), "seeds must be matched"
    by_mask = {}
    for j in jobs:
        by_mask.setdefault(j["mask_id"], set()).add(j["seed_init"])
    for d in man["duels"]:
        sa, sb = by_mask[f"{d['duel_id']}a"], by_mask[f"{d['duel_id']}b"]
        assert sa == sb, f"{d['duel_id']}: the two arms must share seed slots exactly"


def test_pilot_arm_is_the_first_48_jobs():
    """So `retrain --campaign D --limit 48` runs exactly the pilot and nothing else."""
    man = _man()
    jobs = RT.jobs("D")
    n = man["n_pilot"] * 2 * man["pilot_depth"]
    pilot_ids = {j["mask_id"][:4] for j in jobs[:n]}
    expected = {d["duel_id"] for d in man["duels"] if d["arm"] == "pilot"}
    assert pilot_ids == expected, f"{pilot_ids} != {expected}"
