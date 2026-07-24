"""Campaign J (pass-4 confirmation draw) must be a genuinely fresh mask set: disjoint from all
three prior draws G (seed 11), H (seed 4711), I (seed 9973). If J shared even one mask with a
consumed draw, the two-bar out-of-sample confirmation would be contaminated exactly the way B8's
selection-mask bootstrap was (BLOCKERS #20)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from if_repair import retrain as RT


def _masks(seed, prefix):
    ms, _ = RT.fresh_demo_masks(seed=seed, prefix=prefix)
    return [frozenset(m["demos"]) for m in ms]


def test_j_is_24_masks():
    ms, _ = RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_J, prefix="J")
    assert len(ms) == 24
    assert all(m["mask_id"].startswith("J") for m in ms)


def test_all_four_draws_pairwise_disjoint():
    draws = {
        "G": _masks(11, "G"),
        "H": _masks(RT.FRESH_MASK_SEED, "H"),
        "I": _masks(RT.FRESH_MASK_SEED_I, "I"),
        "J": _masks(RT.FRESH_MASK_SEED_J, "J"),
    }
    for name, ms in draws.items():
        assert len(set(ms)) == 24, f"{name} has duplicate masks within the draw"
    names = list(draws)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = set(draws[names[i]]), set(draws[names[j]])
            assert not (a & b), f"{names[i]} and {names[j]} share {len(a & b)} mask(s)"


def test_j_seed_is_frozen():
    # Guard against an accidental change to the frozen confirmation seed.
    assert RT.FRESH_MASK_SEED_J == 20260723
