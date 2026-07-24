"""Campaign M (the pass-7 resolving campaign) must be genuinely fresh, and its SIX sub-draws
must be fresh from each other as well as from G/H/I/J/K/L.

M is unusual in this repo: it is 144 masks, not 24, built as six independent 24-mask sub-draws
of the unmodified generator, because build_demo_masks is hardcoded to K_DEMO = 24. That makes
the disjointness surface twelve draws wide instead of six, and a collision between two SUB-draws
would silently duplicate masks inside a single campaign -- inflating n while adding no
information. Same failure mode as BLOCKERS #20, one level down.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from if_repair import retrain as RT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))
import dataset  # noqa: E402

STRATIFICATION = sorted([8, 8, 8, 8, 8, 7, 7, 7, 7])


def _masks(seed, prefix):
    ms, _ = RT.fresh_demo_masks(seed=seed, prefix=prefix)
    return [frozenset(m["demos"]) for m in ms]


def _all_draws():
    draws = {
        "G": _masks(11, "G"),
        "H": _masks(RT.FRESH_MASK_SEED, "H"),
        "I": _masks(RT.FRESH_MASK_SEED_I, "I"),
        "J": _masks(RT.FRESH_MASK_SEED_J, "J"),
        "K": _masks(RT.FRESH_MASK_SEED_K, "K"),
        "L": _masks(RT.FRESH_MASK_SEED_L, "L"),
    }
    for d, s in enumerate(RT.FRESH_MASK_SEED_M):
        draws[f"M{d}"] = _masks(s, f"M{d}")
    return draws


def test_m_is_144_masks_from_six_subdraws():
    ms = RT.fresh_demo_masks_pooled()
    assert len(ms) == 144
    assert len({m["mask_id"] for m in ms}) == 144, "mask ids must be unique across sub-draws"
    assert len({m["subdraw"] for m in ms}) == 6
    assert Counter(m["subdraw"] for m in ms) == {f"M{d}": 24 for d in range(6)}
    assert all(m["mask_id"].startswith("M") for m in ms)


def test_all_twelve_draws_pairwise_disjoint():
    draws = _all_draws()
    for name, ms in draws.items():
        assert len(set(ms)) == 24, f"{name} has duplicate masks within the draw"
    names = list(draws)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = set(draws[names[i]]), set(draws[names[j]])
            assert not (a & b), f"{names[i]} and {names[j]} share {len(a & b)} mask(s)"


def test_every_m_mask_has_68_demos_correctly_stratified():
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    owner = {d: c for c in cl for d in by_c[c]}
    for m in RT.fresh_demo_masks_pooled():
        assert len(m["demos"]) == 68, f"{m['mask_id']}: {len(m['demos'])} demos"
        assert len(set(m["demos"])) == 68, f"{m['mask_id']}: duplicate demos"
        per_cluster = Counter(owner[d] for d in m["demos"])
        assert sorted(per_cluster.values()) == STRATIFICATION, \
            f"{m['mask_id']}: stratification {sorted(per_cluster.values())}"


def test_m_runs_at_depth_two_for_288_jobs():
    """The W0.2 allocation result is load-bearing: M buys masks, not seeds."""
    assert RT.M_DEPTH == 2
    jobs = RT.jobs("M")
    assert len(jobs) == 144 * 2 == 288
    assert {j["seed_init"] for j in jobs} == set(RT.B_SEEDS[:2])
    assert all(j["seed_init"] == j["seed_order"] for j in jobs), "M is common init/order like A-L"
    assert len({j["run_id"] for j in jobs}) == 288


def test_m_seeds_are_frozen():
    assert RT.FRESH_MASK_SEED_M == (20260726, 20260727, 20260728, 20260729, 20260730, 20260731)


# ---------------------------------------------------------------- wiring smoke tests (pass-3 pattern)
# Run the confirm_mseries scoring path against a draw that ALREADY HAS A COMMITTED ANSWER, before
# campaign M has produced a single run. BLOCKERS #14 was a scorer wired to the wrong ground truth
# and it was caught exactly this way.

def test_mseries_ceiling_is_the_archived_recipe():
    from if_repair import confirm_mseries as CM
    from if_repair import functionals as F
    from if_repair import p7_pooled_oos as P7
    for draw in ("J", "K", "L"):
        raw = P7.raw_outcomes(draw, "C5")
        got = CM.ceiling(raw, CM.STATS["spearman"])
        want = F.split_half_ceiling(raw)["ceiling"]
        assert abs(got - want) < 1e-12, f"{draw}: {got} != {want}"


def test_mseries_scoring_path_reproduces_committed_campaign_l_row():
    """The M scorer, pointed at campaign L, must return L's preregistered published number."""
    from if_repair import confirm_mseries as CM
    from if_repair import p7_pooled_oos as P7
    masks = P7.masks_for("L")
    df = CM.evaluate("L", masks=masks, strata=["L"] * len(masks))
    row = df[df.statistic == "spearman"].iloc[0]
    assert abs(row.lds - 0.21739130434782608) < 1e-12
    assert abs(row.graddot_lds - 0.14260869565217388) < 1e-12
    assert abs(row.ceiling - 0.9584344169497008) < 1e-12
    assert abs(row.paired_delta - 0.0747826086956522) < 1e-12


def test_mseries_family_is_one_so_alpha_is_unadjusted():
    from if_repair import confirm_mseries as CM
    assert len(CM.PREREG_M) == 1
    assert abs(CM.ALPHA_ABS - 0.05) < 1e-12
    assert CM.STATS["kendall_tau_b"] is not None, "primary statistic must be Kendall tau_b"
