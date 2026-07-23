"""The I-series must be a genuinely THIRD draw, disjoint from both prior mask sets.

If it shared masks with G or H, the "out-of-sample" confirmation would be circular.
"""
import numpy as np

from if_repair import data as D


def test_iseries_is_disjoint_from_G_and_H():
    from if_repair import retrain as RT
    D.add_repo_paths()
    import masks as MK
    g = MK.demo_mask_manifest()["masks"]
    h, _ = RT.fresh_demo_masks()                                  # H, seed 4711
    i, _ = RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")
    assert len(i) == 24 and {len(m["demos"]) for m in i} == {68}
    gs = {frozenset(m["demos"]) for m in g}
    hs = {frozenset(m["demos"]) for m in h}
    iset = {frozenset(m["demos"]) for m in i}
    assert not (iset & gs), "I duplicates a G mask"
    assert not (iset & hs), "I duplicates an H mask"
    assert {m["mask_id"][0] for m in i} == {"I"}


def test_iseries_seed_is_not_a_prior_seed():
    from if_repair import retrain as RT
    assert RT.FRESH_MASK_SEED_I not in (11, RT.FRESH_MASK_SEED)


def test_paired_bootstrap_sign_and_significance():
    """The paired test must call a clear win a win and a tie a tie."""
    from if_repair.confirm_iseries import paired_bootstrap
    rng = np.random.default_rng(0)
    n = 24
    truth = rng.normal(size=n)
    strong = truth + rng.normal(0, 0.3, n)      # tracks the outcome
    weak = rng.normal(size=n)                    # independent of it
    d, p, ci = paired_bootstrap(strong, weak, truth, n_boot=2000)
    assert d > 0 and p < 0.05
    d2, p2, _ = paired_bootstrap(weak, weak.copy(), truth, n_boot=500)
    assert abs(d2) < 1e-9                         # identical predictors -> zero difference
