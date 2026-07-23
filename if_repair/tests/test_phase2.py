"""Pass-2 invariants: datamodel, fusion, layer grouping."""
import numpy as np
import pytest

from if_repair import data as D
from if_repair import datamodel as DM
from if_repair import fusion as FU

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds  # noqa: E402


def test_design_matrix_matches_manifest():
    X, demo_ids, mask_ids = DM.design_matrix("bc_s10")
    masks = D.demo_masks()
    assert X.shape == (24, 135)
    assert set(np.unique(X)) <= {0.0, 1.0}
    for r, m in enumerate(masks):
        assert int(X[r].sum()) == len([d for d in m["demos"] if d in set(demo_ids)])


def test_insample_datamodel_lds_is_circular_and_must_not_be_reported():
    """The trap in BLOCKERS #7, pinned as a test so it cannot silently return.

    In-sample datamodel coefficients scored by demo_grain_lds reproduce the fit, and on C5
    that exceeds the noise ceiling (ratio > 1) -- impossible for a real estimator.
    """
    r = DM.fit_datamodel("bc_s10", "C5", "lasso")
    rho, _, _, _, _ = demo_grain_lds(r["scores"], D.demo_masks(),
                                     D.outcomes("bc_s10")["C5"])
    ceil = D.ceilings("bc_s10")["C5"]
    assert rho / ceil > 1.0, "in-sample datamodel should be impossibly good; guard stale?"


def test_reported_datamodel_lds_is_out_of_fold_and_below_ceiling():
    df = DM.evaluate_datamodel(targets=("C1", "C5"), models=("lasso",))
    assert (df.ratio <= 1.0).all(), "an out-of-fold ratio above the ceiling is impossible"
    assert (df.lds != df.lds_insample_INVALID).all()


def test_datamodel_constant_prediction_yields_nan_not_a_pass():
    """C7/C9 collapse to zero coefficients (BLOCKERS #8). NaN must never count as a pass."""
    df = DM.evaluate_datamodel(targets=("C7",), models=("lasso",))
    row = df.iloc[0]
    assert not bool(row.passed)
    if np.isfinite(row.lds):
        pytest.skip("C7 no longer degenerate; nothing to guard")
    assert row.n_nonzero_coef == 0


def test_fusion_rules_are_permutation_invariant():
    rng = np.random.default_rng(0)
    ids = [f"d{i}" for i in range(30)]
    a = {d: float(v) for d, v in zip(ids, rng.normal(size=30))}
    b = {d: float(v) for d, v in zip(ids, rng.normal(size=30))}
    for fn in (FU.borda, FU.zscore_average, FU.median_rank):
        x = fn([a, b], ids)
        y = fn([b, a], ids)
        assert np.allclose([x[d] for d in ids], [y[d] for d in ids])


def test_borda_is_monotone_under_a_shared_ranking():
    ids = [f"d{i}" for i in range(10)]
    a = {d: float(i) for i, d in enumerate(ids)}
    fused = FU.borda([a, a], ids)
    order = [d for d in sorted(fused, key=lambda k: fused[k])]
    assert order == ids


def test_zscore_average_is_scale_and_shift_invariant():
    ids = [f"d{i}" for i in range(20)]
    rng = np.random.default_rng(1)
    a = {d: float(v) for d, v in zip(ids, rng.normal(size=20))}
    b = {d: 3.0 * a[d] + 7.0 for d in ids}
    x, y = FU.zscore_average([a], ids), FU.zscore_average([b], ids)
    assert np.allclose([x[d] for d in ids], [y[d] for d in ids])


def test_fusion_with_datamodel_uses_leave_one_mask_out():
    """Membership is decided by RECIPES, not by the recipe NAME -- `all4` contains the
    datamodel without saying so, and must still take the leave-one-mask-out path."""
    df = FU.evaluate_fusion(targets=("C5",))
    uses_dm = {r for r, members in FU.RECIPES.items()
               if any(m.startswith("datamodel") for m in members)}
    assert uses_dm and len(uses_dm) < len(FU.RECIPES)
    dm = df[df.recipe.isin(uses_dm)]
    assert (dm.eval_mode == "leave_one_mask_out").all()
    assert (df[~df.recipe.isin(uses_dm)].eval_mode == "direct").all()
    assert (dm.ratio <= 1.0).all(), "LOO fusion above the ceiling means circularity leaked"


def test_param_groups_partition_the_model():
    """Blocks/embed/head must exactly partition ALL, with no overlap."""
    import re

    class P:
        def __init__(s, n): s.requires_grad = True; s.numel_ = n
        def numel(s): return s.numel_

    class M:
        def named_parameters(s):
            for n in ["pos", "state_proj.weight", "blocks.0.mlp.0.weight",
                      "blocks.1.attn.in_proj_weight", "head.weight"]:
                yield n, P(4)

    from if_repair.gradients import param_groups
    g = param_groups(M())
    parts = [k for k in g if k.startswith("block_") or k in ("embed", "head")]
    union, seen = [], set()
    for k in parts:
        for n in g[k]:
            assert n not in seen, f"{n} in two groups"
            seen.add(n); union.append(n)
    assert sorted(union) == sorted(g["ALL"])
