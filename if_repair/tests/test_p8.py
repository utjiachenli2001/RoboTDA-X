"""Pass-8 pins. CPU-only and fast; the GPU-dependent parts are covered by the campaign itself."""
import itertools
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from if_repair import p8_masks as P8M  # noqa: E402
from if_repair import confirm_nseries as CN  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


# ------------------------------------------------------------------ mask design
def _man():
    p = os.path.join(RESULTS, "p8_mask_manifest.json")
    if not os.path.exists(p):
        pytest.skip("p8_mask_manifest.json not built")
    return json.load(open(p))


def test_campaign_n_masks_are_disjoint_from_stage_f():
    """The out-of-sample guarantee. Asserted, not commented."""
    man = _man()
    fresh = {tuple(sorted(m["clusters"])) for m in man["masks"]}
    used = P8M.stage_f_signatures()
    assert not (fresh & used), sorted(fresh & used)[:5]


def test_no_duplicate_signatures_within_campaign_n():
    man = _man()
    sigs = [tuple(sorted(m["clusters"])) for m in man["masks"]]
    assert len(sigs) == len(set(sigs))


def test_complete_strata_are_exactly_balanced():
    """A complete enumeration puts every cluster in C(8, per-1) masks -- no repair needed."""
    man = _man()
    for per in (4, 6):
        a = man["audit"][f"{per}of9"]
        assert a["fresh"] == math.comb(9, per)
        assert a["balanced"]
        assert a["inclusion_per_cluster"] == [math.comb(8, per - 1)]


def test_stage_f_repeated_subsets_are_recorded_not_hidden():
    """Stage F's 72 masks carry only 58 distinct signatures; a repeat buys depth, not coverage."""
    a5 = _man()["audit"]["5of9"]
    assert a5["stage_f_masks_nominal"] == 72
    assert a5["stage_f_signatures_distinct"] == 58
    assert a5["stage_f_repeated_masks"] == 14
    assert a5["fresh"] == 126 - 58


def test_mask_demo_count_is_15_per_cluster():
    for m in _man()["masks"]:
        assert m["n_demos"] == 15 * m["n_clusters"]


def test_requesting_more_than_the_space_is_impossible_by_construction():
    """Enumeration cannot return duplicates or exceed C(9,per) -- the cap is structural."""
    masks, audit = P8M.build(strata=(5,), exclude=set())
    assert len(masks) == math.comb(9, 5)
    sigs = {tuple(sorted(m["clusters"])) for m in masks}
    assert len(sigs) == len(masks)


# ------------------------------------------------------------------ the lift
def test_cluster_prediction_is_the_sum_over_the_masks_demos():
    from if_repair import p7_pooled_oos as P7
    sc = {"d1": 1.0, "d2": 2.0, "d3": 5.0}
    masks = [{"mask_id": "x", "demos": ["d1", "d3"]}, {"mask_id": "y", "demos": ["d2"]}]
    assert list(P7.mask_pred(sc, masks)) == [6.0, 2.0]


def test_cluster_prediction_is_invariant_to_demo_order():
    from if_repair import p7_pooled_oos as P7
    sc = {"a": 0.5, "b": -1.5, "c": 2.0}
    m1 = [{"mask_id": "m", "demos": ["a", "b", "c"]}]
    m2 = [{"mask_id": "m", "demos": ["c", "a", "b"]}]
    assert P7.mask_pred(sc, m1)[0] == pytest.approx(P7.mask_pred(sc, m2)[0])


def test_missing_demo_scores_contribute_zero_not_nan():
    from if_repair import p7_pooled_oos as P7
    assert P7.mask_pred({"a": 3.0}, [{"mask_id": "m", "demos": ["a", "absent"]}])[0] == 3.0


# ------------------------------------------------------------------ conditioning
def test_conditional_masks_all_contain_the_target():
    man = _man()
    for t in ("C1", "C5", "C9"):
        cm = CN.conditional_masks(t, man["masks"])
        assert cm and all(t in m["clusters"] for m in cm)


def test_conditional_is_a_strict_subset_of_all_masks():
    man = _man()
    assert len(CN.conditional_masks("C5", man["masks"])) < len(man["masks"])


# ------------------------------------------------------------------ the stopping rule
def test_achieved_depth_is_the_largest_complete_prefix():
    raw = {"m1": {1: 0.0, 2: 0.0, 3: 0.0}, "m2": {1: 0.0, 2: 0.0}, "m3": {1: 0.0, 2: 0.0, 3: 0.0}}
    assert CN.achieved_depth(raw, 3)[0] == 2


def test_achieved_depth_is_zero_when_a_mask_is_wholly_missing():
    raw = {"m1": {1: 0.0}, "m2": {1: 0.0}}
    assert CN.achieved_depth(raw, 3)[0] == 0


def test_achieved_depth_reads_completion_only_never_outcome_values():
    """Replacing every outcome with NaN must not change the depth the rule selects."""
    raw = {"m1": {1: 0.3, 2: 0.9}, "m2": {1: -2.0, 2: 7.0}}
    nan = {m: {s: float("nan") for s in v} for m, v in raw.items()}
    assert CN.achieved_depth(raw, 2) == CN.achieved_depth(nan, 2)


def test_achieved_depth_returns_the_seed_prefix_in_order():
    raw = {"m": {10: 0.0, 20: 0.0, 30: 0.0}}
    d, seeds = CN.achieved_depth(raw, 1)
    assert d == 3 and seeds == [10, 20, 30]


# ------------------------------------------------------------------ score-once
def test_prereg_n_is_a_family_of_one():
    assert len(CN.PREREG_N) == 1 and CN.ALPHA_ABS == 0.05


def test_primary_statistic_is_the_one_prereg_n_names():
    assert CN.PRIMARY_STAT == "kendall_tau_b"


def test_statistic_choice_was_made_without_a_contrast_column():
    """Structural: a later edit must not be able to leak the hypothesis into the choice."""
    p = os.path.join(RESULTS, "p8_design_statistic.csv")
    if not os.path.exists(p):
        pytest.skip("statistic stage not run")
    import pandas as pd
    cols = set(pd.read_csv(p).columns)
    banned = {"delta", "paired_delta", "lds", "rho", "contrast", "graddot_lds", "ratio"}
    assert not (cols & banned), cols & banned


# ------------------------------------------------------------------ even-depth ceiling defect
def test_split_half_ceiling_is_nan_at_odd_depth():
    """The defect analysis_depth() exists to route around. Pinned so it cannot regress silently."""
    from if_repair.confirm_mseries import ceiling, STATS
    rng = np.random.default_rng(0)
    truth = rng.normal(size=30)

    def mk(S):
        return {f"m{i}": {s: float(truth[i] + 0.3 * rng.normal()) for s in range(S)}
                for i in range(30)}
    assert np.isfinite(ceiling(mk(4), STATS["spearman"]))
    assert np.isnan(ceiling(mk(5), STATS["spearman"]))


def test_analysis_depth_is_the_largest_even_prefix():
    assert [CN.analysis_depth(d) for d in (0, 1, 2, 3, 4, 5, 6)] == [0, 0, 2, 2, 4, 4, 6]


def test_analysis_depth_never_exceeds_achieved():
    for d in range(0, 12):
        assert CN.analysis_depth(d) <= d
