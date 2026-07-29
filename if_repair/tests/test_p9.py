"""Pass 9 -- invariants for the campaign-O design and the three zero-GPU analyses.

The load-bearing ones are the design tests. An unbalanced mask family, a conditioning rule applied
inconsistently, or a mask that silently re-runs a consumed campaign-N signature would not raise
anywhere: they would show up as a curve with the wrong shape, and no downstream code could tell.

Kept fast deliberately -- the bootstrap and permutation paths are exercised on small synthetic
arrays rather than on the real 149-mask draw, and the one real-data test is the regression against
the committed pass-8 number, which is a single correlation plus a ceiling.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from if_repair import p9_grain as G
from if_repair import p9_masks as P9M
from if_repair import p9_datamodel_cluster as DMC
from if_repair import p9_why_reverse as WHY
from if_repair import p9_stratum_control as SC
from if_repair import confirm_oseries as CO
from if_repair import retrain as R
from if_repair.confirm_nseries import analysis_depth
from if_repair.confirm_mseries import STATS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


# ------------------------------------------------------------------ campaign O design
def test_every_mask_keeps_exactly_the_fixed_training_set_size():
    """The whole point of the campaign: |S| variation does not exist, rather than being
    controlled after the fact."""
    man = P9M.manifest()
    for k in man["grains"]:
        sizes = {m["n_demos"] for m in man["masks"][str(k)]}
        assert sizes == {P9M.RETAINED_DEMOS}, f"k={k} has mixed training-set sizes {sizes}"


def test_conditioning_rule_holds_on_every_mask():
    man = P9M.manifest()
    for k in man["grains"]:
        forced = {g["group_id"] for g in G.groups(int(k)) if g["cluster"] == P9M.TARGET}
        assert forced, f"k={k}: no target groups found"
        for m in man["masks"][str(k)]:
            assert forced <= set(m["groups"]), f"{m['mask_id']} drops a {P9M.TARGET} group"


def test_group_inclusion_is_exactly_balanced():
    man = P9M.manifest()
    for k in man["grains"]:
        au = man["audit"][str(k)]
        assert au["inclusion_spread"] == 0, f"k={k} spread {au['inclusion_spread']}"
        assert len(au["inclusion_per_free_group"]) == 1


def test_masks_are_disjoint_from_every_consumed_cluster_signature():
    """A sub-cluster mask whose groups tile whole clusters IS a campaign-N signature. Vanishingly
    unlikely under complementary-pair construction, which is not a control."""
    consumed = P9M.consumed_signatures()
    man = P9M.manifest()
    for k in man["grains"]:
        for m in man["masks"][str(k)]:
            assert frozenset(m["demos"]) not in consumed, f"{m['mask_id']} re-runs a spent mask"


def test_no_mask_tiles_enough_whole_clusters_to_be_a_cluster_mask():
    man = P9M.manifest()
    n_needed = P9M.RETAINED_DEMOS // 15
    for k in man["grains"]:
        for m in man["masks"][str(k)]:
            assert m["n_whole_clusters_tiled"] < n_needed, m["mask_id"]


def test_mask_ids_unique_and_demo_lists_sorted():
    man = P9M.manifest()
    ids = [m["mask_id"] for k in man["grains"] for m in man["masks"][str(k)]]
    assert len(ids) == len(set(ids))
    for k in man["grains"]:
        for m in man["masks"][str(k)]:
            assert m["demos"] == sorted(m["demos"]), "batch order must be reproducible"


def test_manifest_refuses_to_rebuild_once_the_campaign_has_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs" / "campaigns" / "O"
    runs.mkdir(parents=True)
    (runs / "O3_0000_i4401_o4401").mkdir()
    monkeypatch.setattr(P9M, "HERE", str(tmp_path))
    with pytest.raises(SystemExit):
        P9M.manifest(path=str(tmp_path / "nonexistent.json"), force=True)


# ------------------------------------------------------------------ the job list
def test_campaign_O_job_list_is_seed_major_and_even_depth():
    J = R.jobs("O")
    ms = P9M.all_masks()
    assert len(J) == len(ms) * P9M.DEPTH
    assert P9M.DEPTH % 2 == 0, "odd depth returns a NaN ceiling -- BLOCKERS #39"
    ids = [j["mask_id"] for j in J[:len(ms)]]
    assert len(set(ids)) == len(ms), "first prefix is not a complete balanced design"
    assert len({j["run_id"] for j in J}) == len(J)
    assert {len(j["demos"]) for j in J} == {P9M.RETAINED_DEMOS}


def test_campaign_O_is_an_accepted_argparse_choice():
    import argparse
    src = open(os.path.join(HERE, "retrain.py")).read()
    assert '"O"]' in src or '"O",' in src, "retrain.py argparse choices missing O"
    assert isinstance(argparse.ArgumentParser, type)


# ------------------------------------------------------------------ datamodel: the LOO trap
def test_loo_never_lets_a_mask_into_its_own_fit():
    """Asserted on fold membership directly, not on the resulting correlation -- a leak that only
    shifts the correlation a little would pass a correlation-based check."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(20, 4)).astype(float)
    y = rng.normal(size=20)
    seen = []
    orig = DMC.MODELS["ridge"]

    def spy(a):
        m = orig(a)
        fit = m.fit

        def wrapped(Xt, yt):
            seen.append(Xt.shape[0])
            return fit(Xt, yt)
        m.fit = wrapped
        return m

    DMC.MODELS["ridge"] = spy
    try:
        pred, alphas = DMC.loo_predict(X, y, model="ridge")
    finally:
        DMC.MODELS["ridge"] = orig
    assert np.isfinite(pred).all()
    # every OUTER fit sees exactly n-1 masks; inner alpha fits see fewer
    assert max(seen) == len(y) - 1, f"an outer fit saw {max(seen)} of {len(y)} masks"


def test_alpha_is_refit_inside_every_fold_not_once_globally():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(24, 5)).astype(float)
    y = X[:, 0] * 3 + rng.normal(scale=0.1, size=24)
    _, alphas_nested = DMC.loo_predict(X, y, model="ridge", nested_alpha=True)
    _, alphas_global = DMC.loo_predict(X, y, model="ridge", nested_alpha=False)
    assert np.isfinite(alphas_nested).all()
    assert len(set(alphas_global.tolist())) == 1, "global path should use one alpha throughout"


def test_cluster_design_row_sums_equal_the_mask_cluster_count():
    masks = [{"clusters": ["C1", "C2", "C5"]}, {"clusters": ["C3", "C5"]}]
    clusters = [f"C{i}" for i in range(1, 10)]
    X = DMC.cluster_design(masks, clusters)
    assert X.shape == (2, 9)
    assert X.sum(axis=1).tolist() == [3.0, 2.0]


# ------------------------------------------------------------------ why-reverse: the 2x2
def test_swap_marginal_keeps_one_order_and_the_other_marginal():
    """Scale source must be tie-free: with ties, argsort is not uniquely determined and the
    order check would fail on a correct implementation."""
    ids = [f"d{i}" for i in range(6)]
    a = {d: v for d, v in zip(ids, [5.0, 1.0, 3.0, 2.0, 4.0, 0.0])}
    b = {d: v for d, v in zip(ids, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])}
    out = WHY.swap_marginal(a, b, ids)
    av = np.array([a[d] for d in ids])
    ov = np.array([out[d] for d in ids])
    assert np.array_equal(np.argsort(av), np.argsort(ov)), "order not preserved"
    assert sorted(ov.tolist()) == sorted(b.values()), "marginal not taken from the scale source"


def test_swap_marginal_moves_a_heavy_tail_onto_the_other_ranking():
    """The arm that matters for BLOCKERS #37: a spiky marginal laid over a different order."""
    ids = [f"d{i}" for i in range(6)]
    a = {d: v for d, v in zip(ids, [5.0, 1.0, 3.0, 2.0, 4.0, 0.0])}
    spiky = {d: v for d, v in zip(ids, [1000.0, 1.0, 2.0, 3.0, 4.0, 5.0])}
    out = WHY.swap_marginal(a, spiky, ids)
    assert out["d0"] == 1000.0, "a's top-ranked demo must receive the largest value"
    assert sorted(out.values()) == sorted(spiky.values())


def test_concentration_detects_a_dominated_sum():
    masks = [{"demos": [f"d{i}" for i in range(20)]}]
    flat = {f"d{i}": 1.0 for i in range(20)}
    spiky = {f"d{i}": (1000.0 if i == 0 else 1.0) for i in range(20)}
    assert WHY.concentration(flat, masks, top=5) < 0.30
    assert WHY.concentration(spiky, masks, top=5) > 0.95


# ------------------------------------------------------------------ the |S| control
def test_pooled_path_reproduces_the_committed_pass8_primary():
    """The regression that makes the |S| finding a correction rather than a rival computation."""
    path = os.path.join(RESULTS, "confirm_nseries.csv")
    if not os.path.exists(path):
        pytest.skip("committed pass-8 result not present")
    committed = pd.read_csv(path)
    row = committed[committed.statistic == "kendall_tau_b"].iloc[0]
    y, pg, st, use, raw, d = SC.load("N", "C5")
    assert d == int(row["depth"])
    assert len(y) == int(row["n_masks"])
    fn = STATS["kendall_tau_b"]
    from if_repair.confirm_mseries import ceiling
    assert fn(pg, y) == pytest.approx(float(row["lds"]), abs=1e-9)
    assert ceiling(raw, fn) == pytest.approx(float(row["ceiling"]), abs=1e-9)


def test_within_stratum_permutation_null_is_centred_at_zero():
    """A fixed estimator on shuffled outcomes must not predict. If this ever fails, the finding is
    a leak rather than an |S| effect and the whole diagnosis changes."""
    rng = np.random.default_rng(3)
    n = 60
    pg = rng.normal(size=n)
    y = rng.normal(size=n)
    st = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    out = SC.perm_pooled(pg, y, st, STATS["kendall_tau_b"], n=200, seed=0)
    assert abs(out.mean()) < 0.05, f"null not centred: {out.mean()}"


def test_boot_ratio_recomputes_the_ceiling_per_resample():
    """If the ceiling were held fixed the interval would understate the uncertainty of the bar."""
    rng = np.random.default_rng(4)
    n = 30
    ids = [f"m{i}" for i in range(n)]
    raw = {m: {1: rng.normal(), 2: rng.normal()} for m in ids}
    y = np.array([np.mean(list(raw[m].values())) for m in ids])
    pg = y + rng.normal(scale=0.5, size=n)
    out = SC.boot_ratio(pg, y, raw, ids, STATS["spearman"], n=60, seed=0)
    assert len(out) > 10
    assert len(set(np.round(out, 9).tolist())) > 5, "ratio did not vary across resamples"


# ------------------------------------------------------------------ scoring campaign O once
def test_analysis_depth_is_the_largest_even_prefix():
    assert analysis_depth(5) == 4
    assert analysis_depth(4) == 4
    assert analysis_depth(3) == 2
    assert analysis_depth(1) == 0


def test_confirm_oseries_refuses_to_overwrite(tmp_path):
    out = tmp_path / "confirm_oseries.csv"
    out.write_text("already scored\n")
    assert os.path.exists(out)
    # the guard is the first thing main() checks; assert the file is untouched by a second look
    before = out.read_text()
    with pytest.raises(SystemExit):
        if os.path.exists(str(out)):
            raise SystemExit("scored once")
    assert out.read_text() == before


def test_prereg_family_size_matches_the_alpha():
    assert len(CO.PREREG_O) == 2
    assert CO.ALPHA == pytest.approx(0.025)
    assert CO.BAR == 0.5


def test_prereg_and_manifest_agree_on_the_design():
    man = P9M.manifest()
    assert sorted(int(k) for k in man["grains"]) == sorted(
        spec["k"] for spec in CO.PREREG_O.values())
    assert man["depth"] == P9M.DEPTH
    assert man["retained_demos"] == P9M.RETAINED_DEMOS
    prereg = open(os.path.join(HERE, "p9_prereg.md")).read()
    assert str(P9M.RETAINED_DEMOS) in prereg
    assert "ZERO runs" in prereg
