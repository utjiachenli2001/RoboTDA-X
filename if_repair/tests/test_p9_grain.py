"""Pass 9 -- the grain ladder's partition invariants.

These are the load-bearing tests of the pass. An unbalanced or overlapping partition would not show
up as an error anywhere downstream: it would show up as a crossover curve with the wrong shape, and
nothing in the scoring path could tell the difference.
"""
import pytest

from if_repair import p9_grain as G

import dataset  # noqa: E402  (path set by if_repair.data.add_repo_paths at import of p9_grain)


def test_group_counts():
    assert G.n_groups(3) == 45
    assert G.n_groups(5) == 27
    assert G.n_groups(15) == 9


def test_every_group_has_exactly_k_demos():
    for k in (3, 5, 15):
        assert {len(g["demos"]) for g in G.groups(k)} == {k}


def test_each_cluster_contributes_the_same_number_of_groups():
    for k in (3, 5, 15):
        per = {}
        for g in G.groups(k):
            per[g["cluster"]] = per.get(g["cluster"], 0) + 1
        assert set(per.values()) == {15 // k}
        assert set(per) == set(dataset.clusters())


def test_partition_is_disjoint_and_covers_the_pool():
    ids, _ = dataset.train_pool()
    for k in (3, 5, 15):
        seen = [d for g in G.groups(k) for d in g["demos"]]
        assert len(seen) == len(set(seen)), f"k={k} groups overlap"
        assert sorted(seen) == sorted(ids), f"k={k} does not cover the train pool exactly"


def test_top_rung_is_exactly_the_cluster():
    """The ladder must NEST: k=15 has to be campaign N's unit, not merely resemble it."""
    _, by_c = dataset.train_pool()
    for g in G.groups(15):
        assert g["demos"] == sorted(by_c[g["cluster"]])


def test_groups_are_not_manifest_order_chunks():
    """The shuffle is the point -- see the module docstring. If chunking were task-major, small
    groups would be systematically more homogeneous than large ones and homogeneity, not size,
    would drive the curve."""
    _, by_c = dataset.train_pool()
    chunked = []
    for c, demos in by_c.items():
        for gi in range(15 // 3):
            chunked.append(sorted(demos[gi * 3:(gi + 1) * 3]))
    actual = [g["demos"] for g in G.groups(3)]
    assert sorted(actual) != sorted(chunked)


def test_mask_demos_unions_the_groups():
    gs = [g["group_id"] for g in G.groups(5)][:12]
    demos = G.mask_demos(gs, 5)
    assert len(demos) == 60
    assert len(set(demos)) == 60
    assert demos == sorted(demos)


def test_mask_demos_rejects_unknown_group():
    with pytest.raises(KeyError):
        G.mask_demos(["g5_NOPE_0"], 5)


def test_partition_is_deterministic():
    a = [dict(g) for g in G.groups(3)]
    G.groups.cache_clear()
    b = [dict(g) for g in G.groups(3)]
    assert a == b


def test_non_divisor_k_is_refused():
    with pytest.raises(ValueError):
        G.groups(4)
    with pytest.raises(ValueError):
        G.groups(8)


def test_across_arm_spans_clusters():
    gs = G.groups(5, across=True)
    assert len(gs) == 27
    assert {len(g["demos"]) for g in gs} == {5}
    seen = [d for g in gs for d in g["demos"]]
    assert len(seen) == len(set(seen)) == 135
    # at least one group must actually span clusters, or the arm is not the arm it claims to be
    _, by_c = dataset.train_pool()
    owner = {d: c for c, ds in by_c.items() for d in ds}
    assert any(len({owner[d] for d in g["demos"]}) > 1 for g in gs)
