"""Estimator-level invariants for the Task 1-3 machinery."""
import numpy as np
import pytest

from if_repair import data as D
from if_repair.aggregate import (AGGREGATORS, aggregate, hodges_lehmann,
                                 normalize_members, per_member_scores)
from if_repair.eval import build_scores, evaluate
from if_repair.shrinkage import james_stein, topk_jaccard

D.add_repo_paths()
from p6_lambda_sweep import scores_at_ridge, DEFAULT_RIDGE  # noqa: E402


@pytest.fixture(scope="module")
def Z():
    return D.gram_e20()


@pytest.mark.parametrize("est", ["IF", "TRAK"])
def test_per_member_mean_reproduces_scores_at_ridge(Z, est):
    """Splitting scores_at_ridge per member and re-averaging must be a no-op."""
    S = per_member_scores(Z, est, ridge_rel=DEFAULT_RIDGE).mean(0)
    ref = scores_at_ridge(Z, DEFAULT_RIDGE)[est]
    tids, tgts = list(Z["train_ids"]), list(Z["targets"])
    got = np.array([[S[i, j] for j in range(len(tgts))] for i in range(len(tids))])
    exp = np.array([[ref[tgts[j]][tids[i]] for j in range(len(tgts))]
                    for i in range(len(tids))])
    assert np.allclose(got, exp, rtol=1e-10, atol=0)


def test_hodges_lehmann_matches_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 5))
    got = hodges_lehmann(x, axis=0)
    for c in range(5):
        walsh = [0.5 * (x[i, c] + x[j, c]) for i in range(7) for j in range(i, 7)]
        assert abs(got[c] - np.median(walsh)) < 1e-12


def test_hodges_lehmann_equals_midpoint_for_pair():
    assert abs(float(hodges_lehmann(np.array([[1.0], [3.0]]), axis=0)[0]) - 2.0) < 1e-12


@pytest.mark.parametrize("agg", AGGREGATORS)
def test_aggregators_shape_and_translation_equivariance(Z, agg):
    S = per_member_scores(Z, "GradDot")[:, :20, :3]
    A = aggregate(S, agg)
    assert A.shape == S.shape[1:]
    assert np.allclose(aggregate(S + 5.0, agg), A + 5.0, rtol=1e-9)


def test_unitL2_normalization_is_per_member_scale_invariant(Z):
    S = per_member_scores(Z, "GradDot")
    scale = np.arange(1, S.shape[0] + 1, dtype=float)[:, None, None]
    assert np.allclose(normalize_members(S, "unitL2"),
                       normalize_members(S * scale, "unitL2"), rtol=1e-10)


def test_james_stein_preserves_cluster_centres_and_contracts():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    cl = np.array(["a", "a", "b", "b"])
    out = james_stein(x, np.full(4, 0.5), "cluster_mean", cl)
    for c in np.unique(cl):
        sel = cl == c
        assert abs(out[sel].mean() - x[sel].mean()) < 1e-9
        assert np.all(np.abs(out[sel] - x[sel].mean())
                      <= np.abs(x[sel] - x[sel].mean()) + 1e-12)


def test_james_stein_with_zero_variance_is_identity():
    x = np.array([1.0, 5.0, 9.0])
    assert np.allclose(james_stein(x, np.zeros(3), "grand_mean"), x)


def test_topk_jaccard_bounds():
    a = np.arange(50.0)
    assert topk_jaccard(a, a, 10) == 1.0
    assert topk_jaccard(a, -a, 10) == 0.0


def test_evaluate_returns_full_record_never_bare_rho():
    sc = build_scores({"kind": "GradDot", "normalize": "unitL2", "aggregator": "mean"},
                      "bc_s10")
    r = evaluate(sc["C1"], "C1", "bc_s10")
    for k in ("lds", "ceiling", "ratio", "p", "ci95", "passed", "n", "bar"):
        assert k in r
    assert abs(r["lds"] - 0.5130434782608695) < 1e-12
    assert r["ci95"][0] < r["lds"] < r["ci95"][1]


def test_frozen_estimators_are_actually_different_on_bc():
    """Guard: the two frozen Task-6 estimators must not silently be the same thing.

    They DO coincide on the diffusion tier (E=5) -- see FINDINGS.md -- so this asserts
    distinctness where it matters, on bc_s10.
    """
    a = build_scores({"kind": "GradDot", "normalize": "dmean", "aggregator": "mean"},
                     "bc_s10")["C1"]
    b = build_scores({"kind": "GradDot", "normalize": "unitL2",
                      "aggregator": "hodges_lehmann"}, "bc_s10")["C1"]
    va = np.array([a[k] for k in sorted(a)])
    vb = np.array([b[k] for k in sorted(b)])
    assert not np.allclose(va / np.linalg.norm(va), vb / np.linalg.norm(vb))


def test_tier_triples_are_bound_correctly():
    assert D.TIERS["diff_s10"]["agg"] == "median"      # BLOCKERS #4
    assert D.TIERS["bc_s10"]["agg"] == "mean"
    assert D.TIERS["bc_s10"]["cache"] == "e20"
    assert D.TIERS["dev_s6"]["cache"] == "e10"


def test_dev_and_holdout_targets_are_disjoint():
    assert not set(D.DEV_TARGETS) & set(D.HOLDOUT_TARGETS)
