"""Task 2 acceptance: the k-sweep must contain both published endpoints exactly."""
import numpy as np
import pytest

from if_repair import data as D
from if_repair import spectral as S

D.add_repo_paths()
from p6_lambda_sweep import scores_at_ridge, demo_grain_lds  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402

TIER = "bc_s10"


@pytest.fixture(scope="module")
def ctx():
    Z = D.gram_e20()
    return Z, D.demo_masks(), D.outcomes(TIER)


def _lds(S_mat, Z, gm, obs, t):
    return S._lds_of(S_mat, Z, gm, obs, t)[0]


@pytest.mark.parametrize("t", ["C1", "C5"])
def test_k0_equals_graddot(ctx, t):
    """k=0 <=> GradDot (identity preconditioner), in the dmean convention."""
    Z, gm, obs = ctx
    got = _lds(S.truncated_if(Z, 0, normalize="dmean"), Z, gm, obs, t)
    ref = demo_grain_lds(scores_graddot(Z, normalize_per_member=True)[t], gm, obs[t])[0]
    assert abs(got - ref) <= 1e-6, f"{t}: truncated_if(k=0)={got} vs GradDot={ref}"


@pytest.mark.parametrize("t", ["C1", "C5"])
def test_kN_equals_exact_if(ctx, t):
    """k=N <=> exact IF (full inverse), in the unnormalized convention of scores_at_ridge."""
    Z, gm, obs = ctx
    N = Z["K"].shape[1]
    got = _lds(S.truncated_if(Z, N, normalize="none"), Z, gm, obs, t)
    ref = demo_grain_lds(scores_at_ridge(Z, 1e-8)["IF"][t], gm, obs[t])[0]
    assert abs(got - ref) <= 1e-3, f"{t}: truncated_if(k=N)={got} vs exact IF={ref}"


@pytest.mark.parametrize("t", ["C1", "C5"])
def test_k0_unnormalized_equals_graddot_unnorm(ctx, t):
    Z, gm, obs = ctx
    got = _lds(S.truncated_if(Z, 0, normalize="none"), Z, gm, obs, t)
    ref = demo_grain_lds(scores_graddot(Z, normalize_per_member=False)[t], gm, obs[t])[0]
    assert abs(got - ref) <= 1e-6


def test_truncation_operator_is_exact_on_well_conditioned_synthetic():
    """P_0 = I and P_N = G^{-1} EXACTLY, proven where conditioning cannot hide a bug.

    The real Gram has cond up to 5.0e6, and eigendecomposition-based inversion there
    carries ~eps*cond^2 (~1e-4) relative error vs an LU solve -- a property of the
    matrix, not of this operator. On a well-conditioned synthetic Gram the same code
    path reproduces both endpoints to ~1e-12, which is what actually establishes that
    the truncation logic is right.
    """
    rng = np.random.default_rng(0)
    M, N, T = 3, 40, 4
    A = rng.normal(size=(M, N, N))
    G = np.stack([a @ a.T + N * np.eye(N) for a in A])      # well-conditioned PD
    K = rng.normal(size=(M, N, T))
    Z = {"G": G, "K": K, "train_ids": np.arange(N).astype(str),
         "targets": np.array([f"C{i}" for i in range(T)]), "members": np.arange(M).astype(str)}
    assert np.allclose(S.truncated_if(Z, 0, normalize="none"), K.mean(0), rtol=1e-12, atol=0)
    exact = np.stack([np.linalg.solve(G[m], K[m]) for m in range(M)]).mean(0)
    assert np.allclose(S.truncated_if(Z, N, normalize="none"), exact, rtol=1e-10, atol=0)


def test_truncation_endpoints_on_real_gram(ctx):
    """Same endpoints on the real (ill-conditioned) Gram: k=0 exact, k=N within ~1e-3."""
    Z, _, _ = ctx
    G, K = np.asarray(Z["G"], float), np.asarray(Z["K"], float)
    N = K.shape[1]
    assert np.allclose(S.truncated_if(Z, 0, normalize="none"), K.mean(0), rtol=1e-12, atol=0)
    for m in range(G.shape[0]):
        Zm = {"G": G[m:m + 1], "K": K[m:m + 1], "train_ids": Z["train_ids"],
              "targets": Z["targets"], "members": Z["members"][m:m + 1]}
        got = S.truncated_if(Zm, N, normalize="none")
        exact = np.linalg.solve(G[m], K[m])
        rel = np.linalg.norm(got - exact) / np.linalg.norm(exact)
        assert rel < 5e-3, f"member {m}: relative reconstruction error {rel:.2e}"


def test_full_inverse_mean_matches_within_cancellation(ctx):
    """Ensemble-mean level, norm-relative (see the cancellation note above)."""
    Z, _, _ = ctx
    G, K = np.asarray(Z["G"], float), np.asarray(Z["K"], float)
    SN = S.truncated_if(Z, K.shape[1], normalize="none")
    exact = np.stack([np.linalg.solve(G[m], K[m]) for m in range(G.shape[0])]).mean(0)
    rel = np.linalg.norm(SN - exact) / np.linalg.norm(exact)
    assert rel < 1e-3, rel


def test_k_must_be_in_range(ctx):
    Z, _, _ = ctx
    with pytest.raises(ValueError):
        S.truncated_if(Z, 136)
    with pytest.raises(ValueError):
        S.truncated_if(Z, -1)


def test_spectrum_shapes(ctx):
    Z, _, _ = ctx
    sp = S.gram_spectrum(Z)
    assert sp.shape == (20, 135)
    assert np.all(sp > 0), "Gram must be PD"
    assert np.all(np.diff(sp, axis=1) <= 1e-6), "eigenvalues must be descending"
