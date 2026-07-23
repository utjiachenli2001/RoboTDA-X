"""Task 2 -- spectral truncated-inverse influence (the headline estimator).

IDEA. The exact IF preconditions the gradient dots by G^{-1}. The Gram's small
eigenvalues are estimated from 135 demos and are mostly sampling noise, yet G^{-1}
weights them by 1/lambda -- i.e. the noisiest directions get the LARGEST weight. That
is a mechanism for exact IF (LDS -0.16) to lose to the unpreconditioned dot product
(GradDot, +0.59). Truncating the inverse to the top-k eigendirections and acting as the
IDENTITY on the rest interpolates between the two:

    P_k = sum_{i<=k} (1/lambda_i) v_i v_i^T  +  (I - sum_{i<=k} v_i v_i^T)
    score_m = P_k K_m   (optionally / d_m),   then aggregate over members

    k = 0  ->  P_0 = I      -> GradDot          (no preconditioning at all)
    k = N  ->  P_N = G^{-1} -> exact IF         (full preconditioning)

One curve, both published endpoints. An INTERIOR maximum is the claim "the noise part
of the preconditioner hurts"; a monotone curve would refute it.

NORMALIZATION (see BLOCKERS.md #6). The two endpoints live in DIFFERENT per-member
scale conventions and cannot both be reproduced by a single one:

  normalize="dmean"  k=0 == p6_lambda_extend.scores_graddot(normalize_per_member=True)
                          == the true lambda->inf limit (C1 E=20: 0.5930)
  normalize="none"   k=N == p6_lambda_sweep.scores_at_ridge(lambda->0)["IF"]
                          == exact IF (C1 E=20: -0.1626)

Both curves are computed and both endpoint identities are asserted in
tests/test_spectral.py, each against its matching reference.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair.aggregate import aggregate as _aggregate  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds, ALPHA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")


def _eigh_desc(Gm):
    """Symmetric PSD eigendecomposition, eigenvalues DESCENDING."""
    w, V = np.linalg.eigh(Gm)          # ascending
    return w[::-1], V[:, ::-1]


def truncated_if(Z, k: int, aggregate: str = "mean", normalize: str = "dmean") -> np.ndarray:
    """-> (N,T) ensemble score matrix using only the top-k eigendirections of each G_m."""
    G, K = np.asarray(Z["G"], float), np.asarray(Z["K"], float)
    M, N, _ = K.shape
    if not (0 <= k <= N):
        raise ValueError(f"k must be in [0,{N}], got {k}")
    S = np.empty_like(K)
    for m in range(M):
        Km = K[m]
        if k == 0:
            Sm = Km.copy()
        else:
            w, V = _eigh_desc(G[m])
            Vk, wk = V[:, :k], w[:k]
            C = Vk.T @ Km                       # (k,T) coordinates in the top-k subspace
            Sm = Vk @ (C / wk[:, None]) + (Km - Vk @ C)
        if normalize == "dmean":
            Sm = Sm / float(np.mean(np.diag(G[m])))
        elif normalize not in (None, "none"):
            raise KeyError(normalize)
        S[m] = Sm
    return _aggregate(S, aggregate)


def _lds_of(Sm, Z, gmasks, obs, target):
    tids = list(Z["train_ids"])
    j = D.target_index(Z, target)
    sc = {tids[i]: float(Sm[i, j]) for i in range(len(tids))}
    return demo_grain_lds(sc, gmasks, obs[target])


def k_sweep(Z, tier, targets=("C1", "C5"), ks=None, aggregate="mean",
            normalize="dmean") -> dict:
    N = Z["K"].shape[1]
    ks = list(ks) if ks is not None else list(range(0, N + 1))
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    curves = {t: [] for t in targets}
    for k in ks:
        S = truncated_if(Z, k, aggregate=aggregate, normalize=normalize)
        for t in targets:
            rho, p, n, _, _ = _lds_of(S, Z, gm, obs, t)
            c = float(ceil[t])
            curves[t].append({"k": k, "lds": float(rho), "p": float(p), "n": int(n),
                              "ceiling": c, "ratio": float(rho) / c,
                              "passed": bool(np.isfinite(rho) and rho >= 0.5 * c
                                             and p < ALPHA)})
    return {"ks": ks, "tier": tier, "aggregate": aggregate, "normalize": normalize,
            "curves": curves}


# ------------------------------------------------------------------ spectrum + null
def gram_spectrum(Z) -> np.ndarray:
    """-> (M,N) eigenvalues, descending, per member."""
    G = np.asarray(Z["G"], float)
    return np.stack([_eigh_desc(G[m])[0] for m in range(G.shape[0])])


def spectrum_null(Z, n_perm=200, seed=0, q=95.0) -> dict:
    """Parallel-analysis noise floor for the Gram spectrum.

    Only G is cached (never Phi), so the null is built by permuting the OFF-DIAGONAL
    entries of each G_m -- destroying the demo-to-demo correlation structure while
    preserving the diagonal (each demo's own gradient norm) and the overall scale.
    Eigenvalues above the q-th percentile of this null are carrying structure that
    random demo pairing does not produce.

    k* = number of leading eigenvalues exceeding the null envelope (median over members).
    """
    rng = np.random.default_rng(seed)
    G = np.asarray(Z["G"], float)
    M, N, _ = G.shape
    iu = np.triu_indices(N, k=1)
    per_member_kstar, null_curves = [], []
    for m in range(M):
        Gm = G[m]
        real = _eigh_desc(Gm)[0]
        off = Gm[iu]
        null = np.empty((n_perm, N))
        for b in range(n_perm):
            P = np.zeros((N, N))
            P[iu] = rng.permutation(off)
            P = P + P.T
            np.fill_diagonal(P, np.diag(Gm))
            null[b] = _eigh_desc(P)[0]
        env = np.percentile(null, q, axis=0)
        above = real > env
        kstar = int(np.argmin(above)) if not above.all() else N
        per_member_kstar.append(kstar)
        null_curves.append(env)
    return {"k_star_per_member": per_member_kstar,
            "k_star_median": int(np.median(per_member_kstar)),
            "k_star_min": int(np.min(per_member_kstar)),
            "k_star_max": int(np.max(per_member_kstar)),
            "null_envelope_median_over_members": np.median(np.stack(null_curves),
                                                           axis=0).tolist(),
            "n_perm": n_perm, "percentile": q}


# ------------------------------------------------------------------ damped variant
def damped_if(Z, gamma: float, mode: str = "diag", aggregate: str = "mean",
              normalize: str = "dmean") -> np.ndarray:
    """Companion to truncation: shrink the inverse instead of truncating it.

    mode="diag": (G + gamma*diag(G))^{-1} K      -- per-direction relative damping
    mode="muI" : (G + gamma*mean(diag G)*I)^{-1} K -- the repo's adaptive ridge form
    """
    G, K = np.asarray(Z["G"], float), np.asarray(Z["K"], float)
    M, N, _ = K.shape
    S = np.empty_like(K)
    for m in range(M):
        Gm, Km = G[m], K[m]
        if mode == "diag":
            A = Gm + gamma * np.diag(np.diag(Gm))
        elif mode == "muI":
            A = Gm + gamma * float(np.mean(np.diag(Gm))) * np.eye(N)
        else:
            raise KeyError(mode)
        Sm = np.linalg.solve(A, Km)
        if normalize == "dmean":
            Sm = Sm / float(np.mean(np.diag(Gm)))
        S[m] = Sm
    return _aggregate(S, aggregate)


def damped_cv(Z, tier, gammas=None, tune_on="C1", eval_on="C5", mode="diag",
              normalize="dmean") -> dict:
    """gamma chosen by the repo's CV protocol: tune on one focal, freeze, report the other."""
    gammas = list(gammas if gammas is not None else
                  [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    best = None
    for g in gammas:
        S = damped_if(Z, g, mode=mode, normalize=normalize)
        rho = _lds_of(S, Z, gm, obs, tune_on)[0]
        if np.isfinite(rho) and (best is None or rho > best[0]):
            best = (float(rho), g)
    tuned_rho, g = best
    S = damped_if(Z, g, mode=mode, normalize=normalize)
    rho, p, n, _, _ = _lds_of(S, Z, gm, obs, eval_on)
    c = float(ceil[eval_on])
    return {"mode": mode, "frozen_gamma": g, "tuned_on": tune_on,
            "tuned_lds": tuned_rho, "evaluated_on": eval_on, "lds": float(rho),
            "p": float(p), "n": int(n), "ceiling": c, "ratio": float(rho) / c,
            "passed": bool(rho >= 0.5 * c and p < ALPHA)}


# ------------------------------------------------------------------ driver
def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    Z = D.gram_e20()
    tier = "bc_s10"
    N = Z["K"].shape[1]

    null = spectrum_null(Z, n_perm=200, seed=0)
    print(f"[spectrum] k* (parallel analysis, 200 perms): median={null['k_star_median']} "
          f"min={null['k_star_min']} max={null['k_star_max']}")

    out = {}
    for nrm in ("dmean", "none"):
        sw = k_sweep(Z, tier, ks=list(range(0, N + 1)), normalize=nrm)
        out[nrm] = sw
        for t in ("C1", "C5"):
            c = sw["curves"][t]
            best = max(c, key=lambda r: r["lds"])
            print(f"[k_sweep {nrm:6s}] {t}: k=0 -> {c[0]['lds']:+.4f} | "
                  f"k={N} -> {c[-1]['lds']:+.4f} | best k={best['k']} "
                  f"lds={best['lds']:+.4f} ratio={best['ratio']:.3f} "
                  f"p={best['p']:.4f} pass={best['passed']}")
        with open(os.path.join(RESULTS, f"k_sweep_{nrm}.json"), "w") as f:
            json.dump(sw, f, indent=1)
    for t in ("C1", "C5"):
        with open(os.path.join(RESULTS, f"k_sweep_{t}.json"), "w") as f:
            json.dump({"target": t, "tier": tier, "k_star": null["k_star_median"],
                       "dmean": out["dmean"]["curves"][t],
                       "none": out["none"]["curves"][t]}, f, indent=1)
    with open(os.path.join(RESULTS, "spectrum_null.json"), "w") as f:
        json.dump(null, f, indent=1)

    print("\n[damped] CV-frozen gamma:")
    for mode in ("diag", "muI"):
        for a, b in (("C1", "C5"), ("C5", "C1")):
            r = damped_cv(Z, tier, tune_on=a, eval_on=b, mode=mode)
            print(f"  {mode:5s} tune {a}->eval {b}: gamma={r['frozen_gamma']:.0e} "
                  f"lds={r['lds']:+.4f} ratio={r['ratio']:.3f} p={r['p']:.4f} "
                  f"pass={r['passed']}")


if __name__ == "__main__":
    main()
