"""Task 1 -- robust aggregation over ensemble members, plus a member-tail diagnostic.

The champion GradDot averages the per-member score vectors. If a few members are
outliers (heavy-tailed across members), the MEAN is the wrong summary and a robust
aggregator should recover LDS for free -- no new information, just a better estimate
of the same population quantity.

The per-member normalization and the aggregator are ORTHOGONAL choices and are varied
independently here. Holding normalization fixed while only the aggregator changes is
what makes the comparison honest: `mean` + the canonical normalization reproduces the
published baseline exactly (asserted in tests/test_estimators.py).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds, ALPHA, DEFAULT_RIDGE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

AGGREGATORS = ("mean", "median", "trimmed_0.1", "trimmed_0.2", "hodges_lehmann")


# ------------------------------------------------------------------ per-member scores
def per_member_scores(Z, estimator: str, ridge_rel: float = DEFAULT_RIDGE) -> np.ndarray:
    """-> (M, N, T) per-member score matrices, BEFORE any normalization or aggregation.

    IF / TRAK use the same closed forms as p6_lambda_sweep.scores_at_ridge, but kept
    per-member instead of summed (scores_at_ridge accumulates over m internally, so the
    per-member pieces cannot be recovered from it).
    """
    G, K = np.asarray(Z["G"], float), np.asarray(Z["K"], float)
    M, N, _T = K.shape
    out = np.empty_like(K)
    for m in range(M):
        Gm, Km = G[m], K[m]
        if estimator == "GradDot":
            out[m] = Km
            continue
        lam = ridge_rel * float(np.mean(np.diag(Gm)))
        I = np.eye(N)
        if estimator == "TRAK":
            out[m] = np.linalg.solve(Gm + lam * I, Km)
        elif estimator == "IF":
            inner = np.linalg.solve(lam * N * I + Gm, Km)
            out[m] = (Km - Gm @ inner) / lam
        else:
            raise KeyError(estimator)
    return out


def normalize_members(S: np.ndarray, how: str | None, Z=None) -> np.ndarray:
    """Per-member scale normalization. S is (M,N,T)."""
    if how in (None, "none"):
        return S
    if how == "unitL2":
        nrm = np.linalg.norm(S, axis=1, keepdims=True)
        if np.any(nrm == 0):
            raise RuntimeError("zero-norm member score vector")
        return S / nrm
    if how == "dmean":
        G = np.asarray(Z["G"], float)
        d = np.array([np.mean(np.diag(G[m])) for m in range(S.shape[0])])
        return S / d[:, None, None]
    raise KeyError(how)


# ------------------------------------------------------------------ aggregators
def hodges_lehmann(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median of all pairwise Walsh averages (i<=j) along `axis`."""
    x = np.moveaxis(np.asarray(x, float), axis, 0)
    m = x.shape[0]
    iu, ju = np.triu_indices(m, k=0)
    walsh = 0.5 * (x[iu] + x[ju])              # (n_pairs, ...)
    return np.median(walsh, axis=0)


def aggregate(S: np.ndarray, how: str) -> np.ndarray:
    """(M,N,T) -> (N,T) by aggregating over members."""
    if how == "mean":
        return S.mean(0)
    if how == "median":
        return np.median(S, axis=0)
    if how.startswith("trimmed_"):
        p = float(how.split("_")[1])
        return stats.trim_mean(S, p, axis=0)
    if how == "hodges_lehmann":
        return hodges_lehmann(S, axis=0)
    raise KeyError(how)


# ------------------------------------------------------------------ tail diagnostic
def member_tail_stats(S: np.ndarray, Z, target: str) -> dict:
    """Per-demo heavy-tailedness across members, for one target column."""
    j = D.target_index(Z, target)
    X = S[:, :, j]                                     # (M, N)
    # scale-free per demo: standardize each demo's member vector before shape stats
    mu, sd = X.mean(0), X.std(0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    Xs = (X - mu) / sd
    kurt = stats.kurtosis(Xs, axis=0, fisher=True, bias=False)   # 0 = Gaussian
    spread = (X.max(0) - X.min(0)) / np.where(np.abs(mu) == 0, np.nan, np.abs(mu))
    return {
        "target": target, "n_members": int(X.shape[0]), "n_demos": int(X.shape[1]),
        "excess_kurtosis_median": float(np.nanmedian(kurt)),
        "excess_kurtosis_p90": float(np.nanpercentile(kurt, 90)),
        "excess_kurtosis_max": float(np.nanmax(kurt)),
        "frac_demos_kurtosis_gt_1": float(np.nanmean(kurt > 1.0)),
        "maxmin_over_absmean_median": float(np.nanmedian(spread)),
        "maxmin_over_absmean_p90": float(np.nanpercentile(spread, 90)),
    }


# ------------------------------------------------------------------ experiment
CONFIGS_E20 = [
    ("GradDot", "unitL2", None),
    ("GradDot", "dmean", None),
    ("IF", "none", DEFAULT_RIDGE),
    ("IF", "none", 1e1),
    ("IF", "unitL2", 1e1),
    ("TRAK", "none", DEFAULT_RIDGE),
    ("TRAK", "unitL2", 1e1),
]


def exp_01(tier="bc_s10", targets=("C1", "C5"), configs=None, write=True) -> pd.DataFrame:
    Z = D.cache_for(tier)
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    tids = list(Z["train_ids"])
    rows = []
    for est, nrm, rr in (configs or CONFIGS_E20):
        S0 = per_member_scores(Z, est, ridge_rel=(rr if rr is not None else DEFAULT_RIDGE))
        S = normalize_members(S0, nrm, Z)
        for agg in AGGREGATORS:
            A = aggregate(S, agg)
            for t in targets:
                j = D.target_index(Z, t)
                sc = {tids[i]: float(A[i, j]) for i in range(len(tids))}
                rho, p, n, _, _ = demo_grain_lds(sc, gm, obs[t])
                c = float(ceil[t])
                rows.append({
                    "tier": tier, "estimator": est, "normalize": nrm,
                    "ridge_rel": rr, "aggregator": agg, "target": t,
                    "lds": float(rho), "ceiling": c, "ratio": float(rho) / c,
                    "bar": 0.5 * c, "p": float(p), "n": n,
                    "passed": bool(np.isfinite(rho) and rho >= 0.5 * c and p < ALPHA),
                })
    df = pd.DataFrame(rows)
    if write:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        df.to_csv(os.path.join(HERE, "results", f"exp_01_{tier}.csv"), index=False)
    return df


def tail_report(tier="bc_s10", targets=("C1", "C5")) -> pd.DataFrame:
    Z = D.cache_for(tier)
    rows = []
    for est, nrm, rr in [("GradDot", "unitL2", None), ("GradDot", "dmean", None),
                         ("IF", "none", 1e1), ("TRAK", "none", DEFAULT_RIDGE)]:
        S = normalize_members(
            per_member_scores(Z, est, ridge_rel=(rr if rr is not None else DEFAULT_RIDGE)),
            nrm, Z)
        for t in targets:
            r = member_tail_stats(S, Z, t)
            r.update(estimator=est, normalize=nrm, tier=tier)
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    for tier in ("bc_s10", "dev_s6"):
        df = exp_01(tier)
        print("=" * 104)
        print(f"EXP_01 -- aggregator sweep, tier={tier} "
              f"(E={D.cache_for(tier)['K'].shape[0]}), demo grain, n=24")
        print("=" * 104)
        for (est, nrm, rr), sub in df.groupby(["estimator", "normalize", "ridge_rel"],
                                              dropna=False):
            lbl = f"{est}/{nrm}" + (f"@{rr:.0e}" if rr is not None and rr == rr else "")
            piv = sub.pivot_table(index="aggregator", columns="target", values="ratio")
            piv = piv.reindex(AGGREGATORS)
            base = piv.loc["mean"]
            print(f"\n  {lbl}")
            for a in piv.index:
                cells = "  ".join(f"{t} ratio={piv.loc[a, t]:+.3f} "
                                  f"(vs mean {piv.loc[a, t]-base[t]:+.3f})"
                                  for t in piv.columns)
                print(f"    {a:<16s} {cells}")
        # gate
        c1 = df[df.target == "C1"]
        best = c1.loc[c1.lds.idxmax()]
        mean_best = c1[c1.aggregator == "mean"].lds.max()
        print(f"\nGATE tier={tier}: best C1 LDS over all aggregators = {best.lds:+.4f} "
              f"({best.estimator}/{best.normalize}, {best.aggregator}); "
              f"best with mean = {mean_best:+.4f}; "
              f"robust aggregator helps = {bool(best.lds > mean_best + 1e-9)}")
    print("\n" + "=" * 104)
    print("MEMBER TAIL DIAGNOSTIC")
    print("=" * 104)
    td = tail_report("bc_s10")
    print(td[["estimator", "normalize", "target", "excess_kurtosis_median",
              "excess_kurtosis_p90", "frac_demos_kurtosis_gt_1",
              "maxmin_over_absmean_median"]].to_string(index=False))
    td.to_csv(os.path.join(HERE, "results", "exp_01_tail_stats.csv"), index=False)


if __name__ == "__main__":
    main()
