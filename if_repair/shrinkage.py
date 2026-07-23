"""Task 3 -- variance-aware shrinkage across ensemble members.

Each demo's score is an ensemble mean over M members with its own cross-member
variance. Demos whose members disagree carry little information; James-Stein shrinks
those toward a pooled centre (the demo's cluster mean), keeping confident demos put.

Reported with BOTH faithfulness (LDS/ceiling) and stability (top-k Jaccard across
disjoint sub-ensembles) -- a shrinkage that buys stability by flattening the ranking
would show up as stable-but-not-faithful, which is not an improvement.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair.aggregate import per_member_scores, normalize_members  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds, ALPHA, DEFAULT_RIDGE  # noqa: E402
import dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def cluster_vector(train_ids):
    """-> array of cluster label per demo, via the repo's own task->cluster map."""
    c_of = dataset.cluster_of_task()
    out = []
    for d in train_ids:
        suite, task, _demo = dataset.parse_did(d)
        out.append(c_of.get(task, c_of.get(f"{suite}/{task}", "UNK")))
    return np.array(out)


def james_stein(scores: np.ndarray, member_var: np.ndarray, toward="cluster_mean",
                clusters=None, cap=1.0) -> np.ndarray:
    """scores: (N,) ensemble means. member_var: (N,) variance OF THE MEAN across members.

    JS factor per demo: b = 1 - (k-2)*sigma_i^2 / ||x - centre||^2, clipped to [0, cap].
    """
    x = np.asarray(scores, float)
    v = np.asarray(member_var, float)
    N = x.size
    if toward == "cluster_mean":
        if clusters is None:
            raise ValueError("cluster_mean needs clusters")
        centre = np.empty(N)
        for c in np.unique(clusters):
            sel = clusters == c
            centre[sel] = x[sel].mean()
    elif toward == "grand_mean":
        centre = np.full(N, x.mean())
    elif toward == "zero":
        centre = np.zeros(N)
    else:
        raise KeyError(toward)
    diff = x - centre
    ss = float(np.sum(diff ** 2))
    if ss <= 0:
        return x.copy()
    b = 1.0 - (max(N - 2, 1) * float(np.mean(v))) / ss
    b = float(np.clip(b, 0.0, cap))
    return centre + b * diff


def _lds(vec, train_ids, gmasks, obs_t):
    sc = {train_ids[i]: float(vec[i]) for i in range(len(train_ids))}
    return demo_grain_lds(sc, gmasks, obs_t)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k=20) -> float:
    ia, ib = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(ia & ib) / len(ia | ib)


def exp_02(tier="bc_s10", targets=("C1", "C5"), k_top=20, seed=0) -> pd.DataFrame:
    Z = D.cache_for(tier)
    gm, obs, ceil = D.demo_masks(), D.outcomes(tier), D.ceilings(tier)
    tids = list(Z["train_ids"])
    clusters = cluster_vector(tids)
    M = Z["K"].shape[0]
    rng = np.random.default_rng(seed)
    rows = []
    for est, nrm, rr in [("GradDot", "unitL2", None), ("GradDot", "dmean", None),
                         ("IF", "none", 1e1)]:
        S = normalize_members(
            per_member_scores(Z, est, ridge_rel=(rr or DEFAULT_RIDGE)), nrm, Z)
        for t in targets:
            j = D.target_index(Z, t)
            X = S[:, :, j]                                  # (M,N)
            mean = X.mean(0)
            var_of_mean = X.var(0, ddof=1) / M
            c = float(ceil[t])
            variants = {"baseline_mean": mean}
            for toward in ("cluster_mean", "grand_mean", "zero"):
                variants[f"js_{toward}"] = james_stein(mean, var_of_mean, toward,
                                                       clusters)
            # stability: disjoint halves of the ensemble, 20 random splits
            for name, vec in variants.items():
                rho, p, n, _, _ = _lds(vec, tids, gm, obs[t])
                jac = []
                for _ in range(20):
                    perm = rng.permutation(M)
                    h1, h2 = perm[: M // 2], perm[M // 2:]
                    v1, v2 = X[h1].mean(0), X[h2].mean(0)
                    if name != "baseline_mean":
                        tw = name[3:]
                        v1 = james_stein(v1, X[h1].var(0, ddof=1) / len(h1), tw, clusters)
                        v2 = james_stein(v2, X[h2].var(0, ddof=1) / len(h2), tw, clusters)
                    jac.append(topk_jaccard(v1, v2, k_top))
                rows.append({"tier": tier, "estimator": est, "normalize": nrm,
                             "variant": name, "target": t, "lds": float(rho),
                             "ceiling": c, "ratio": float(rho) / c, "p": float(p),
                             "n": int(n),
                             "passed": bool(np.isfinite(rho) and rho >= .5 * c
                                            and p < ALPHA),
                             f"jaccard_top{k_top}_mean": float(np.mean(jac)),
                             f"jaccard_top{k_top}_sd": float(np.std(jac))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", f"exp_02_{tier}.csv"), index=False)
    return df


def main():
    for tier in ("bc_s10",):
        df = exp_02(tier)
        print("=" * 100)
        print(f"EXP_02 -- variance-aware shrinkage, tier={tier}")
        print("=" * 100)
        print(df[["estimator", "normalize", "variant", "target", "lds", "ratio", "p",
                  "passed", "jaccard_top20_mean"]].to_string(index=False))
        for t in ("C1", "C5"):
            s = df[df.target == t]
            base = s[s.variant == "baseline_mean"].lds.max()
            print(f"  GATE {t}: best shrunk LDS={s[s.variant!='baseline_mean'].lds.max():+.4f} "
                  f"vs baseline {base:+.4f} -> "
                  f"helps={bool(s[s.variant!='baseline_mean'].lds.max() > base + 1e-9)}")


if __name__ == "__main__":
    main()
