"""W6 -- hybrid gradient-prior datamodel.

The datamodel is the ONLY estimator that beats GradDot out of sample (FINDINGS), but it costs
one retrain per mask -- 240 GPU-jobs for the 24-mask LDS. The pragmatic form of "repairing IF"
is therefore: can the gradient scores, as a PRIOR on the datamodel's coefficients, buy the same
LDS with fewer masks? If a gradient-primed datamodel at 12 masks matches the plain datamodel at
24, gradients have halved the retrain budget -- win condition 2.

Three priors, all reducing to an sklearn Ridge/Lasso by a change of variables, prior strength
tuned by CV-MSE over the training masks (never on the LDS):

  adaptive_lasso   penalty w_d|beta_d|, w_d = 1/(|s_d|+eps)^gamma   (low-GradDot demos penalized
                   harder). Fit lasso on X/w, beta = gamma/w.
  kernel_ridge     Tikhonov ||y-Xb||^2 + lam b^T Sigma^-1 b, Sigma = a*Ghat + (1-a)*I,
                   Ghat = member-mean cached Gram (PSD). Cholesky Sigma=LL^T; ridge on X@L; b=L@g.
  priormean_ridge  ||y-Xb||^2 + lam||b - c*s||^2. Ridge on residual y-c*(X@s); b = delta + c*s.

s = GradDot_dmean per target, from the E=20 cached Gram (bc_s10). Evaluation: subsample K masks
from the pooled G/H/I universe (campaign A/B/I outcomes, same construction), LOO-LDS of each model
vs plain datamodel vs GradDot alone, R resamples, per target C1/C2/C5. Deliverable: LDS-vs-K curve.
Gate: hybrid@12 >= plain@24 on C2 or C5.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402
from if_repair.eval import build_scores  # noqa: E402

D.add_repo_paths()
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

K_GRID = (6, 9, 12, 16, 20, 24)
N_RESAMPLE = 20
TARGETS = ("C1", "C2", "C5")
ALPHA_L = (1e-3, 1e-2, 1e-1, 1e0)
ALPHA_R = (1e-1, 1e0, 1e1, 1e2)
GAMMA = (0.5, 1.0, 2.0)
KA = (0.3, 0.6, 0.9)          # kernel blend alpha
CPRIOR = (0.0, 0.5, 1.0, 2.0)  # prior-mean strength (0 -> plain ridge)


def prior_scores():
    """GradDot_dmean per target from the E=20 cached Gram -> {target: (135,) aligned to demo_ids}."""
    sc = build_scores({"kind": "GradDot", "normalize": "dmean", "aggregator": "mean"}, "bc_s10")
    Z = D.cache_for("bc_s10")
    demo_ids = list(Z["train_ids"])
    return {t: np.array([sc[t][d] for d in demo_ids]) for t in sc}, demo_ids


def ghat_chol():
    """member-mean cached Gram, scale-normalized, -> function a -> chol(a*Ghat+(1-a)I)."""
    Z = D.cache_for("bc_s10")
    G = np.asarray(Z["G"], float).mean(0)                 # (135,135), PSD
    G = 0.5 * (G + G.T)
    G = G / np.mean(np.diag(G))                            # unit mean-diagonal
    n = G.shape[0]
    def chol(a):
        S = a * G + (1 - a) * np.eye(n)
        return np.linalg.cholesky(S + 1e-8 * np.eye(n))
    return chol


def universe(target):
    """Pooled G/H/I masks with finite campaign outcomes -> (list[demos], outcome vector)."""
    from if_repair import retrain as RT
    draws = [("G", D.demo_masks(), "A"),
             ("H", RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0], "B"),
             ("I", RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0], "I")]
    masks, ys = [], []
    for _, ms, camp in draws:
        raw = F.campaign_outcomes(camp, "plain", targets=(target,))[target]
        obs = F.seed_mean(raw)
        for m in ms:
            mid = m["mask_id"]
            if mid in obs and np.isfinite(obs[mid]):
                masks.append(m["demos"]); ys.append(obs[mid])
    return masks, np.array(ys, float)


def design(mask_demos, demo_ids):
    idx = {d: i for i, d in enumerate(demo_ids)}
    X = np.zeros((len(mask_demos), len(demo_ids)))
    for r, demos in enumerate(mask_demos):
        for d in demos:
            if d in idx:
                X[r, idx[d]] = 1.0
    return X


# ------------------------------------------------------------------ fitters (return beta)
def fit_plain_lasso(X, y, a):
    return Lasso(alpha=a, fit_intercept=True, max_iter=50000).fit(X, y).coef_

def fit_plain_ridge(X, y, a):
    return Ridge(alpha=a, fit_intercept=True).fit(X, y).coef_

def fit_adaptive_lasso(X, y, a, s, gamma):
    """penalty w_d|beta_d|, w_d = 1/(|s_d|/scale + 0.1)^gamma, mean-normalized so the overall
    lasso strength is comparable to plain lasso. Fit on X/w (=X*(...)^gamma), beta=coef/w."""
    scale = np.median(np.abs(s)) + 1e-12
    w = 1.0 / (np.abs(s) / scale + 0.1) ** gamma
    w = w / w.mean()
    g = Lasso(alpha=a, fit_intercept=True, max_iter=50000).fit(X / w[None, :], y).coef_
    return g / w

def fit_kernel_ridge(X, y, a, L):
    g = Ridge(alpha=a, fit_intercept=True).fit(X @ L, y).coef_
    return L @ g

def fit_priormean_ridge(X, y, a, s, c):
    resid = y - c * (X @ s)
    delta = Ridge(alpha=a, fit_intercept=True).fit(X, resid).coef_
    return delta + c * s


def cv_mse(X, y, fitter, params, folds=5, seed=0):
    """out-of-fold MSE of mask-outcome prediction; never touches the LDS."""
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold = np.array_split(order, min(folds, n))
    errs = []
    for te in fold:
        tr = np.setdiff1d(np.arange(n), te)
        if len(tr) < 3:
            continue
        b = fitter(X[tr], y[tr], *params)
        mu = y[tr].mean()
        pred = (X[te] - X[tr].mean(0)) @ b + mu     # intercept via centering
        errs.append(np.mean((pred - y[te]) ** 2))
    return np.mean(errs) if errs else np.inf


def select(X, y, fitter, grid):
    best, bs = grid[0], np.inf
    for p in grid:
        m = cv_mse(X, y, fitter, p)
        if m < bs:
            bs, best = m, p
    return best


def loo_lds(X, y, fitter, params):
    """each mask predicted by a refit that excluded it -> Spearman(pred, out)."""
    n = len(y)
    pred = np.empty(n)
    for r in range(n):
        keep = np.arange(n) != r
        b = fitter(X[keep], y[keep], *params)
        mu = y[keep].mean()
        pred[r] = (X[r] - X[keep].mean(0)) @ b + mu
    return spearman(pred, y)


def main():
    priors, demo_ids = prior_scores()
    L_of = ghat_chol()
    rng = np.random.default_rng(0)
    rows = []
    for t in TARGETS:
        mask_demos, yfull = universe(t)
        Xfull = design(mask_demos, demo_ids)
        s = priors[t]
        La, Lb, Lc = L_of(0.3), L_of(0.6), L_of(0.9)
        Ls = {0.3: La, 0.6: Lb, 0.9: Lc}
        for K in K_GRID:
            acc = {k: [] for k in ("GradDot", "plain_lasso", "plain_ridge",
                                   "adaptive_lasso", "kernel_ridge", "priormean_ridge")}
            for _ in range(N_RESAMPLE):
                sel = rng.choice(len(yfull), K, replace=False)
                X, y = Xfull[sel], yfull[sel]
                # GradDot alone (score fixed, LDS over these K masks)
                acc["GradDot"].append(spearman(X @ s, y))
                # plain lasso / ridge
                a = select(X, y, fit_plain_lasso, [(v,) for v in ALPHA_L])
                acc["plain_lasso"].append(loo_lds(X, y, fit_plain_lasso, a))
                a = select(X, y, fit_plain_ridge, [(v,) for v in ALPHA_R])
                acc["plain_ridge"].append(loo_lds(X, y, fit_plain_ridge, a))
                # adaptive lasso
                p = select(X, y, lambda Xt, yt, aa, gg: fit_adaptive_lasso(Xt, yt, aa, s, gg),
                           [(aa, gg) for aa in ALPHA_L for gg in GAMMA])
                acc["adaptive_lasso"].append(
                    loo_lds(X, y, lambda Xt, yt, aa, gg: fit_adaptive_lasso(Xt, yt, aa, s, gg), p))
                # kernel ridge
                p = select(X, y, lambda Xt, yt, aa, ka: fit_kernel_ridge(Xt, yt, aa, Ls[ka]),
                           [(aa, ka) for aa in ALPHA_R for ka in KA])
                acc["kernel_ridge"].append(
                    loo_lds(X, y, lambda Xt, yt, aa, ka: fit_kernel_ridge(Xt, yt, aa, Ls[ka]), p))
                # prior-mean ridge
                p = select(X, y, lambda Xt, yt, aa, cc: fit_priormean_ridge(Xt, yt, aa, s, cc),
                           [(aa, cc) for aa in ALPHA_R for cc in CPRIOR])
                acc["priormean_ridge"].append(
                    loo_lds(X, y, lambda Xt, yt, aa, cc: fit_priormean_ridge(Xt, yt, aa, s, cc), p))
            for est, vals in acc.items():
                v = np.array([x for x in vals if np.isfinite(x)])
                rows.append({"target": t, "K": K, "estimator": est,
                             "lds_mean": float(v.mean()) if len(v) else np.nan,
                             "lds_sd": float(v.std()) if len(v) else np.nan,
                             "n_ok": len(v)})
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b16_hybrid_lds_vs_k.csv"), index=False)

    for t in TARGETS:
        print("=" * 96)
        print(f"W6 -- {t}: LDS vs #masks (mean +- sd over {N_RESAMPLE} resamples, pooled G/H/I "
              f"universe)")
        print("=" * 96)
        sub = df[df.target == t]
        piv = sub.pivot(index="K", columns="estimator", values="lds_mean")
        cols = ["GradDot", "plain_lasso", "plain_ridge", "adaptive_lasso", "kernel_ridge",
                "priormean_ridge"]
        print(piv[cols].round(3).to_string())
        plain24 = sub[(sub.K == 24) & (sub.estimator == "plain_lasso")].lds_mean.iloc[0]
        plainr24 = sub[(sub.K == 24) & (sub.estimator == "plain_ridge")].lds_mean.iloc[0]
        best_plain24 = max(plain24, plainr24)
        print(f"\n  plain datamodel @24: lasso {plain24:.3f}, ridge {plainr24:.3f}  "
              f"(gate target = {best_plain24:.3f})")
        print("  hybrid @12 vs that gate:")
        for est in ("adaptive_lasso", "kernel_ridge", "priormean_ridge"):
            h12 = sub[(sub.K == 12) & (sub.estimator == est)].lds_mean.iloc[0]
            ok = "PASS" if h12 >= best_plain24 else "no"
            print(f"    {est:18s} @12 = {h12:.3f}   -> {ok}")
        # secondary: does the prior regularize the datamodel at low K, and where does the
        # datamodel first beat GradDot-alone?
        gd = {int(r.K): r.lds_mean for _, r in sub[sub.estimator == "GradDot"].iterrows()}
        best_dm = {int(k): max(sub[(sub.K == k) & (sub.estimator.isin(
            ["plain_lasso", "plain_ridge", "kernel_ridge", "priormean_ridge", "adaptive_lasso"]))]
            .lds_mean) for k in K_GRID}
        cross = next((k for k in K_GRID if best_dm[k] > gd[k]), None)
        print(f"  best datamodel first beats GradDot-alone at K = {cross}"
              f" (GradDot flat ~{np.mean(list(gd.values())):.2f})")
        pl12 = sub[(sub.K == 12) & (sub.estimator == "plain_lasso")].lds_mean.iloc[0]
        kr12 = sub[(sub.K == 12) & (sub.estimator == "kernel_ridge")].lds_mean.iloc[0]
        print(f"  prior regularization @12: kernel_ridge {kr12:+.3f} vs plain_lasso {pl12:+.3f} "
              f"(delta {kr12-pl12:+.3f})")


if __name__ == "__main__":
    main()
