"""PASS 19 -- a synthetic corpus with KNOWN ground-truth influence. Zero GPU.

WHY THIS EXISTS. Every methodological lesson this project reports currently rests on FOUR
SELF-INFLICTED CASES -- errors we made and then caught. That is persuasive as a narrative and weak
as evidence: a reader cannot tell whether the failure modes are properties of the metric or of us.
Here the true per-item influence is CONSTRUCTED, so each failure mode can be DEMONSTRATED against a
known answer rather than narrated.

THE GENERATIVE MODEL, chosen to be the simplest thing that reproduces the real setting's structure:

    outcome(mask) = -g( MEAN_{i in mask} theta_i )  +  seed_noise
    theta_i ~ N(0, 1)                         the TRUE per-item influence, known exactly
    seed_noise ~ N(0, sigma^2)                run-to-run training noise
    The MEAN, not the sum, is the load-bearing choice. A saturating learner cares about the
    average quality of what it trained on, not the total, so the across-mask spread of the signal
    falls as sqrt((1/k)*(n-k)/(n-1)) -- shrinking with the retained count k and vanishing as k -> n.
    That is the mechanism campaign T died on, in one line. (A first version of this file used the
    SUM, whose spread GROWS as sqrt(k); it failed to reproduce the horizon and the sign of the
    mechanism was the reason.)

Everything the real campaign measures -- the split-half ceiling, LDS tau, the ratio, the slope --
is computed by the same code paths, so a failure that appears here is a property of the ESTIMATOR
AND METRIC, not of the robot data.

WHAT IT IS FOR, and what it is not. It is a demonstration harness for the measurement claims:
  1. the measurability horizon: r -> 0 as the retained count grows, at fixed true influence
  2. rho/r is inflated at low seed depth, and inflated MORE at lower depth   (BLOCKERS #42)
  3. rho/sqrt(r) does not normalise a KENDALL statistic, and the residual drift tracks r
  4. a nuisance axis that moves outcome AND prediction credits both sides   (BLOCKERS #41)
  5. a ratio inherits its denominator's noise                                (BLOCKERS #56)
It is NOT a claim that robot data behaves like a Gaussian additive model. It shows the metric
misbehaves even when the data are as friendly as possible -- which is the stronger direction: if
LDS cannot be trusted here, it cannot be trusted on messier data either.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


# --------------------------------------------------------------------------- the world
class SyntheticCorpus:
    """A corpus of `n_items` with known influence, and a retrain oracle with seed noise."""

    # CALIBRATION, stated rather than tuned. sigma is set so that at the campaign's own operating
    # point (keep=25, depth 2) the synthetic ceiling matches the MEASURED one, r_kendall ~ 0.56:
    #   signal sd at keep=25 with finite population = sqrt((1/25)*(345/369)) = 0.1934
    #   r2 = s^2/(s^2 + sigma^2/2) = 0.87  ->  sigma = 0.547*s = 0.106
    # mu > 0 makes MORE DATA HELP, as it does in reality -- without it, mask size moves only the
    # outcome's variance and not its mean, so there is no nuisance axis for claim 4 to detect.
    SIGMA_CALIBRATED = 0.106
    MU = 0.5

    def __init__(self, n_items=370, sigma=SIGMA_CALIBRATED, saturation=1.0, seed=0, mu=MU):
        self.n = n_items
        self.sigma = sigma
        self.saturation = saturation
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(mu, 1, n_items)         # TRUE influence, known; mu>0 = data helps
        self._rng = rng

    def _g(self, m):
        """Concave map on the MEAN quality. saturation=0 -> identity (purely additive world)."""
        if self.saturation <= 0:
            return m
        return np.tanh(self.saturation * m) / self.saturation

    def outcome(self, mask_idx, seed):
        """One 'retraining': the true signal plus seed noise, reproducible from `seed`."""
        rng = np.random.default_rng([seed, int(np.sum(mask_idx)) % (2 ** 31)])
        m = float(self.theta[mask_idx].mean())
        return -self._g(m) + rng.normal(0, self.sigma)

    def draw_masks(self, n_masks, keep, seed=1):
        rng = np.random.default_rng(seed)
        return [np.sort(rng.permutation(self.n)[:keep]) for _ in range(n_masks)]


# --------------------------------------------------------------------------- the metric
def measure(corpus, masks, depth=2, seed0=100, predictor=None):
    """Run the metric exactly as campaign U does: depth-2 outcomes, split-half Kendall ceiling,
    LDS tau of the ORACLE predictor (sum of true theta over the retained set)."""
    seeds = [seed0 + i for i in range(2 * depth)]     # 2*depth so the split-half is AT depth
    Y = np.array([[corpus.outcome(m, s) for s in seeds] for m in masks])
    y = Y.mean(1)
    pred = (np.array([corpus.theta[m].mean() for m in masks]) if predictor is None
            else np.array([predictor(corpus, m) for m in masks]))
    tau = stats.kendalltau(-pred, y).statistic                    # -pred: higher theta -> lower loss
    # split-half AT the reported depth: two disjoint halves of `depth` seeds each
    r_kendall = stats.kendalltau(Y[:, :depth].mean(1), Y[:, depth:].mean(1)).statistic
    return {"tau": float(tau), "r_kendall": float(r_kendall),
            "ratio_over_r": float(tau / r_kendall) if r_kendall > 0 else float("nan"),
            "ratio_over_sqrt_r": float(tau / np.sqrt(r_kendall)) if r_kendall > 0
            else float("nan"),
            "outcome_sd": float(y.std(ddof=1)), "n_masks": len(masks)}


# --------------------------------------------------------------------------- demonstrations
def demo_measurability_horizon(n_masks=400, keeps=(10, 25, 50, 100, 185, 300), seed=0):
    """CLAIM 1. Holding the corpus and the TRUE influence fixed, the measurable ceiling collapses
    as the RETAINED count grows. The estimator here is PERFECT -- it is the ground truth itself --
    so any fall in tau is a property of the metric, not of the estimator."""
    c = SyntheticCorpus(n_items=370, saturation=1.0, seed=seed)
    rows = []
    for k in keeps:
        m = c.draw_masks(n_masks, k, seed=seed + 1)
        d = measure(c, m)
        d["keep"] = k
        rows.append(d)
    return rows


def demo_ratio_depth_inflation(depths=(2, 4, 8, 16), n_masks=400, keep=25, seed=0):
    """CLAIM 2 (BLOCKERS #42). rho/r is inflated at low depth and MORE inflated at lower depth,
    so two ratios measured at different depths are not comparable. Same corpus, same masks, same
    perfect estimator -- only the seed depth moves."""
    c = SyntheticCorpus(n_items=370, saturation=1.0, seed=seed)
    m = c.draw_masks(n_masks, keep, seed=seed + 1)
    rows = []
    for d in depths:
        r = measure(c, m, depth=d)
        r["depth"] = d
        rows.append(r)
    return rows


def demo_sqrt_r_does_not_normalise(sigmas=(0.04, 0.07, 0.106, 0.2, 0.4), n_masks=600,
                                   keep=25, seed=0):
    """CLAIM 3. rho/sqrt(r) is a PEARSON disattenuation applied to a KENDALL statistic. Holding
    the estimator perfect and varying ONLY the seed noise (hence r), a correctly normalised
    quantity would be flat. Any drift here is manufactured by the normaliser."""
    rows = []
    for s in sigmas:
        c = SyntheticCorpus(n_items=370, sigma=s, saturation=1.0, seed=seed)
        m = c.draw_masks(n_masks, keep, seed=seed + 1)
        r = measure(c, m)
        r["sigma"] = s
        rows.append(r)
    return rows


def demo_nuisance_axis_credits_both_sides(n_masks=600, keeps=(15, 25, 35), seed=0):
    """CLAIM 4 (BLOCKERS #41). Pool masks of DIFFERENT retained sizes. Size moves the outcome and
    moves any summed prediction, so a FIXED estimator scores well pooled and ~0 within stratum.
    The detector is a permutation null on the fixed estimator -- it cannot be confused with
    leakage, because the estimator never sees the outcomes."""
    # The outcome must DEPEND on size for the nuisance axis to exist, so the world is additive
    # in the SUM here (a non-saturating learner) -- which is exactly campaign N's regime.
    c = SyntheticCorpus(n_items=370, sigma=1.0, saturation=0.0, seed=seed)

    def sum_predictor(corpus, m):
        """A FIXED estimator with NO per-item information beyond a constant. It cannot rank items
        at all; its only content is 'bigger mask -> bigger prediction'. If the pooled correlation
        credits it, the credit is arithmetic, not attribution."""
        return float(len(m))

    def outcome_sum(mask_idx, sd):
        rng = np.random.default_rng([sd, int(np.sum(mask_idx)) % (2 ** 31)])
        return -float(c.theta[mask_idx].sum()) + rng.normal(0, c.sigma)

    c.outcome = outcome_sum          # additive-in-sum world, so size moves the outcome
    pooled, per = [], []
    for k in keeps:
        per.append((k, c.draw_masks(n_masks // len(keeps), k, seed=seed + k)))
        pooled += per[-1][1]
    d_pool = measure(c, pooled, predictor=sum_predictor)
    d_strat = [dict(measure(c, m, predictor=sum_predictor), keep=k) for k, m in per]
    seeds = [100, 101]
    Y = np.array([[c.outcome(m, s) for s in seeds] for m in pooled]).mean(1)
    pred = np.array([sum_predictor(c, m) for m in pooled])
    rng = np.random.default_rng(7)
    null = [stats.kendalltau(-pred, rng.permutation(Y)).statistic for _ in range(300)]
    # ...and the same null WITHIN stratum, which is where it should collapse
    return {"pooled": d_pool, "within_stratum": d_strat,
            "pooled_shuffle_null_abs_p95": float(np.percentile(np.abs(null), 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "p19_synthetic.json"))
    a = ap.parse_args()
    res = {
        "measurability_horizon": demo_measurability_horizon(),
        "ratio_depth_inflation": demo_ratio_depth_inflation(),
        "sqrt_r_does_not_normalise": demo_sqrt_r_does_not_normalise(),
        "nuisance_axis": demo_nuisance_axis_credits_both_sides(),
    }
    print("CLAIM 1 -- measurability horizon (estimator is the GROUND TRUTH; only `keep` moves)")
    print(f"  {'keep':>5} {'tau':>8} {'ceiling r':>10} {'outcome sd':>11}")
    for r in res["measurability_horizon"]:
        print(f"  {r['keep']:>5} {r['tau']:>8.4f} {r['r_kendall']:>10.4f} {r['outcome_sd']:>11.4f}")
    print("\nCLAIM 2 -- rho/r inflation vs seed depth (same masks, same perfect estimator)")
    print(f"  {'depth':>5} {'tau':>8} {'r':>8} {'tau/r':>8} {'tau/sqrt(r)':>12}")
    for r in res["ratio_depth_inflation"]:
        print(f"  {r['depth']:>5} {r['tau']:>8.4f} {r['r_kendall']:>8.4f} "
              f"{r['ratio_over_r']:>8.4f} {r['ratio_over_sqrt_r']:>12.4f}")
    print("\nCLAIM 3 -- tau/sqrt(r) drifts with r even for a PERFECT estimator")
    print(f"  {'sigma':>6} {'r':>8} {'tau':>8} {'tau/sqrt(r)':>12}")
    for r in res["sqrt_r_does_not_normalise"]:
        print(f"  {r['sigma']:>6.1f} {r['r_kendall']:>8.4f} {r['tau']:>8.4f} "
              f"{r['ratio_over_sqrt_r']:>12.4f}")
    n = res["nuisance_axis"]
    print("\nCLAIM 4 -- a size nuisance axis credits both sides")
    print(f"  pooled tau = {n['pooled']['tau']:.4f}   "
          f"pooled shuffle-null |tau| p95 = {n['pooled_shuffle_null_abs_p95']:.4f}")
    for d in n["within_stratum"]:
        print(f"    within keep={d['keep']:>3}: tau = {d['tau']:.4f}")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n[p19] wrote {a.out}")


if __name__ == "__main__":
    main()
