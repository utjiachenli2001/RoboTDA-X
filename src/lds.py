"""LDS scoring, noise ceiling, and bootstrap CIs (spec §7).

Predicted mask score = sum of the attributor's per-demo influence over the demos in the mask.

Primary metric = CONDITIONAL LDS: Spearman(predicted, outcome) over ONLY the masks that
INCLUDE the target cluster (40 of 72 at cluster grain). The full-72 LDS is reported as
secondary and labelled "inflated by target inclusion" -- with the target excluded, success
collapses to the floor, so a predictor that only knows "is the target in the mask" would
score a high LDS while knowing nothing about which OUTSIDERS matter.

Outcome transform (spec §3): success p -> logit(clamp(p, 1/(2n), 1 - 1/(2n))), n = #episodes.

Noise ceiling (spec §7): from the 12 replicate masks x 4 seeds, take all 3 disjoint seed
pairings; for each, Spearman between the two pair-mean outcome vectors over the 12 masks;
average. A 2-vs-2 split correlation is exactly the reliability of a 2-SEED mean, which is the
granularity of the Stage-F outcomes -- so this average IS the ceiling for Stage F, and is
reported as `ceiling`. The Spearman-Brown extrapolation 2r/(1+r) (reliability of a 4-seed
mean) is also reported as `ceiling_sb` for completeness. LDS is judged against the ceiling,
never against 1.0.
"""
import numpy as np
from scipy import stats


def logit_success(p, n_episodes):
    """Spec §3: clamp at 1/(2n) then logit. n=30 -> clamp [1/60, 59/60]."""
    lo = 1.0 / (2 * n_episodes)
    p = np.clip(np.asarray(p, dtype=float), lo, 1 - lo)
    return np.log(p / (1 - p))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    if np.all(a[ok] == a[ok][0]) or np.all(b[ok] == b[ok][0]):
        return np.nan          # a constant vector has no rank information
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def bootstrap_spearman_ci(pred, out, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI by resampling MASKS (the unit of the LDS)."""
    rng = np.random.default_rng(seed)
    pred, out = np.asarray(pred, float), np.asarray(out, float)
    n = len(pred)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        r = spearman(pred[i], out[i])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def spearman_p_onesided(rho, n):
    """One-sided p-value (H1: rho > 0) via the t approximation, as used for the Bonferroni gate."""
    if not np.isfinite(rho) or n < 4:
        return np.nan
    r = min(max(rho, -0.999999), 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return float(stats.t.sf(t, df=n - 2))


def mask_pred_score(demo_scores, mask_demos):
    """Sum of per-demo attribution over the demos present in the mask."""
    return float(sum(demo_scores.get(d, 0.0) for d in mask_demos))


def conditional_lds(demo_scores, masks, outcomes, target_cluster, cluster_of_demo=None,
                    include_only_target_masks=True):
    """masks: [{'mask_id','demos','clusters'?}], outcomes: {mask_id -> float outcome}.

    Returns dict with rho, n, p_onesided, CI, and the paired vectors.
    """
    pred, out, ids = [], [], []
    for m in masks:
        if include_only_target_masks:
            if "clusters" in m:
                if target_cluster not in m["clusters"]:
                    continue
            elif cluster_of_demo is not None:
                if not any(cluster_of_demo[d] == target_cluster for d in m["demos"]):
                    continue
        if m["mask_id"] not in outcomes or not np.isfinite(outcomes[m["mask_id"]]):
            continue
        pred.append(mask_pred_score(demo_scores, m["demos"]))
        out.append(outcomes[m["mask_id"]])
        ids.append(m["mask_id"])
    rho = spearman(pred, out)
    ci = bootstrap_spearman_ci(pred, out) if len(pred) >= 4 else (np.nan, np.nan)
    return {"rho": rho, "n_masks": len(pred), "p_onesided": spearman_p_onesided(rho, len(pred)),
            "ci95": ci, "mask_ids": ids, "pred": pred, "outcome": out}


def noise_ceiling(outcome_by_mask_seed, seeds=(301, 302, 303, 304), n_boot=2000, seed=0):
    """outcome_by_mask_seed: {mask_id: {seed: outcome}} over the replicate masks.

    Returns {'ceiling', 'ceiling_sb', 'per_pairing', 'n_masks', 'ci95'}.
    """
    mids = [m for m, d in outcome_by_mask_seed.items()
            if all(s in d and np.isfinite(d[s]) for s in seeds)]
    if len(mids) < 4:
        return {"ceiling": np.nan, "ceiling_sb": np.nan, "per_pairing": [],
                "n_masks": len(mids), "ci95": (np.nan, np.nan)}
    s = list(seeds)
    pairings = [((s[0], s[1]), (s[2], s[3])),
                ((s[0], s[2]), (s[1], s[3])),
                ((s[0], s[3]), (s[1], s[2]))]
    rhos, vecs = [], []
    for (a, b) in pairings:
        va = np.array([np.mean([outcome_by_mask_seed[m][x] for x in a]) for m in mids])
        vb = np.array([np.mean([outcome_by_mask_seed[m][x] for x in b]) for m in mids])
        rhos.append(spearman(va, vb))
        vecs.append((va, vb))
    r = float(np.nanmean(rhos))
    sb = (2 * r / (1 + r)) if np.isfinite(r) and (1 + r) != 0 else np.nan
    # bootstrap over masks, averaging the 3 pairings within each resample
    rng = np.random.default_rng(seed)
    n = len(mids)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rr = [spearman(va[i], vb[i]) for va, vb in vecs]
        if np.all(np.isfinite(rr)):
            bs.append(np.mean(rr))
    ci = ((float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs
          else (np.nan, np.nan))
    return {"ceiling": r, "ceiling_sb": float(sb), "per_pairing": [float(x) for x in rhos],
            "n_masks": n, "ci95": ci}
