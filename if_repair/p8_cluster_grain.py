"""PASS 8 W1 -- CLUSTER-GRAIN out-of-sample scoring of every frozen config. ZERO GPU.

The observation this rests on is the same one that made pass 7's central result free, applied to
the one draw in the repo nobody has ever attributed against.

`runs/stage_F` holds 168 completed retrains: 72 balanced cluster masks (5 of 9 clusters = 75
demos) x seeds 301/302, plus 12 noise-ceiling masks x seeds 303/304. `src/stage_efg.py` built
them as ATTRIBUTION-AGNOSTIC ground truth, and pass 7's HANDOFF (open thread #3) records that
they were never used for attribution. Every estimator config frozen in `p7_pooled_oos.CONFIGS` is
therefore out-of-sample on all 72 masks by construction -- there is no discovery draw to remove
and no winner's curse to discount, which is a cleaner provenance than anything pass 7 had.

WHY THE GRAIN CHANGES THE QUESTION. BLOCKERS #33/#35 put the demo grain below the noise floor:
swapping one demo of 68 moves the outcome by 0.44 seed-noise sd, and the two leading estimators
rank-correlate 0.547. A cluster is 15 demos, so a cluster mask moves 15 units at once. If
influence functions are weak because the unit is too small, cluster grain lifts them; if they are
weak because influence functions are weak, it does not. Pass 7 could not tell those apart.

THE CONDITIONAL IS THE HONEST NUMBER. `src/analysis.py` already names the full-72 correlation
`lds_full72_INFLATED`, and the reason is structural: a mask that drops the target cluster entirely
has its outcome dominated by that removal rather than by the ordering of the remaining demos.
Conditioning on target-in-mask leaves exactly 40 of 72 masks for every target (the manifest's
balance constraint, asserted below), against 8 free predictors -- far better conditioned than demo
grain's 135 demos against 24-192 masks. Both are written out; only the conditional one is
reported as the estimate.

The outcome functional is `plain`, matching `p7_pooled_oos.raw_outcomes`, so cluster-grain and
demo-grain numbers in this repo answer the same question about the same quantity.

NO BAR IS ATTACHED HERE. This module reports; `p8_prereg.md` chooses the hypothesis afterwards,
and `confirm_nseries.py` tests it on a fresh draw. Attaching a bar to a scan of every config on a
single draw is how passes 4-6 manufactured a +0.34 that was really +0.06 (BLOCKERS #28).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import p7_pooled_oos as P7  # noqa: E402

P7.D.add_repo_paths()
import lds  # noqa: E402
import masks as MK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPO = os.path.dirname(HERE)

F_SEEDS = [301, 302]
N_EPISODES = 30
N_BOOT = P7.N_BOOT
BOOT_SEED = P7.BOOT_SEED

# `plain` matches p7_pooled_oos.raw_outcomes, so the two grains measure the same functional.
PRIMARY_OUTCOME = "neg_plain_loss"
OUTCOMES = ["neg_plain_loss", "logit_success", "neg_transport_loss", "neg_interaction_loss"]

STATS = {
    "kendall_tau_b": lambda p, o: float(stats.kendalltau(p, o, variant="b").statistic),
    "spearman": lambda p, o: lds.spearman(np.asarray(p, float), np.asarray(o, float)),
}


def _p_onesided(name, pred, obs):
    if name == "spearman":
        return float(lds.spearman_p_onesided(STATS["spearman"](pred, obs), len(obs)))
    return float(stats.kendalltau(pred, obs, alternative="greater").pvalue)


# ------------------------------------------------------------------ Stage F outcomes
def outcome_value(row, key):
    if key == "logit_success":
        return float(lds.logit_success(row["success_rate"], N_EPISODES))
    return -float(row[key.replace("neg_", "")])


def stage_f_table(path=None):
    """The 168-retrain outcome table, oriented so higher = better (src/analysis.py convention)."""
    path = path or os.path.join(REPO, "results", "stage_F_outcomes.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- regenerate with `python -m src.stage_efg --stage F --collect_only` "
            "(artifact-only collection over the existing run dirs, no GPU)")
    df = pd.read_parquet(path)
    for k in OUTCOMES:
        df[k] = df.apply(lambda r: outcome_value(r, k), axis=1)
    return df


def seedmean(df, target, key, seeds=F_SEEDS):
    """{mask_id: mean outcome over `seeds`} for one target."""
    sub = df[(df.target == target) & (df.seed.isin(seeds))]
    return {m: float(v) for m, v in sub.groupby("mask_id")[key].mean().items()}


def cluster_masks():
    """The 72 Stage F masks, each carrying its demo list and its 5 included clusters."""
    man = MK.cluster_mask_manifest()
    return [{"mask_id": m["mask_id"], "demos": m["demos"], "clusters": set(m["clusters"])}
            for m in man["masks"]]


# ------------------------------------------------------------------ the lift
def paired_rows(scores, base_scores, masks, obs, target, statname, conditional=True):
    """Cluster-grain correlation of one config and its baseline on the SAME masks.

    The lift to cluster grain is `P7.mask_pred`: a cluster mask is a demo list like any other, so
    the prediction is the sum of per-demo scores over the 75 demos the mask keeps. No estimator is
    re-implemented -- that is the gap BLOCKERS #14 came through.
    """
    use = [m for m in masks if (not conditional) or (target in m["clusters"])]
    use = [m for m in use if m["mask_id"] in obs]
    if len(use) < 5:
        return None
    o = np.array([obs[m["mask_id"]] for m in use], float)
    p = P7.mask_pred(scores, use)
    b = P7.mask_pred(base_scores, use)
    ok = np.isfinite(o) & np.isfinite(p) & np.isfinite(b)
    o, p, b, use = o[ok], p[ok], b[ok], [m for m, k in zip(use, ok) if k]
    if len(o) < 5 or np.std(p) == 0 or np.std(b) == 0:
        return None
    fn = STATS[statname]
    r_cfg, r_base = fn(p, o), fn(b, o)

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(o), size=(N_BOOT, len(o)))
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = idx[i]
        if np.std(p[j]) == 0 or np.std(b[j]) == 0 or np.std(o[j]) == 0:
            deltas[i] = np.nan
            continue
        deltas[i] = fn(p[j], o[j]) - fn(b[j], o[j])
    d = deltas[np.isfinite(deltas)]
    return {
        "n_masks": len(o), "conditional": conditional,
        "rho": r_cfg, "rho_baseline": r_base, "delta": r_cfg - r_base,
        "delta_ci_lo": float(np.percentile(d, 2.5)) if len(d) else np.nan,
        "delta_ci_hi": float(np.percentile(d, 97.5)) if len(d) else np.nan,
        "delta_p_boot_onesided": float(np.mean(d <= 0)) if len(d) else np.nan,
        "p_onesided_rho": _p_onesided(statname, p, o),
    }


def scan(outcomes=(PRIMARY_OUTCOME,), statnames=("kendall_tau_b", "spearman")):
    """Every frozen config x its own target x outcome x statistic, on all 72 Stage F masks."""
    df = stage_f_table()
    masks = cluster_masks()
    _assert_balance(masks)
    rows = []
    for name, cfg in P7.CONFIGS.items():
        target = cfg["target"]
        sc = cfg["scores"]()
        base = P7._graddot(cfg["baseline"])[target]
        for key in outcomes:
            obs = seedmean(df, target, key)
            for statname in statnames:
                for cond in (True, False):
                    r = paired_rows(sc, base, masks, obs, target, statname, conditional=cond)
                    if r is None:
                        continue
                    rows.append({
                        "config": name, "label": cfg["label"], "frozen": cfg["frozen"],
                        "target": target, "outcome": key, "statistic": statname,
                        "grain": "cluster", "draw": "stage_F",
                        "provenance": "OOS by construction (Stage F never used for attribution)",
                        **r})
    return pd.DataFrame(rows)


def _assert_balance(masks):
    """Every cluster sits in exactly 40 of the 72 masks -- fails loudly if the manifest changed."""
    from collections import Counter
    c = Counter(cl for m in masks for cl in m["clusters"])
    assert len(masks) == 72, f"expected 72 Stage F masks, got {len(masks)}"
    bad = {k: v for k, v in c.items() if v != 40}
    assert not bad, f"cluster inclusion counts off the 72*5/9=40 design: {bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_outcomes", action="store_true",
                    help="scan all four outcome keys, not just the primary `plain` functional")
    ap.add_argument("--out", default=os.path.join(RESULTS, "p8_stageF_oos.csv"))
    a = ap.parse_args()

    outs = tuple(OUTCOMES) if a.all_outcomes else (PRIMARY_OUTCOME,)
    df = scan(outcomes=outs)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"[p8/W1] cluster-grain OOS scan -> {a.out}  ({len(df)} rows)")
    show = df[(df.conditional) & (df.outcome == PRIMARY_OUTCOME)
              & (df.statistic == "kendall_tau_b")]
    if len(show):
        print("\nCONDITIONAL (target-in-mask), primary outcome `plain`, Kendall tau_b:")
        print(f"  {'config':16s} {'tgt':4s} {'n':>3s} {'rho':>8s} {'base':>8s} {'delta':>8s}  95% CI")
        for _, r in show.iterrows():
            print(f"  {r.config:16s} {r.target:4s} {r.n_masks:3d} {r.rho:8.3f} "
                  f"{r.rho_baseline:8.3f} {r.delta:+8.3f}  "
                  f"[{r.delta_ci_lo:+.3f}, {r.delta_ci_hi:+.3f}]")
    print("\nNo bar is attached here by design -- p8_prereg.md chooses the hypothesis, "
          "confirm_nseries.py tests it on a fresh draw.")


if __name__ == "__main__":
    main()
