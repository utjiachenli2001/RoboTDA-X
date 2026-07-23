"""RQ2 moderator regression (spec §7).

Regress each target's INSIDER-ADVANTAGE AUC (probability a random insider outranks a random
outsider, under the best-performing attributor) on trajectory-space similarity + within-target
redundancy. Trajectory-space only -- NO image/DINO features anywhere.

Quantity's moderator evidence comes from Stage C (within-target), not from this regression.

n = 9 targets, so this is a descriptive fit, not a powered test; it is reported as such.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RESULTS
import dataset
import moderators as MO


def main():
    infl = pd.read_parquet(os.path.join(RESULTS, "influence_table.parquet"))
    best = json.load(open(os.path.join(RESULTS, "best_attributor_by_target.json")))
    M, cl = MO.cluster_matrices()
    _, by_c = dataset.train_pool()
    red = MO.within_target_redundancy()

    rows = []
    for j, t in enumerate(cl):
        attr = best[t]
        sub = infl[(infl.attributor == attr) & (infl.functional == "plain") & (infl.target == t)]
        ins = sub[sub.cluster_of_demo == t].score.values
        out = sub[sub.cluster_of_demo != t].score.values
        auc = float((out[None, :] < ins[:, None]).mean()
                    + 0.5 * (out[None, :] == ins[:, None]).mean())
        # similarity of the OUTSIDERS to the target: mean over the 8 other clusters
        others = [i for i in range(9) if i != j]
        rows.append({
            "target": t, "attributor": attr, "insider_advantage_auc": auc,
            "mean_outsider_dtw": float(np.mean([M["dtw"][i, j] for i in others])),
            "mean_outsider_mmd": float(np.mean([M["mmd"][i, j] for i in others])),
            "mean_outsider_bddl": float(np.mean([M["bddl"][i, j] for i in others])),
            "within_target_redundancy_dtw": float(red[t]),
        })
    df = pd.DataFrame(rows)

    from scipy import stats
    preds = ["mean_outsider_dtw", "mean_outsider_mmd", "mean_outsider_bddl",
             "within_target_redundancy_dtw"]
    out = {"n_targets": len(df), "note": "n=9 -> descriptive, not a powered test",
           "per_target": rows, "univariate": {}}
    print("=== RQ2 moderator regression: insider-advantage AUC ~ trajectory-space similarity ===")
    print(f"(n = {len(df)} targets; descriptive only. Quantity's evidence is Stage C, not this.)\n")
    print(f"{'predictor':>30} {'Pearson r':>10} {'p':>8} {'Spearman':>10} {'slope':>10}")
    for p in preds:
        r, pv = stats.pearsonr(df[p], df.insider_advantage_auc)
        sr = stats.spearmanr(df[p], df.insider_advantage_auc).statistic
        sl = np.polyfit(df[p], df.insider_advantage_auc, 1)[0]
        out["univariate"][p] = {"pearson_r": float(r), "p": float(pv),
                                "spearman": float(sr), "slope": float(sl)}
        print(f"{p:>30} {r:>+10.3f} {pv:>8.3f} {sr:>+10.3f} {sl:>+10.3f}")

    # multivariate OLS (DTW + redundancy), the spec's model
    X = np.column_stack([np.ones(len(df)), df.mean_outsider_dtw, df.within_target_redundancy_dtw])
    y = df.insider_advantage_auc.values
    beta, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out["multivariate_auc_on_dtw_and_redundancy"] = {
        "intercept": float(beta[0]), "beta_mean_outsider_dtw": float(beta[1]),
        "beta_within_target_redundancy": float(beta[2]), "r2": float(r2), "n": len(df)}
    print(f"\nOLS: AUC ~ 1 + mean_outsider_DTW + within_target_redundancy")
    print(f"  intercept={beta[0]:+.3f}  beta_DTW={beta[1]:+.3f}  beta_redundancy={beta[2]:+.3f}"
          f"  R2={r2:.3f}  (n={len(df)})")

    df.to_csv(os.path.join(RESULTS, "rq2_moderators.csv"), index=False)
    json.dump(out, open(os.path.join(RESULTS, "rq2_moderators.json"), "w"), indent=1)
    print(f"\nwrote results/rq2_moderators.csv / .json")


if __name__ == "__main__":
    main()
