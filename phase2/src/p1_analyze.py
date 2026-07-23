"""P1 analysis: seed-ensembled (S=6) demo-grain LDS vs a seed-ensembled noise ceiling.

Phase 1's Gate 1 demanded rho > 0.50 against a 2-seed ceiling of 0.570 -- i.e. 88% of oracle.
P1 raises the CEILING (6 seeds per mask), not the bar, and re-asks the question.

PREREGISTERED CRITERION (preregistration_phase2.json, P1):
  on the held-out L2 outcome (PRIMARY), 6-seed mean, focal targets C1 and C5,
  attribution is USABLE at demo grain iff ANY attributor reaches
      rho >= 0.5 * (that target's measured 6-seed L2 ceiling)
  with one-sided p < 0.025 (Bonferroni-2 over the two focal targets).

Ceiling: all 10 distinct 3-vs-3 splits of seeds {401..406}; Spearman between the two half-mean
outcome vectors across the 24 masks; averaged (= reliability of a 3-seed mean); Spearman-Brown
corrected 3 -> 6 as r6 = 2*r3/(1+r3).
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import RESULTS, ROOT  # noqa: E402
from lds import (spearman, spearman_p_onesided, bootstrap_spearman_ci,  # noqa: E402
                 logit_success, mask_pred_score)

P2 = os.path.join(ROOT, "phase2")
SEEDS_OLD = [401, 402]
SEEDS_NEW = [403, 404, 405, 406]
ALL_SEEDS = SEEDS_OLD + SEEDS_NEW
FOCAL = ["C1", "C5"]
ATTRS = ["IF", "TRAK", "TracIn"]
ALPHA_BONF2 = 0.025


def collect_new(jobs):
    rows, missing = [], []
    for j in jobs:
        p = os.path.join(j["run_dir"], "outcomes.json")
        if not os.path.exists(p):
            missing.append(os.path.basename(j["run_dir"]))
            continue
        for c, v in json.load(open(p))["outcomes"].items():
            rows.append({"stage": "P1_stage_G6", "run": os.path.basename(j["run_dir"]),
                         "mask_id": j["mask_id"], "seed": j["seed"], "target": c,
                         **{k: v[k] for k in ("success_rate", "n_episodes", "plain_loss",
                                              "transport_loss", "interaction_loss")}})
    return pd.DataFrame(rows), missing


def ceiling_6seed(sub, col):
    """sub: rows for ONE target, indexed by (mask_id, seed). -> (r3_mean, r6_sb, per_split)."""
    piv = sub.pivot_table(index="mask_id", columns="seed", values=col)
    piv = piv[[s for s in ALL_SEEDS if s in piv.columns]].dropna()
    if piv.shape[1] < 6 or piv.shape[0] < 4:
        return np.nan, np.nan, [], piv.shape
    seeds = list(piv.columns)
    seen, vals = set(), []
    for half in itertools.combinations(seeds, 3):
        other = tuple(s for s in seeds if s not in half)
        key = frozenset([half, other])
        if key in seen:
            continue
        seen.add(key)
        r = spearman(piv[list(half)].mean(1).values, piv[list(other)].mean(1).values)
        if np.isfinite(r):
            vals.append(float(r))
    assert len(vals) == 10, f"expected 10 distinct 3v3 splits, got {len(vals)}"
    r3 = float(np.mean(vals))
    r6 = 2 * r3 / (1 + r3) if (1 + r3) != 0 else np.nan   # Spearman-Brown 3 -> 6
    return r3, r6, vals, piv.shape


def main():
    jobs = json.load(open(f"{P2}/results/p1_jobs.json"))
    new, missing = collect_new(jobs)
    old = pd.read_parquet(f"{RESULTS}/stage_G_outcomes.parquet")
    if missing:
        print(f"[P1] WARNING: {len(missing)} runs missing outcomes: {missing[:6]}")

    df = pd.concat([old, new], ignore_index=True)
    df["logit_success"] = logit_success(df.success_rate.values, df.n_episodes.values)
    df["neg_plain_loss"] = -df.plain_loss
    df["neg_transport_loss"] = -df.transport_loss
    df["neg_interaction_loss"] = -df.interaction_loss
    df.to_parquet(f"{P2}/results/stage_G6_outcomes.parquet", index=False)
    n_seed = df.groupby("mask_id").seed.nunique()
    print(f"[P1] merged outcomes: {len(df)} rows, {df.mask_id.nunique()} masks, "
          f"seeds/mask min={n_seed.min()} max={n_seed.max()}, "
          f"runs={df.run.nunique()} -> stage_G6_outcomes.parquet")

    man = json.load(open(f"{RESULTS}/demo_mask_manifest.json"))
    masks = man["masks"]
    inf = pd.read_parquet(f"{RESULTS}/influence_table.parquet")

    OUTCOMES = ["neg_plain_loss", "logit_success",
                "neg_transport_loss", "neg_interaction_loss"]
    res, rows = {}, []
    for tgt in sorted(df.target.unique()):
        sub = df[df.target == tgt]
        res[tgt] = {}
        for oc in OUTCOMES:
            r3, r6, splits, shape = ceiling_6seed(sub, oc)
            piv = sub.pivot_table(index="mask_id", columns="seed", values=oc)
            piv = piv[[s for s in ALL_SEEDS if s in piv.columns]].dropna()
            mean6 = piv.mean(1)                                   # 6-seed mean per mask
            # 1-seed and 2-seed reliabilities on the SAME masks, for the SB prediction check
            r1 = float(np.mean([spearman(piv[a].values, piv[b].values)
                                for a, b in itertools.combinations(piv.columns, 2)]))
            entry = {"ceiling_r3_splithalf": r3, "ceiling_6seed_SB": r6,
                     "mean_pairwise_1seed_r": r1,
                     "predicted_6seed_from_1seed_SB": 6 * r1 / (1 + 5 * r1) if (1 + 5 * r1) else np.nan,
                     "n_masks": int(len(mean6)), "n_seeds": int(piv.shape[1]),
                     "per_split": splits, "attributors": {}}
            for attr in ATTRS:
                sc = inf[(inf.attributor == attr) & (inf.functional == "plain")
                         & (inf.target == tgt)]
                sc = dict(zip(sc.demo_id, sc.score))
                pred = np.array([mask_pred_score(sc, m["demos"]) for m in masks])
                out = np.array([mean6.get(m["mask_id"], np.nan) for m in masks])
                ok = np.isfinite(out)
                rho = spearman(pred[ok], out[ok])
                p1 = spearman_p_onesided(rho, int(ok.sum()))
                lo, hi = bootstrap_spearman_ci(pred[ok], out[ok])
                ratio = rho / r6 if (np.isfinite(r6) and r6 > 0) else np.nan
                entry["attributors"][attr] = {
                    "rho": rho, "n": int(ok.sum()), "p_onesided": p1,
                    "ci95": [lo, hi], "ratio_to_ceiling": ratio,
                    "meets_half_ceiling": bool(np.isfinite(ratio) and ratio >= 0.5),
                    "p_lt_0.025": bool(np.isfinite(p1) and p1 < ALPHA_BONF2),
                    "PASS": bool(np.isfinite(ratio) and ratio >= 0.5
                                 and np.isfinite(p1) and p1 < ALPHA_BONF2),
                    "PASS_bonf6_robustness": bool(np.isfinite(ratio) and ratio >= 0.5
                                                  and np.isfinite(p1) and p1 < 0.05 / 6),
                }
                rows.append({"target": tgt, "outcome": oc, "attributor": attr, "rho": rho,
                             "ceiling_6seed": r6, "ratio": ratio, "p_onesided": p1,
                             "half_ceiling_bar": 0.5 * r6 if np.isfinite(r6) else np.nan,
                             "focal": tgt in FOCAL,
                             "PASS": entry["attributors"][attr]["PASS"]})
            res[tgt][oc] = entry

    tab = pd.DataFrame(rows)
    tab.to_csv(f"{P2}/results/p1_lds_table.csv", index=False)

    # -------------------------------------------------------------- preregistered verdict
    prim = "neg_plain_loss"
    verdict = {}
    for t in FOCAL:
        e = res[t][prim]
        best = max(ATTRS, key=lambda a: e["attributors"][a]["rho"])
        verdict[t] = {
            "ceiling_6seed": e["ceiling_6seed_SB"],
            "bar_half_ceiling": 0.5 * e["ceiling_6seed_SB"],
            "best_attributor": best,
            "best_rho": e["attributors"][best]["rho"],
            "best_ratio": e["attributors"][best]["ratio_to_ceiling"],
            "best_p_onesided": e["attributors"][best]["p_onesided"],
            "any_attributor_PASS": any(e["attributors"][a]["PASS"] for a in ATTRS),
            "per_attributor": {a: e["attributors"][a] for a in ATTRS},
        }
    overall = any(verdict[t]["any_attributor_PASS"] for t in FOCAL)

    out = {
        "stage": "P1",
        "criterion": ("held-out L2 (neg_plain_loss), 6-seed mean, focal C1/C5: any attributor "
                      "rho >= 0.5 * measured 6-seed ceiling AND one-sided p < 0.025 (Bonferroni-2)"),
        "n_new_retrains": len(jobs), "n_missing": len(missing), "missing": missing,
        "seeds": ALL_SEEDS, "S": 6, "n_masks": len(masks),
        "PRIMARY_OUTCOME": prim,
        "focal_verdict": verdict,
        "PASS": bool(overall),
        "interpretation": ("PASS => Phase-1 Gate-1 failure was the noise floor. "
                           "FAIL at a high ceiling => attribution is genuinely unfaithful at "
                           "demo grain in this regime."),
        "all_targets": res,
    }
    json.dump(out, open(f"{P2}/results/p1_demo_grain.json", "w"), indent=1, default=float)

    print("\n" + "=" * 92)
    print("P1 -- SEED-ENSEMBLED DEMO-GRAIN LDS (primary outcome: held-out L2, 6-seed mean)")
    print("=" * 92)
    print(f"{'target':7s} {'ceil(1s)':>9s} {'pred6(SB)':>10s} {'ceil6(meas)':>12s} "
          f"{'bar=.5c':>8s} {'best attr':>10s} {'rho':>7s} {'ratio':>6s} {'p1':>7s}  verdict")
    for t in sorted(res):
        e = res[t][prim]
        best = max(ATTRS, key=lambda a: e["attributors"][a]["rho"])
        b = e["attributors"][best]
        mark = "FOCAL" if t in FOCAL else "     "
        v = ("PASS" if b["PASS"] else "fail") if t in FOCAL else "-"
        print(f"{t:7s} {e['mean_pairwise_1seed_r']:9.3f} {e['predicted_6seed_from_1seed_SB']:10.3f} "
              f"{e['ceiling_6seed_SB']:12.3f} {0.5*e['ceiling_6seed_SB']:8.3f} {best:>10s} "
              f"{b['rho']:+7.3f} {b['ratio_to_ceiling']:6.2f} {b['p_onesided']:7.4f}  {mark} {v}")
    print("-" * 92)
    for t in FOCAL:
        e = res[t][prim]
        for a in ATTRS:
            x = e["attributors"][a]
            print(f"  {t} {a:7s} rho={x['rho']:+.3f} ratio={x['ratio_to_ceiling']:+.2f} "
                  f"p1={x['p_onesided']:.4f} ci95=[{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}] "
                  f"{'PASS' if x['PASS'] else 'fail'}")
    print("=" * 92)
    print(f"PREREGISTERED VERDICT: {'PASS' if overall else 'FAIL'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
