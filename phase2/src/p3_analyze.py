"""P3 analysis: the regime boundary -- does ground truth AND attribution get more reliable with Q?

Per Q in {15, 50, 150}:
  (a) 4-seed split-half noise ceiling over the 12 masks -- all 3 disjoint 2-vs-2 seed pairings,
      Spearman between the pair-mean outcome vectors, averaged (= reliability of a 2-seed mean);
      Spearman-Brown 2->4 also reported. Computed for held-out L2 AND logit-success.
  (b) LDS: Spearman(predicted mask score, 4-seed-mean outcome) over the 12 masks, per attributor,
      with attribution computed from that Q's OWN 5 full-Q models.
  (c) ratio = best LDS / ceiling.

PREREGISTERED TESTS (one-sided alpha = 0.05 each), Page's trend test for a monotone INCREASE in Q:
  (i)  the SUCCESS-outcome ceiling,   blocks = the 3 disjoint 2-vs-2 seed pairings
  (ii) the best LDS/ceiling ratio,    blocks = the 3 attributors
Both are k=3 treatments x n=3 blocks. The p-value is EXACT: the null distribution of Page's L is
enumerated over all (3!)^3 = 216 within-block rank assignments, not approximated.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT  # noqa: E402
from lds import (spearman, spearman_p_onesided, bootstrap_spearman_ci,  # noqa: E402
                 logit_success, mask_pred_score)

P2 = os.path.join(ROOT, "phase2")
QS = [15, 50, 150]
SEEDS = [601, 602, 603, 604]
ATTRS = ["IF", "TRAK", "TracIn"]
PAIRINGS = [((601, 602), (603, 604)),      # the 3 disjoint 2-vs-2 splits of 4 seeds
            ((601, 603), (602, 604)),
            ((601, 604), (602, 603))]


def page_L_exact(mat):
    """mat[block][treatment], treatments in the hypothesised INCREASING order.

    Page's L = sum_j (j+1) * R_j, R_j = sum over blocks of the within-block rank of treatment j
    (rank 1 = smallest). Exact one-sided p = P(L >= L_obs) under H0 (all within-block orderings
    equally likely), enumerated over (k!)^n assignments.
    """
    mat = np.asarray(mat, float)
    n, k = mat.shape
    if not np.isfinite(mat).all():
        return {"L": np.nan, "p_onesided_exact": np.nan, "n_blocks": n, "k": k}
    from scipy.stats import rankdata
    ranks = np.stack([rankdata(row) for row in mat])            # (n,k), 1=smallest
    w = np.arange(1, k + 1)
    L_obs = float((w * ranks.sum(0)).sum())

    perms = list(itertools.permutations(range(1, k + 1)))
    tot, ge = 0, 0
    for combo in itertools.product(perms, repeat=n):
        R = np.sum(combo, axis=0)
        L = float((w * R).sum())
        tot += 1
        if L >= L_obs - 1e-9:
            ge += 1
    return {"L": L_obs, "p_onesided_exact": ge / tot, "n_perms": tot,
            "n_blocks": n, "k": k, "ranks": ranks.tolist()}


def collect():
    jobs = json.load(open(f"{P2}/results/p3_jobs.json"))
    rows, missing = [], []
    for j in jobs:
        rd = j["run_dir"]
        hl = os.path.join(rd, "heldout_losses.json")
        ce = os.path.join(rd, "cluster_eval.json")
        if not (os.path.exists(hl) and os.path.exists(ce)):
            missing.append(os.path.basename(rd))
            continue
        L = json.load(open(hl))["losses"]["C1"]
        c = json.load(open(ce))
        rows.append({"Q": j["Q"], "mask_id": j["mask_id"], "kind": j["kind"], "seed": j["seed"],
                     "run": os.path.basename(rd),
                     "plain_loss": L["plain_loss"],
                     "transport_loss": L["transport_loss"],
                     "interaction_loss": L["interaction_loss"],
                     "success_rate": c["cluster_success"], "n_episodes": c["n_episodes"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["neg_plain_loss"] = -df.plain_loss
        df["logit_success"] = logit_success(df.success_rate.values, df.n_episodes.values)
    return df, missing


def ceiling_4seed(sub, col):
    """3 disjoint 2v2 pairings over the 12 masks -> reliability of a 2-seed mean; SB 2->4."""
    piv = sub.pivot_table(index="mask_id", columns="seed", values=col)
    piv = piv[[s for s in SEEDS if s in piv.columns]].dropna()
    if piv.shape[1] < 4 or piv.shape[0] < 4:
        return np.nan, np.nan, [], piv
    per = []
    for (a, b) in PAIRINGS:
        per.append(float(spearman(piv[list(a)].mean(1).values, piv[list(b)].mean(1).values)))
    r2 = float(np.mean(per))
    r4 = 2 * r2 / (1 + r2) if (1 + r2) != 0 else np.nan
    return r2, r4, per, piv


def main():
    df, missing = collect()
    if missing:
        print(f"[P3] WARNING: {len(missing)} runs missing outcomes: {missing[:6]}")
    if df.empty:
        print("[P3] no outcomes yet -- did not run")
        return
    df.to_parquet(f"{P2}/results/p3_outcomes.parquet", index=False)

    man = json.load(open(f"{P2}/results/p3_mask_manifest.json"))
    OUT = ["neg_plain_loss", "logit_success"]
    res, rows = {}, []
    ceil_by_pairing = {oc: {} for oc in OUT}     # for Page test (i): per-pairing ceilings

    for q in QS:
        sub = df[(df.Q == q) & (df.kind == "mask")]
        masks = man["Q"][str(q)]["masks"]
        res[str(q)] = {"n_mask_runs": int(len(sub)), "n_masks": int(sub.mask_id.nunique()),
                       "n_seeds": int(sub.seed.nunique()), "outcomes": {}}
        infp = f"{P2}/results/p3_influence_Q{q}.parquet"
        inf = pd.read_parquet(infp) if os.path.exists(infp) else None

        for oc in OUT:
            r2, r4, per, piv = ceiling_4seed(sub, oc)
            ceil_by_pairing[oc][q] = per
            mean4 = piv.mean(1) if len(piv) else pd.Series(dtype=float)
            e = {"ceiling_2seed_splithalf": r2, "ceiling_4seed_SB": r4,
                 "per_pairing": per, "n_masks": int(len(mean4)), "attributors": {}}
            for attr in ATTRS:
                if inf is None:
                    e["attributors"][attr] = "attribution did not run"
                    continue
                s = inf[(inf.attributor == attr) & (inf.functional == "plain")]
                sc = dict(zip(s.demo_id, s.score))
                pred = np.array([mask_pred_score(sc, m["demos"]) for m in masks])
                out = np.array([mean4.get(m["mask_id"], np.nan) for m in masks])
                ok = np.isfinite(out)
                rho = spearman(pred[ok], out[ok])
                p1 = spearman_p_onesided(rho, int(ok.sum()))
                lo, hi = bootstrap_spearman_ci(pred[ok], out[ok])
                ratio = rho / r4 if (np.isfinite(r4) and r4 > 0) else np.nan
                e["attributors"][attr] = {"rho": rho, "n": int(ok.sum()), "p_onesided": p1,
                                          "ci95": [lo, hi], "ratio_to_ceiling": ratio}
                rows.append({"Q": q, "outcome": oc, "attributor": attr, "rho": rho,
                             "ceiling_4seed": r4, "ratio": ratio, "p_onesided": p1})
            res[str(q)]["outcomes"][oc] = e

    tab = pd.DataFrame(rows)
    if len(tab):
        tab.to_csv(f"{P2}/results/p3_lds_table.csv", index=False)

    # ---------------------------------------------------- PREREGISTERED TEST (i): success ceiling
    m_i = [[ceil_by_pairing["logit_success"][q][b] if len(ceil_by_pairing["logit_success"][q]) == 3
            else np.nan for q in QS] for b in range(3)]
    page_i = page_L_exact(m_i)

    # -------------------------------------------- PREREGISTERED TEST (ii): best LDS/ceiling ratio
    m_ii = []
    for attr in ATTRS:
        row = []
        for q in QS:
            a = res[str(q)]["outcomes"]["neg_plain_loss"]["attributors"].get(attr)
            row.append(a["ratio_to_ceiling"] if isinstance(a, dict) else np.nan)
        m_ii.append(row)
    page_ii = page_L_exact(m_ii)

    # ---- DESCRIPTIVE (not preregistered): the same Page tests in the DECREASING direction.
    # Both preregistered one-sided tests are for an INCREASE. If the truth is a DECREASE, those
    # tests correctly FAIL, but reporting only "no increase" would hide the direction. Reversing
    # the treatment order tests monotone DECREASE with the identical machinery.
    page_i_dec = page_L_exact([row[::-1] for row in m_i])
    page_ii_dec = page_L_exact([row[::-1] for row in m_ii])

    # ---- DESCRIPTIVE: why the success ceiling behaves as it does -- signal vs seed noise.
    # ceiling is a RATIO problem: between-mask spread (the data-composition signal LDS must rank)
    # against within-mask across-seed spread (the noise). Reported per Q per outcome.
    snr = {}
    for q in QS:
        sub = df[(df.Q == q) & (df.kind == "mask")]
        snr[str(q)] = {}
        for col, name in [("success_rate", "success"), ("plain_loss", "L2")]:
            piv = sub.pivot_table(index="mask_id", columns="seed", values=col)
            piv = piv[[s for s in SEEDS if s in piv.columns]].dropna()
            between = float(piv.mean(1).std())
            within = float(piv.std(1).mean())
            snr[str(q)][name] = {
                "between_mask_sd": between, "within_mask_seed_sd": within,
                "signal_to_noise": between / within if within > 0 else np.nan,
                "mask_mean_min": float(piv.mean(1).min()), "mask_mean_max": float(piv.mean(1).max()),
            }

    # secondary: Spearman of the statistic vs Q
    def sp_vs_q(vals):
        ok = np.isfinite(vals)
        return spearman(np.array(QS)[ok], np.array(vals)[ok]) if ok.sum() >= 3 else np.nan

    ceil_succ = [res[str(q)]["outcomes"]["logit_success"]["ceiling_4seed_SB"] for q in QS]
    ceil_l2 = [res[str(q)]["outcomes"]["neg_plain_loss"]["ceiling_4seed_SB"] for q in QS]
    best_ratio = [np.nanmax([res[str(q)]["outcomes"]["neg_plain_loss"]["attributors"][a]["ratio_to_ceiling"]
                             for a in ATTRS
                             if isinstance(res[str(q)]["outcomes"]["neg_plain_loss"]["attributors"][a], dict)]
                            or [np.nan]) for q in QS]

    out = {
        "stage": "P3", "Q_values": QS, "seeds": SEEDS,
        "n_missing_runs": len(missing), "missing": missing,
        "per_Q": res,
        "PREREGISTERED_TEST_i_success_ceiling_rises_with_Q": {
            "test": "Page's trend test, blocks = 3 disjoint 2v2 seed pairings, k=3 Q levels, exact p",
            "matrix_blocks_x_Q": m_i, **page_i,
            "PASS": bool(np.isfinite(page_i["p_onesided_exact"])
                         and page_i["p_onesided_exact"] < 0.05),
        },
        "PREREGISTERED_TEST_ii_LDS_ceiling_ratio_rises_with_Q": {
            "test": "Page's trend test, blocks = 3 attributors, k=3 Q levels, exact p",
            "matrix_blocks_x_Q": m_ii, **page_ii,
            "PASS": bool(np.isfinite(page_ii["p_onesided_exact"])
                         and page_ii["p_onesided_exact"] < 0.05),
        },
        "DESCRIPTIVE_reverse_direction_tests": {
            "note": ("NOT preregistered. The preregistered tests are one-sided for an INCREASE; "
                     "these test a monotone DECREASE with the same exact machinery, so the "
                     "direction of the effect is reported rather than merely 'no increase'."),
            "i_success_ceiling_DECREASES_with_Q": {**page_i_dec,
                                                   "significant_at_05": bool(page_i_dec["p_onesided_exact"] < 0.05)},
            "ii_LDS_ratio_DECREASES_with_Q": {**page_ii_dec,
                                              "significant_at_05": bool(page_ii_dec["p_onesided_exact"] < 0.05)},
        },
        "DESCRIPTIVE_signal_vs_seed_noise": {
            "note": ("the mechanism: a ceiling is a signal/noise ratio. between_mask_sd is the "
                     "data-composition signal any attributor must rank; within_mask_seed_sd is "
                     "the seed noise at fixed data. As Q grows the policy saturates, so removing "
                     "40% of demos moves it less -- the SIGNAL shrinks while the noise does not."),
            "per_Q": snr,
        },
        "secondary_spearman_vs_Q": {
            "success_ceiling": sp_vs_q(np.array(ceil_succ)),
            "L2_ceiling": sp_vs_q(np.array(ceil_l2)),
            "best_LDS_ceiling_ratio": sp_vs_q(np.array(best_ratio)),
        },
        "summary": {"Q": QS, "ceiling_success_4seed": ceil_succ,
                    "ceiling_L2_4seed": ceil_l2, "best_LDS_ratio_L2": best_ratio},
    }
    json.dump(out, open(f"{P2}/results/p3_regime_boundary.json", "w"), indent=1, default=float)

    print("=" * 90)
    print("P3 -- REGIME BOUNDARY")
    print("=" * 90)
    print(f"{'Q':>5s} {'ceil L2(4s)':>12s} {'ceil succ(4s)':>14s} {'best LDS(L2)':>13s} {'ratio':>7s}")
    for i, q in enumerate(QS):
        e = res[str(q)]["outcomes"]["neg_plain_loss"]
        best = np.nanmax([e["attributors"][a]["rho"] for a in ATTRS
                          if isinstance(e["attributors"][a], dict)] or [np.nan])
        print(f"{q:5d} {ceil_l2[i]:12.3f} {ceil_succ[i]:14.3f} {best:13.3f} {best_ratio[i]:7.2f}")
    print("-" * 90)
    print(f"(i)  success ceiling rises with Q:  L={page_i['L']}, exact one-sided p="
          f"{page_i['p_onesided_exact']:.4f}  -> "
          f"{'PASS' if out['PREREGISTERED_TEST_i_success_ceiling_rises_with_Q']['PASS'] else 'FAIL'}")
    print(f"(ii) LDS/ceiling ratio rises with Q: L={page_ii['L']}, exact one-sided p="
          f"{page_ii['p_onesided_exact']:.4f}  -> "
          f"{'PASS' if out['PREREGISTERED_TEST_ii_LDS_ceiling_ratio_rises_with_Q']['PASS'] else 'FAIL'}")
    print("-" * 90)
    print("DESCRIPTIVE (not preregistered) -- the DECREASING direction:")
    print(f"  (i)  success ceiling DECREASES with Q: exact p={page_i_dec['p_onesided_exact']:.4f}")
    print(f"  (ii) LDS/ceiling ratio DECREASES with Q: exact p={page_ii_dec['p_onesided_exact']:.4f}")
    print("-" * 90)
    print("MECHANISM -- ceiling is signal/noise; the SIGNAL shrinks as the policy saturates:")
    print(f"  {'Q':>5s} {'succ between-mask sd':>21s} {'succ seed sd':>13s} {'S/N':>6s} "
          f"{'succ range':>11s}")
    for q in QS:
        s = snr[str(q)]["success"]
        print(f"  {q:5d} {s['between_mask_sd']:21.4f} {s['within_mask_seed_sd']:13.4f} "
              f"{s['signal_to_noise']:6.2f} "
              f"{s['mask_mean_min']:.3f}-{s['mask_mean_max']:.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
