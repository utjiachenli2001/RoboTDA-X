"""P10 at S = 6 -- the remedy for the collapsed S=4 ceiling (PHASE3_DEFECT.md).

WHAT CHANGES vs p10_analyze.py: ONLY the number of ground-truth seeds, 4 -> 6 (601..606), and
therefore the ceiling ESTIMATOR: the same 3v3 / 10-distinct-split estimator Phase-2 P1 used on the
BC arm, instead of the 3-pairing 2v2 estimator that produced negative ceilings at S=4.

WHAT DOES NOT CHANGE: the criterion. Focal C1/C5; PRIMARY outcome neg_plain_loss (held-out L2 on
the executed DDIM action, matched to the BC arm); any preregistered attributor ({TracIn, TRAK})
must reach rho >= 0.5 x the MEASURED ceiling with one-sided p < 0.025 (Bonferroni-2).

This makes the two policy classes strictly comparable: same 24 masks, same S=6, same estimator,
same outcome, same criterion.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, RESULTS, P2_RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci  # noqa: E402

SEEDS = [601, 602, 603, 604, 605, 606]
FOCAL = ["C1", "C5"]
PREREG_ATTRS = ["TracIn", "TRAK"]
DESC_ATTRS = ["IF"]
ALPHA = 0.025
OUTCOMES = ["neg_plain_loss", "neg_denoise_loss", "logit_success"]


def collect():
    jobs = json.load(open(os.path.join(P3_RESULTS, "p10b_jobs.json")))
    jobs += json.load(open(os.path.join(P3_RESULTS, "p10b_jobs_seeds56.json")))
    rows, missing = [], []
    for j in jobs:
        oc = L.read_outcomes(j["run_dir"], required=False)      # MARKER-GATED
        if oc is None:
            missing.append(os.path.basename(j["run_dir"]))
            continue
        for c, v in oc.items():
            rows.append({"run": os.path.basename(j["run_dir"]), "mask_id": j["mask_id"],
                         "seed": j["seed"], "target": c,
                         "success_rate": v["success_rate"], "n_episodes": v["n_episodes"],
                         "plain_loss": v["plain_loss"], "denoise_loss": v["denoise_loss"]})
    df = pd.DataFrame(rows)
    df["logit_success"] = L.logit_success_rowwise(df.success_rate.values, df.n_episodes.values)
    df["neg_plain_loss"] = -df.plain_loss
    df["neg_denoise_loss"] = -df.denoise_loss
    return df, missing


def ceiling_6seed(sub, col):
    """The Phase-2 P1 estimator: all 10 distinct 3v3 splits; SB-corrected 3 -> 6."""
    piv = sub.pivot_table(index="mask_id", columns="seed", values=col)
    piv = piv[[s for s in SEEDS if s in piv.columns]].dropna()
    if piv.shape[1] < 6 or piv.shape[0] < 4:
        return np.nan, np.nan, [], piv
    seeds = list(piv.columns)
    seen, vals = set(), []
    for half in itertools.combinations(seeds, 3):
        other = tuple(s for s in seeds if s not in half)
        k = frozenset([half, other])
        if k in seen:
            continue
        seen.add(k)
        r = spearman(piv[list(half)].mean(1).values, piv[list(other)].mean(1).values)
        if np.isfinite(r):
            vals.append(float(r))
    assert len(vals) == 10, f"expected 10 distinct 3v3 splits, got {len(vals)}"
    r3 = float(np.mean(vals))
    r6 = 2 * r3 / (1 + r3) if (1 + r3) != 0 else np.nan
    return r3, float(r6), vals, piv


def main():
    df, missing = collect()
    if missing:
        print(f"[P10-S6] WARNING: {len(missing)} runs missing outcomes: {missing[:6]}")
    df.to_parquet(os.path.join(P3_RESULTS, "p10b_outcomes_S6.parquet"), index=False)
    print(f"[P10-S6] {len(df)} rows, {df.mask_id.nunique()} masks, {df.seed.nunique()} seeds, "
          f"{df.run.nunique()} runs")

    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    infl = pd.read_parquet(os.path.join(P3_RESULTS, "p10_influence.parquet"))
    clusters = dataset.clusters()

    res, rows = {}, []
    for t in clusters:
        sub = df[df.target == t]
        res[t] = {}
        for oc in OUTCOMES:
            r3, r6, splits, piv = ceiling_6seed(sub, oc)
            mean6 = piv.mean(1)
            # SB consistency check (the one that failed at S=4)
            r1 = float(np.mean([spearman(piv[a].values, piv[b].values)
                                for a, b in itertools.combinations(piv.columns, 2)]))
            pred6 = 6 * r1 / (1 + 5 * r1) if (1 + 5 * r1) else np.nan
            e = {"ceiling_r3_splithalf": r3, "ceiling_6seed_SB": r6, "per_split": splits,
                 "mean_pairwise_1seed_r": r1, "predicted_6seed_from_1seed_SB": pred6,
                 "SB_consistency_gap": (abs(pred6 - r6) if np.isfinite(pred6) and np.isfinite(r6)
                                        else np.nan),
                 "n_masks": int(len(mean6)), "n_seeds": int(piv.shape[1]), "attributors": {}}
            for a in PREREG_ATTRS + DESC_ATTRS:
                sc = infl[(infl.attributor == a) & (infl.target == t)]
                sc = dict(zip(sc.demo_id, sc.score))
                if not sc:
                    continue
                pred = np.array([sum(sc.get(d, 0.0) for d in m["demos"]) for m in gman])
                out = np.array([mean6.get(m["mask_id"], np.nan) for m in gman])
                ok = np.isfinite(out)
                rho = spearman(pred[ok], out[ok])
                p = spearman_p_onesided(rho, int(ok.sum()))
                lo, hi = bootstrap_spearman_ci(pred[ok], out[ok])
                ratio = rho / r6 if (np.isfinite(r6) and r6 > 0) else np.nan
                pas = bool(np.isfinite(ratio) and ratio >= 0.5 and np.isfinite(p) and p < ALPHA)
                e["attributors"][a] = {"rho": rho, "n": int(ok.sum()), "p_onesided": p,
                                       "ci95": [lo, hi], "ratio_to_ceiling": ratio, "PASS": pas,
                                       "in_preregistered_criterion": a in PREREG_ATTRS}
                rows.append({"target": t, "outcome": oc, "attributor": a, "rho": rho,
                             "ceiling_6seed": r6, "ratio": ratio, "p_onesided": p,
                             "bar": 0.5 * r6 if np.isfinite(r6) else np.nan,
                             "focal": t in FOCAL, "PASS": pas,
                             "in_criterion": a in PREREG_ATTRS})
            res[t][oc] = e

    pd.DataFrame(rows).to_csv(os.path.join(P3_RESULTS, "p10_lds_table_S6.csv"), index=False)

    prim = "neg_plain_loss"
    verdict = {}
    for t in FOCAL:
        e = res[t][prim]
        cand = {a: v for a, v in e["attributors"].items() if v["in_preregistered_criterion"]}
        best = max(cand, key=lambda a: cand[a]["rho"])
        verdict[t] = {"ceiling_6seed_measured": e["ceiling_6seed_SB"],
                      "ceiling_1seed": e["mean_pairwise_1seed_r"],
                      "ceiling_6seed_SB_predicted_from_1seed": e["predicted_6seed_from_1seed_SB"],
                      "SB_consistency_gap": e["SB_consistency_gap"],
                      "bar_half_ceiling": 0.5 * e["ceiling_6seed_SB"],
                      "best_attributor": best, "best_rho": cand[best]["rho"],
                      "best_ratio": cand[best]["ratio_to_ceiling"],
                      "best_p_onesided": cand[best]["p_onesided"],
                      "any_preregistered_attributor_PASS": any(v["PASS"] for v in cand.values()),
                      "per_attributor": e["attributors"]}
    overall = any(verdict[t]["any_preregistered_attributor_PASS"] for t in FOCAL)
    ceil_ok = all(np.isfinite(verdict[t]["ceiling_6seed_measured"])
                  and verdict[t]["ceiling_6seed_measured"] > 0.3 for t in FOCAL)

    out = {
        "stage": "P10 at S=6 (the PHASE3_DEFECT.md remedy)",
        "deviation": ("DISCLOSED: the preregistered S=4 ceiling collapsed (4 of 9 negative; SB "
                      "consistency off by 0.37), so the ratio-to-ceiling criterion had no power. "
                      "Two seeds (605, 606) were added -> S=6, matching the BC arm exactly. The "
                      "CRITERION IS UNCHANGED; only S changes. Both S=4 and S=6 are reported."),
        "s4_result_archived": "phase3/results/p10_verdict_S4_PREREGISTERED.json",
        "seeds": SEEDS, "S": 6, "n_masks": int(df.mask_id.nunique()),
        "n_retrains_total": int(df.run.nunique()),
        "ceiling_estimator": "all 10 distinct 3v3 splits, SB-corrected 3 -> 6 (Phase-2 P1's)",
        "PRIMARY_OUTCOME": prim,
        "CRITERION": ("focal C1/C5, 6-seed mean held-out L2: any preregistered attributor "
                      "rho >= 0.5 x measured ceiling AND one-sided p < 0.025 (Bonferroni-2)"),
        "focal_verdict": verdict,
        "PASS": bool(overall),
        "CEILING_IS_USABLE": bool(ceil_ok),
        "INTERPRETATION": None,
        "all_targets": res,
    }
    if not ceil_ok:
        out["INTERPRETATION"] = (
            "INCONCLUSIVE. Even at S=6 the executed-action ground truth for the diffusion policy "
            "does not have a usable ceiling. The preregistered symmetric interpretation is NOT "
            "invoked: this is a MEASUREMENT FAILURE, not evidence about attribution. The diffusion "
            "policy is deterministic but not seed-stable in action space (PHASE3_DEFECT.md).")
    elif overall:
        out["INTERPRETATION"] = (
            "PASS => faithfulness is POLICY-CLASS-DEPENDENT. Demo-grain attribution is faithful "
            "for the diffusion policy where it is not for the BC-Transformer, so the Phase-1/2 "
            "null must be stated as a property of the BC-Transformer regime, not of data "
            "attribution in general.")
    else:
        out["INTERPRETATION"] = (
            "FAIL at a usable ceiling => the null GENERALIZES ACROSS POLICY CLASS. It is not an "
            "artifact of the BC-Transformer's GMM head or training objective.")

    L.atomic_write_json(os.path.join(P3_RESULTS, "p10_verdict_S6.json"), out)

    print("\n" + "=" * 104)
    print("P10 at S=6 -- DIFFUSION DEMO-GRAIN LDS (primary: held-out L2 on the executed DDIM action)")
    print("=" * 104)
    print(f"{'target':7s} {'1seed':>7s} {'pred6(SB)':>10s} {'ceil6(meas)':>12s} {'gap':>6s} "
          f"{'bar':>7s} {'best':>8s} {'rho':>8s} {'ratio':>6s} {'p1':>7s}  verdict")
    for t in clusters:
        e = res[t][prim]
        cand = {a: v for a, v in e["attributors"].items() if v["in_preregistered_criterion"]}
        b = max(cand, key=lambda a: cand[a]["rho"])
        v = cand[b]
        mark = "FOCAL" if t in FOCAL else "     "
        st = ("PASS" if v["PASS"] else "fail") if t in FOCAL else "-"
        rt = v["ratio_to_ceiling"]
        print(f"{t:7s} {e['mean_pairwise_1seed_r']:7.3f} "
              f"{e['predicted_6seed_from_1seed_SB']:10.3f} {e['ceiling_6seed_SB']:12.3f} "
              f"{e['SB_consistency_gap']:6.3f} {0.5*e['ceiling_6seed_SB']:7.3f} {b:>8s} "
              f"{v['rho']:+8.3f} {rt:6.2f} {v['p_onesided']:7.4f}  {mark} {st}")
    print("=" * 104)
    print(f"CEILING USABLE: {ceil_ok}   PREREGISTERED CRITERION: {'PASS' if overall else 'FAIL'}")
    print(out["INTERPRETATION"])
    print("=" * 104)


if __name__ == "__main__":
    main()
