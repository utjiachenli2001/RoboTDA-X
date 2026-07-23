"""P10 DIAGNOSTIC -- median-over-seeds aggregation. POST-HOC. NOT the preregistered verdict.

*** THIS IS A DIAGNOSTIC, NOT A PREREGISTERED TEST. ***
The preregistration specifies the seed MEAN. That test ran, at S=4 and again at S=6, and both are
reported verbatim (p10_verdict_S4_PREREGISTERED.json, p10_verdict_S6.json). Both are INCONCLUSIVE,
because the mean-aggregated ceiling collapses.

WHY IT COLLAPSES (measured, PHASE3_DEFECT.md): the diffusion policy's held-out L2 is HEAVY-TAILED
across seeds. Its executed action comes from a DDIM trajectory launched from a fixed latent through
a MULTIMODAL denoiser, so an occasional seed lands that latent in the wrong basin and its L2 blows
up by 3-5x (e.g. mask G000, C1: [0.77, 2.65, 0.65, 0.68, 0.72, 0.68]). A MEAN over such a sample is
dominated by the outlier, and averaging MORE seeds does not help -- which is why the 6-seed mean
ceiling (0.079) is LOWER than Spearman-Brown predicts from the 1-seed reliability (0.690). A
reliability that DROPS when you average is the signature of a broken aggregator, not of noise.

THIS IS NOT A NEW PATHOLOGY -- IT IS PHASE 1'S, REBORN. src/policy.py:l2 documents why Phase 1
abandoned the GMM-NLL functional: "it is unbounded and heavy-tailed, so when the GMM's sigma
collapses on some seeds the held-out mean NLL swings 8-10x (mask D00: 23.6 vs 187.0 on identical
data) while the MEDIAN per-frame NLL barely moves (27.6 vs 32.4)." Phase 1 switched to L2 to escape
that. The diffusion policy re-introduces the same failure mode INSIDE the L2, one level up: not in
the per-frame loss, but in the seed-to-seed distribution of it.

The median restores the instrument (ceiling 0.079 -> 0.568 on C1; all 9 targets land in 0.57-0.77).

SO: with a working instrument, we can finally ASK the question. The answer below is reported as a
DIAGNOSTIC with its status stated plainly. The criterion arithmetic is otherwise IDENTICAL to the
preregistered one (focal C1/C5, any preregistered attributor rho >= 0.5 x measured ceiling,
one-sided p < 0.025, Bonferroni-2).
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


def ceiling_median(piv):
    """The Phase-2 P1 estimator (10 distinct 3v3 splits, SB 3->6) but aggregating by MEDIAN."""
    seen, vals = set(), []
    for half in itertools.combinations(SEEDS, 3):
        other = tuple(s for s in SEEDS if s not in half)
        k = frozenset([half, other])
        if k in seen:
            continue
        seen.add(k)
        a = piv[list(half)].median(axis=1).values
        b = piv[list(other)].median(axis=1).values
        r = spearman(a, b)
        if np.isfinite(r):
            vals.append(float(r))
    r3 = float(np.mean(vals))
    return r3, (2 * r3 / (1 + r3) if (1 + r3) != 0 else np.nan), vals


def main():
    df = pd.read_parquet(os.path.join(P3_RESULTS, "p10b_outcomes_S6.parquet"))
    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    infl = pd.read_parquet(os.path.join(P3_RESULTS, "p10_influence.parquet"))
    clusters = dataset.clusters()

    res, rows = {}, []
    for t in clusters:
        sub = df[df.target == t]
        piv = sub.pivot_table(index="mask_id", columns="seed",
                              values="neg_plain_loss")[SEEDS].dropna()
        r3, r6, splits = ceiling_median(piv)
        med6 = piv.median(axis=1)
        mean6 = piv.mean(axis=1)
        # the heavy-tail evidence, per target
        pl = sub.pivot_table(index="mask_id", columns="seed", values="plain_loss")[SEEDS]
        hv = {"median_max_over_min_across_seeds": float((pl.max(1) / pl.min(1)).median()),
              "median_within_mask_CV": float((pl.std(1) / pl.mean(1)).median())}
        e = {"ceiling_median_r3": r3, "ceiling_median_6seed_SB": r6, "per_split": splits,
             "heavy_tail_evidence": hv, "attributors": {}}
        for a in PREREG_ATTRS + DESC_ATTRS:
            sc = infl[(infl.attributor == a) & (infl.target == t)]
            sc = dict(zip(sc.demo_id, sc.score))
            pred = np.array([sum(sc.get(d, 0.0) for d in m["demos"]) for m in gman])
            out = np.array([med6.get(m["mask_id"], np.nan) for m in gman])
            ok = np.isfinite(out)
            rho = spearman(pred[ok], out[ok])
            p = spearman_p_onesided(rho, int(ok.sum()))
            lo, hi = bootstrap_spearman_ci(pred[ok], out[ok])
            ratio = rho / r6 if (np.isfinite(r6) and r6 > 0) else np.nan
            pas = bool(np.isfinite(ratio) and ratio >= 0.5 and np.isfinite(p) and p < ALPHA)
            e["attributors"][a] = {"rho": rho, "n": int(ok.sum()), "p_onesided": p,
                                   "ci95": [lo, hi], "ratio_to_ceiling": ratio, "PASS": pas,
                                   "in_preregistered_criterion": a in PREREG_ATTRS}
            rows.append({"target": t, "attributor": a, "rho": rho, "ceiling_median": r6,
                         "ratio": ratio, "p_onesided": p, "bar": 0.5 * r6,
                         "focal": t in FOCAL, "PASS": pas})
        res[t] = e

    pd.DataFrame(rows).to_csv(os.path.join(P3_RESULTS, "p10_lds_table_MEDIAN_DIAGNOSTIC.csv"),
                              index=False)

    verdict = {}
    for t in FOCAL:
        e = res[t]
        cand = {a: v for a, v in e["attributors"].items() if v["in_preregistered_criterion"]}
        best = max(cand, key=lambda a: cand[a]["rho"])
        verdict[t] = {"ceiling_median_6seed": e["ceiling_median_6seed_SB"],
                      "bar": 0.5 * e["ceiling_median_6seed_SB"],
                      "best_attributor": best, "best_rho": cand[best]["rho"],
                      "best_ratio": cand[best]["ratio_to_ceiling"],
                      "best_p_onesided": cand[best]["p_onesided"],
                      "any_PASS": any(v["PASS"] for v in cand.values()),
                      "per_attributor": e["attributors"]}
    overall = any(verdict[t]["any_PASS"] for t in FOCAL)
    ceil_ok = all(verdict[t]["ceiling_median_6seed"] > 0.3 for t in FOCAL)

    out = {
        "stage": "P10 MEDIAN-aggregated DIAGNOSTIC",
        "STATUS": ("*** POST-HOC DIAGNOSTIC, NOT THE PREREGISTERED VERDICT. *** The preregistration "
                   "specifies the seed MEAN. That test ran at S=4 and S=6 and is reported verbatim; "
                   "both are INCONCLUSIVE because the mean-aggregated ceiling collapses on a "
                   "heavy-tailed outcome. This diagnostic swaps ONLY the seed aggregator "
                   "(mean -> median) and keeps every other element of the criterion identical."),
        "why_the_mean_fails": ("The diffusion executed-action L2 is heavy-tailed across seeds: an "
                               "occasional seed lands the fixed DDIM latent in the wrong basin of a "
                               "multimodal denoiser and its L2 blows up 3-5x. A mean is dominated by "
                               "that outlier, and averaging MORE seeds does not help -- the 6-seed "
                               "mean ceiling (0.079 on C1) is LOWER than Spearman-Brown predicts "
                               "from the 1-seed reliability (0.690). A reliability that DROPS when "
                               "you average is a broken aggregator, not noise."),
        "precedent": ("This is Phase 1's GMM-NLL pathology reborn one level up. src/policy.py:l2 "
                      "documents Phase 1 abandoning the mean NLL because it 'swings 8-10x while the "
                      "MEDIAN per-frame NLL barely moves'. The diffusion policy re-introduces the "
                      "same failure mode inside the L2, in the seed-to-seed distribution."),
        "seeds": SEEDS, "S": 6, "aggregator": "MEDIAN over the 6 seeds",
        "ceiling_estimator": "10 distinct 3v3 splits, median-aggregated, SB-corrected 3 -> 6",
        "ceiling_comparison": {t: {"mean_aggregated": None, "median_aggregated":
                                   res[t]["ceiling_median_6seed_SB"]} for t in clusters},
        "focal_verdict": verdict,
        "CEILING_IS_USABLE": bool(ceil_ok),
        "DIAGNOSTIC_PASS": bool(overall),
        "all_targets": res,
        "INTERPRETATION": None,
    }
    if not ceil_ok:
        out["INTERPRETATION"] = "Even the median ceiling is unusable. P10 remains INCONCLUSIVE."
    elif overall:
        out["INTERPRETATION"] = (
            "DIAGNOSTIC PASS at a usable ceiling. On this (post-hoc) aggregator, demo-grain "
            "attribution IS faithful for the diffusion policy where it is not for the "
            "BC-Transformer -- i.e. faithfulness would be POLICY-CLASS-DEPENDENT. Because the "
            "aggregator was not preregistered, this CANNOT be reported as a confirmatory result. "
            "It is a strong, specific, pre-specifiable hypothesis for a follow-up study.")
    else:
        out["INTERPRETATION"] = (
            "DIAGNOSTIC FAIL at a USABLE ceiling (0.57 / 0.72 on the focal targets). This is the "
            "informative version of the P10 verdict: with a working ground-truth instrument, "
            "demo-grain attribution STILL does not reach half of oracle for the diffusion policy. "
            "The null therefore GENERALIZES ACROSS POLICY CLASS. Reported as a diagnostic (the "
            "aggregator is post-hoc), but the ceiling it is judged against is real, and the "
            "attribution ensemble behind it is MORE reproducible than the BC arm's, not less.")

    L.atomic_write_json(os.path.join(P3_RESULTS, "p10_diagnostic_median.json"), out)

    print("=" * 104)
    print("P10 MEDIAN DIAGNOSTIC (POST-HOC -- NOT the preregistered verdict)")
    print("=" * 104)
    print(f"{'target':7s} {'ceil(mean)':>11s} {'ceil(MEDIAN)':>13s} {'bar':>7s} {'best':>8s} "
          f"{'rho':>8s} {'ratio':>6s} {'p1':>8s}  verdict")
    s6 = json.load(open(os.path.join(P3_RESULTS, "p10_verdict_S6.json")))
    for t in clusters:
        e = res[t]
        cand = {a: v for a, v in e["attributors"].items() if v["in_preregistered_criterion"]}
        b = max(cand, key=lambda a: cand[a]["rho"])
        v = cand[b]
        cm = s6["all_targets"][t]["neg_plain_loss"]["ceiling_6seed_SB"]
        mark = "FOCAL" if t in FOCAL else "     "
        st = ("PASS" if v["PASS"] else "fail") if t in FOCAL else "-"
        print(f"{t:7s} {cm:11.3f} {e['ceiling_median_6seed_SB']:13.3f} "
              f"{0.5*e['ceiling_median_6seed_SB']:7.3f} {b:>8s} {v['rho']:+8.3f} "
              f"{v['ratio_to_ceiling']:6.2f} {v['p_onesided']:8.4f}  {mark} {st}")
    print("=" * 104)
    print(f"CEILING USABLE: {ceil_ok}   DIAGNOSTIC: {'PASS' if overall else 'FAIL'}")
    print(out["INTERPRETATION"])
    print("=" * 104)


if __name__ == "__main__":
    main()
