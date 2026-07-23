"""P10 -- the DIFFUSION verdict. Does the core null replicate on a second policy class?

PREREGISTERED CRITERION (identical in FORM to Phase-2 P1, so the two policy classes are judged by
the same instrument):
    focal targets C1 and C5; PRIMARY outcome neg_plain_loss (held-out L2 on the EXECUTED DDIM
    action), 4-seed mean; attribution is USABLE at demo grain iff ANY preregistered attributor
    ({TracIn, TRAK}) reaches rho >= 0.5 x the MEASURED 4-seed ceiling with one-sided p < 0.025
    (Bonferroni-2).

The PRIMARY outcome is the L2-on-executed-action -- the SAME outcome on which the BC-Transformer
null was established. That choice is what makes the replication meaningful: a diffusion-native
outcome would have made the two classes incomparable. The denoising-loss outcome is reported
alongside as the diffusion-native secondary.

CEILING: 2v2 split-half over the 4 seeds (all 3 disjoint pairings), Spearman between the two
pair-mean outcome vectors across the 24 masks, averaged; Spearman-Brown corrected 2 -> 4 as
r4 = 2*r2/(1+r2). The S = 1..4 seed ladder is also reported.

PREREGISTERED SYMMETRIC INTERPRETATION (verbatim from the preregistration):
    "FAIL again => the null generalizes across policy class; PASS => faithfulness is
     policy-class-dependent -- report symmetrically."
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci  # noqa: E402

SEEDS = [601, 602, 603, 604]
FOCAL = ["C1", "C5"]
PREREG_ATTRS = ["TracIn", "TRAK"]      # the preregistered criterion ranges over THESE only
DESCRIPTIVE_ATTRS = ["IF"]             # free from the same G,K; reported, NOT in the criterion
ALPHA = 0.025
OUTCOMES = ["neg_plain_loss", "neg_denoise_loss", "logit_success"]


def collect():
    jobs = json.load(open(os.path.join(P3_RESULTS, "p10b_jobs.json")))
    rows, missing = [], []
    for j in jobs:
        oc = L.read_outcomes(j["run_dir"], required=False)     # MARKER-GATED
        if oc is None:
            missing.append(os.path.basename(j["run_dir"]))
            continue
        for c, v in oc.items():
            rows.append({"run": os.path.basename(j["run_dir"]), "mask_id": j["mask_id"],
                         "seed": j["seed"], "target": c,
                         "success_rate": v["success_rate"], "n_episodes": v["n_episodes"],
                         "plain_loss": v["plain_loss"], "denoise_loss": v["denoise_loss"],
                         "transport_loss": v["transport_loss"],
                         "interaction_loss": v["interaction_loss"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["logit_success"] = L.logit_success_rowwise(df.success_rate.values,
                                                      df.n_episodes.values)
        df["neg_plain_loss"] = -df.plain_loss
        df["neg_denoise_loss"] = -df.denoise_loss
    return df, missing


def ceiling_4seed(sub, col):
    """2v2 split-half over the 4 seeds -> reliability of a 2-seed mean; SB-corrected 2 -> 4."""
    piv = sub.pivot_table(index="mask_id", columns="seed", values=col)
    piv = piv[[s for s in SEEDS if s in piv.columns]].dropna()
    if piv.shape[1] < 4 or piv.shape[0] < 4:
        return np.nan, np.nan, [], piv
    s = list(piv.columns)
    pairings = [((s[0], s[1]), (s[2], s[3])), ((s[0], s[2]), (s[1], s[3])),
                ((s[0], s[3]), (s[1], s[2]))]
    rs = [spearman(piv[list(a)].mean(1).values, piv[list(b)].mean(1).values)
          for a, b in pairings]
    r2 = float(np.nanmean(rs))
    r4 = 2 * r2 / (1 + r2) if (1 + r2) != 0 else np.nan
    return r2, float(r4), [float(x) for x in rs], piv


def seed_ladder(sub, col):
    """Measured ceiling at S = 1, 2 (and the SB extrapolation to 4)."""
    piv = sub.pivot_table(index="mask_id", columns="seed", values=col)
    piv = piv[[s for s in SEEDS if s in piv.columns]].dropna()
    out = {}
    r1 = float(np.mean([spearman(piv[a].values, piv[b].values)
                        for a, b in itertools.combinations(piv.columns, 2)]))
    out["S1_measured"] = r1
    r2, r4, _, _ = ceiling_4seed(sub, col)
    out["S2_measured"] = r2
    out["S4_SB_from_S2"] = r4
    out["S4_SB_from_S1"] = 4 * r1 / (1 + 3 * r1) if (1 + 3 * r1) else np.nan
    return out


def main():
    df, missing = collect()
    if not len(df):
        raise SystemExit("[P10] no outcomes found -- P10b has not run")
    if missing:
        print(f"[P10] WARNING: {len(missing)} runs missing outcomes: {missing[:6]}")
    df.to_parquet(os.path.join(P3_RESULTS, "p10b_outcomes.parquet"), index=False)
    print(f"[P10] {len(df)} rows, {df.mask_id.nunique()} masks, {df.seed.nunique()} seeds, "
          f"{df.run.nunique()} runs")

    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    infl = pd.read_parquet(os.path.join(P3_RESULTS, "p10_influence.parquet"))
    clusters = dataset.clusters()

    res, rows = {}, []
    for t in clusters:
        sub = df[df.target == t]
        res[t] = {}
        for oc in OUTCOMES:
            r2, r4, splits, piv = ceiling_4seed(sub, oc)
            mean4 = piv.mean(1)
            entry = {"ceiling_2seed_splithalf": r2, "ceiling_4seed_SB": r4,
                     "per_pairing": splits, "seed_ladder": seed_ladder(sub, oc),
                     "n_masks": int(len(mean4)), "attributors": {}}
            for a in PREREG_ATTRS + DESCRIPTIVE_ATTRS:
                sc = infl[(infl.attributor == a) & (infl.target == t)]
                sc = dict(zip(sc.demo_id, sc.score))
                if not sc:
                    continue
                pred = np.array([sum(sc.get(d, 0.0) for d in m["demos"]) for m in gman])
                out = np.array([mean4.get(m["mask_id"], np.nan) for m in gman])
                ok = np.isfinite(out)
                rho = spearman(pred[ok], out[ok])
                p = spearman_p_onesided(rho, int(ok.sum()))
                lo, hi = bootstrap_spearman_ci(pred[ok], out[ok])
                ratio = rho / r4 if (np.isfinite(r4) and r4 > 0) else np.nan
                pas = bool(np.isfinite(ratio) and ratio >= 0.5 and np.isfinite(p) and p < ALPHA)
                entry["attributors"][a] = {
                    "rho": rho, "n": int(ok.sum()), "p_onesided": p, "ci95": [lo, hi],
                    "ratio_to_ceiling": ratio, "PASS": pas,
                    "in_preregistered_criterion": a in PREREG_ATTRS,
                }
                rows.append({"target": t, "outcome": oc, "attributor": a, "rho": rho,
                             "ceiling_4seed": r4, "ratio": ratio, "p_onesided": p,
                             "bar": 0.5 * r4 if np.isfinite(r4) else np.nan,
                             "focal": t in FOCAL, "PASS": pas,
                             "in_criterion": a in PREREG_ATTRS})
            res[t][oc] = entry

    pd.DataFrame(rows).to_csv(os.path.join(P3_RESULTS, "p10_lds_table.csv"), index=False)

    # ---------------- PREREGISTERED VERDICT
    prim = "neg_plain_loss"
    verdict = {}
    for t in FOCAL:
        e = res[t][prim]
        cand = {a: v for a, v in e["attributors"].items() if v["in_preregistered_criterion"]}
        best = max(cand, key=lambda a: cand[a]["rho"])
        verdict[t] = {
            "ceiling_4seed_measured": e["ceiling_4seed_SB"],
            "bar_half_ceiling": 0.5 * e["ceiling_4seed_SB"],
            "best_attributor": best, "best_rho": cand[best]["rho"],
            "best_ratio": cand[best]["ratio_to_ceiling"],
            "best_p_onesided": cand[best]["p_onesided"],
            "any_preregistered_attributor_PASS": any(v["PASS"] for v in cand.values()),
            "per_attributor": e["attributors"],
            "seed_ladder": e["seed_ladder"],
        }
    overall = any(verdict[t]["any_preregistered_attributor_PASS"] for t in FOCAL)

    out = {
        "stage": "P10 diffusion-policy replication",
        "policy": "state-based diffusion policy, DDPM training, deterministic DDIM (eta=0) eval",
        "n_retrains_P10b": int(df.run.nunique()), "n_missing": len(missing),
        "seeds": SEEDS, "S": 4, "n_masks": int(df.mask_id.nunique()),
        "masks": "the EXISTING 24 Stage-G demo masks, NOT resampled",
        "PRIMARY_OUTCOME": prim,
        "primary_outcome_rationale": ("L2 on the EXECUTED action -- the SAME outcome on which the "
                                      "BC-Transformer null was established. A diffusion-native "
                                      "outcome would have made the two policy classes "
                                      "incomparable."),
        "preregistered_attributors": PREREG_ATTRS,
        "descriptive_attributors_not_in_criterion": DESCRIPTIVE_ATTRS,
        "CRITERION": ("focal C1/C5, 4-seed mean held-out L2: any preregistered attributor "
                      "rho >= 0.5 x measured 4-seed ceiling AND one-sided p < 0.025 (Bonferroni-2)"),
        "focal_verdict": verdict,
        "PASS": bool(overall),
        "PREREGISTERED_SYMMETRIC_INTERPRETATION": (
            "FAIL again => the null generalizes across policy class; PASS => faithfulness is "
            "policy-class-dependent -- report symmetrically."),
        "VERDICT_TEXT": (
            "PASS: demo-grain attribution IS faithful for the diffusion policy where it is not for "
            "the BC-Transformer. Faithfulness is therefore POLICY-CLASS-DEPENDENT, and the "
            "Phase-1/2 null must be stated as a property of the BC-Transformer regime, not of data "
            "attribution in general."
            if overall else
            "FAIL: demo-grain attribution is unfaithful for the diffusion policy too. The null "
            "GENERALIZES ACROSS POLICY CLASS -- it is not an artifact of the BC-Transformer's GMM "
            "head or of its training objective."),
        "all_targets": res,
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p10_verdict.json"), out)

    print("\n" + "=" * 100)
    print("P10 -- DIFFUSION DEMO-GRAIN LDS (primary: held-out L2 on the executed DDIM action)")
    print("=" * 100)
    print(f"{'target':7s} {'ceil(S1)':>9s} {'ceil(S2)':>9s} {'ceil4(SB)':>10s} {'bar':>7s} "
          f"{'best':>8s} {'rho':>8s} {'ratio':>6s} {'p1':>7s}  verdict")
    for t in clusters:
        e = res[t][prim]
        cand = {a: v for a, v in e["attributors"].items() if v["in_preregistered_criterion"]}
        if not cand:
            continue
        b = max(cand, key=lambda a: cand[a]["rho"])
        v = cand[b]
        mark = "FOCAL" if t in FOCAL else "     "
        st = ("PASS" if v["PASS"] else "fail") if t in FOCAL else "-"
        print(f"{t:7s} {e['seed_ladder']['S1_measured']:9.3f} "
              f"{e['ceiling_2seed_splithalf']:9.3f} {e['ceiling_4seed_SB']:10.3f} "
              f"{0.5*e['ceiling_4seed_SB']:7.3f} {b:>8s} {v['rho']:+8.3f} "
              f"{v['ratio_to_ceiling']:6.2f} {v['p_onesided']:7.4f}  {mark} {st}")
    print("=" * 100)
    print(f"PREREGISTERED VERDICT: {'PASS' if overall else 'FAIL'}")
    print(out["VERDICT_TEXT"])
    print("=" * 100)


if __name__ == "__main__":
    main()
