"""P11 -- THE CHAMPION CONFIRMATORY TEST ON THE BC-TRANSFORMER. Zero retrains.

Phases 2-3 never tested the configuration their own evidence says is best. They tested the
DEFAULT: IF/TRAK/TracIn at E=10, unnormalized, at a ridge nobody had tuned. Phase 3 then
established two things about the estimator side -- neither of them derived from the statistic
this test computes:

  * TracIn is the ONLY reproducible attributor (per-demo rank split-half reliability 0.831 at
    E=20, vs IF 0.254; ~16 members suffice for r=0.8 vs ~235 for IF)             [P9]
  * per-member SCALE NORMALIZATION is what the lambda->inf analysis showed matters (0.504 vs
    0.397 on C1)                                                                  [P6.5]

The CHAMPION is those two facts, composed, and FROZEN in the preregistration before any Phase-4
number existed:

    TracIn, E = 20 (members 201-220), each member's 135-demo score vector normalized to unit L2
    BEFORE averaging over members.

SECONDARY (preregistered, Bonferroni-4): the exact lambda->infinity limit estimator -- the raw
gradient dot product K, per-member unit-L2 normalized, same E=20.

GROUND TRUTH: the 24 Stage-G masks at the DEEPEST available seed count (S=10 from P12; S=6 if
P12 did not run), held-out L2, seed MEAN. The seed mean is the preregistered aggregator here and
its justification is on record and predates this test: the BC-Transformer's outcome is not
heavy-tailed, and Phase-2's SB consistency check on this exact instrument agreed to 0.014.
(P12 re-runs that check at S=10 as an up-front gate.)

CRITERION (one shot): focal C1 and C5; champion rho >= 0.5 * measured ceiling AND one-sided
p < 0.025 (Bonferroni-2 across the two focal targets). Exact rational arithmetic on the ratio.
All 9 targets reported DESCRIPTIVELY.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, P3_RESULTS, P2_RESULTS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci, mask_pred_score  # noqa: E402

FOCAL = ["C1", "C5"]
ALPHA_PRIMARY = 0.025          # Bonferroni-2 (2 focal targets), CHAMPION
ALPHA_SECONDARY = 0.0125       # Bonferroni-4 (2 targets x 2 estimator families), SECONDARY
MEMBERS_OLD = [f"ens_s{s}" for s in range(201, 211)]
MEMBERS_NEW = [f"ens_s{s}" for s in range(211, 221)]
MEMBERS = MEMBERS_OLD + MEMBERS_NEW            # E = 20
OUTCOME = "neg_plain_loss"


def load_per_member():
    """The E=20 per-member TracIn/TRAK/IF scores at functional 'plain', from the TWO archives."""
    a = pd.read_parquet(os.path.join(RESULTS, "influence_table_per_member.parquet"))
    a = a[a.functional == "plain"][["attributor", "target", "demo_id", "member", "score"]]
    b = pd.read_parquet(os.path.join(P3_RESULTS, "p9_influence_new_members.parquet"))
    b = b[b.functional == "plain"][["attributor", "target", "demo_id", "member", "score"]]
    df = pd.concat([a, b], ignore_index=True)
    got = sorted(df.member.unique())
    assert got == sorted(MEMBERS), f"expected E=20 members {MEMBERS}, got {got}"
    return df


def graddot_frame(train_ids, clusters):
    """The lambda->inf limit input: the RAW gradient dot product K, as a per-member frame.

    K_m for members 201-210 comes from phase3/results/p6_gram_cache.npz (P6.5's cache);
    K_m for members 211-220 from phase4/results/p11_gram_cache_new_members.npz (P11's pass).
    """
    rows = []
    for path in (os.path.join(P3_RESULTS, "p6_gram_cache.npz"),
                 os.path.join(P4_RESULTS, "p11_gram_cache_new_members.npz")):
        Z = np.load(path, allow_pickle=True)
        K, G = Z["K"], Z["G"]
        mem, tids, tgts = list(Z["members"]), list(Z["train_ids"]), list(Z["targets"])
        assert tids == list(train_ids), f"{path}: train_ids order differs"
        assert tgts == list(clusters), f"{path}: target order differs"
        d = np.array([np.mean(np.diag(G[m])) for m in range(K.shape[0])])   # mean diag(G_m)
        for mi, m in enumerate(mem):
            for j, c in enumerate(tgts):
                for i, t in enumerate(tids):
                    rows.append(("GradDot", c, t, str(m), float(K[mi, i, j])))
                    rows.append(("GradDot_dmean", c, t, str(m), float(K[mi, i, j] / d[mi])))
    return pd.DataFrame(rows, columns=["attributor", "target", "demo_id", "member", "score"])


def main():
    L.assert_prereg_locked()
    clusters = dataset.clusters()
    train_ids, _ = dataset.train_pool()
    masks = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    assert len(masks) == 24

    # ---------------------------------------------------------------- ground truth (deepest S)
    p12 = os.path.join(P4_RESULTS, "p12_ceilings.json")
    if os.path.exists(p12):
        C = json.load(open(p12))
        if not C["INSTRUMENT_GATE"]["PASS"]:
            raise SystemExit("[P11] P12's instrument gate FAILED -- no verdict may be computed. "
                             "See phase4/PHASE4_DEFECT.md.")
        df = pd.read_parquet(os.path.join(P4_RESULTS, "p12_outcomes_S10.parquet"))
        SEEDS = C["seeds"]
        ceil_key = f"ceiling_{len(SEEDS)}seed_SB"
        ceilings = {t: C["targets"][t][ceil_key] for t in clusters}
        gt_src = f"P12 S={len(SEEDS)} (phase4/results/p12_outcomes_S10.parquet)"
    else:
        raise SystemExit("[P11] p12_ceilings.json absent. P11's preregistration names S=10 as "
                         "primary; run p12_analyze.py first. (The S=6 fallback is only for the "
                         "case where P12 did not run at all, and must be taken deliberately.)")

    print(f"[P11] ground truth: {gt_src}, seeds {SEEDS}, aggregator = seed MEAN")
    obs = {t: df[df.target == t].groupby("mask_id")[OUTCOME].mean().to_dict() for t in clusters}

    # ---------------------------------------------------------------- estimators
    pm = load_per_member()
    gd = graddot_frame(train_ids, clusters)

    ESTIMATORS = [
        # (id, frame, attributor, normalize, family)
        ("TracIn_E20_normalized",   pm, "TracIn",        True,  "CHAMPION"),
        ("GradDot_E20_normalized",  gd, "GradDot",       True,  "SECONDARY"),
        # ---- descriptive only, never a verdict
        ("TracIn_E20_unnormalized", pm, "TracIn",        False, "descriptive"),
        ("GradDot_E20_dmean",       gd, "GradDot_dmean", False, "descriptive"),
        ("IF_E20_normalized",       pm, "IF",            True,  "descriptive"),
        ("IF_E20_unnormalized",     pm, "IF",            False, "descriptive"),
        ("TRAK_E20_normalized",     pm, "TRAK",          True,  "descriptive"),
        ("TRAK_E20_unnormalized",   pm, "TRAK",          False, "descriptive"),
    ]

    rows, res = [], {}
    for t in clusters:
        res[t] = {"ceiling": ceilings[t], "bar_half_ceiling": 0.5 * ceilings[t],
                  "estimators": {}}
        out_v = np.array([obs[t].get(m["mask_id"], np.nan) for m in masks])
        for eid, frame, attr, norm, fam in ESTIMATORS:
            sc = L.normalized_ensemble_scores(frame, attr, t, train_ids, MEMBERS, normalize=norm)
            pred = np.array([mask_pred_score(sc, m["demos"]) for m in masks])
            ok = np.isfinite(out_v) & np.isfinite(pred)
            rho = spearman(pred[ok], out_v[ok])
            n = int(ok.sum())
            p1 = spearman_p_onesided(rho, n)
            lo, hi = bootstrap_spearman_ci(pred[ok], out_v[ok])
            alpha = (ALPHA_PRIMARY if fam == "CHAMPION"
                     else ALPHA_SECONDARY if fam == "SECONDARY" else np.nan)
            meets = L.meets_half_ceiling(rho, ceilings[t])
            sig = bool(np.isfinite(p1) and np.isfinite(alpha) and p1 < alpha)
            e = {"estimator": eid, "family": fam, "attributor": attr, "normalized": norm,
                 "rho": rho, "n_masks": n, "p_onesided": p1, "ci95": [lo, hi],
                 "ratio_to_ceiling": (rho / ceilings[t]) if ceilings[t] else np.nan,
                 "ratio_exact": L.ratio_exact_str(rho, ceilings[t]),
                 "meets_half_ceiling_EXACT": meets,
                 "alpha_bonferroni": (None if not np.isfinite(alpha) else alpha),
                 "p_lt_alpha": sig,
                 "PASS": bool(meets and sig) if fam in ("CHAMPION", "SECONDARY") else None}
            res[t]["estimators"][eid] = e
            rows.append({"target": t, "focal": t in FOCAL, "estimator": eid, "family": fam,
                         "rho": rho, "ceiling": ceilings[t], "bar": 0.5 * ceilings[t],
                         "ratio": e["ratio_to_ceiling"], "p_onesided": p1,
                         "PASS": e["PASS"]})

    pd.DataFrame(rows).to_csv(os.path.join(P4_RESULTS, "p11_lds_table.csv"), index=False)

    # ---------------------------------------------------------------- PREREGISTERED VERDICT
    champ = {t: res[t]["estimators"]["TracIn_E20_normalized"] for t in FOCAL}
    seco = {t: res[t]["estimators"]["GradDot_E20_normalized"] for t in FOCAL}
    champ_pass = any(champ[t]["PASS"] for t in FOCAL)
    seco_pass = any(seco[t]["PASS"] for t in FOCAL)

    INTERP_PASS = ("a stability-selected, normalization-corrected attributor IS demo-grain "
                   "faithful where default-config attribution was not -- the Phase-2/3 null is a "
                   "statement about attribution AS PRACTICED, not about attribution's ceiling.")
    INTERP_FAIL = ("the null now covers the best configuration the evidence could assemble -- "
                   "champion-tested, not just default-tested.")

    verdict = {
        "stage": "P11",
        "preregistration_sha256": L.PREREG_SHA,
        "ground_truth": gt_src, "seeds": SEEDS, "S": len(SEEDS),
        "aggregator": "seed MEAN",
        "PRIMARY_OUTCOME": OUTCOME,
        "n_masks": 24,
        "CHAMPION": "TracIn, E=20 (members 201-220), per-member unit-L2 normalization, MEAN",
        "SECONDARY": "GradDot (exact lambda->inf limit), E=20, per-member unit-L2 normalization",
        "criterion": ("focal C1/C5: champion rho >= 0.5 * measured ceiling at the analysis seed "
                      "count AND one-sided p < 0.025 (Bonferroni-2). Secondary at Bonferroni-4 "
                      "(p < 0.0125). Ratio compared with EXACT rational arithmetic."),
        "focal": {t: {"ceiling": ceilings[t], "bar": 0.5 * ceilings[t],
                      "CHAMPION": champ[t], "SECONDARY": seco[t]} for t in FOCAL},
        "CHAMPION_PASS_any_focal": bool(champ_pass),
        "SECONDARY_PASS_any_focal": bool(seco_pass),
        "VERDICT": "PASS" if champ_pass else "FAIL",
        "PREREGISTERED_INTERPRETATION": INTERP_PASS if champ_pass else INTERP_FAIL,
        "all_targets_DESCRIPTIVE": res,
    }
    L.atomic_write_json(os.path.join(P4_RESULTS, "p11_verdict.json"), verdict)

    # ---------------------------------------------------------------- print
    print("\n" + "=" * 104)
    print(f"P11 -- CHAMPION CONFIRMATORY TEST  (BC-Transformer, S={len(SEEDS)}, held-out L2, "
          f"seed mean, n=24 masks)")
    print("=" * 104)
    print(f"{'target':7s} {'ceiling':>8s} {'bar':>7s} | "
          f"{'CHAMPION rho':>12s} {'ratio':>6s} {'p1':>8s} {'':>5s} | "
          f"{'SECONDARY rho':>13s} {'ratio':>6s} {'p1':>8s}")
    for t in clusters:
        c, s = res[t]["estimators"]["TracIn_E20_normalized"], \
               res[t]["estimators"]["GradDot_E20_normalized"]
        mark = "FOCAL" if t in FOCAL else ""
        v = ("PASS" if c["PASS"] else "fail") if t in FOCAL else "-"
        print(f"{t:7s} {ceilings[t]:8.3f} {0.5*ceilings[t]:7.3f} | "
              f"{c['rho']:+12.3f} {c['ratio_to_ceiling']:6.2f} {c['p_onesided']:8.4f} "
              f"{v:>5s} | {s['rho']:+13.3f} {s['ratio_to_ceiling']:6.2f} {s['p_onesided']:8.4f}"
              f"  {mark}")
    print("-" * 104)
    print("descriptive contrasts on the focal targets (normalization on/off):")
    for t in FOCAL:
        for eid in ("TracIn_E20_normalized", "TracIn_E20_unnormalized", "GradDot_E20_normalized",
                    "GradDot_E20_dmean", "IF_E20_normalized", "TRAK_E20_normalized"):
            e = res[t]["estimators"][eid]
            print(f"  {t} {eid:26s} rho={e['rho']:+.3f} ratio={e['ratio_to_ceiling']:+.2f} "
                  f"p1={e['p_onesided']:.4f}")
    print("=" * 104)
    print(f"CHAMPION VERDICT (any focal target): {'PASS' if champ_pass else 'FAIL'}")
    print(f"  -> {verdict['PREREGISTERED_INTERPRETATION']}")
    print(f"SECONDARY (Bonferroni-4): {'PASS' if seco_pass else 'FAIL'}")
    print("=" * 104)


if __name__ == "__main__":
    main()
