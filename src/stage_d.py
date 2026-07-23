"""STAGE D -- GATE 1: attributor sanity (STOP/GO for attribution trust). Spec §5.

Design:
  C1 only. K=12 fixed-size demo masks, each exactly 8 of C1's 15 training demos, each demo
  included in 6 or 7 masks (12*8 = 96 = 9 demos x6 + 6 demos x7). Deterministic seed.
  x 2 seeds = 24 small retrains. Each is evaluated on C1's probe battery.

  Attributors are computed from FULL-C1-trained models (Stage B's C1_target_s101..s105, which
  are trained on exactly C1's 15 demos and carry 5 checkpoints each) toward the C1 held-out
  plain-loss functional.

  predicted mask score = sum of per-demo attributions over the mask's 8 demos
  Spearman(predicted, retrained outcome) over the 12 masks (seed-mean outcome).

SIGN CONVENTION: a positive influence score means "training on this demo REDUCES the target's
held-out loss". So the loss outcome is reported as NEGATIVE loss ("utility"), making a
faithful attributor give a POSITIVE Spearman for both outcomes (loss-utility and success).

PASS iff any attributor reaches Spearman > 0.50.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS, RESULTS
import dataset
import orchestrator as O
import lds

SEEDS = [101, 102]
K = 12
PER_MASK = 8
MASK_SEED = 23
N_ROLLOUTS = 10
PASS_RHO = 0.50


def build_masks():
    """K=12 masks x 8 of C1's 15 demos; every demo in 6 or 7 masks (deterministic)."""
    _, by_c = dataset.train_pool()
    demos = list(by_c["C1"])
    n = len(demos)
    rng = np.random.default_rng(MASK_SEED)
    cnt = np.zeros(n, dtype=int)
    masks = []
    for k in range(K):
        order = np.lexsort((rng.random(n), cnt))     # fewest inclusions first, random ties
        sel = sorted(order[:PER_MASK])
        cnt[sel] += 1
        masks.append({"mask_id": f"D{k:02d}", "demos": [demos[i] for i in sel]})
    assert cnt.min() >= 6 and cnt.max() <= 7, f"inclusion counts {cnt}"
    assert all(len(m["demos"]) == PER_MASK for m in masks)
    return masks, {demos[i]: int(cnt[i]) for i in range(n)}


def build_jobs(masks):
    return [{"run_dir": os.path.join(RUNS, "stage_D", f"{m['mask_id']}_s{s}"),
             "demos": m["demos"], "seed": s, "n_rollouts": N_ROLLOUTS,
             "eval": "probe", "clusters": ["C1"], "workers": 8}
            for m in masks for s in SEEDS]


def outcomes(masks):
    """{mask_id: {neg_plain_loss, success, logit_success}} averaged over the 2 seeds."""
    out = {}
    for m in masks:
        pl, sc = [], []
        for s in SEEDS:
            p = os.path.join(RUNS, "stage_D", f"{m['mask_id']}_s{s}", "outcomes.json")
            if not os.path.exists(p):
                continue
            o = json.load(open(p))["outcomes"]["C1"]
            pl.append(o["plain_loss"])
            sc.append(o["success_rate"])
        if not pl:
            continue
        out[m["mask_id"]] = {
            "neg_plain_loss": float(-np.mean(pl)),
            "success": float(np.mean(sc)),
            "logit_success": float(np.mean(lds.logit_success(sc, 30))),
            "n_seeds": len(pl),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--attr_glob", default=os.path.join(RUNS, "stage_B", "C1_target_s*"))
    a = ap.parse_args()

    masks, counts = build_masks()
    json.dump({"seed": MASK_SEED, "K": K, "per_mask": PER_MASK,
               "per_demo_inclusion": counts, "masks": masks},
              open(os.path.join(RESULTS, "stage_D_mask_manifest.json"), "w"), indent=1)

    if not a.analyze_only:
        O.run_jobs(build_jobs(masks), "stage_D")

    # ---- attribution from full-C1 models (Stage B target-only runs)
    import attribution as ATT
    _, by_c = dataset.train_pool()
    c1 = by_c["C1"]
    parq = os.path.join(RESULTS, "stage_D_influence_C1.parquet")
    if not os.path.exists(parq):
        dirs = sorted(glob.glob(a.attr_glob))
        dirs = [d for d in dirs if os.path.exists(os.path.join(d, "final.pt"))]
        if not dirs:
            raise SystemExit(f"no full-C1 models at {a.attr_glob} (Stage B must run first)")
        print(f"[stage_D] attribution from {len(dirs)} full-C1 models: "
              f"{[os.path.basename(d) for d in dirs]}")
        ATT.compute(dirs, parq, train_ids=c1,
                    targets=[("C1", f) for f in ATT.FUNCTIONALS])
    import pandas as pd
    df = pd.read_parquet(parq)

    obs = outcomes(masks)

    # --- NOISE CEILING for Gate 1 (judge LDS against the ceiling, never against 1.0).
    # With only 2 seeds we estimate the 1-seed reliability as the Spearman between the two
    # seeds' outcome vectors across the 12 masks, then Spearman-Brown it up to the 2-seed mean
    # that the LDS is actually predicting: r2 = 2r/(1+r).
    import numpy as _np
    ceil = {}
    for key, getter in (("neg_plain_loss", lambda o: -o["plain_loss"]),
                        ("logit_success", lambda o: o["success_rate"])):
        v = {s: [] for s in SEEDS}
        for m in masks:
            ok = True
            row = {}
            for s in SEEDS:
                p = os.path.join(RUNS, "stage_D", f"{m['mask_id']}_s{s}", "outcomes.json")
                if not os.path.exists(p):
                    ok = False
                    break
                row[s] = getter(json.load(open(p))["outcomes"]["C1"])
            if ok:
                for s in SEEDS:
                    v[s].append(row[s])
        r1 = lds.spearman(v[SEEDS[0]], v[SEEDS[1]])
        r2 = (2 * r1 / (1 + r1)) if (_np.isfinite(r1) and (1 + r1) != 0) else float("nan")
        ceil[key] = {"seed_pair_rho_1seed": float(r1), "ceiling_2seed_SpearmanBrown": float(r2),
                     "n_masks": len(v[SEEDS[0]])}

    res = {"gate": "GATE 1 (Stage D)", "criterion": f"any attributor Spearman > {PASS_RHO}",
           "K": K, "n_masks_with_outcomes": len(obs), "seeds": SEEDS,
           "loss_functional": "L2 on the executed action (the GMM-NLL functional was unusable; "
                              "see policy.l2)",
           "noise_ceiling": ceil, "mask_outcomes": obs, "attributors": {}}

    for attr in sorted(df.attributor.unique()):
        sub = df[(df.attributor == attr) & (df.functional == "plain") & (df.target == "C1")]
        scores = dict(zip(sub.demo_id, sub.score))
        entry = {}
        for outcome_key in ("neg_plain_loss", "logit_success"):
            o = {m: obs[m][outcome_key] for m in obs}
            r = lds.conditional_lds(scores, [m for m in masks if m["mask_id"] in obs], o,
                                    "C1", include_only_target_masks=False)
            entry[outcome_key] = {"rho": r["rho"], "n": r["n_masks"],
                                  "p_onesided": r["p_onesided"], "ci95": r["ci95"]}
        entry["PASS_primary"] = bool(entry["neg_plain_loss"]["rho"] is not None
                                     and entry["neg_plain_loss"]["rho"] > PASS_RHO)
        res["attributors"][attr] = entry

    res["any_pass"] = any(v["PASS_primary"] for v in res["attributors"].values())
    res["PASS"] = res["any_pass"]
    json.dump(res, open(os.path.join(RESULTS, "stage_D_gate1.json"), "w"), indent=1)

    print("\n=== GATE 1: attributor sanity (C1, 12 masks x 8 demos, 2 seeds) ===")
    print(f"  loss functional: L2 on the executed action (GMM NLL was unusable)")
    for k, c in ceil.items():
        print(f"  NOISE CEILING [{k}]: 1-seed rho={c['seed_pair_rho_1seed']:+.3f} -> "
              f"2-seed ceiling={c['ceiling_2seed_SpearmanBrown']:+.3f} "
              f"(the max an oracle attributor could score)")
    print(f"\n{'attributor':>10} | {'rho (held-out L2, PRIMARY)':>30} | {'rho (success)':>16}")
    for attr, v in res["attributors"].items():
        a1, a2 = v["neg_plain_loss"], v["logit_success"]
        print(f"{attr:>10} | {a1['rho']:>+8.3f}  (p={a1['p_onesided']:.3f}, n={a1['n']})   "
              f"| {a2['rho']:>+8.3f}")
    print(f"\n  criterion: any attributor rho > {PASS_RHO} on the primary outcome")
    print(f"  VERDICT: {'PASS' if res['PASS'] else 'FAIL'}")
    if not res["PASS"]:
        print("  -> Stages E-G still run (retrains are attribution-agnostic ground truth),")
        print("     but Stage H conclusions STOP and GATE1_FAIL.md is written.")
    return res


if __name__ == "__main__":
    main()
