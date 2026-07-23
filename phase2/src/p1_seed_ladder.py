"""P1 supplement: the seed ladder -- the direct answer to "attribution, or the noise floor?"

For S = 1..6 seeds averaged into the ground truth, on the SAME 24 Stage-G masks:
    ceiling(S)  = split-half reliability of an S-seed mean (Spearman-Brown corrected to S)
    LDS(S)      = Spearman(predicted mask score, S-seed-mean outcome)
averaged over many random draws of which S of the 6 seeds are used.

If Phase 1's Gate-1 failure was the NOISE FLOOR, LDS(S) must climb with the ceiling.
If LDS(S) plateaus while ceiling(S) climbs, the residual gap is ATTRIBUTION ERROR, and the
negative result is intrinsic to the attributor -- not a measurement artefact.

Descriptive supplement to the preregistered P1 criterion; it changes no verdict.
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
from lds import spearman, mask_pred_score  # noqa: E402

P2 = os.path.join(ROOT, "phase2")
SEEDS = [401, 402, 403, 404, 405, 406]
ATTRS = ["IF", "TRAK", "TracIn"]
TARGETS = ["C1", "C5"]
OC = "neg_plain_loss"
RNG = np.random.default_rng(0)


def main():
    df = pd.read_parquet(f"{P2}/results/stage_G6_outcomes.parquet")
    man = json.load(open(f"{RESULTS}/demo_mask_manifest.json"))
    masks = man["masks"]
    inf = pd.read_parquet(f"{RESULTS}/influence_table.parquet")

    out = {"outcome": OC, "n_masks": len(masks), "seeds": SEEDS, "ladder": {}}
    for tgt in TARGETS:
        piv = df[df.target == tgt].pivot_table(index="mask_id", columns="seed", values=OC)
        piv = piv[SEEDS].dropna()
        mids = list(piv.index)

        pred = {}
        for a in ATTRS:
            s = inf[(inf.attributor == a) & (inf.functional == "plain") & (inf.target == tgt)]
            sc = dict(zip(s.demo_id, s.score))
            byid = {m["mask_id"]: mask_pred_score(sc, m["demos"]) for m in masks}
            pred[a] = np.array([byid[m] for m in mids])

        out["ladder"][tgt] = {}
        for S in range(1, 7):
            # ceiling of an S-seed mean: disjoint S-vs-S halves need 2S <= 6, so measure the
            # split-half at floor(6/2)=3 max; for each S use all disjoint S|S pairs available
            ceils = []
            for half in itertools.combinations(SEEDS, S):
                rest = [s for s in SEEDS if s not in half]
                if len(rest) < S:
                    continue
                for other in itertools.combinations(rest, S):
                    r = spearman(piv[list(half)].mean(1).values, piv[list(other)].mean(1).values)
                    if np.isfinite(r):
                        ceils.append(float(r))
            rS = float(np.mean(ceils))                     # reliability of an S-seed mean
            # Spearman-Brown is not needed: rS IS the S-vs-S split reliability of an S-seed mean.

            ldss = {a: [] for a in ATTRS}
            draws = list(itertools.combinations(SEEDS, S))
            for combo in draws:
                mean_s = piv[list(combo)].mean(1).values
                for a in ATTRS:
                    r = spearman(pred[a], mean_s)
                    if np.isfinite(r):
                        ldss[a].append(float(r))
            out["ladder"][tgt][S] = {
                "ceiling_SvS": rS, "n_ceiling_pairs": len(ceils),
                "n_draws": len(draws),
                "lds_mean": {a: float(np.mean(ldss[a])) for a in ATTRS},
                "lds_sd": {a: float(np.std(ldss[a])) for a in ATTRS},
                "best_lds_mean": float(max(np.mean(ldss[a]) for a in ATTRS)),
            }

    json.dump(out, open(f"{P2}/results/p1_seed_ladder.json", "w"), indent=1, default=float)

    print("=" * 88)
    print("P1 SUPPLEMENT -- SEED LADDER (held-out L2, same 24 masks)")
    print("  ceiling = reliability of an S-seed mean (S-vs-S split half)")
    print("  LDS(S)  = mean Spearman(attribution, S-seed-mean outcome) over all draws of S seeds")
    print("=" * 88)
    for tgt in TARGETS:
        print(f"\n  {tgt}:")
        print(f"    {'S':>2s} {'ceiling':>8s} {'IF':>7s} {'TRAK':>7s} {'TracIn':>7s} {'best':>7s} "
              f"{'best/ceil':>10s}")
        for S in range(1, 7):
            e = out["ladder"][tgt][S]
            c = e["ceiling_SvS"]
            l = e["lds_mean"]
            b = e["best_lds_mean"]
            ratio = b / c if c > 0 else float("nan")
            cs = f"{c:8.3f}" if np.isfinite(c) else "     n/a"
            print(f"    {S:2d} {cs} {l['IF']:+7.3f} {l['TRAK']:+7.3f} {l['TracIn']:+7.3f} "
                  f"{b:+7.3f} {ratio:10.2f}")
    print("\n" + "=" * 88)
    print("READ: if the ceiling column climbs but the LDS columns stay flat, the residual gap is")
    print("      ATTRIBUTION ERROR, not measurement noise.")
    print("=" * 88)


if __name__ == "__main__":
    main()
