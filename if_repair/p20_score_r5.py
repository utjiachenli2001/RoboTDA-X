"""R5 scoring -- preregistered in `p20_prereg_r5.md`. Zero GPU. Scored once.

THE TEST, as preregistered: the RANDOM arm's 10 replicates ARE the null distribution. A fixed arm
is declared different if its seed-mean falls outside the 2.5th-97.5th percentile of the 10 random
replicate means. No distributional assumption -- deliberately, because success rates are binomial
and losses are not, and the two must be judged the same way.

SUCCESS IS PRIMARY. Held-out loss is reported but cannot carry the headline: campaign U just spent
11,386 retrainings showing that this exact loss has a measurability horizon, so adjudicating a
selection claim on it alone would be circular.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import p20_r5 as R5  # noqa: E402

OUT = os.path.join(R5.RESULTS, "confirm_r5.csv")


def load():
    loss, succ = {}, {}
    for f in sorted(glob.glob(os.path.join(R5.CKPT, "*.json"))):
        if f.endswith(".rollout.json"):
            d = json.load(open(f))
            succ.setdefault(d["arm"], {})[d["seed"]] = d["success_rate"]
        else:
            d = json.load(open(f))
            loss.setdefault(d["arm"], {})[d["seed"]] = d["plain_loss"]
    return loss, succ


def band(per_arm, arms):
    """The RANDOM replicates' arm-means are the null distribution."""
    vals = [float(np.mean(list(per_arm[a].values()))) for a in arms if a in per_arm]
    return np.percentile(vals, [2.5, 97.5]), vals


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists -- R5 is SCORED ONCE. Refusing to overwrite.")
    loss, succ = load()
    rnd = sorted(a for a in loss if a.startswith("RANDOM"))
    for name, tbl, lbl in (("loss", loss, "held-out loss (lower better)"),
                           ("success", succ, "rollout success (higher better)")):
        (lo, hi), vals = band(tbl, rnd)
        print(f"\n=== {lbl} ===")
        print(f"  RANDOM null: n={len(vals)} mean={np.mean(vals):.4f} "
              f"band[2.5,97.5]=[{lo:.4f}, {hi:.4f}]")
        for arm in ("DM-top", "GD-top", "DM-bottom"):
            if arm not in tbl:
                continue
            v = float(np.mean(list(tbl[arm].values())))
            n = len(tbl[arm])
            out = "OUTSIDE" if (v < lo or v > hi) else "inside"
            print(f"  {arm:10s} n={n} mean={v:.4f}  -> {out} the random band")
    res = {"loss": {a: float(np.mean(list(v.values()))) for a, v in loss.items()},
           "success": {a: float(np.mean(list(v.values()))) for a, v in succ.items()},
           "n_seeds_loss": {a: len(v) for a, v in loss.items()},
           "n_seeds_success": {a: len(v) for a, v in succ.items()}}
    json.dump(res, open(OUT.replace(".csv", ".json"), "w"), indent=1)
    with open(OUT, "w") as fh:
        fh.write("arm,mean_loss,mean_success,n_seeds_loss,n_seeds_success\n")
        for a in sorted(loss):
            fh.write(f"{a},{res['loss'][a]:.6f},{res['success'].get(a, float('nan')):.6f},"
                     f"{res['n_seeds_loss'][a]},{res['n_seeds_success'].get(a, 0)}\n")
    print(f"\n[r5] wrote {OUT}")


if __name__ == "__main__":
    main()
