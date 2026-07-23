"""Gate-0 diagnostic: does co-training move the HELD-OUT ACTION LOSS?

The gate's outcome is closed-loop success on 200 episodes, whose training-seed SD turned out to
be ~16 points. The held-out action loss is a far more sensitive probe of the same models (it
averages thousands of frames instead of 200 binary episodes), so it can separate

    "co-training does nothing"                       (loss margin ~ 0 too)
from
    "co-training helps, but success at n=5 seeds cannot resolve it"   (loss margin < 0, consistent)

This re-uses the ALREADY-TRAINED Stage-B checkpoints -- no new training, no rollouts.
Writes results/stage_B_loss_diagnostic.json.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS, RESULTS
import evaluate as EV

SEEDS = [101, 102, 103, 104, 105]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="C1")
    a = ap.parse_args()
    from scipy import stats
    out = {}
    for tgt in a.targets.split(","):
        rows, margins = [], []
        for s in SEEDS:
            r = {}
            for cond in ("target", "cotrain"):
                p = os.path.join(RUNS, "stage_B", f"{tgt}_{cond}_s{s}", "final.pt")
                if not os.path.exists(p):
                    continue
                m = EV.load_model(p, device="cuda")
                r[cond] = EV.heldout_losses(m, device="cuda")[tgt]
            if len(r) != 2:
                continue
            d = r["cotrain"]["plain_loss"] - r["target"]["plain_loss"]
            margins.append(d)
            rows.append({"seed": s,
                         "target_only_plain_loss": r["target"]["plain_loss"],
                         "cotrain_plain_loss": r["cotrain"]["plain_loss"],
                         "loss_margin": d,
                         "target_only_transport": r["target"]["transport_loss"],
                         "cotrain_transport": r["cotrain"]["transport_loss"],
                         "target_only_interaction": r["target"]["interaction_loss"],
                         "cotrain_interaction": r["cotrain"]["interaction_loss"]})
            print(f"  {tgt} seed {s}: plain loss  target-only={r['target']['plain_loss']:.4f}  "
                  f"co-train={r['cotrain']['plain_loss']:.4f}  margin={d:+.4f}", flush=True)
        if len(margins) > 1:
            m = np.array(margins)
            t, p2 = stats.ttest_1samp(m, 0.0)
            p1 = p2 / 2 if t < 0 else 1 - p2 / 2      # H1: co-train LOWERS the loss
            out[tgt] = {"per_seed": rows, "mean_loss_margin": float(m.mean()),
                        "sd_loss_margin": float(m.std(ddof=1)), "t": float(t),
                        "p_onesided_cotrain_lowers_loss": float(p1),
                        "n_seeds": len(m),
                        "interpretation": "negative margin = co-training LOWERS the target's "
                                          "held-out action loss (helps)"}
            print(f"\n  {tgt}: mean loss margin = {m.mean():+.4f} (SD {m.std(ddof=1):.4f}), "
                  f"one-sided p(co-train lowers loss) = {p1:.4f}")
    json.dump(out, open(os.path.join(RESULTS, "stage_B_loss_diagnostic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
