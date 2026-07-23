"""STAGE C -- in-target quantity sweep (spec §5, primary moderator for RQ2-quantity).

Q in {15, 50, 490} in-target (libero_goal) demos.
  * supersets SHARE the Q=15 selection (Q=15 subset of Q=50 subset of Q=490)
  * Q=490 is the spec's "Q=500 (full suite)" MINUS C1's 10 held-out probe demos, which must
    stay unseen -- the spec itself reserves "the full suite minus C1's held-out 10" for this
    stage, so 490 is the largest legitimate value. Labelled Q=490 everywhere, not 500.

Conditions {target-only(Q), co-train(Q + the fixed 120 outsider demos)} x 3 seeds = 18 runs.
Each model is evaluated on EVERY task of the goal suite x 20 rollouts (200 episodes).
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RUNS, RESULTS
import dataset
import orchestrator as O
from clusters import suite_task_names

SEEDS = [101, 102, 103]
QS = [15, 50, 490]
N_ROLLOUTS = 20
SUITE = "libero_goal"


def goal_demo_ladder():
    """Nested selections: Q=15 (the C1 train pool) subset Q=50 subset Q=490.

    Continues the SAME deterministic rule as the corpus (alphabetical tasks, round-robin,
    lowest demo index first), skipping C1's 10 held-out demos.
    """
    _, by_c = dataset.train_pool()
    _, ho_c = dataset.heldout_pool()
    base = list(by_c["C1"])                       # the 15, in corpus order
    held = set(ho_c["C1"])
    tasks = sorted(suite_task_names(SUITE))
    taken = set(base)
    ladder = {15: list(base)}
    # round-robin over remaining demo indices
    extra = []
    for rank in range(50):
        for t in tasks:
            d = dataset.did(SUITE, t, f"demo_{rank}")
            if d in taken or d in held:
                continue
            extra.append(d)
            taken.add(d)
    ladder[50] = base + extra[:50 - len(base)]
    ladder[490] = base + extra
    for q in QS:
        assert len(ladder[q]) == q, f"Q={q}: got {len(ladder[q])}"
        assert not (set(ladder[q]) & held), f"Q={q} leaks held-out demos"
    assert set(ladder[15]) <= set(ladder[50]) <= set(ladder[490]), "ladder not nested"
    return ladder


def outsiders():
    """The fixed 120 outsider demos (C2..C9 train pools)."""
    _, by_c = dataset.train_pool()
    return [d for c in dataset.clusters() if c != "C1" for d in by_c[c]]


def build_jobs():
    ladder = goal_demo_ladder()
    outs = outsiders()
    assert len(outs) == 120, len(outs)
    jobs = []
    for q in QS:
        for cond in ("target", "cotrain"):
            demos = ladder[q] if cond == "target" else ladder[q] + outs
            for s in SEEDS:
                jobs.append({
                    "run_dir": os.path.join(RUNS, "stage_C", f"Q{q}_{cond}_s{s}"),
                    "demos": demos, "seed": s, "n_rollouts": N_ROLLOUTS,
                    "eval": "cluster_tasks", "target": "C1", "workers": 12,
                    "Q": q, "cond": cond,
                })
    return jobs


def analyze():
    rows = []
    for q in QS:
        for cond in ("target", "cotrain"):
            for s in SEEDS:
                p = os.path.join(RUNS, "stage_C", f"Q{q}_{cond}_s{s}", "cluster_eval.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                rows.append({"Q": q, "cond": cond, "seed": s,
                             "success_pct": 100 * d["cluster_success"],
                             "n_episodes": d["n_episodes"]})
    out = {"Q_values": QS, "seeds": SEEDS, "runs": rows, "by_Q": {}}
    for q in QS:
        t = [r["success_pct"] for r in rows if r["Q"] == q and r["cond"] == "target"]
        c = [r["success_pct"] for r in rows if r["Q"] == q and r["cond"] == "cotrain"]
        # paired by seed
        margins = []
        for s in SEEDS:
            tt = [r["success_pct"] for r in rows if r["Q"] == q and r["cond"] == "target" and r["seed"] == s]
            cc = [r["success_pct"] for r in rows if r["Q"] == q and r["cond"] == "cotrain" and r["seed"] == s]
            if tt and cc:
                margins.append(cc[0] - tt[0])
        out["by_Q"][str(q)] = {
            "target_only_pct_mean": float(np.mean(t)) if t else None,
            "target_only_pct_sd": float(np.std(t, ddof=1)) if len(t) > 1 else None,
            "cotrain_pct_mean": float(np.mean(c)) if c else None,
            "cotrain_pct_sd": float(np.std(c, ddof=1)) if len(c) > 1 else None,
            "margin_pts_mean": float(np.mean(margins)) if margins else None,
            "margin_pts_sd": float(np.std(margins, ddof=1)) if len(margins) > 1 else None,
            "per_seed_margins": margins, "n_runs": len(t) + len(c),
        }
    json.dump(out, open(os.path.join(RESULTS, "stage_C_quantity.json"), "w"), indent=1)
    print("\n=== STAGE C: co-train margin vs in-target quantity Q ===")
    print(f"{'Q':>5} {'target-only %':>15} {'co-train %':>13} {'margin (pts)':>14}")
    for q in QS:
        b = out["by_Q"][str(q)]
        if b["target_only_pct_mean"] is None:
            print(f"{q:>5}   (did not run)")
            continue
        print(f"{q:>5} {b['target_only_pct_mean']:>10.1f} +- {b['target_only_pct_sd'] or 0:<4.1f} "
              f"{b['cotrain_pct_mean']:>8.1f} {b['margin_pts_mean']:>+13.1f}")
    return out


def intrusion_sweep():
    """Spec §5: TracIn toward C1 (plain functional) on each CO-TRAIN model's checkpoints ->
    outsider intrusion rate (fraction of the 120 outsiders above the median insider) at each Q.

    Insiders = that model's Q goal demos; outsiders = the fixed 120.
    """
    import attribution as ATT
    ladder = goal_demo_ladder()
    outs = set(outsiders())
    res = {}
    for q in QS:
        rates = []
        for s in SEEDS:
            rd = os.path.join(RUNS, "stage_C", f"Q{q}_cotrain_s{s}")
            if not os.path.exists(os.path.join(rd, "final.pt")):
                continue
            demos = json.load(open(os.path.join(rd, "demos.json")))["demos"]
            sc = ATT.tracin_scores(rd, demos, [("C1", "plain")])[("C1", "plain")]
            ins = np.array([sc[d] for d in ladder[q] if d in sc])
            out = np.array([sc[d] for d in demos if d in outs and d in sc])
            if len(ins) == 0 or len(out) == 0:
                continue
            med = float(np.median(ins))
            rates.append({"seed": s,
                          "intrusion_above_median": float((out > med).mean()),
                          "intrusion_above_p75": float((out > np.percentile(ins, 75)).mean()),
                          "n_insiders": int(len(ins)), "n_outsiders": int(len(out))})
            print(f"[stage_C] Q={q} seed={s}: intrusion="
                  f"{rates[-1]['intrusion_above_median']:.3f}", flush=True)
        if rates:
            v = [r["intrusion_above_median"] for r in rates]
            res[str(q)] = {
                "intrusion_above_median": float(np.mean(v)),
                "intrusion_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "intrusion_above_p75": float(np.mean([r["intrusion_above_p75"] for r in rates])),
                "per_seed": rates, "attributor": "TracIn", "functional": "plain",
            }
    json.dump(res, open(os.path.join(RESULTS, "stage_C_intrusion.json"), "w"), indent=1)
    print("\n=== STAGE C: outsider intrusion vs Q (TracIn -> C1) ===")
    for q in QS:
        r = res.get(str(q))
        print(f"  Q={q:>3}: intrusion above median insider = "
              f"{r['intrusion_above_median']:.3f} +- {r['intrusion_sd']:.3f}" if r
              else f"  Q={q:>3}: did not run")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--skip_intrusion", action="store_true")
    a = ap.parse_args()
    if not a.analyze_only:
        O.run_jobs(build_jobs(), "stage_C")
    analyze()
    if not a.skip_intrusion:
        intrusion_sweep()


if __name__ == "__main__":
    main()
