"""P4 analysis: the closed-loop success reliability curve. Purely DESCRIPTIVE (no pass/fail).

Question the robotics-curation literature (CUPID et al.) leaves unanswered: closed-loop return is
the functional they curate against, but nobody validates it as GROUND TRUTH. Phase 1 found it
unusable at demo grain (30-episode success ceiling = -0.93 on C1). What WOULD it take?

Reliability = split-half Spearman of the success outcome across the 24 Stage-G masks, as a
function of
    episodes per estimate : 10 / 30 / 50 rollouts per probe task (subsampled from the 50 run)
                            = 30 / 90 / 150 DISTINCT episodes per cluster estimate
    seeds averaged        : 1 / 2 / 6
    cluster               : C1 (near-floor) vs C2 (mid-range)

NOTE (PHASE2_DEFECT.md): the brief said 90 rollouts/task. A LIBERO task has only 50 init states
and rollout.py indexes them ep % 50 under a deterministic policy, so 50 is every distinct episode
the instrument can supply. The ladder is 10/30/50, not 10/30/90.

Split-half over SEEDS (the reliability of an S-seed mean at a given episode budget): for a given
S, draw disjoint S|S seed halves, average each half's success rate per mask, Spearman the two
mask-vectors. Spearman-Brown is then used to EXTRAPOLATE the budget needed for a ceiling of
0.5 / 0.8 -- and every extrapolation is LABELLED as such.
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
from lds import spearman, logit_success  # noqa: E402

P2 = os.path.join(ROOT, "phase2")
P4 = os.path.join(P2, "runs/P4")
CLUSTERS = ["C1", "C2"]
ROLLOUT_LADDER = [10, 30, 50]         # per probe task
SEED_LADDER = [1, 2, 3, 6]   # S=1,2,3 measurable with 6 seeds (needs 2S); S=6 is SB-extrapolated


def load():
    rows = []
    if not os.path.isdir(P4):
        return pd.DataFrame()
    for d in sorted(os.listdir(P4)):
        p = os.path.join(P4, d, "success50.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        owner = r["cluster_of_task"]
        for t, v in r["per_task"].items():
            rows.append({"run": d, "mask_id": r["mask_id"], "seed": r["seed"],
                         "task": t, "cluster": owner[t], "succ": v})
    return pd.DataFrame(rows)


def rate(df, cluster, mask, seed, R):
    """success rate over the cluster's 3 probe tasks using the FIRST R episodes of each."""
    s = df[(df.cluster == cluster) & (df.mask_id == mask) & (df.seed == seed)]
    if len(s) != 3:
        return np.nan, 0
    hits = sum(sum(v[:R]) for v in s.succ)
    n = 3 * R
    return hits / n, n


def main():
    df = load()
    if df.empty:
        print("[P4] no results -- did not run")
        return
    seeds = sorted(df.seed.unique())
    masks = sorted(df.mask_id.unique())
    print(f"[P4] {df.run.nunique()} models, {len(masks)} masks, seeds={seeds}")

    grid, rows = {}, []
    for c in CLUSTERS:
        grid[c] = {}
        for R in ROLLOUT_LADDER:
            grid[c][R] = {}
            # per (mask, seed) logit-success at this episode budget
            tab = {}
            for m in masks:
                for s in seeds:
                    p, n = rate(df, c, m, s, R)
                    if np.isfinite(p):
                        tab[(m, s)] = float(logit_success(np.array([p]), np.array([n]))[0])
            for S in SEED_LADDER:
                if 2 * S > len(seeds):
                    # A split-half estimate of an S-seed mean needs 2S seeds. Where that is not
                    # available, Spearman-Brown EXTRAPOLATES from the largest measurable S.
                    # (S=6 needs 12 seeds and is therefore ALWAYS an extrapolation here.)
                    Smax = len(seeds) // 2
                    base = grid[c][R].get(Smax, {}).get("reliability")
                    k = S / Smax if Smax else None
                    est = (k * base / (1 + (k - 1) * base)
                           if (base is not None and np.isfinite(base) and base > 0 and k) else None)
                    grid[c][R][S] = {
                        "reliability": est,
                        "status": (f"EXTRAPOLATION (Spearman-Brown from measured S={Smax}); "
                                   f"a split-half of an S-seed mean needs {2*S} seeds, "
                                   f"only {len(seeds)} exist"),
                        "is_extrapolation": True, "extrapolated_from_S": Smax,
                        "episodes_per_estimate": 3 * R,
                    }
                    rows.append({"cluster": c, "rollouts_per_task": R,
                                 "episodes_per_estimate": 3 * R, "seeds_averaged": S,
                                 "reliability": est, "is_extrapolation": True})
                    continue
                vals, seen = [], set()
                for half in itertools.combinations(seeds, S):
                    rest = [x for x in seeds if x not in half]
                    for other in itertools.combinations(rest, S):
                        key = frozenset([half, other])
                        if key in seen:
                            continue
                        seen.add(key)
                        a = [np.mean([tab[(m, s)] for s in half if (m, s) in tab]) for m in masks]
                        b = [np.mean([tab[(m, s)] for s in other if (m, s) in tab]) for m in masks]
                        r = spearman(a, b)
                        if np.isfinite(r):
                            vals.append(float(r))
                rel = float(np.mean(vals)) if vals else np.nan
                grid[c][R][S] = {"reliability": rel, "n_splits": len(vals),
                                 "episodes_per_estimate": 3 * R, "status": "measured",
                                 "is_extrapolation": False}
                rows.append({"cluster": c, "rollouts_per_task": R, "episodes_per_estimate": 3 * R,
                             "seeds_averaged": S, "reliability": rel, "is_extrapolation": False})

    tab = pd.DataFrame(rows)
    tab.to_csv(f"{P2}/results/p4_reliability.csv", index=False)

    # ---- Spearman-Brown EXTRAPOLATION: what budget would reach a ceiling of 0.5 / 0.8?
    # SB: r_k = k*r_1 / (1 + (k-1)*r_1)  =>  k = r_t (1 - r_1) / (r_1 (1 - r_t))
    def k_needed(r1, target):
        if not np.isfinite(r1) or r1 <= 0 or r1 >= 1:
            return None
        if target >= 1:
            return None
        return float(target * (1 - r1) / (r1 * (1 - target)))

    extrap = {}
    for c in CLUSTERS:
        base = grid[c][max(ROLLOUT_LADDER)].get(1, {}).get("reliability")
        e = {"basis": f"measured 1-seed reliability at {max(ROLLOUT_LADDER)} rollouts/task "
                      f"({3*max(ROLLOUT_LADDER)} episodes)",
             "r1_measured": base}
        for target in (0.5, 0.8):
            k = k_needed(base, target)
            e[f"seeds_needed_for_{target}"] = (
                {"k_seeds": k, "LABEL": "EXTRAPOLATION (Spearman-Brown), not measured"}
                if k else {"k_seeds": None,
                           "LABEL": "NOT EXTRAPOLABLE: 1-seed reliability is <= 0"})
        extrap[c] = e

    # ---- THE ACTIONABLE COMPARISON: is reliability bought with EPISODES or with SEEDS?
    # Both axes cost compute. Which one actually moves the ceiling?
    marginal = {}
    for c in CLUSTERS:
        Rlo, Rhi = min(ROLLOUT_LADDER), max(ROLLOUT_LADDER)
        Smax = len(seeds) // 2                      # largest MEASURED S
        d_ep = (grid[c][Rhi][1]["reliability"] - grid[c][Rlo][1]["reliability"])
        d_sd = (grid[c][Rhi][Smax]["reliability"] - grid[c][Rhi][1]["reliability"])
        marginal[c] = {
            "episodes_axis": {
                "from": f"{3*Rlo} episodes (S=1)", "to": f"{3*Rhi} episodes (S=1)",
                "cost_multiple": Rhi / Rlo, "delta_reliability": d_ep,
            },
            "seeds_axis": {
                "from": f"S=1 (at {3*Rhi} episodes)", "to": f"S={Smax} (at {3*Rhi} episodes)",
                "cost_multiple": float(Smax), "delta_reliability": d_sd,
            },
            "seeds_beat_episodes": bool(d_sd > d_ep),
            "ratio_seed_gain_to_episode_gain": (d_sd / d_ep) if d_ep > 1e-9 else None,
        }

    out = {
        "stage": "P4", "descriptive_only": True,
        "HEADLINE_episodes_vs_seeds": marginal,
        "episode_ladder_rollouts_per_task": ROLLOUT_LADDER,
        "episode_ladder_episodes_per_cluster_estimate": [3 * R for R in ROLLOUT_LADDER],
        "DEFECT_NOTE": ("the brief's 90-rollout arm is impossible: a LIBERO task has 50 init "
                        "states and rollout.py indexes ep % 50 under a deterministic policy, so "
                        "episodes 50-89 are bit-identical replays. See PHASE2_DEFECT.md."),
        "seed_ladder": SEED_LADDER, "n_models": int(df.run.nunique()),
        "grid": grid, "extrapolation": extrap,
        "hard_instrument_limit": ("beyond 50 episodes/task no further INDEPENDENT closed-loop "
                                  "samples exist for that task; more must come from more tasks, "
                                  "more seeds, or new initial states."),
    }
    json.dump(out, open(f"{P2}/results/p4_success_reliability.json", "w"), indent=1, default=float)

    print("=" * 88)
    print("P4 -- CLOSED-LOOP SUCCESS RELIABILITY (split-half over 24 masks; DESCRIPTIVE)")
    print("=" * 88)
    for c in CLUSTERS:
        print(f"\n  {c} ({'near-floor' if c == 'C1' else 'mid-range'})")
        print(f"    {'rollouts/task':>14s} {'episodes':>9s} " +
              "".join(f"{'S=' + str(S):>10s}" for S in SEED_LADDER))
        for R in ROLLOUT_LADDER:
            cells = []
            for S in SEED_LADDER:
                g = grid[c][R][S]
                v = g["reliability"]
                if v is None or not np.isfinite(v):
                    cells.append(f"{'n/a':>10s}")
                else:
                    cells.append(f"{v:+9.3f}{'*' if g.get('is_extrapolation') else ' '}")
            print(f"    {R:14d} {3*R:9d} " + "".join(cells))
    print("\n  (* = EXTRAPOLATION, Spearman-Brown; a split-half of an S-seed mean needs 2S seeds)")
    print("\n  EPISODES vs SEEDS -- which axis actually buys reliability?")
    for c in CLUSTERS:
        m = marginal[c]
        e, sd = m["episodes_axis"], m["seeds_axis"]
        print(f"    {c}: {e['cost_multiple']:.0f}x the EPISODES ({e['from']} -> {e['to']}) "
              f"buys {e['delta_reliability']:+.3f}")
        print(f"        {sd['cost_multiple']:.0f}x the SEEDS    ({sd['from']} -> {sd['to']}) "
              f"buys {sd['delta_reliability']:+.3f}"
              f"   <- {'SEEDS WIN' if m['seeds_beat_episodes'] else 'episodes win'}")
    print("\n  Spearman-Brown EXTRAPOLATIONS (labelled as extrapolations, not measurements):")
    for c in CLUSTERS:
        e = extrap[c]
        r1 = e["r1_measured"]
        r1s = f"{r1:+.3f}" if r1 is not None and np.isfinite(r1) else "n/a"
        print(f"    {c}: 1-seed reliability at 150 episodes = {r1s}")
        for t in (0.5, 0.8):
            k = e[f"seeds_needed_for_{t}"]["k_seeds"]
            if k:
                print(f"        -> ceiling {t}: ~{k:.1f}x the seeds (EXTRAPOLATION)")
            else:
                print(f"        -> ceiling {t}: NOT EXTRAPOLABLE ({e[f'seeds_needed_for_{t}']['LABEL']})")
    print("=" * 88)


if __name__ == "__main__":
    main()
