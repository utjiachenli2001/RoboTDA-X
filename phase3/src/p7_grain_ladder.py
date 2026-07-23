"""P7 -- the GRAIN-RESOLUTION LADDER. Zero retrains, zero GPU.

Phases 1-2 established two endpoints: attribution at CLUSTER grain (15 demos) is partially
faithful; at DEMO grain (1 demo) it is not. P7 asks the question those two endpoints pose:
WHERE does it break? It interpolates the grain and measures the resolution limit.

THE COARSENED PREDICTOR (preregistered). Partition each cluster's 15 training demos into 15/g
groups of size g. An attributor that can only resolve GROUPS -- with no within-group resolution
-- would predict a mask's outcome as

    group_score(G)          = sum of the per-demo attribution over the demos in G
    predicted_mask_score(M) = sum over ALL groups G of  group_score(G) * |G intersect M| / g

The (|G ∩ M| / g) factor is the fraction of the group that survives in the mask: the coarse
attributor knows the group's total worth and assumes it is spread evenly inside the group.

  * at g = 1  this is EXACTLY the Phase-2 demo predictor (sum of per-demo scores over the mask)
  * at g = 15 this is the cluster predictor with fractional inclusion

so the ladder runs continuously between the two endpoints the study has already measured, and
the SAME ground truth (the 24 Stage-G masks, 6-seed mean) and the SAME ceilings judge every rung.
The ceiling is a property of the ground truth, not of the predictor, so it does not move with g.

GROUPING RULES (both preregistered)
  (a) PRIMARY   random within cluster, default_rng(707), chunked into blocks of g
  (b) SECONDARY complete-linkage on the Phase-1 DTW matrix, balanced to exact size g

If (b) beats (a) at the same g, group STRUCTURE carries signal -- i.e. the attributor is not
merely resolution-limited, it is resolving the wrong thing.

PREREGISTERED READ-OUT: per focal target (C1, C5), the SMALLEST g whose BEST attributor reaches
LDS >= 0.5 x ceiling with one-sided p < 0.025 (Bonferroni-2). Rule (a) is primary.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_FIGURES, RESULTS, P2_RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci  # noqa: E402

GRAINS = [1, 3, 5, 15]
RULE_A_SEED = 707
FOCAL = ["C1", "C5"]
ATTRS = ["IF", "TRAK", "TracIn"]
ALPHA = 0.025


# ---------------------------------------------------------------- grouping rules
def groups_random(demos, g, seed=RULE_A_SEED):
    """(a) PRIMARY: random within cluster, fixed seed, chunked into consecutive blocks of g."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(demos))
    ids = [demos[i] for i in perm]
    return [ids[i:i + g] for i in range(0, len(ids), g)]


def groups_dtw(demos, g, D):
    """(b) SECONDARY: complete-linkage on DTW, cut to 15/g groups, BALANCED to exact size g.

    Complete linkage does not give equal-sized groups, so the preregistered deterministic
    post-pass balances them: while some group is oversized, take the demo FARTHEST from its own
    group (max mean DTW to its groupmates) and move it into the nearest undersized group (min
    mean DTW). All ties are broken lexicographically by demo id -- no RNG anywhere.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    n, k = len(demos), len(demos) // g
    if k == 1:
        return [list(demos)]
    if g == 1:
        return [[d] for d in demos]
    Z = linkage(squareform(D, checks=False), method="complete")
    lab = fcluster(Z, t=k, criterion="maxclust")
    grp = {c: [demos[i] for i in range(n) if lab[i] == c] for c in sorted(set(lab))}
    # ensure exactly k groups (fcluster can return fewer on ties)
    while len(grp) < k:
        big = max(grp, key=lambda c: (len(grp[c]), -min(grp[c] and [0])))
        new = max(grp) + 1
        grp[new] = [sorted(grp[big])[-1]]
        grp[big] = [d for d in grp[big] if d != grp[new][0]]

    idx = {d: i for i, d in enumerate(demos)}

    def mean_d(d, members):
        m = [x for x in members if x != d]
        return float(np.mean([D[idx[d], idx[x]] for x in m])) if m else 0.0

    guard = 0
    while any(len(v) != g for v in grp.values()):
        guard += 1
        if guard > 1000:
            raise RuntimeError("DTW group balancing did not converge")
        over = sorted([c for c in grp if len(grp[c]) > g],
                      key=lambda c: (-len(grp[c]), sorted(grp[c])[0]))
        under = sorted([c for c in grp if len(grp[c]) < g],
                       key=lambda c: (len(grp[c]), sorted(grp[c])[0]))
        if not over or not under:
            break
        src = over[0]
        # the demo farthest from its own group (ties -> lexicographic demo id)
        mover = sorted(grp[src], key=lambda d: (-mean_d(d, grp[src]), d))[0]
        # the nearest undersized group (ties -> lexicographic first member)
        dst = sorted(under, key=lambda c: (mean_d(mover, grp[c]), sorted(grp[c])[0]))[0]
        grp[src].remove(mover)
        grp[dst].append(mover)
    return [sorted(v) for _, v in sorted(grp.items())]


# ---------------------------------------------------------------- coarsened predictor
def coarse_pred(mask_demos, all_groups, scores, g):
    """sum over groups of group_score * (fraction of the group present in the mask)."""
    M = set(mask_demos)
    tot = 0.0
    for grp in all_groups:
        gs = sum(scores.get(d, 0.0) for d in grp)
        tot += gs * (len(M & set(grp)) / float(g))
    return tot


def build_groups(rule, g, by_c, DTW, demo_ids):
    """All groups across all 9 clusters, at grain g, under the given rule."""
    out = []
    for c, demos in by_c.items():
        if rule == "random":
            out.extend(groups_random(demos, g))
        else:
            ii = [demo_ids.index(d) for d in demos]
            D = DTW[np.ix_(ii, ii)]
            out.extend(groups_dtw(demos, g, D))
    return out


def main():
    # ---- attribution (PRIMARY: the archived table, default ridge -- as preregistered)
    infl = pd.read_parquet(os.path.join(RESULTS, "influence_table.parquet"))
    infl = infl[infl.functional == "plain"]

    # ---- ground truth: the 24 Stage-G masks, 6-seed mean, held-out L2 (Phase-2 P1 primary)
    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    G6 = pd.read_parquet(os.path.join(P2_RESULTS, "stage_G6_outcomes.parquet"))
    p1 = json.load(open(os.path.join(P2_RESULTS, "p1_demo_grain.json")))

    clusters = dataset.clusters()
    _, by_c = dataset.train_pool()
    ceil = {t: p1["all_targets"][t]["neg_plain_loss"]["ceiling_6seed_SB"] for t in clusters}
    obs = {t: G6[G6.target == t].groupby("mask_id")["neg_plain_loss"].mean().to_dict()
           for t in clusters}

    sim = np.load(os.path.join(RESULTS, "similarity.npz"))
    DTW = sim["dtw"]
    demo_ids = list(sim["demo_ids"])

    # sanity: the corpus order in similarity.npz must match the training pool
    tp, _ = dataset.train_pool()
    assert set(demo_ids) == set(tp), "similarity.npz demo_ids != training pool"

    rows = []
    for rule in ("random", "dtw"):
        for g in GRAINS:
            groups = build_groups(rule, g, by_c, DTW, demo_ids)
            assert all(len(x) == g for x in groups), f"bad group sizes at g={g} rule={rule}"
            assert sum(len(x) for x in groups) == 135
            for t in clusters:
                for a in ATTRS:
                    sc = infl[(infl.attributor == a) & (infl.target == t)]
                    sc = dict(zip(sc.demo_id, sc.score))
                    pred = np.array([coarse_pred(m["demos"], groups, sc, g) for m in gman])
                    out = np.array([obs[t].get(m["mask_id"], np.nan) for m in gman])
                    ok = np.isfinite(out)
                    rho = spearman(pred[ok], out[ok])
                    p = spearman_p_onesided(rho, int(ok.sum()))
                    lo, hi = bootstrap_spearman_ci(pred[ok], out[ok])
                    rows.append({
                        "rule": rule, "g": g, "target": t, "attributor": a, "focal": t in FOCAL,
                        "lds": rho, "p_onesided": p, "n_masks": int(ok.sum()),
                        "ci_lo": lo, "ci_hi": hi,
                        "ceiling": ceil[t], "bar_half_ceiling": 0.5 * ceil[t],
                        "ratio": rho / ceil[t] if ceil[t] > 0 else np.nan,
                        "PASS": bool(np.isfinite(rho) and rho >= 0.5 * ceil[t]
                                     and np.isfinite(p) and p < ALPHA),
                    })
            print(f"[P7] rule={rule:6s} g={g:2d}: {len(groups)} groups", flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(P3_RESULTS, "p7_grain_ladder.csv"), index=False)

    # ---------------- PREREGISTERED READ-OUT: smallest qualifying g, rule (a), focal targets
    readout = {}
    for t in FOCAL:
        s = T[(T.rule == "random") & (T.target == t)]
        qual = sorted(s[s.PASS].g.unique())
        per_g = {}
        for g in GRAINS:
            sg = s[s.g == g]
            b = sg.loc[sg.lds.idxmax()]
            per_g[str(g)] = {"best_attributor": str(b.attributor), "lds": float(b.lds),
                             "ratio": float(b.ratio), "p_onesided": float(b.p_onesided),
                             "bar": float(b.bar_half_ceiling), "PASS": bool(b.PASS)}
        readout[t] = {
            "ceiling_6seed": ceil[t], "bar_half_ceiling": 0.5 * ceil[t],
            "per_grain": per_g,
            "SMALLEST_QUALIFYING_g": int(qual[0]) if qual else None,
            "qualifying_grains": [int(x) for x in qual],
            "NO_g_QUALIFIES": len(qual) == 0,
        }

    # ---------------- rule (b) vs rule (a): does group STRUCTURE carry signal?
    contrast = {}
    for t in clusters:
        contrast[t] = {}
        for g in GRAINS:
            a_best = T[(T.rule == "random") & (T.target == t) & (T.g == g)].lds.max()
            d_best = T[(T.rule == "dtw") & (T.target == t) & (T.g == g)].lds.max()
            contrast[t][str(g)] = {"random_best_lds": float(a_best),
                                   "dtw_best_lds": float(d_best),
                                   "dtw_minus_random": float(d_best - a_best)}
    dtw_wins = {str(g): int(sum(1 for t in clusters if contrast[t][str(g)]["dtw_minus_random"] > 0))
                for g in GRAINS}
    dtw_mean = {str(g): float(np.mean([contrast[t][str(g)]["dtw_minus_random"] for t in clusters]))
                for g in GRAINS}

    any_qual = any(readout[t]["SMALLEST_QUALIFYING_g"] is not None for t in FOCAL)
    out = {
        "stage": "P7 grain-resolution ladder",
        "n_retrains": 0, "n_gpu_h": 0,
        "grains": GRAINS,
        "grouping_rule_primary": f"random within cluster, default_rng({RULE_A_SEED})",
        "grouping_rule_secondary": "complete-linkage on the Phase-1 DTW matrix, balanced to size g",
        "predictor": ("group_score = sum of per-demo attribution over the group; "
                      "predicted_mask_score = sum_groups group_score * |G n M| / g"),
        "ground_truth": ("phase2/results/stage_G6_outcomes.parquet, 6-seed mean, neg_plain_loss "
                         "(held-out L2), the 24 existing Stage-G demo masks"),
        "ceilings_reused": ceil,
        "endpoints_check": {
            "g1_is_the_phase2_demo_predictor": True,
            "g15_is_the_cluster_predictor_with_fractional_inclusion": True,
        },
        "PREREGISTERED_READOUT_focal": readout,
        "rule_b_vs_a_contrast_all_targets": contrast,
        "dtw_beats_random_n_targets_of_9": dtw_wins,
        "dtw_minus_random_mean_over_9_targets": dtw_mean,
        "VERDICT": None,
    }
    if not any_qual:
        out["VERDICT"] = "NO_GRAIN_QUALIFIES"
        out["VERDICT_TEXT"] = (
            "No grain in {1,3,5,15} reaches half of the seed-ensembled ceiling on either focal "
            "target with p < 0.025. Even the fully-coarsened predictor (g=15, i.e. the cluster "
            "predictor) fails to rank the DEMO-grain masks. The resolution limit is therefore not "
            "located between 1 and 15 demos: coarsening the attribution does not rescue it, so "
            "the failure is not one of RESOLUTION but of the attribution SIGNAL itself.")
    else:
        gs = {t: readout[t]["SMALLEST_QUALIFYING_g"] for t in FOCAL}
        out["VERDICT"] = "RESOLUTION_LIMIT_LOCATED"
        out["VERDICT_TEXT"] = (
            f"The measured resolution limit (smallest qualifying grain, rule (a)): {gs}. Below "
            f"this grain the attributor cannot predict counterfactual outcomes; at or above it, "
            f"it can. This is the paper's cleanest positive framing of the null: attribution in "
            f"this regime carries information about GROUPS of demonstrations, not about "
            f"individual demonstrations.")

    L.atomic_write_json(os.path.join(P3_RESULTS, "p7_grain_ladder.json"), out)

    # ---------------- print
    print("\n" + "=" * 100)
    print("P7 -- GRAIN LADDER (rule (a) random; outcome = held-out L2, 6-seed ground truth)")
    print("=" * 100)
    print(f"{'target':7s} {'g':>3s} {'best attr':>10s} {'LDS':>8s} {'bar':>7s} {'ratio':>6s} "
          f"{'p1':>7s}  {'PASS':>5s}")
    for t in clusters:
        for g in GRAINS:
            s = T[(T.rule == "random") & (T.target == t) & (T.g == g)]
            b = s.loc[s.lds.idxmax()]
            mark = " FOCAL" if t in FOCAL else ""
            print(f"{t:7s} {g:3d} {b.attributor:>10s} {b.lds:+8.3f} {b.bar_half_ceiling:7.3f} "
                  f"{b.ratio:6.2f} {b.p_onesided:7.4f}  {str(bool(b.PASS)):>5s}{mark}")
        print("-" * 100)
    print("\nPREREGISTERED READ-OUT (focal targets, rule (a)):")
    for t in FOCAL:
        r = readout[t]
        print(f"  {t}: smallest qualifying g = {r['SMALLEST_QUALIFYING_g']} "
              f"(qualifying: {r['qualifying_grains']})")
    print("\nrule (b) DTW-grouped minus rule (a) random-grouped, best LDS, mean over 9 targets:")
    for g in GRAINS:
        print(f"  g={g:2d}: {dtw_mean[str(g)]:+.4f}  (DTW wins on {dtw_wins[str(g)]}/9 targets)")
    print("=" * 100)
    print(f"VERDICT: {out['VERDICT']}")
    print(out["VERDICT_TEXT"])
    print("=" * 100)


if __name__ == "__main__":
    main()
