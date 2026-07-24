"""W2 -- the duel design: a second measurement instrument for the same question.

THE PROBLEM WITH LDS HERE. A random 68-of-135 mask changes about half the corpus, so the two
estimators predict nearly the same thing for it and most outcome variation carries no
information about which is right. Power is spent on masks where they agree.

THE DESIGN. Build pairs of masks differing in EXACTLY ONE DEMO, chosen where the two estimators
disagree about which of the two demos matters more:

  - a and b are in the SAME cluster, so both masks keep the 8/8/8/8/8/7/7/7/7 stratification and
    remain valid draws from the same design family as every other campaign.
  - selection requires sign(rank_relatif(a) - rank_relatif(b)) != sign(rank_graddot(a) - ...),
    with BOTH within-cluster rank gaps >= MIN_RANK_GAP, so both estimators are confident and
    they contradict each other.
  - train base+{a} and base+{b} at MATCHED seed slots, read the sign of the C5 held-out loss
    difference.
  - each duel is one binary datum: which estimator's ordering was right. Over n duels that is a
    one-sided SIGN TEST against a clean binomial null -- no ceiling, no Spearman, no n=24
    resolution limit.

WHY MATCHED SEEDS ARE LEGITIMATE HERE AND NOT A RE-RUN OF B5. B5 retired common random seeds for
the general mask-outcome-variance question because the binding noise is the mask x init
interaction, which seeding does not remove. Within a duel the two masks share 67 of 68 demos, so
most of that interaction is SHARED and differences out. That is an empirical claim, so the pilot
MEASURES it rather than asserting it.

HONEST CAVEAT, to be carried into every write-up: the duel measures LOCAL PAIRWISE ORDERING on
deliberately selected disagreement pairs. It is not the LDS estimand and its mask distribution is
not the uniform one, so it must NEVER be pooled with campaign M's numbers. It answers a different
and more interesting question: when the two estimators disagree, who is right?

Stages:
  --stage select   build and freeze the duel manifest (no GPU). Commit before training.
  --stage pilot    analyse the pilot duels, apply the KILL RULE.
  --stage full     analyse the full arm (only if the pilot passes).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402
from if_repair import p7_pooled_oos as P7  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MANIFEST = os.path.join(RESULTS, "p7_duel_manifest.json")

# ---------------------------------------------------------------- FROZEN selection rule
TARGET = "C5"

# MIN_RANK_GAP was set to 4 (of a possible 14 within a 15-demo cluster) on FEASIBILITY grounds,
# decided and frozen before any duel outcome existed. The brief assumed 32-40 duels were
# available; they are not, and the reason is a finding in its own right:
#
#   RelatIF and GradDot_dmean rank-correlate 0.547 over the 135 demos (0.566 within cluster),
#   so they mostly AGREE. Of the 9 x C(15,2) = 945 possible within-cluster pairs, only 6.6%
#   disagree at gap >= 4, and demo-disjointness (each demo in at most one duel, which is what
#   makes the duels independent and the sign test valid) cuts that to 18 usable duels:
#
#     gap >=  3    4    5    6    7    8    9
#     pairs  108   62   37   23   11    3    1
#     duels   29   18   12   11    5    1    1
#
# Gap 3 is not a confident disagreement, and gaps >= 7 leave n = 5. Gap 4 keeps duels where the
# two demos sit at least ~27% of the cluster apart in BOTH rankings while retaining n = 18.
# This caps the design's power (a sign test at n = 18 needs 13/18 to clear one-sided 0.05), and
# the write-up must say so rather than presenting n = 18 as if it were the intended size.
MIN_RANK_GAP = 4
N_PILOT = 6
N_FULL = 18
PILOT_DEPTH = 4
FULL_DEPTH = 4
SELECT_SEED = 20260726
DUEL_SEEDS = (4401, 4402, 4403, 4404, 4405)   # matched slots; first PILOT_DEPTH / FULL_DEPTH used


def _scores():
    rel = P7.CONFIGS["relatif_C5"]["scores"]()
    gd = P7._graddot("cached")[TARGET]
    return rel, gd


def candidates():
    """Every within-cluster pair on which the two estimators confidently disagree."""
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    rel, gd = _scores()
    out = []
    for c in cl:
        ds = list(by_c[c])
        rr = stats.rankdata([rel[d] for d in ds])
        rg = stats.rankdata([gd[d] for d in ds])
        for i, j in itertools.combinations(range(len(ds)), 2):
            drel, dgd = rr[i] - rr[j], rg[i] - rg[j]
            if np.sign(drel) == np.sign(dgd) or drel == 0 or dgd == 0:
                continue                      # they agree -> the duel carries no information
            g = min(abs(drel), abs(dgd))
            if g < MIN_RANK_GAP:
                continue
            out.append({"cluster": c, "a": ds[i], "b": ds[j],
                        "gap_relatif": float(drel), "gap_graddot": float(dgd),
                        "min_gap": float(g),
                        "relatif_prefers": ds[i] if drel > 0 else ds[j],
                        "graddot_prefers": ds[i] if dgd > 0 else ds[j]})
    return out


def availability():
    """How many duels exist at each threshold, and the demo-level agreement that limits them.

    A deliverable in its own right: it explains WHY the LDS cannot separate these two estimators
    -- on a random 68-demo mask their scores nearly coincide.
    """
    from lds import spearman
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    rel, gd = _scores()
    tids = list(D.cache_for("bc_s10")["train_ids"])
    rows = []
    for th in range(3, 10):
        pairs, used, nd = 0, set(), 0
        allc = []
        for c in cl:
            ds = list(by_c[c])
            rr = stats.rankdata([rel[d] for d in ds])
            rg = stats.rankdata([gd[d] for d in ds])
            for i, j in itertools.combinations(range(len(ds)), 2):
                dr, dg = rr[i] - rr[j], rg[i] - rg[j]
                if np.sign(dr) == np.sign(dg) or dr == 0 or dg == 0:
                    continue
                g = min(abs(dr), abs(dg))
                if g < th:
                    continue
                pairs += 1
                allc.append((g, ds[i], ds[j]))
        for g, a, b in sorted(allc, key=lambda x: -x[0]):
            if a in used or b in used:
                continue
            used.update((a, b))
            nd += 1
        rows.append({"min_rank_gap": th, "n_pairs": pairs, "n_duels_demo_disjoint": nd})
    return {"demo_level_rank_corr_relatif_vs_graddot":
            float(spearman([rel[d] for d in tids], [gd[d] for d in tids])),
            "n_possible_within_cluster_pairs": len(cl) * 105, "by_threshold": rows}


def build_pair(a, b, cluster, rng):
    """base+{a} and base+{b}: two valid 68-demo stratified masks differing in exactly one demo."""
    cl = dataset.clusters()
    _, by_c = dataset.train_pool()
    eights = set(rng.choice(len(cl), 5, replace=False).tolist())
    demos = []
    for k, c in enumerate(cl):
        q = 8 if k in eights else 7
        pool = [d for d in by_c[c] if d not in (a, b)]
        if c == cluster:
            q -= 1                            # a or b will supply the last slot
        demos += [pool[i] for i in rng.choice(len(pool), q, replace=False)]
    ma, mb = sorted(demos + [a]), sorted(demos + [b])
    assert len(ma) == len(mb) == 68 and len(set(ma) ^ set(mb)) == 2
    return ma, mb


def select():
    cand = candidates()
    rng = np.random.default_rng(SELECT_SEED)
    # rank by confidence (min rank gap), ties broken by a seeded shuffle -> fully reproducible
    order = np.lexsort((rng.random(len(cand)), -np.array([c["min_gap"] for c in cand])))
    chosen, used = [], set()
    for i in order:
        c = cand[i]
        if c["a"] in used or c["b"] in used:
            continue                          # each demo appears in at most one duel
        used.update((c["a"], c["b"]))
        chosen.append(c)
        if len(chosen) >= N_FULL:
            break
    duels = []
    for k, c in enumerate(chosen):
        ma, mb = build_pair(c["a"], c["b"], c["cluster"], rng)
        duels.append({**c, "duel_id": f"D{k:03d}", "arm": "pilot" if k < N_PILOT else "full",
                      "mask_a": ma, "mask_b": mb})
    man = {"target": TARGET, "min_rank_gap": MIN_RANK_GAP, "select_seed": SELECT_SEED,
           "n_pilot": N_PILOT, "n_full": N_FULL, "pilot_depth": PILOT_DEPTH,
           "full_depth": FULL_DEPTH, "duel_seeds": list(DUEL_SEEDS),
           "n_candidates": len(cand), "availability": availability(), "duels": duels}
    os.makedirs(RESULTS, exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(man, fh, indent=1)
    return man


def load():
    with open(MANIFEST) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- analysis
def _outcomes(arm):
    man = load()
    raw = F.campaign_outcomes("D", "plain", targets=(man["target"],))[man["target"]]
    rows = []
    for d in man["duels"]:
        if d["arm"] != arm and arm != "all":
            continue
        ka, kb = f"{d['duel_id']}a", f"{d['duel_id']}b"
        if ka not in raw or kb not in raw:
            continue
        sa = dict(raw[ka])
        sb = dict(raw[kb])
        common = sorted(set(sa) & set(sb), key=str)
        if not common:
            continue
        va = np.array([sa[s] for s in common])
        vb = np.array([sb[s] for s in common])
        # positive outcome = higher neg-loss = better. a wins if it is the more helpful demo.
        diff = float(va.mean() - vb.mean())
        winner = d["relatif_prefers"] if diff > 0 else d["graddot_prefers"]
        rows.append({"duel_id": d["duel_id"], "arm": d["arm"], "cluster": d["cluster"],
                     "a": d["a"], "b": d["b"], "min_gap": d["min_gap"], "n_seeds": len(common),
                     "outcome_a": float(va.mean()), "outcome_b": float(vb.mean()),
                     "diff": diff, "abs_diff": abs(diff),
                     "seed_sd": float(np.std(np.concatenate([va, vb]), ddof=1)),
                     "paired_sd": float(np.std(va - vb, ddof=1)) if len(common) > 1 else np.nan,
                     "unpaired_sd": float(np.sqrt(np.var(va, ddof=1) + np.var(vb, ddof=1)))
                     if len(common) > 1 else np.nan,
                     "relatif_correct": bool(winner == d["relatif_prefers"])})
    return pd.DataFrame(rows)


def analyse(arm):
    df = _outcomes(arm)
    if not len(df):
        print(f"no {arm} duel outcomes on disk yet")
        return df, {}
    n = len(df)
    k = int(df.relatif_correct.sum())
    p = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)
    signal = float((df.abs_diff / df.seed_sd).mean())
    verdict = {"arm": arm, "n_duels": n, "relatif_correct": k, "sign_test_p": p,
               "mean_abs_diff_over_seed_sd": signal,
               "mean_paired_sd": float(df.paired_sd.mean()),
               "mean_unpaired_sd": float(df.unpaired_sd.mean()),
               "pairing_gain": float(df.unpaired_sd.mean() / df.paired_sd.mean())
               if df.paired_sd.mean() else np.nan,
               "KILL": bool(signal < 1.0)}
    return df, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="select", choices=["select", "pilot", "full"])
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.stage == "select":
        man = select()
        av = man["availability"]
        npil = min(N_PILOT, len(man["duels"]))
        nfull = max(0, len(man["duels"]) - npil)
        print(f"demo-level rank corr(RelatIF, GradDot) = "
              f"{av['demo_level_rank_corr_relatif_vs_graddot']:.3f} -- they mostly AGREE, which "
              f"is why confident disagreements are scarce")
        print(pd.DataFrame(av["by_threshold"]).to_string(index=False))
        print(f"\n{man['n_candidates']} candidate pairs at min rank gap >= {MIN_RANK_GAP}; "
              f"{len(man['duels'])} demo-disjoint duels selected ({npil} pilot + {nfull} full)")
        print(f"pilot cost: {npil} x 2 x {PILOT_DEPTH} = {npil * 2 * PILOT_DEPTH} retrains")
        print(f"full  cost: {nfull} x 2 x {FULL_DEPTH} = {nfull * 2 * FULL_DEPTH} retrains")
        print("wrote", MANIFEST)
        for d in man["duels"][:N_PILOT]:
            print(f"  {d['duel_id']} {d['cluster']}: relatif prefers "
                  f"...{d['relatif_prefers'][-28:]}, graddot prefers "
                  f"...{d['graddot_prefers'][-28:]} (gaps {d['gap_relatif']:+.0f}/"
                  f"{d['gap_graddot']:+.0f})")
    else:
        df, v = analyse(a.stage)
        if not len(df):
            return
        df.to_csv(os.path.join(RESULTS, f"p7_duels_{a.stage}.csv"), index=False)
        print("=" * 110)
        print(f"W2 DUEL {a.stage.upper()}")
        print("=" * 110)
        print(df.round(5).to_string(index=False))
        print()
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")
        if a.stage == "pilot":
            print("\n  KILL RULE: within-duel signal < 1 seed-noise sd -> kill W2.")
            print("  VERDICT:", "KILL -- write up as a negative design result and return the "
                  "budget" if v["KILL"] else "PASS -- proceed to the full arm")


if __name__ == "__main__":
    main()
