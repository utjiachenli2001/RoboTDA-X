"""PASS 10 -- complete the k=15 rung as a CENSUS. Descriptive only, no alpha, no bar test.

WHY THIS IS NOT A TEST, AND MUST NOT BECOME ONE.

Pass 9's k=15 rung rests on campaign N's 37 C5-conditional 5of9 masks. Conditional on C5 the
population of such masks is C(8,4) = 70 -- a combinatorial cap, not a sampling budget. The other 33
are Stage F's. They are NOT a fresh draw: `p8_cluster_grain.py`'s W1 scan scored every frozen config
on Stage F's 72 masks, and `p8_prereg.md` selected the cluster-grain hypothesis *from that scan*. So
the ancestor of everything this rung measures was discovered on these 33. Re-running them through
`retrain.heldout_frame_losses` redraws the SEED NOISE but not the true per-mask outcome the scan
selected on, so "never scored through the campaign pipeline" is a pipeline distinction and not an
inferential one. BLOCKERS #28 measured the winner's curse on a discovery draw at the difference
between +0.083 and -0.044; #31 says the honest draw set is per-config and excludes it.

**Campaign N's 37 were the only winner's-curse-free masks of this kind that will ever exist on this
corpus, and pass 9 spent them.** No alpha is available here again. This module therefore reports a
census and labels the split.

WHY THE CENSUS IS STILL WORTH 132 RETRAINS. Two reasons, neither inferential. It puts the complete
conditional population on ONE outcome pipeline at ONE depth, which is the record the project should
have. And the 33-vs-37 gap is a direct, on-corpus measurement of how much the Stage F scan's
selection pressure inflated this hypothesis -- a methods datum the project has only ever estimated.
Reported side by side, never averaged into a single number that hides it.

WHY A BOOTSTRAP-OVER-MASKS CI IS THE WRONG UNCERTAINTY HERE. With all 70 masks in hand there is no
mask sampling left to resample -- the conditional population is exhausted. Resampling masks would
answer "what if we had drawn a different 70 from the 70", which is not a question. What remains is
SEED noise (the outcome of each mask is a 4-seed mean of a noisy retrain) and the CONDITIONING choice
(C5-in-mask, 5of9, 75 demos). This module reports a seed-resampling interval and leaves the
conditioning sensitivity to the caller.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import p8_masks as P8M  # noqa: E402
from if_repair.confirm_mseries import ceiling, STATS  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import masks as MK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TARGET = "C5"
STRATUM = 5           # 5 of 9 clusters
RETAINED = 75
DEPTH = 4             # campaign N's 37 already have depth 4 on disk
SEED_SLOTS = 4        # slots {4401..4404} for the 33
PRIMARY_STAT = "kendall_tau_b"


def enumerate_population(target=TARGET, per=STRATUM):
    """All C(8,4) = 70 cluster subsets of size 5 that contain the target."""
    cl = sorted(dataset.clusters())
    others = [c for c in cl if c != target]
    subs = [tuple(sorted((target,) + combo))
            for combo in itertools.combinations(others, per - 1)]
    assert len(subs) == len(set(subs))
    return subs


def campaign_n_conditional(target=TARGET):
    """Campaign N's C5-conditional 5of9 masks -- the winner's-curse-free 37."""
    return {tuple(sorted(m["clusters"])): m
            for m in P8M.manifest()["masks"]
            if m["stratum"] == f"{STRATUM}of9" and target in m["clusters"]}


def stage_f_signatures():
    return {tuple(sorted(m["clusters"])) for m in MK.cluster_mask_manifest()["masks"]}


def split():
    """(population, campaign-N 37, the 33 discovery-draw remainder). Asserts the arithmetic."""
    pop = enumerate_population()
    n37 = campaign_n_conditional()
    sf = stage_f_signatures()
    rest = [s for s in pop if s not in n37]

    assert len(pop) == 70, f"population is {len(pop)}, expected 70"
    assert set(n37) <= set(pop), "campaign N holds a mask outside the conditional population"
    assert len(n37) == 37, f"campaign N holds {len(n37)} conditional 5of9 masks, expected 37"
    assert len(rest) == 33, f"remainder is {len(rest)}, expected 33"
    not_sf = [s for s in rest if s not in sf]
    assert not not_sf, f"{len(not_sf)} of the remainder are NOT Stage F signatures: {not_sf[:3]}"
    return pop, n37, rest


def masks_for_the_33():
    """The 33 as retrain-ready mask dicts. Demo lists come from the cluster pool, sorted."""
    _, _, rest = split()
    _, by_c = dataset.train_pool()
    out = []
    for i, sig in enumerate(sorted(rest)):
        demos = sorted(d for c in sig for d in by_c[c])
        assert len(demos) == RETAINED, f"{sig} has {len(demos)} demos"
        out.append({"mask_id": f"P{STRATUM}_{i:03d}", "clusters": list(sig),
                    "n_clusters": STRATUM, "n_demos": len(demos), "demos": demos,
                    "stratum": f"{STRATUM}of9", "provenance": "stage_f_discovery_draw"})
    return out


def manifest(path=None, force=False):
    path = path or os.path.join(RESULTS, "p10_k15_manifest.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    pop, n37, rest = split()
    ms = masks_for_the_33()
    out = {"pass": 10, "campaign": "P", "grain": "cluster", "target": TARGET,
           "stratum": f"{STRATUM}of9", "retained_demos": RETAINED, "depth": DEPTH,
           "population": len(pop), "campaign_n_holds": len(n37), "fresh_this_campaign": 0,
           "discovery_draw_masks": len(ms),
           "inference_note": "DESCRIPTIVE CENSUS ONLY. These 33 are the Stage F discovery draw for "
                             "the cluster-grain hypothesis (p8_cluster_grain W1 scan). No alpha, no "
                             "bar test. See BLOCKERS #28, #31.",
           "masks": ms}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


# ------------------------------------------------------------------ the census read
def merged_outcomes(target=TARGET, campaigns=("N", "P")):
    """Per-mask {seed: value} merged ACROSS campaigns.

    `functionals.campaign_outcomes` takes one campaign, and the census spans two: campaign N holds
    the 37 and campaign P the 33. Mask ids are disjoint by construction, so the merge is a dict
    update rather than a reconciliation.
    """
    out = {}
    for c in campaigns:
        try:
            got = F.campaign_outcomes(c, "plain", targets=(target,))[target]
        except Exception:
            continue
        for m, v in got.items():
            out.setdefault(m, {}).update(v)
    return out


def _read(masks, raw_all, depth, fn):
    raw = {m["mask_id"]: raw_all[m["mask_id"]] for m in masks if m["mask_id"] in raw_all}
    raw = {m: v for m, v in raw.items() if len(v) >= depth}
    if len(raw) < 5:
        return None
    keep = {m: dict(sorted(v.items())[:depth]) for m, v in raw.items()}
    obs = F.seed_mean(keep)
    use = [m for m in masks if m["mask_id"] in obs]
    o = np.array([obs[m["mask_id"]] for m in use], float)
    pg = P7.mask_pred(P7._graddot("cached")[TARGET], use)
    ok = np.isfinite(o) & np.isfinite(pg)
    if ok.sum() < 5 or np.std(pg[ok]) == 0:
        return None
    c = ceiling({m["mask_id"]: keep[m["mask_id"]] for m, k in zip(use, ok) if k}, fn)
    lds = fn(pg[ok], o[ok])
    return {"n_masks": int(ok.sum()), "lds": lds, "ceiling": c,
            "ratio": lds / c if np.isfinite(c) and c else np.nan,
            "ratio_sqrt": lds / np.sqrt(c) if np.isfinite(c) and c > 0 else np.nan}


def evaluate(depth=DEPTH):
    """Census, plus the 37-only and 33-only reads side by side so the selection gap is visible."""
    pop, n37, rest = split()
    _, by_c = dataset.train_pool()
    n37_masks = [dict(m, provenance="campaign_N_unselected") for m in n37.values()]
    p33_masks = masks_for_the_33()
    raw_all = merged_outcomes()

    rows = []
    for label, ms, note in (
        ("37 only (campaign N, winner's-curse-FREE)", n37_masks,
         "the only unselected-upon masks of this kind that will ever exist; pass 9 scored them once"),
        ("33 only (Stage F DISCOVERY draw)", p33_masks,
         "the cluster-grain hypothesis was selected on these; expected to read HIGH (#28)"),
        ("70 census (complete conditional population)", n37_masks + p33_masks,
         "no mask sampling remains -- the population is exhausted; uncertainty is seed noise"),
    ):
        for sname, fn in STATS.items():
            r = _read(ms, raw_all, depth, fn)
            if r is None:
                rows.append({"subset": label, "statistic": sname, "depth": depth,
                             "n_masks": 0, "note": "no outcomes at this depth yet"})
                continue
            rows.append(dict(r, subset=label, statistic=sname, depth=depth,
                             primary=sname == PRIMARY_STAT, note=note))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--out", default=os.path.join(RESULTS, "p10_k15_census.csv"))
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args()

    man = manifest()
    print(f"[p10/census] population={man['population']} "
          f"campaign_N={man['campaign_n_holds']} discovery_draw={man['discovery_draw_masks']} "
          f"depth={man['depth']}")
    print(f"  {man['inference_note']}")
    if a.manifest_only:
        return
    df = evaluate(depth=a.depth)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        cols = [c for c in ["subset", "statistic", "n_masks", "depth", "lds", "ceiling", "ratio",
                            "ratio_sqrt"] if c in df.columns]
        print(df[df.get("primary", True) == True][cols].to_string(index=False))  # noqa: E712
    print(f"\n[p10/census] -> {a.out}")


if __name__ == "__main__":
    main()
