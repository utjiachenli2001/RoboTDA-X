"""Pass 3 -- the FRESH-MASK confirmatory family. Preregistered, computed ONCE.

The pass-2 hold-out (C2/C4/C7/C9 on the 24 archived masks) has been consumed. Re-running it
cannot add evidence: the masks and the outcomes are the same numbers that every dev iteration
has already seen. The only way to adjudicate a claim now is a fresh mask draw, which campaign B
supplies -- 24 masks from the repo's own Stage-G generator at seed 4711, retrained at 6 seeds,
sharing not one mask with the archived 24 (tests/test_pass3.py pins that).

PROTOCOL. The PREREG dict below is frozen BEFORE campaign B has produced a single run, and
committed in that state; `git log` on this file is the evidence. Every entry names the exact
estimator, its configuration, and the single target it is tested on. Nothing is chosen after
the fact, nothing is re-run, and a NaN or a negative is a failure, not a missing value.

Family size 5 -> Bonferroni alpha = 0.025 / 5 = 0.005.

Each hypothesis was selected on DEV (C1/C5 on the archived masks), where iterating freely is
allowed, and every one of them is a maximum over a sweep -- B4 over 180 cells, B3 over 110, B2
over 18. B1's `block_01`/C5 = 0.757 was exactly such a maximum and collapsed to -0.142 on
hold-out. That is the specific failure this family exists to catch.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402
from if_repair import spectral as SP  # noqa: E402

D.add_repo_paths()
from p6_lambda_sweep import ALPHA  # noqa: E402
from p6_lambda_extend import scores_graddot  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

PREREG = {
    "H1_datamodel_C2": {
        "why": "the only hold-out pass in pass 2 (ratio 0.639, p=0.00115). Needs fresh masks "
               "because it is fitted ON mask outcomes.",
        "kind": "datamodel_loo", "model": "lasso", "target": "C2",
    },
    "H2_tracin_head_C1": {
        "why": "B4 dev max: TracIn over all 5 checkpoints, LR-weighted, Phi restricted to the "
               "action head -> C1 ratio 0.602 vs GradDot(head) 0.506.",
        "kind": "tracin", "group": "head", "density": "last5", "lr_weighted": True,
        "target": "C1",
    },
    "H3_kfac_embed_C5": {
        "why": "B3 dev max: KFAC at lambda_rel=1e-4 on the embed group -> C5 ratio 0.615 vs "
               "identity 0.485 on the same restricted Phi.",
        "kind": "kfac", "group": "embed", "method": "kfac", "lam_rel": 1e-4, "target": "C5",
    },
    "H4_interaction_functional_C5": {
        "why": "B2 dev: the interaction-phase functional beats the plain one on C5 "
               "(0.474 vs 0.401) with a comparable ceiling.",
        "kind": "functional", "weighting": "interaction", "estimator": "GradDot_dmean",
        "target": "C5",
    },
    "H5_graddot_head_C1": {
        "why": "B1's structural claim -- the action head (0.2% of parameters) carries the C1 "
               "signal -- restated as a testable pass: GradDot on head-restricted Phi.",
        "kind": "graddot_group", "group": "head", "target": "C1",
    },
}
BONF = ALPHA / len(PREREG)

# Reference rows, NOT part of the family and not Bonferroni-corrected: without them a pass or a
# failure cannot be read as "better or worse than what we already had".
REFERENCE = {
    "REF_graddot_ALL": {"kind": "graddot_group", "group": "ALL"},
    "REF_tracin_head_last1": {"kind": "tracin", "group": "head", "density": "last1",
                              "lr_weighted": False},
}


# --------------------------------------------------------------------- fresh ground truth
def fresh_masks():
    from if_repair import retrain as RT
    ms, _ = RT.fresh_demo_masks()
    return ms


def fresh_truth(campaign="B", weighting="plain", targets=D.ALL_TARGETS):
    """-> (masks, {target: {mask: seed-mean outcome}}, {target: ceiling}) from campaign B.

    Keyed by WEIGHTING. A hypothesis about a redesigned target functional must be scored
    against that functional's own outcome and its own ceiling; scoring it against the plain
    outcome measures the mismatch between two functionals, which is precisely the error B2
    exists to avoid (and which this file's smoke check caught).
    """
    raw = F.campaign_outcomes(campaign, weighting, targets=targets)
    obs = {t: F.seed_mean(raw[t]) for t in raw}
    ceil = {t: F.split_half_ceiling(raw[t])["ceiling"] for t in raw}
    return fresh_masks(), obs, ceil


def lds_on(scores, masks, obs_t):
    pred = np.array([sum(scores.get(d, 0.0) for d in m["demos"]) for m in masks])
    out = np.array([obs_t.get(m["mask_id"], np.nan) for m in masks])
    ok = np.isfinite(out)
    rho = spearman(pred[ok], out[ok])
    return rho, spearman_p_onesided(rho, int(ok.sum())), int(ok.sum())


# --------------------------------------------------------------------- estimators
def _graddot_group(group):
    from if_repair import b1_layerwise as B1
    from if_repair import gradients as GR
    import glob
    members = sorted(os.path.basename(d) for d in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(d, "final.pt")))
    ens = B1.build_ensemble(members)
    return scores_graddot(ens[group], normalize_per_member=True)


def _tracin(group, density, lr_weighted):
    from if_repair import b4_tracin as B4
    z = np.load(B4.CACHE, allow_pickle=True)
    index = json.loads(str(z["index"]))
    n = len(index["ckpts"])
    ck = {"last1": [n - 1], "last2": list(range(n - 2, n)), "last3": list(range(n - 3, n)),
          "last5": list(range(n))}[density]
    Z = B4.tracin_Z(index, z, group, ck, lr_weight=lr_weighted)
    return scores_graddot(Z, normalize_per_member=True)


def _kfac(group, method, lam_rel):
    from if_repair import b3_kfac as B3
    ens = B3.build_ensemble()
    return scores_graddot(ens[(group, lam_rel, method)], normalize_per_member=True)


def _functional(weighting, estimator):
    from if_repair import b2_functionals as B2
    _, ens = B2.build_ensemble(F.WEIGHTINGS)
    Z = ens[weighting]
    if estimator == "GradDot_dmean":
        return scores_graddot(Z, normalize_per_member=True)
    k = int(estimator.split("_k")[1])
    S = SP.truncated_if(Z, k, normalize="dmean")
    tids = list(Z["train_ids"])
    return {tg: {tids[i]: float(S[i, j]) for i in range(len(tids))}
            for j, tg in enumerate(list(Z["targets"]))}


def _datamodel_loo(model, target, masks, obs_t):
    """Leave-one-mask-out on the FRESH masks (BLOCKERS #7: never the in-sample path)."""
    from if_repair.datamodel import MODELS, select_alpha
    Z = D.cache_for("bc_s10")
    demo_ids = list(Z["train_ids"])
    idx = {d: i for i, d in enumerate(demo_ids)}
    X = np.zeros((len(masks), len(demo_ids)))
    for r, m in enumerate(masks):
        for d in m["demos"]:
            if d in idx:
                X[r, idx[d]] = 1.0
    y = np.array([obs_t.get(m["mask_id"], np.nan) for m in masks], float)
    ok = np.isfinite(y)
    alpha, _ = select_alpha(X[ok], y[ok], model, folds=5)
    pred, out = [], []
    for r in range(len(masks)):
        if not np.isfinite(y[r]):
            continue
        keep = np.array([q != r and np.isfinite(y[q]) for q in range(len(masks))])
        fit = MODELS[model](alpha).fit(X[keep], y[keep])
        pred.append(float(np.dot(X[r], fit.coef_)))
        out.append(y[r])
    rho = spearman(pred, out)
    return rho, spearman_p_onesided(rho, len(pred)), len(pred), alpha


def evaluate_spec(name, spec, masks, truth):
    """truth: {weighting: (obs, ceil)}. The spec's own weighting selects the ground truth."""
    t = spec.get("target")
    kind = spec["kind"]
    obs, ceil = truth[spec.get("weighting", "plain")]
    extra = {}
    if kind == "datamodel_loo":
        rho, p, n, alpha = _datamodel_loo(spec["model"], t, masks, obs[t])
        extra["alpha"] = alpha
    else:
        if kind == "graddot_group":
            sc = _graddot_group(spec["group"])
        elif kind == "tracin":
            sc = _tracin(spec["group"], spec["density"], spec["lr_weighted"])
        elif kind == "kfac":
            sc = _kfac(spec["group"], spec["method"], spec["lam_rel"])
        elif kind == "functional":
            sc = _functional(spec["weighting"], spec["estimator"])
        else:
            raise KeyError(kind)
        rho, p, n = lds_on(sc[t], masks, obs[t])
    c = float(ceil[t])
    return {"name": name, "target": t, "kind": kind, "lds": float(rho), "ceiling": c,
            "ratio": float(rho) / c if c else np.nan, "bar": 0.5 * c, "p": float(p),
            "n": int(n), **extra}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="B")
    a = ap.parse_args()

    targets = sorted({s["target"] for s in PREREG.values()} | set(D.DEV_TARGETS))
    wneeded = sorted({s.get("weighting", "plain") for s in PREREG.values()} | {"plain"})
    truth = {}
    for w in wneeded:
        masks, obs, ceil = fresh_truth(a.campaign, weighting=w, targets=tuple(targets))
        truth[w] = (obs, ceil)

    print("=" * 104)
    print(f"PASS 3 -- FRESH-MASK CONFIRMATORY FAMILY ({len(PREREG)} hypotheses, "
          f"Bonferroni alpha = {BONF:.4f}), computed once")
    print("=" * 104)
    print(f"masks: {len(masks)} fresh (seed 4711), outcomes: campaign {a.campaign}")
    print("\nceilings of the fresh outcome, per functional (a hypothesis on a target whose "
          f"ceiling is below {F.GATE} is unadjudicable):")
    for w in wneeded:
        row = "  ".join(f"{t}={truth[w][1][t]:.3f}"
                        + ("!" if truth[w][1][t] < F.GATE else "") for t in targets)
        print(f"  {w:12s} {row}")

    rows = []
    for name, spec in PREREG.items():
        r = evaluate_spec(name, spec, masks, truth)
        r["family"] = "prereg"; r["alpha"] = BONF
        r["weighting"] = spec.get("weighting", "plain")
        r["PASS"] = bool(np.isfinite(r["lds"]) and r["ratio"] >= 0.5 and r["p"] < BONF)
        rows.append(r)
    for name, spec in REFERENCE.items():
        for t in targets:
            s = dict(spec); s["target"] = t
            r = evaluate_spec(name, s, masks, truth)
            r["family"] = "reference"; r["alpha"] = ALPHA
            r["weighting"] = "plain"
            r["PASS"] = bool(np.isfinite(r["lds"]) and r["ratio"] >= 0.5 and r["p"] < ALPHA)
            rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "confirm3_fresh_masks.csv"), index=False)
    print("\n--- preregistered family ---")
    print(df[df.family == "prereg"][["name", "target", "weighting", "lds", "ceiling",
                                     "ratio", "bar", "p", "alpha",
                                     "PASS"]].to_string(index=False))
    print("\n--- reference (not in the family, uncorrected alpha) ---")
    print(df[df.family == "reference"][["name", "target", "lds", "ceiling", "ratio", "p",
                                        "PASS"]].to_string(index=False))
    np_ = int(df[df.family == "prereg"].PASS.sum())
    print(f"\nPREREGISTERED PASSES: {np_}/{len(PREREG)}")


if __name__ == "__main__":
    main()
