"""if_repair.data -- loaders for cached Grams, outcomes, ceilings and manifests.

Nothing here recomputes science. Every artifact is read from the repo exactly as
archived. The only construction is the E=20 ensemble, which is by definition the
concatenation of the p6 base cache (10 members) and the p11 confirmatory cache
(10 members) along the member axis.

The (cache, outcome, ceiling) triples are defined once, here, in TIERS -- so that
an estimator can never be silently scored against a mismatched ground truth or a
mismatched seed depth.
"""
from __future__ import annotations

import functools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEV_TARGETS = ("C1", "C5")
HOLDOUT_TARGETS = ("C2", "C4", "C7", "C9")
EXPLORATORY_TARGETS = ("C3", "C8")
ALL_TARGETS = tuple(f"C{i}" for i in range(1, 10))


def add_repo_paths() -> None:
    """Put the repo's own modules on sys.path, exactly as the phase scripts do."""
    for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "phase3", "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


# --------------------------------------------------------------------------- caches
P6 = os.path.join(ROOT, "phase3", "results", "p6_gram_cache.npz")
P11 = os.path.join(ROOT, "phase4", "results", "p11_gram_cache_new_members.npz")
P17 = os.path.join(ROOT, "phase5", "results", "p17_diffusion_gram_cache.npz")


def _as_dict(z) -> dict:
    """np.load object -> plain dict. scores_at_ridge/scores_graddot only index it."""
    return {k: z[k] for k in ("G", "K", "members", "train_ids", "targets")}


@functools.lru_cache(maxsize=None)
def gram_e10() -> dict:
    """Base ensemble, E=10 (phase-3 p6 cache)."""
    return _as_dict(np.load(P6, allow_pickle=True))


@functools.lru_cache(maxsize=None)
def gram_e20() -> dict:
    """E=20 = p6 (base 10) + p11 (confirmatory 10), concatenated on the member axis.

    The paper's headline GradDot = 0.513 is at this depth.
    """
    a = np.load(P6, allow_pickle=True)
    b = np.load(P11, allow_pickle=True)
    if not np.array_equal(a["train_ids"], b["train_ids"]):
        raise RuntimeError("p6/p11 train_ids differ -- caches are not concatenable")
    if not np.array_equal(a["targets"], b["targets"]):
        raise RuntimeError("p6/p11 targets differ -- caches are not concatenable")
    return {
        "G": np.concatenate([a["G"], b["G"]], axis=0),
        "K": np.concatenate([a["K"], b["K"]], axis=0),
        "members": np.concatenate([a["members"], b["members"]], axis=0),
        "train_ids": a["train_ids"],
        "targets": a["targets"],
    }


@functools.lru_cache(maxsize=None)
def gram_diffusion() -> dict:
    """Diffusion-policy arm (phase-5 p17). NOTE: E=5 members, not 10."""
    return _as_dict(np.load(P17, allow_pickle=True))


# --------------------------------------------------------------------------- masks
@functools.lru_cache(maxsize=None)
def demo_masks() -> list:
    """The 24 demo-grain masks (the LDS unit for every number we report)."""
    return json.load(open(os.path.join(ROOT, "results", "demo_mask_manifest.json")))["masks"]


@functools.lru_cache(maxsize=None)
def cluster_masks() -> list:
    m = json.load(open(os.path.join(ROOT, "results", "mask_manifest.json")))
    return [{"mask_id": x["mask_id"], "demos": x["demos"], "clusters": x["clusters"]}
            for x in m["masks"]]


# --------------------------------------------------------------------------- tiers
# A tier fixes the (cache, outcome table, seed set, seed aggregator, ceiling) triple.
# Mixing these is the single easiest way to manufacture a fake improvement, so the
# binding happens here and nowhere else.
TIERS = {
    "bc_s10": {
        "cache": "e20",
        "outcomes": os.path.join(ROOT, "phase4", "results", "p12_outcomes_S10.parquet"),
        "seeds": tuple(range(401, 411)),
        "agg": "mean",
        "functional": "neg_plain_loss",
        "ceiling_file": os.path.join(ROOT, "phase4", "results", "p12_ceilings.json"),
        "ceiling_key": "ceiling_10seed_SB",
        "desc": "BC policy, E=20 Gram, 10-seed mean of neg_plain_loss, 10-seed SB ceiling",
    },
    "dev_s6": {
        "cache": "e10",
        "outcomes": os.path.join(ROOT, "phase2", "results", "stage_G6_outcomes.parquet"),
        "seeds": None,
        "agg": "mean",
        "functional": "neg_plain_loss",
        "ceiling_file": os.path.join(ROOT, "phase2", "results", "p1_demo_grain.json"),
        "ceiling_key": "ceiling_6seed_SB",
        "desc": "BC policy, E=10 Gram, 6-seed mean of neg_plain_loss, 6-seed SB ceiling",
    },
    "diff_s10": {
        "cache": "diffusion",
        "outcomes": os.path.join(ROOT, "phase5", "results", "p15_outcomes_S10.parquet"),
        "seeds": tuple(range(601, 611)),
        # phase 5 established that the SEED-MEAN outcome is broken for the diffusion
        # arm (p15_verdict.seed_mean_brokenness_series: the mean-aggregated ceiling
        # collapses to 0.20 while the median-aggregated one is 0.83). The
        # preregistered aggregator there is the MEDIAN. Match it, or the ceiling and
        # the prediction are not measuring the same quantity.
        "agg": "median",
        "functional": "neg_plain_loss",
        "ceiling_file": os.path.join(ROOT, "phase5", "results", "p15_verdict.json"),
        "ceiling_key": "ceiling_median_10seed_SB",
        "desc": "Diffusion policy, E=5 Gram, 10-seed MEDIAN of neg_plain_loss, median 10-seed SB ceiling",
    },
}


def cache_for(tier: str) -> dict:
    which = TIERS[tier]["cache"]
    return {"e10": gram_e10, "e20": gram_e20, "diffusion": gram_diffusion}[which]()


@functools.lru_cache(maxsize=None)
def outcomes(tier: str, functional: str | None = None) -> dict:
    """-> {target: {mask_id: observed outcome}} at this tier's seed depth/aggregator."""
    spec = TIERS[tier]
    func = functional or spec["functional"]
    df = pd.read_parquet(spec["outcomes"])
    if spec["seeds"] is not None:
        if "seed" not in df.columns:
            raise RuntimeError(f"{spec['outcomes']} has no seed column")
        have = set(df.seed.unique())
        want = set(spec["seeds"])
        if not want.issubset(have):
            raise RuntimeError(f"tier {tier}: missing seeds {sorted(want - have)}")
        df = df[df.seed.isin(spec["seeds"])]
    if func not in df.columns:
        raise RuntimeError(f"tier {tier}: functional {func!r} not in {list(df.columns)}")
    agg = spec["agg"]
    out = {}
    for t, sub in df.groupby("target"):
        g = sub.groupby("mask_id")[func]
        out[str(t)] = (g.mean() if agg == "mean" else g.median()).to_dict()
    return out


@functools.lru_cache(maxsize=None)
def ceilings(tier: str) -> dict:
    """-> {target: ceiling}. Read verbatim from the archived ceiling artifact."""
    spec = TIERS[tier]
    j = json.load(open(spec["ceiling_file"]))
    key = spec["ceiling_key"]
    if tier == "bc_s10":
        return {t: float(v[key]) for t, v in j["targets"].items()}
    if tier == "dev_s6":
        return {t: float(v["neg_plain_loss"][key]) for t, v in j["all_targets"].items()}
    if tier == "diff_s10":
        return {t: float(v[key]) for t, v in j["all_targets_DESCRIPTIVE"].items()}
    raise KeyError(tier)


def bar(tier: str) -> dict:
    """Half-ceiling pass bar per target."""
    return {t: 0.5 * c for t, c in ceilings(tier).items()}


def score_matrix_to_dict(S, Z, j) -> dict:
    """Column j of an (N,T) score matrix -> {demo_id: score}, the form the LDS wants."""
    train_ids = list(Z["train_ids"])
    return {train_ids[i]: float(S[i, j]) for i in range(len(train_ids))}


def target_index(Z, target: str) -> int:
    return list(Z["targets"]).index(target)
