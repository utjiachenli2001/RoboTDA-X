"""Wiring tests for pass-7 W0.1 -- the pass-3 pattern that caught BLOCKERS #14.

Every scorer is exercised on masks/outcomes that ALREADY have a committed answer, before it is
allowed near a pooled number. The strongest available check is exact reproduction of the
campaign J / K / L confirmation CSVs, which were preregistered and scored once.

The surrogate-LOO scorer needs the regenerated checkpoints and a GPU (b12_headloo has no disk
cache), so its reproduction test is gated behind P7_SLOW=1 to keep `pytest if_repair/tests -q`
in the seconds range. The GPU-free scorers (RelatIF and the leverage family, both on the cached
E=20 Gram) cover the J, K and L wiring on their own.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from if_repair import data as D
from if_repair import functionals as F
from if_repair import p7_pooled_oos as P7

SLOW = os.environ.get("P7_SLOW") == "1"

# committed, preregistered, scored once -- results/confirm_{j,k,l}series.csv
COMMITTED = {
    ("relatif_C5", "J"): dict(lds=0.5156521739130434, graddot_lds=0.17478260869565218,
                              ceiling=0.9678053575817153),
    ("relatif_C5", "L"): dict(lds=0.21739130434782608, graddot_lds=0.14260869565217388,
                              ceiling=0.9584344169497008),
    ("leverage_C7", "K"): dict(lds=0.4156521739130434, graddot_lds=0.12956521739130433,
                               ceiling=0.9695340501792113),
    ("surrogate_C5", "J"): dict(lds=0.46347826086956523, graddot_lds=0.12347826086956522,
                                ceiling=0.9678053575817153),
    ("surrogate_C5", "L"): dict(lds=0.1269565217391304, graddot_lds=0.2121739130434782,
                                ceiling=0.9584344169497008),
}
GPU_FREE = ("relatif_C5", "leverage_C7")


@pytest.mark.parametrize("cfg_name,draw", [k for k in COMMITTED if k[0] in GPU_FREE])
def test_reproduces_committed_confirmation_rows(cfg_name, draw):
    """The p7 scoring path must reproduce the preregistered confirmation numbers bit-for-bit."""
    row = P7.analyse_per_draw(cfg_name, P7.CONFIGS[cfg_name], [draw])[0]
    exp = COMMITTED[(cfg_name, draw)]
    for k, v in exp.items():
        assert abs(row[k] - v) < 1e-12, f"{cfg_name}/{draw} {k}: {row[k]} != {v}"


@pytest.mark.skipif(not SLOW, reason="needs regen checkpoints + GPU; set P7_SLOW=1")
@pytest.mark.parametrize("cfg_name,draw", [k for k in COMMITTED if k[0] not in GPU_FREE])
def test_reproduces_committed_confirmation_rows_slow(cfg_name, draw):
    row = P7.analyse_per_draw(cfg_name, P7.CONFIGS[cfg_name], [draw])[0]
    exp = COMMITTED[(cfg_name, draw)]
    for k, v in exp.items():
        assert abs(row[k] - v) < 1e-12, f"{cfg_name}/{draw} {k}: {row[k]} != {v}"


@pytest.mark.parametrize("draw", ["J", "K", "L"])
def test_pooled_ceiling_raw_matches_archived_recipe_single_draw(draw):
    """rule='raw' on one draw IS functionals.split_half_ceiling -- the archived recipe."""
    got = P7.pooled_ceiling((draw,), "C5", "raw")["ceiling"]
    want = F.split_half_ceiling(P7.raw_outcomes(draw, "C5"))["ceiling"]
    assert abs(got - want) < 1e-12


def test_pooled_ceiling_raw_matches_archived_recipe_on_union():
    """Pooling must not change the recipe: the union ceiling is the archived one on merged data."""
    merged = {}
    for dr in ("J", "K", "L"):
        merged.update(P7.raw_outcomes(dr, "C5"))
    got = P7.pooled_ceiling(("J", "K", "L"), "C5", "raw")["ceiling"]
    want = F.split_half_ceiling(merged)["ceiling"]
    assert abs(got - want) < 1e-12


def test_within_draw_rank_is_monotone_so_single_draw_spearman_is_rule_invariant():
    """A single draw's Spearman cannot depend on the raw/rank rule (ranking is monotone)."""
    from lds import spearman
    px, pg, o, di = P7.assemble(P7.CONFIGS["relatif_C5"], ["J"])
    r_raw = spearman(px, o)
    r_rank = spearman(P7._rank_within(px, di), P7._rank_within(o, di))
    assert abs(r_raw - r_rank) < 1e-12


@pytest.mark.parametrize("cfg_name", GPU_FREE)
def test_scores_cover_every_demo_and_are_finite(cfg_name):
    tids = list(D.cache_for("bc_s10")["train_ids"])
    sc = P7.CONFIGS[cfg_name]["scores"]()
    assert set(map(str, sc)) == set(map(str, tids))
    assert np.all(np.isfinite(np.array([sc[d] for d in sc], float)))


def test_oos_draws_exclude_every_selection_draw():
    """No config may claim a draw it was selected on. Selection was G/H/I for all four."""
    for name, cfg in P7.CONFIGS.items():
        assert not (set(cfg["oos"]) & set(P7.DEV_DRAWS)), name


def test_pooling_rule_is_decided_on_the_frozen_draw_set():
    """The raw/rank rule must be keyed on J/K/L only -- not on whatever set is being pooled."""
    assert P7.RULE_DRAWS == ("J", "K", "L")
    for t in ("C5", "C7"):
        assert P7.rule_for(t) == P7.icc_by_draw(t, P7.RULE_DRAWS)["rule"]
