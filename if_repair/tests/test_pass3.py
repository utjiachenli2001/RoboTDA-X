"""Pass-3 guards: the claims that, if they silently broke, would make the numbers wrong.

Each test pins a specific way this pass could lie:
  * the local trainer drifting from src/train.py (every campaign outcome would be off-protocol)
  * the ceiling recipe drifting from the archived one (every ratio has a ceiling in the denominator)
  * a frame weighting that is not a weighting (negative or unnormalised)
  * TracIn's density knob not containing GradDot as its k=1 endpoint (the comparison would be
    against a different estimator, not a baseline)
  * per-frame losses not reproducing the archived aggregate functionals
  * an outcome-fitted estimator reaching eval.evaluate instead of the leave-one-mask-out path
"""
import glob
import json
import os

import numpy as np
import pytest

from if_repair import data as D
from if_repair import functionals as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGNS = os.path.join(HERE, "runs", "campaigns")
has_cuda = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="needs a pinned GPU")


def _campaign_runs(c):
    return sorted(glob.glob(os.path.join(CAMPAIGNS, c, "*.npz")))


# --------------------------------------------------------------------- the trainer
@has_cuda
def test_split_seed_trainer_reproduces_src_train_bitwise():
    """retrain.train_one(s, s) must BE src/train.py:train(s), not merely resemble it.

    If this drifts, every campaign outcome is measured under a different protocol than the
    archived one, and the comparison to p12 stops meaning anything.
    """
    import tempfile
    import torch
    from if_repair import retrain as RT
    D.add_repo_paths()
    import train as TR
    import masks as MK

    cfg = TR.load_cfg()
    cfg["total_steps"] = 60
    demos = MK.demo_mask_manifest()["masks"][0]["demos"]
    tmp = tempfile.mkdtemp()
    TR.train(tmp, demos, 401, dict(cfg))
    legacy = torch.load(os.path.join(tmp, "final.pt"), map_location="cpu",
                        weights_only=False)["model"]
    model, _ = RT.train_one(demos, 401, 401, dict(cfg))
    got = model.state_dict()
    for k in legacy:
        assert torch.equal(legacy[k].cpu(), got[k].cpu()), f"{k} differs"


# --------------------------------------------------------------------- the ceiling
def test_ceiling_recipe_reproduces_archived_ceilings():
    """Every ratio in this project divides by one of these. They must be the archived numbers."""
    arch = json.load(open(D.TIERS["bc_s10"]["ceiling_file"]))["targets"]
    obs = F.archived_outcomes("plain")
    for t in D.ALL_TARGETS:
        got = F.split_half_ceiling(obs[t], max_splits=1000)
        assert got["n_splits"] == 126, f"{t}: {got['n_splits']} splits, expected C(10,5)/2"
        assert abs(got["ceiling"] - arch[t]["ceiling_10seed_SB"]) < 1e-12, t
        assert abs(got["half"] - arch[t]["ceiling_5v5_splithalf_uncorrected"]) < 1e-12, t


def test_ceiling_gate_is_applied_before_scoring():
    """A functional below the gate must not appear in the scored table."""
    gate = F.ceiling_table("archived", weightings=("plain", "transport", "interaction"),
                           targets=("C1", "C5"))
    assert (gate.ceiling > F.GATE).all()
    assert gate.gate_pass.all()


# --------------------------------------------------------------------- the weightings
@pytest.mark.parametrize("name", ["plain", "transport", "interaction"])
def test_weightings_are_nonnegative_and_mean_one(name):
    w = F.weights(name)
    assert (w >= 0).all()
    assert abs(w.mean() - 1.0) < 1e-9
    assert np.isfinite(w).all()


def test_plain_weighting_is_the_study_functional():
    assert np.allclose(F.weights("plain"), 1.0)


# --------------------------------------------------------------------- TracIn endpoint
def test_tracin_last1_unweighted_is_graddot():
    """TracIn's density sweep must contain the baseline exactly at its 1-checkpoint endpoint.

    Otherwise "TracIn beats GradDot" would be comparing two things that differ in more than the
    trajectory term.
    """
    from if_repair import b4_tracin as B4
    if not os.path.exists(B4.CACHE):
        pytest.skip("b4 cache not built")
    z = np.load(B4.CACHE, allow_pickle=True)
    index = json.loads(str(z["index"]))
    last = len(index["ckpts"]) - 1
    Z = B4.tracin_Z(index, z, "ALL", [last], lr_weight=False)
    assert np.array_equal(Z["K"], np.stack([z[f"ALL|{m}|{last}|K"]
                                            for m in range(len(index["members"]))]))


# --------------------------------------------------------------------- per-frame losses
def test_perframe_losses_reproduce_archived_aggregates():
    """The stored per-frame l2 must rebuild the three archived functionals for its own run."""
    runs = _campaign_runs("A")
    if not runs:
        pytest.skip("campaign A not run")
    z = np.load(runs[0], allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    fm = F.frame_meta()
    l2 = z["l2"].astype(np.float64)
    for c, rec in meta["outcomes"].items():
        rows = fm["cluster_of_row"] == c
        assert abs(float(l2[rows].mean()) - rec["plain_loss"]) < 1e-5, c
        for kind in ("transport", "interaction"):
            m = fm[kind][rows]
            got = float((l2[rows] * m).sum() / max(m.sum(), 1))
            assert abs(got - rec[f"{kind}_loss"]) < 1e-5, f"{c}/{kind}"


def test_campaign_plain_outcome_is_negative_mean_l2():
    runs = _campaign_runs("A")
    if not runs:
        pytest.skip("campaign A not run")
    out = F.campaign_outcomes("A", "plain", targets=("C1",))["C1"]
    z = np.load(runs[0], allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    key = (meta["seed_init"], meta["seed_order"])
    assert abs(out[meta["mask_id"]][key] + meta["outcomes"]["C1"]["plain_loss"]) < 1e-5


# --------------------------------------------------------------------- fresh masks
def test_fresh_masks_are_actually_fresh():
    """Campaign B must not silently re-test the consumed masks."""
    from if_repair import retrain as RT
    D.add_repo_paths()
    import masks as MK
    fresh, _ = RT.fresh_demo_masks()
    old = MK.demo_mask_manifest()["masks"]
    assert len(fresh) == len(old)
    fs = {frozenset(m["demos"]) for m in fresh}
    os_ = {frozenset(m["demos"]) for m in old}
    assert len(fs & os_) == 0, "a fresh mask duplicates an archived one"
    for m in fresh:
        assert len(m["demos"]) == 68


# --------------------------------------------------------------------- variance algebra
def test_anova_components_partition_total():
    from if_repair import b5_variance as B5
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(6, 3, 3))
    ss = B5.anova(Y)
    parts = sum(ss[k] for k in ("mask", "init", "order", "mask_x_init", "mask_x_order",
                                "init_x_order", "resid_MxIxO"))
    assert abs(parts - ss["total"]) < 1e-8 * max(1.0, ss["total"])


def test_anova_recovers_a_pure_mask_effect():
    from if_repair import b5_variance as B5
    Y = np.tile(np.arange(6.0)[:, None, None], (1, 3, 3))
    ss = B5.anova(Y)
    assert abs(ss["mask"] - ss["total"]) < 1e-9


# --------------------------------------------------------------------- pooled mask draws
def test_pooled_mask_sets_are_exchangeable_in_construction():
    """B8 pools the G- and H-series masks. They must be the same KIND of object to pool.

    Same generator, same mask size, same demo universe, disjoint ids. If a future draw changes
    any of that, pooling silently compares different designs.
    """
    from if_repair import retrain as RT
    D.add_repo_paths()
    import masks as MK
    g = MK.demo_mask_manifest()["masks"]
    h, _ = RT.fresh_demo_masks()
    assert len(g) == len(h) == 24
    assert {len(m["demos"]) for m in g} == {len(m["demos"]) for m in h} == {68}
    gu = {d for m in g for d in m["demos"]}
    hu = {d for m in h for d in m["demos"]}
    assert gu == hu, "the two draws span different demo universes"
    assert not ({m["mask_id"] for m in g} & {m["mask_id"] for m in h})


def test_paired_difference_is_better_determined_than_the_level():
    """The claim B8 rests on: mask-draw noise is shared, so it cancels in a difference.

    Synthetic, so it tests the algebra rather than the corpus: two estimators whose scores differ
    by a small constant shift plus independent noise, evaluated on common subsets.
    """
    rng = np.random.default_rng(0)
    draw_effect = rng.normal(0, 0.15, 400)          # the shared nuisance
    a = draw_effect + rng.normal(0, 0.02, 400)
    b = draw_effect + 0.10 + rng.normal(0, 0.02, 400)
    assert a.std() > 0.1 and b.std() > 0.1          # levels are noisy
    d = b - a
    assert d.std() < 0.05                           # the difference is not
    assert (d > 0).mean() > 0.95
