"""Task 0 acceptance: the paper's anchors must reproduce before any new estimator counts."""
import numpy as np
import pandas as pd
import pytest

from if_repair import anchors as A
from if_repair import data as D

D.add_repo_paths()
from p6_lambda_sweep import demo_grain_lds  # noqa: E402


@pytest.fixture(scope="module")
def anch():
    return A.run()


def test_graddot_champion_reproduces_0_513(anch):
    """The spec's headline anchor: GradDot, C1, E=20 -> 0.513, within 0.02."""
    got = anch["A1_graddot_unitL2_C1_E20_bc_s10"]["lds"]
    assert abs(0.513 - got) < 0.02, f"GradDot champion C1 E=20 = {got}, expected ~0.513"


def test_graddot_champion_is_bit_exact(anch):
    """Stronger than the spec asks: identical to the archived p16 constant."""
    got = anch["A1_graddot_unitL2_C1_E20_bc_s10"]["lds"]
    assert abs(got - A.C1_GRADDOT_ARCHIVED) < 1e-12


def test_graddot_champion_ratio_and_p(anch):
    r = anch["A1_graddot_unitL2_C1_E20_bc_s10"]
    assert abs(r["ratio"] - 0.54) < 0.01, r["ratio"]
    assert abs(r["p"] - 0.005) < 0.002, r["p"]


def test_cv_if_is_about_0_40(anch):
    """Cross-validated exact IF (tuned on C1, frozen, evaluated held-out) ~ 0.40, and FAILS."""
    for key in ("A3_cv_if_tuneC1_evalC5_E20_bc_s10", "A3b_cv_if_tuneC1_evalC5_E10_dev_s6"):
        got = anch[key]["lds"]
        assert abs(0.40 - got) < 0.03, f"{key} = {got}, expected ~0.40"
        assert not anch[key]["passed"], f"{key} must FAIL the half-ceiling+p bar"


def test_cv_if_e10_bit_exact(anch):
    """Reproduces phase3/results/p6_lambda_sweep.json MANDATORY_CROSS_VALIDATION exactly."""
    got = anch["A3b_cv_if_tuneC1_evalC5_E10_dev_s6"]["lds"]
    assert abs(got - A.CV_IF_ARCHIVED_E10) < 1e-12


def test_dev_anchor_0_504(anch):
    """E=10 dev triple: GradDot_dmean C1 = 0.504 (the p6_lambda_extend number)."""
    got = anch["A5_graddot_dmean_C1_E10_dev_s6"]["lds"]
    assert abs(0.504 - got) < 0.01, got


def test_lambda_to_infinity_collapse(anch):
    """IF and TRAK must both converge onto GradDot_dmean as ridge_rel -> inf."""
    c = anch["A4_lambda_collapse_E20_bc_s10"]
    assert c["max_gap_at_largest_ridge"] < 1e-9, c["sweep"][-1]


def test_graddot_variants_are_distinct(anch):
    """Guard against the conflation documented in BLOCKERS.md #1."""
    l2 = anch["A1_graddot_unitL2_C1_E20_bc_s10"]["lds"]
    dm = anch["A2_graddot_dmean_C1_E20_bc_s10"]["lds"]
    assert abs(l2 - dm) > 0.05, "unitL2 and dmean should NOT be the same estimator"


def test_both_graddot_variants_match_archived_p16():
    """Bit-for-bit against phase5/results/p16_lds_table.csv on all 7 non-focal targets."""
    Z = D.gram_e20()
    gm, obs = D.demo_masks(), D.outcomes("bc_s10")
    l2, dm = A.graddot_unit_l2(Z), A.graddot_dmean(Z)
    p16 = pd.read_csv(f"{D.ROOT}/phase5/results/p16_lds_table.csv")
    for est, scores in (("GradDot_E20_normalized", l2), ("GradDot_E20_dmean", dm)):
        sub = p16[p16.estimator == est]
        assert len(sub) == 7
        for r in sub.itertuples():
            got = demo_grain_lds(scores[r.target], gm, obs[r.target])[0]
            assert abs(got - r.rho) < 1e-12, f"{est}/{r.target}: {got} vs archived {r.rho}"


def test_e20_is_p6_plus_p11():
    Z10, Z20 = D.gram_e10(), D.gram_e20()
    assert Z20["G"].shape == (20, 135, 135)
    assert Z20["K"].shape == (20, 135, 9)
    assert np.array_equal(Z20["G"][:10], Z10["G"])
    assert list(Z20["members"])[:10] == list(Z10["members"])
