"""PARITY IS THE GATE.

The reward only becomes user-editable once RewardSpec.v5() is proven to
reproduce the protected abs_env.py's numbers exactly. These tests compare
against the REAL constants and _terminal_g_shape imported from that file --
not a recorded snapshot -- so if anyone ever edits the protected copy, or the
extraction drifts, this fails loudly instead of silently changing what every
past result meant.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_spec import RewardSpec, PRESETS
import abs_env
from abs_env import _terminal_g_shape

# A grid that covers every branch of the shape: below the pivot, the linear
# ramp, exactly at the gatekeeper, above it, and past the safety clamp.
G_GRID = [0.0, 0.05, 0.1, 0.29, 0.3, 0.4, 0.49, 0.5, 0.51, 0.7, 0.9, 1.0,
          1.02, 1.0315, 1.05, 1.0599, 1.06, 1.0601, 1.1, 1.1884, 1.3, 1.5,
          2.0, 2.4, 2.5, 2.6, 5.0]
YAW_GRID = [0.0, 0.01, 0.05, 0.0501, 0.1, 0.2, 0.5, 1.0, 3.0]


def test_v5_constants_match_the_protected_file_exactly():
    v5 = RewardSpec.v5()
    assert v5.ramp_pivot == 0.5
    assert v5.ramp_neg_k == abs_env.TERMINAL_RAMP_NEG_K
    assert v5.ramp_pos_k == abs_env.TERMINAL_RAMP_POS_K
    assert v5.gatekeeper == abs_env.GATEKEEPER_G
    assert v5.step_bonus == abs_env.TERMINAL_STEP_BONUS
    assert v5.quad_k == abs_env.TERMINAL_QUAD_K
    assert v5.per_step_k == abs_env.PER_STEP_K
    assert v5.per_step_g_gate == abs_env.PER_STEP_G_GATE
    assert v5.yaw_bonus_k_step == abs_env.YAW_BONUS_K_STEP
    assert v5.yaw_bonus_alpha == abs_env.YAW_BONUS_ALPHA
    assert v5.yaw_bonus_k_terminal == abs_env.YAW_BONUS_K_TERMINAL
    assert v5.yaw_bonus_threshold == abs_env.YAW_BONUS_THRESHOLD
    assert v5.yaw_pen_k_terminal == abs_env.YAW_PEN_K_TERMINAL
    assert v5.yaw_rate_deadzone == abs_env.YAW_RATE_DEADZONE_RAD_S
    assert v5.crash_penalty == abs_env.CRASH_PENALTY
    assert v5.normalize is False


@pytest.mark.parametrize("g", G_GRID)
def test_v5_g_shape_is_bit_identical_to_the_protected_shape(g):
    assert RewardSpec.v5().g_shape(g) == _terminal_g_shape(g)


@pytest.mark.parametrize("g", G_GRID)
def test_v5_per_step_g_reward_matches_the_env_formula(g):
    # abs_env_incar.py:366 -- step_g_rew = PER_STEP_K * _terminal_g_shape(g)
    expected = abs_env.PER_STEP_K * _terminal_g_shape(g)
    assert RewardSpec.v5().step_g_reward(g) == expected


@pytest.mark.parametrize("g", [0.0, 0.29, 0.3, 0.31, 1.0])
@pytest.mark.parametrize("yaw_error", YAW_GRID)
def test_v5_step_yaw_bonus_matches_the_env_formula(g, yaw_error):
    # abs_env_incar.py:379-382
    if g > abs_env.PER_STEP_G_GATE:
        expected = abs_env.YAW_BONUS_K_STEP * math.exp(-yaw_error * abs_env.YAW_BONUS_ALPHA)
    else:
        expected = 0.0
    assert RewardSpec.v5().step_yaw_bonus(g, yaw_error) == expected


@pytest.mark.parametrize("avg_g", [0.0, 0.3, 0.31, 1.0315, 1.1884, 2.0])
@pytest.mark.parametrize("yaw_abs_sum", [0.0, 0.05, 0.1, 0.5])
@pytest.mark.parametrize("yaw_sq_sum", [0.0, 0.001, 0.05])
def test_v5_terminal_reward_matches_the_env_formula(avg_g, yaw_abs_sum, yaw_sq_sum):
    # abs_env_incar.py:477-489
    expected_g = _terminal_g_shape(avg_g)
    if avg_g > abs_env.PER_STEP_G_GATE:
        clean = max(0.0, 1.0 - yaw_abs_sum / abs_env.YAW_BONUS_THRESHOLD)
        expected_yaw = abs_env.YAW_BONUS_K_TERMINAL * clean
    else:
        expected_yaw = 0.0
    expected_acc = -abs_env.YAW_PEN_K_TERMINAL * yaw_sq_sum
    expected = expected_g + expected_yaw + expected_acc
    got = RewardSpec.v5().terminal_reward(avg_g, yaw_abs_sum, yaw_sq_sum)
    assert got == expected


@pytest.mark.parametrize("yaw_sq_sum", [0.0, 0.001, 0.05, 1.0])
def test_v5_crash_and_timeout_match_the_env_formulas(yaw_sq_sum):
    # abs_env_incar.py:431-444
    acc = -abs_env.YAW_PEN_K_TERMINAL * yaw_sq_sum
    v5 = RewardSpec.v5()
    assert v5.timeout_reward(yaw_sq_sum) == acc
    assert v5.crash_reward(yaw_sq_sum) == abs_env.CRASH_PENALTY + acc


# --- the normalized preset ------------------------------------------------

REFS = (1.0315, 1.1884)   # the measured etk800 @ 60 mph references


def test_normalized_scores_zero_at_stock_and_negative_at_lockup():
    spec = RewardSpec.normalized()
    slam_g, stock_g = REFS
    assert spec.g_shape(stock_g, REFS) == pytest.approx(0.0, abs=1e-9)
    assert spec.g_shape(slam_g, REFS) < -500        # lockup is clearly punished


def test_normalized_fixes_what_v5_got_wrong_on_the_measured_data():
    """The whole reason this preset exists: under v5.0 locking the wheels pays
    +966 and the project's best result (1.043 g) pays +987 -- indistinguishable.
    Normalized must separate them and put stock at zero."""
    v5, norm = RewardSpec.v5(), RewardSpec.normalized()
    slam_g, stock_g = REFS
    best = 1.043

    assert v5.g_shape(slam_g) > 900              # v5 pays ~+966 for lockup
    assert abs(v5.g_shape(best) - v5.g_shape(slam_g)) < 50   # can't tell them apart

    assert norm.g_shape(slam_g, REFS) < 0        # lockup punished
    assert norm.g_shape(best, REFS) > norm.g_shape(slam_g, REFS)   # progress visible
    assert norm.g_shape(best, REFS) < 0          # ...but still short of stock
    assert norm.g_shape(stock_g, REFS) == pytest.approx(0.0, abs=1e-9)


def test_normalized_gatekeeper_fires_only_above_stock():
    spec = RewardSpec.normalized()
    slam_g, stock_g = REFS
    gap = stock_g - slam_g
    just_under = stock_g + 0.05 * gap    # +5% of stock's margin
    well_over = stock_g + 0.30 * gap     # +30%
    assert spec.g_shape(just_under, REFS) < spec.step_bonus
    assert spec.g_shape(well_over, REFS) > spec.step_bonus


def test_normalized_refuses_to_score_without_references():
    """Scoring a normalized spec against absolute anchors is precisely the bug
    normalization exists to fix -- it must raise, never silently fall back."""
    with pytest.raises(ValueError):
        RewardSpec.normalized().g_shape(1.0)


def test_v5_does_not_need_references():
    RewardSpec.v5().g_shape(1.0)   # must not raise


# --- provenance -----------------------------------------------------------

def test_hash_is_stable_and_detects_any_edit():
    a, b = RewardSpec.v5(), RewardSpec.v5()
    assert a.hash() == b.hash()
    b.gatekeeper = 1.07
    assert a.hash() != b.hash()


def test_is_default_flags_customized_specs():
    assert RewardSpec.v5().is_default()
    assert not RewardSpec.normalized().is_default()
    tweaked = RewardSpec.v5()
    tweaked.per_step_k = 0.002
    assert not tweaked.is_default()


def test_roundtrip_through_dict():
    for factory in PRESETS.values():
        spec = factory()
        assert RewardSpec.from_dict(spec.to_dict()).hash() == spec.hash()
