"""PARITY IS THE GATE.

The reward only becomes user-editable once RewardSpec.v5() is proven to
reproduce the protected abs_env.py's numbers exactly. These tests compare
against the REAL constants and _terminal_g_shape imported from that file --
not a recorded snapshot, so if anyone ever edits the protected copy, or the
extraction drifts, this fails loudly instead of silently changing what every
past result meant.
"""
import math
import os
import sys
import time

import numpy as np
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
    # abs_env_incar.py:366, step_g_rew = PER_STEP_K * _terminal_g_shape(g)
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
    +966 and the project's best result (1.043 g) pays +987, indistinguishable.
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
    normalization exists to fix, it must raise, never silently fall back."""
    with pytest.raises(ValueError):
        RewardSpec.normalized().g_shape(1.0)


def test_v5_does_not_need_references():
    RewardSpec.v5().g_shape(1.0)   # must not raise


# --- the target-free v6 preset -------------------------------------------

def _v6_dense_total(samples, dt=0.01):
    tracker = RewardSpec.v6().make_step_tracker(dt)
    return sum(tracker.push(g) for g in samples)


def test_v6_keeps_yaw_out_of_dense_shaping_and_uses_the_compact_scale():
    spec = RewardSpec.v6()
    assert spec.step_yaw_bonus(1.0, 0.0) == 0.0
    assert spec.step_yaw_bonus(1.0, 99.0) == 0.0
    assert spec.accumulated_yaw_penalty(99.0) == 0.0
    assert spec.terminal_reward(1.0, 0.0, 0.0) == 25.0
    assert spec.terminal_reward(1.0, 99.0, 99.0) == 22.0
    assert spec.crash_reward(99.0) == -50.0
    assert spec.timeout_reward(99.0) == -30.0


def test_v6_terminal_stability_guard_is_deadzoned_graded_and_capped():
    spec = RewardSpec.v6()
    assert spec.terminal_stability_penalty(0.0) == 0.0
    assert spec.terminal_stability_penalty(0.08) == 0.0
    assert spec.terminal_stability_penalty(0.14) == pytest.approx(-0.75)
    assert spec.terminal_stability_penalty(0.20) == -3.0
    assert spec.terminal_stability_penalty(99.0) == -3.0
    assert spec.terminal_reward(1.0, 0.14, 0.0) == pytest.approx(24.25)


def test_v6_terminal_reward_is_monotonic_and_has_no_upper_plateau():
    spec = RewardSpec.v6()
    assert spec.g_shape(0.9) < spec.g_shape(1.0) < spec.g_shape(1.5)
    assert spec.g_shape(1.5) < spec.g_shape(2.5) < spec.g_shape(5.0)


def test_terminal_component_log_exactly_reproduces_both_frozen_rewards():
    for spec in (RewardSpec.v5(), RewardSpec.v6()):
        components = spec.terminal_reward_components(1.03, 0.14, 0.02)
        assert components["total"] == pytest.approx(
            spec.terminal_reward(1.03, 0.14, 0.02))
        assert components["total"] == pytest.approx(sum(
            value for key, value in components.items() if key != "total"))


def test_v6_rewards_steady_g_over_an_equal_mean_oscillation():
    steady = _v6_dense_total([0.9] * 200)
    oscillating = _v6_dense_total([1.5, 0.3] * 100)
    assert sum([0.9] * 200) == pytest.approx(sum([1.5, 0.3] * 100))
    assert steady > oscillating


def test_v6_still_rewards_higher_steady_g():
    assert _v6_dense_total([1.5] * 200) > _v6_dense_total([1.0] * 200)


def test_v6_dense_reward_uses_seconds_not_step_count():
    at_100_hz = _v6_dense_total([1.0] * 100, dt=0.01)
    at_200_hz = _v6_dense_total([1.0] * 200, dt=0.005)
    assert at_100_hz == pytest.approx(1.0)
    assert at_200_hz == pytest.approx(at_100_hz)


def test_v6_higher_g_stop_wins_even_when_its_episode_is_shorter():
    """Fixed initial speed: higher G needs fewer samples, but must still win."""
    spec = RewardSpec.v6()
    start_speed_ms = 80 * 0.44704
    slow_steps = round(start_speed_ms / (1.0 * 9.81 * 0.01))
    fast_steps = round(start_speed_ms / (1.5 * 9.81 * 0.01))
    slow = _v6_dense_total([1.0] * slow_steps) + spec.terminal_reward(1.0, 0, 0)
    fast = _v6_dense_total([1.5] * fast_steps) + spec.terminal_reward(1.5, 0, 0)
    assert fast_steps < slow_steps
    assert fast > slow


def test_v6_stability_guard_is_bounded_relative_to_primary_terminal_g():
    spec = RewardSpec.v6()
    primary_at_one_g = spec.terminal_reward_components(1.0, 0, 0)["terminal_g"]
    worst_stability = abs(spec.terminal_stability_penalty(float("inf")))
    assert worst_stability == 3.0
    assert worst_stability < 0.15 * primary_at_one_g


@pytest.mark.parametrize("g", [-1e6, -1.0, 0.0, 1.0, 1e6])
@pytest.mark.parametrize("yaw", [0.0, 0.2, 1e6])
def test_v6_reward_is_finite_at_realistic_and_boundary_inputs(g, yaw):
    spec = RewardSpec.v6()
    tracker = spec.make_step_tracker(0.01)
    assert math.isfinite(tracker.push(g))
    assert math.isfinite(spec.terminal_reward(g, yaw, yaw * yaw))


def test_v6_dense_reward_ignores_negative_longitudinal_g():
    tracker = RewardSpec.v6().make_step_tracker(0.01)
    assert tracker.push(-5.0) == 0.0
    assert tracker.rolling_mean == 0.0


def test_v6_refuses_stateless_dense_scoring():
    with pytest.raises(RuntimeError, match="rolling reward"):
        RewardSpec.v6().step_g_reward(1.0)


# --- provenance -----------------------------------------------------------

def test_hash_is_stable_and_detects_any_edit():
    a, b = RewardSpec.v5(), RewardSpec.v5()
    assert a.hash() == "f9c8677e389bffd6"  # adding v6 must not rename old runs
    assert a.hash() == b.hash()
    b.gatekeeper = 1.07
    assert a.hash() != b.hash()


def test_implementation_hash_is_stable_strong_and_includes_coefficients():
    a, b = RewardSpec.v6(), RewardSpec.v6()
    assert len(a.implementation_hash()) == 64
    assert a.implementation_hash() == b.implementation_hash()
    b.consistency_k += 0.01
    assert a.implementation_hash() != b.implementation_hash()


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


# ---------------------------------------------------------------- v7.0
def test_v7_terminal_g_matches_v6_when_the_stop_is_straight():
    v6, v7 = RewardSpec.v6(), RewardSpec.v7()
    for g in (0.5, 1.0, 1.2, 1.5):
        assert v7.terminal_reward(g, 0.0, 0.0) == v6.terminal_reward(g, 0.0, 0.0)
    assert v7.timeout_penalty == v6.timeout_penalty
    assert v7.crash_penalty == v6.crash_penalty


def test_v7_yaw_guard_is_worth_a_whole_1g_stop_and_the_backstop_keeps_growing():
    spec = RewardSpec.v7()
    assert spec.terminal_stability_penalty(0.08) == 0.0          # deadzone kept
    assert spec.terminal_stability_penalty(0.20) == -25.0        # == terminal G at 1.0 g
    assert spec.terminal_stability_penalty(0.50) == -25.0        # capped
    assert spec.accumulated_yaw_penalty(0.10) == -20.0
    assert spec.accumulated_yaw_penalty(1.00) == -200.0          # uncapped


def test_v7_a_spin_cannot_outscore_a_straight_stop_even_with_perfect_slip_and_g():
    spec = RewardSpec.v7()
    best_slip = sum(spec.step_slip_reward([0.2] * 4, 0.01) for _ in range(300))  # +30
    spun = best_slip + spec.terminal_reward(1.5, 1.0, 1.0)       # 1.5 g but 1 rad of yaw
    straight = spec.terminal_reward(0.9, 0.0, 0.0)               # a mediocre straight stop
    assert spun < 0 < straight
    crash = best_slip + spec.crash_reward(1.0)
    assert crash < spun


def test_v7_has_no_dense_g_per_step():
    tracker = RewardSpec.v7().make_step_tracker(0.01)
    assert all(tracker.push(g) == 0.0 for g in (0.0, 0.5, 1.0, 1.5))
    assert RewardSpec.v7().step_yaw_bonus(1.0, 0.0) == 0.0


def test_v7_slip_bands_reward_penalty_and_triple_lock_penalty():
    spec = RewardSpec.v7()
    dt = 0.01
    ok = spec.step_slip_reward([0.50] * 4, dt)
    pen = spec.step_slip_reward([0.51] * 4, dt)
    lock = spec.step_slip_reward([0.99] * 4, dt)
    assert 0 > ok > pen > lock
    assert ok == -5.0 * dt
    assert pen == -10.0 * dt
    assert lock == -30.0 * dt
    assert abs(lock / pen - 3.0) < 1e-12
    assert spec.step_slip_reward([0.98] * 4, dt) == pen
    assert spec.step_slip_reward([1.00] * 4, dt) == lock
    assert spec.step_slip_reward([0.0] * 4, dt) == ok


def test_v7_slip_term_is_wheel_averaged_and_dt_scaled():
    spec = RewardSpec.v7()
    mixed = spec.step_slip_reward([0.1, 0.6, 1.0, 0.5], 0.01)
    assert abs(mixed - 0.01 * (-5.0 - 10.0 - 30.0 - 5.0) / 4) < 1e-12
    assert spec.step_slip_reward([0.1] * 4, 0.02) == 2 * spec.step_slip_reward([0.1] * 4, 0.01)
    assert spec.step_slip_reward([], 0.01) == 0.0
    with pytest.raises(ValueError):
        spec.step_slip_reward([0.1] * 4, 0.0)


def test_v7_per_step_term_is_never_positive():
    spec = RewardSpec.v7()
    for slip in (0.0, 0.2, 0.5, 0.51, 0.98, 0.99, 1.0):
        assert spec.step_slip_reward([slip] * 4, 0.01) < 0.0
    assert spec.step_yaw_bonus(1.0, 0.0) == 0.0
    assert spec.make_step_tracker(0.01).push(1.0) == 0.0


def test_v7_shortest_clean_stop_beats_coasting_and_slamming():
    """PPO-49 learned 24 s coasts because +10/s ok-band paid for duration."""
    spec = RewardSpec.v7()

    def episode(seconds, slip, avg_g, outcome="STOP"):
        steps = int(round(seconds / 0.01))
        dense = sum(spec.step_slip_reward([slip] * 4, 0.01) for _ in range(steps))
        if outcome == "STOP":
            return dense + spec.terminal_reward(avg_g, 0.0, 0.0)
        return dense + spec.timeout_reward(0.0)

    optimal = episode(3.3, 0.20, 1.10)
    gentle = episode(6.0, 0.05, 0.60)
    coast = episode(24.0, 0.01, 0.0, outcome="TIMEOUT")
    slam = episode(3.5, 1.00, 0.95)
    assert optimal > gentle > coast
    assert optimal > gentle > slam
    assert optimal > 0


def test_slip_term_is_inert_on_every_older_preset():
    for name, factory in PRESETS.items():
        if name in ("v7.0", "v8.0", "v10.0"):
            continue
        spec = factory()
        assert not spec.slip_term_active()
        assert spec.step_slip_reward([1.0] * 4, 0.01) == 0.0


def test_adding_v7_did_not_rename_v6_or_v5():
    assert RewardSpec.v5().hash() == "f9c8677e389bffd6"
    assert RewardSpec.v6().hash() == "e2c9721354b99f3a"
    assert RewardSpec.v7().hash() not in ("f9c8677e389bffd6", "e2c9721354b99f3a")
    assert RewardSpec.v6().implementation_hash() != RewardSpec.v7().implementation_hash()


def test_v7_hash_detects_a_band_edit():
    a, b = RewardSpec.v7(), RewardSpec.v7()
    assert a.hash() == b.hash()
    b.slip_ok_max = 0.45
    assert a.hash() != b.hash()


def test_cosim_true_slip_is_ground_speed_referenced_and_reward_only():
    import abs_env_cosim
    from abs_env_cosim import ABSCoSimEnv
    v = [0.0] * len(abs_env_cosim.SIG_TO)
    v[abs_env_cosim.I_GS] = 20.0
    v[abs_env_cosim.I_WS0:abs_env_cosim.I_WS0 + 4] = [20.0, 10.0, 0.0, 30.0]
    v[abs_env_cosim.I_FUSED] = 40.0   # a wrong fused estimate must not leak into the reward
    slips = ABSCoSimEnv._true_slips(v, 20.0)
    assert list(slips) == [0.0, 0.5, 1.0, 0.0]
    assert ABSCoSimEnv._true_slips(v, 0.5) == ()


# ---------------------------------------------------------------- v7 wheels
def test_wheel_release_mapping_is_fully_independent():
    from residual_core import wheel_release_to_brakes, axle_view, BRAKE_FLOOR
    fr, fl, rr, rl = wheel_release_to_brakes([0.0, 0.5, 1.0, 0.25], pedal=1.0)
    assert (fr, fl, rr, rl) == (1.0, 0.5, BRAKE_FLOOR, 0.75)
    assert fr != fl and rr != rl
    assert axle_view([0.2, 0.4, 0.6, 1.0]) == pytest.approx((0.3, 0.8))
    assert axle_view([0.3, 0.8]) == (0.3, 0.8)


def test_cosim_env_obs_is_a_35x64_past_window_for_every_wheel_mode():
    import abs_env_cosim as E
    from abs_env_cosim import ABSCoSimEnv, obs_dim_for, act_dim_for
    assert E.FRAME_DIM == 35 and E.N_STACK == 64 and E.OBS_DIM == 2240
    assert (obs_dim_for("axle"), act_dim_for("axle")) == (2240, 2)
    assert (obs_dim_for("independent"), act_dim_for("independent")) == (2240, 4)
    env = ABSCoSimEnv.__new__(ABSCoSimEnv)
    env._configure_wheel_mode("independent")
    assert env.action_space.shape == (4,) and env.observation_space.shape == (2240,)
    assert env._action_to_brakes([0.0, 0.5, 1.0, 0.25])[1] == 0.5

    def packet(ws, gy=5.0, pitch=0.0):
        v = [0.0] * len(E.SIG_TO)
        v[E.I_GS] = 20.0
        v[E.I_FUSED] = 20.0
        v[E.I_WS0:E.I_WS0 + 4] = ws
        v[E.I_WGY] = gy
        v[E.I_WGYMIN] = gy - 1
        v[E.I_WGYMAX] = gy + 1
        v[E.I_RPM] = 3000.0
        v[E.I_GEAR] = 2.0
        v[E.I_PITCH] = pitch
        v[E.I_BRK] = 1.0
        v[E.I_ABSSPD] = 20.0
        v[E.I_BRKA0:E.I_BRKA0 + 4] = [3100.0, 3000.0, 1700.0, 1600.0]
        return v

    env._prev_brakes = (0.9, 0.8, 0.7, 0.6)
    obs1 = env._obs(packet([20.0, 18.0, 16.0, 20.0]))
    assert obs1.shape == (2240,)
    # zero history: only the newest slot (last 31) is populated after reset
    assert not obs1[:35 * 63].any()
    f1 = obs1[-35:]
    assert list(f1[0:4]) == [20.0, 18.0, 16.0, 20.0]
    assert f1[4] == 5.0 and f1[11] == 4.0 and f1[12] == 6.0
    assert list(f1[7:11]) == [0.9, 0.8, 0.7, 0.6]
    assert f1[13] == 3000.0 and f1[14] == 2.0 and f1[18] == 1.0
    assert not f1[21:27].any()                      # no previous frame: rates are 0
    assert abs(f1[28] - 0.1) < 1e-6 and f1[27] == 0.0  # slips vs the ABS speed (20 m/s)
    assert list(f1[31:35]) == [3100.0, 3000.0, 1700.0, 1600.0]  # applied brake Nm
    obs2 = env._obs(packet([19.0, 18.0, 16.0, 20.0], pitch=0.01))
    # the older frame moved one slot toward the front, newest is last
    assert np.array_equal(obs2[35 * 62:35 * 63], f1)
    f2 = obs2[-35:]
    assert abs(f2[23] - (-1.0 / E.DT)) < 1e-3       # FR wheel accel from PAST frame
    assert abs(f2[21] - (0.01 / E.DT)) < 1e-3       # pitch rate from PAST frame
    assert f2[0] == 19.0
    for _ in range(70):
        env._obs(packet([10.0] * 4))
    obs = env._obs(packet([9.0] * 4))
    assert obs[-35] == 9.0 and obs[0] == 10.0        # oldest-first, newest-last


def test_cosim_frame_matches_machinetrainerboy_layout_and_bounds():
    import abs_env_cosim as E
    assert E.FRAME_LOW.shape == (35,) and E.FRAME_HIGH.shape == (35,)
    assert float(E.FRAME_HIGH[13]) == 15000.0
    assert np.isclose(float(E.FRAME_LOW[16]), -np.pi)
    env = E.ABSCoSimEnv.__new__(E.ABSCoSimEnv)
    env._configure_wheel_mode("axle")
    v = [0.0] * len(E.SIG_TO)
    v[E.I_RPM] = 1e9
    v[E.I_WS0] = float("nan")
    frame = env._frame(v)
    assert frame[13] == 15000.0 and frame[0] == 0.0 and np.all(np.isfinite(frame))


def test_v7_defaults_to_independent_wheels_and_older_presets_keep_the_axle_lock():
    from train_cosim import parse_wheel_mode
    assert parse_wheel_mode(None, "v7.0") == "independent"
    assert parse_wheel_mode("", "v7.0") == "independent"
    assert parse_wheel_mode(None, "v6.0") == "axle"
    assert parse_wheel_mode(None, "v6.1") == "independent"
    assert parse_wheel_mode(None, "v8.0") == "independent"
    assert parse_wheel_mode(None, "v3.3") == "independent"
    assert parse_wheel_mode("axle", "v7.0") == "axle"
    assert parse_wheel_mode("Independent", "v6.0") == "independent"
    with pytest.raises(SystemExit):
        parse_wheel_mode("diagonal", "v7.0")


def test_v6_1_is_v6_scoring_with_a_distinct_name_and_hash():
    v6, v61 = RewardSpec.v6(), RewardSpec.v6_1()
    assert v61.name == "v6.1" and PRESETS["v6.1"]().name == "v6.1"
    assert v61.hash() != v6.hash() and v6.hash() == "e2c9721354b99f3a"
    assert {k: v for k, v in v61.to_dict().items() if k != "name"} ==         {k: v for k, v in v6.to_dict().items() if k != "name"}
    for g in (0.3, 0.9, 1.2):
        assert v61.terminal_reward(g, 0.05, 0.0) == v6.terminal_reward(g, 0.05, 0.0)
    assert not v61.slip_term_active()


# ---------------------------------------------------------------- v8.0
def test_v8_adds_a_capped_window_g_reward_on_top_of_v7_and_stays_negative_per_step():
    v7, v8 = RewardSpec.v7(), RewardSpec.v8()
    assert v8.name == "v8.0" and PRESETS["v8.0"]().name == "v8.0"
    assert v8.step_mode == "window_g" and v8.dense_g_source == "window"
    assert v8.dense_g_k == 3.0 and v8.consistency_k == 0.0
    assert v8.dense_g_k < -v8.slip_ok_k          # G reward can never out-pay the time cost
    for k in ("slip_ok_k", "slip_pen_k", "slip_lock_k", "slip_ok_max", "slip_lock_min",
              "yaw_pen_k_terminal", "yaw_stability_penalty_max", "crash_penalty",
              "timeout_penalty", "ramp_pos_k"):
        assert getattr(v8, k) == getattr(v7, k)
    tracker = v8.make_step_tracker(0.01)
    assert abs(tracker.push(1.0) - 0.03) < 1e-12
    assert tracker.push(-2.0) == 0.0
    for g in (0.0, 0.5, 1.0, 1.3, 1.6):
        per_step = tracker.push(g) + v8.step_slip_reward([0.2] * 4, 0.01)
        assert per_step < 0.0, (g, per_step)
    # v7 unchanged and distinct: it never pays per-step G
    assert v7.make_step_tracker(0.01).push(1.0) == 0.0
    assert v8.hash() != v7.hash()
    assert v7.hash() == "e613d15d1c21cf31" and RewardSpec.v6().hash() == "e2c9721354b99f3a"


def test_v8_orders_hard_clean_stop_above_coast_timeout_and_slam():
    spec = RewardSpec.v8()

    def episode(seconds, slip, g, avg_g, outcome="STOP"):
        tracker = spec.make_step_tracker(0.01)
        steps = int(round(seconds / 0.01))
        dense = sum(tracker.push(g) + spec.step_slip_reward([slip] * 4, 0.01)
                    for _ in range(steps))
        if outcome == "STOP":
            return dense + spec.terminal_reward(avg_g, 0.0, 0.0)
        return dense + spec.timeout_reward(0.0)

    optimal = episode(3.3, 0.20, 1.10, 1.10)
    gentle = episode(6.0, 0.05, 0.60, 0.60)
    coast = episode(24.0, 0.01, 0.15, 0.0, outcome="TIMEOUT")
    slam = episode(3.5, 1.00, 0.95, 0.95)
    assert optimal > gentle > coast and optimal > gentle > slam and optimal > 0
    # the per-step gradient v7 lacks: at equal duration, harder G pays more every step
    t = spec.make_step_tracker(0.01)
    assert t.push(1.0) > t.push(0.6) > t.push(0.2)


def test_v8_yaw_guard_ignores_integrated_noise_but_still_caps_at_25():
    spec = RewardSpec.v8()
    assert spec.terminal_stability_penalty(0.21) == 0.0      # v7 pinned -25 here
    assert spec.terminal_stability_penalty(0.30) == 0.0
    assert -25.0 < spec.terminal_stability_penalty(0.40) < 0.0
    assert spec.terminal_stability_penalty(0.45) == -25.0
    assert spec.terminal_stability_penalty(2.0) == -25.0
    assert spec.accumulated_yaw_penalty(0.1) == -20.0        # backstop untouched


def test_cosim_env_feeds_window_g_only_to_window_presets():
    import abs_env_cosim as E
    env = E.ABSCoSimEnv.__new__(E.ABSCoSimEnv)
    v = [0.0] * len(E.SIG_TO)
    v[E.I_WGY] = 9.81 * 0.8
    env._reward = RewardSpec.v8()
    assert abs(env._dense_g(v, 0.3) - 0.8) < 1e-9
    env._reward = RewardSpec.v7()
    assert env._dense_g(v, 0.3) == 0.3
    env._reward = RewardSpec.v6()
    assert env._dense_g(v, 0.3) == 0.3


# ---------------------------------------------------------------- v3.3 (FastTrain clone)
def test_v33_terminal_shape_matches_fasttrain_v33_numbers():
    spec = RewardSpec.v3_3()
    assert spec.name == "v3.3" and PRESETS["v3.3"]().name == "v3.3"
    # -140 at/below 0.4 g, breakeven 0.5 g, +2100 at/above 2 g, log gate above 1.06 g
    assert spec.g_shape(0.0) == -140.0 and spec.g_shape(0.4) == -140.0
    assert abs(spec.g_shape(0.5)) < 1e-9
    top = 2100.0 + math.log1p((2.0 - 1.06) * 10.0) * 2500.0     # gate keeps paying to the 2 g clamp
    assert abs(spec.g_shape(2.0) - top) < 1e-9 and spec.g_shape(3.0) == spec.g_shape(2.0)
    assert spec.g_shape(1.06) == spec.v33_affine(1.06)
    expected_116 = spec.v33_affine(1.16) + math.log1p(1.0) * 2500.0
    assert abs(spec.g_shape(1.16) - expected_116) < 1e-9
    assert spec.terminal_reward(1.16, 0.5, 0.5) == spec.g_shape(1.16)   # no terminal yaw terms
    assert spec.crash_reward(1.0) == -500.0 and spec.timeout_reward(1.0) == 0.0


def test_v33_per_step_keeps_fasttrain_per_second_weight_at_100hz():
    spec = RewardSpec.v3_3()
    assert spec.step_mode == "v33_step" and spec.dense_g_source == "window"
    tracker = spec.make_step_tracker(0.01)
    per_step_1g = tracker.push(1.0)
    fasttrain_160hz = 0.000225 * (-140.0 * (1 - 0.375) + 2100.0 * 0.375)
    assert abs(per_step_1g * 100.0 - fasttrain_160hz * 160.0) < 1e-9
    assert tracker.push(0.4) < 0.0 < tracker.push(0.6)                    # time cost below 0.5 g
    assert spec.step_slip_reward([1.0] * 4, 0.01) == 0.0                # no slip term


def test_v33_heading_penalty_and_recovery_are_per_step_and_inert_elsewhere():
    spec = RewardSpec.v3_3()
    assert spec.step_heading_penalty(0.1, 0.1) == -1000.0 * 0.01
    assert abs(spec.step_heading_penalty(0.1, 0.12) - (-10.0 + 500.0 * 0.02)) < 1e-9  # shrinking error pays
    assert abs(spec.step_heading_penalty(0.12, 0.1) - (-1000.0 * 0.0144 - 10.0)) < 1e-9
    for name, factory in PRESETS.items():
        if name != "v3.3":
            assert factory().step_heading_penalty(0.5, 0.0) == 0.0


def test_v33_does_not_touch_other_hashes_and_orders_hard_stop_first():
    assert RewardSpec.v6().hash() == "e2c9721354b99f3a"
    assert RewardSpec.v7().hash() == "e613d15d1c21cf31"
    v1 = RewardSpec.v3_3()
    assert v1.hash() not in {RewardSpec.v7().hash(), RewardSpec.v8().hash(), RewardSpec.v5().hash()}

    def episode(seconds, g, avg_g, outcome="STOP"):
        t = v1.make_step_tracker(0.01)
        dense = sum(t.push(g) for _ in range(int(round(seconds / 0.01))))
        return dense + (v1.terminal_reward(avg_g, 0.0, 0.0) if outcome == "STOP"
                        else v1.timeout_reward(0.0))

    optimal = episode(3.3, 1.10, 1.10)
    gentle = episode(6.0, 0.60, 0.60)
    coast = episode(24.0, 0.15, 0.0, outcome="TIMEOUT")
    slam = episode(3.5, 0.95, 0.95)
    assert optimal > slam and optimal > gentle > coast and coast < 0
    fastest = episode(3.0, 1.20, 1.20)
    assert fastest > optimal                                             # no upper plateau


def test_v9_dense_term_is_negative_distance():
    """The v9 dense sum must equal -speed_cost_k * metres travelled."""
    spec = RewardSpec.v9()
    dt = 0.01
    tracker = spec.make_step_tracker(dt)
    speeds = [35.76 - 6.0 * i * dt for i in range(500)]
    speeds = [s for s in speeds if s > 0]
    dense = sum(tracker.push(0.0, speed=s) for s in speeds)
    metres = sum(s * dt for s in speeds)
    assert dense == pytest.approx(-spec.speed_cost_k * metres)


def test_v9_total_reward_is_monotone_in_g():
    """Harder stops must score strictly higher, lockup to well past stock."""
    spec = RewardSpec.v9()
    dt = 0.01
    totals = []
    for g in (0.60, 0.80, 0.979, 1.06, 1.10, 1.15, 1.19, 1.25):
        a = g * 9.81
        v0 = 35.76
        tracker = spec.make_step_tracker(dt)
        dense = 0.0
        v = v0
        while v > 0:
            dense += tracker.push(0.0, speed=v)
            v -= a * dt
        totals.append(dense + spec.terminal_reward(g, 0.15, 0.02))
    assert totals == sorted(totals)
    assert all(b - a > 1.0 for a, b in zip(totals, totals[1:]))


def test_v9_does_not_change_older_preset_hashes():
    assert RewardSpec.v5().hash() == "f9c8677e389bffd6"
    assert RewardSpec.v7().hash() == "e613d15d1c21cf31"
    assert RewardSpec.v8().hash() == "bd4a2cd2482a36e1"
    assert RewardSpec.v3_3().hash() == "4726e7c1a5d39118"


def test_v9_failures_score_below_the_worst_legitimate_stop():
    """Not stopping must never beat stopping badly.

    A timeout crawls to a halt, so its dense cost is small (speed decays, and the
    cost is proportional to speed). Without an explicit penalty the timeout
    outscored a genuine 0.66 g stop, and PPO-60 exploited it within 11 episodes.
    """
    for spec in (RewardSpec.v9(), RewardSpec.v10()):
        dt = 0.01
        tracker = spec.make_step_tracker(dt)
        v, a = 35.76, 0.4 * 9.81
        dense_stop = 0.0
        while v > 0:
            dense_stop += tracker.push(0.0, speed=v)
            v -= a * dt
        worst_stop = dense_stop + spec.terminal_reward(0.4, 0.15, 0.02)

        tracker = spec.make_step_tracker(dt)
        v, a = 35.76, 0.06 * 9.81          # a 24 s crawl that never stops
        dense_timeout = 0.0
        for _ in range(2400):
            dense_timeout += tracker.push(0.0, speed=max(0.0, v))
            v -= a * dt
        timeout = dense_timeout + spec.timeout_reward_components(0.02)["total"]
        crash = dense_timeout + spec.crash_reward_components(0.02)["total"]

        assert timeout < worst_stop, spec.name
        assert crash < worst_stop, spec.name


def test_v11_charges_distance_only_inside_the_scored_window():
    """v11's dense sum must equal minus the SCORED distance, not the whole run."""
    spec = RewardSpec.v11()
    assert spec.metric_window_only is True
    assert spec.approach_time_k > 0.0
    dt = 0.01
    tracker = spec.make_step_tracker(dt)
    # the approach contributes no distance cost at all
    assert tracker.push(0.0, speed=0.0) == 0.0
    scored = [30.0, 20.0, 10.0]
    dense = sum(tracker.push(0.0, speed=s) for s in scored)
    assert dense == pytest.approx(-spec.speed_cost_k * sum(s * dt for s in scored))


def test_v11_keeps_v9_ordering_and_v9_is_untouched():
    v9, v11 = RewardSpec.v9(), RewardSpec.v11()
    assert v9.metric_window_only is False and v9.approach_time_k == 0.0
    for g in (0.6, 0.979, 1.06, 1.19):
        assert v11.terminal_reward(g, 0.15, 0.02) == v9.terminal_reward(g, 0.15, 0.02)


def test_cosim_link_drain_discards_queued_packets():
    """A queued backlog must not be served to the next episode as live data."""
    import socket
    import cosim_link
    link = cosim_link.CoSimLink([("a", "b")], [("c", "d")], time_3rd_party=0.0025)
    link.open()
    try:
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for _ in range(25):
            tx.sendto(b"stale", (cosim_link.SEND_IP, cosim_link.SEND_PORT))
        tx.close()
        time.sleep(0.05)
        assert link.drain() >= 1
        assert link.drain() == 0          # queue is empty and drain does not block
    finally:
        link.close()
