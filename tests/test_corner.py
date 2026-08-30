"""Corner geometry. The HeadingTracker tests matter most: they are what stands
between a 90 deg corner and a fictional CRASH terminal caused by the raw heading
channel wrapping at +-pi."""
import math
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corner import (
    LEFT, RIGHT, RADIUS_MIN, RADIUS_MAX, STEERING_MAX,
    CornerSpec, HeadingTracker, arc_radians, heading_branch_is_safe,
    initial_steering_guess, parse_corner_spec, radius_from_yaw,
    steering_seek_converged, steering_seek_update, target_yaw_rate, wrap_pi,
)


def _crash_heading():
    """abs_env.CRASH_HEADING, read from source rather than imported -- abs_env
    pulls in gymnasium/beamngpy, which the pure-math suite deliberately does
    not need. Still reads the real value, so a change there reaches this test."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "abs_env.py"), encoding="utf-8").read()
    return float(re.search(r"^CRASH_HEADING\s*=\s*([0-9.]+)", src, re.M).group(1))


# ---------------------------------------------------------------- wrap_pi
@pytest.mark.parametrize("raw,expect", [
    (0.0, 0.0),
    (math.pi, math.pi),
    (-math.pi, math.pi),          # (-pi, pi] -- the open end is the negative one
    (math.pi + 0.1, -math.pi + 0.1),
    (3 * math.pi, math.pi),
    (-3.0 * math.pi + 0.2, 0.2 - math.pi),
])
def test_wrap_pi(raw, expect):
    assert wrap_pi(raw) == pytest.approx(expect, abs=1e-9)


# ------------------------------------------------------- target yaw rate
def test_straight_line_target_is_exactly_zero():
    """v5.0's whole yaw reward is |yaw - target| with target 0. A straight
    episode must produce a hard 0.0, not a small float."""
    assert target_yaw_rate(26.8, None) == 0.0
    assert target_yaw_rate(26.8, 0) == 0.0


def test_target_yaw_rate_is_v_over_r_and_signed_by_direction():
    assert target_yaw_rate(26.8, 50.0, LEFT) == pytest.approx(0.536)
    assert target_yaw_rate(26.8, 50.0, RIGHT) == pytest.approx(-0.536)


def test_target_yaw_rate_shrinks_with_speed():
    """The reason it is recomputed every step: a fixed entry-speed value would
    demand 0.54 rad/s from a car doing 5 m/s."""
    fast = target_yaw_rate(26.8, 50.0)
    slow = target_yaw_rate(5.0, 50.0)
    assert slow < fast
    assert slow == pytest.approx(0.1)


def test_radius_from_yaw_round_trips():
    assert radius_from_yaw(26.8, 0.536) == pytest.approx(50.0, rel=1e-3)


def test_radius_from_yaw_is_none_when_not_turning():
    assert radius_from_yaw(26.8, 0.0) is None
    assert radius_from_yaw(26.8, 1e-6) is None


# ------------------------------------------------------------ arc extent
def test_arc_radians_for_a_realistic_stop():
    """60 mph, ~1 g, R=50: about 41 deg of rotation over the stop."""
    arc = arc_radians(26.8, 50.0, 9.81)
    assert arc == pytest.approx(0.732, abs=0.01)
    assert math.degrees(arc) < 90.0


def test_arc_radians_is_zero_for_a_straight():
    assert arc_radians(26.8, None, 9.81) == 0.0


def test_branch_check_passes_near_zero_heading_and_fails_near_pi():
    assert heading_branch_is_safe(0.0, 0.75, LEFT)
    assert not heading_branch_is_safe(math.pi - 0.1, 0.75, LEFT)
    # sweeping AWAY from the cut is fine even when starting close to it
    assert heading_branch_is_safe(math.pi - 0.5, 0.2, RIGHT)


# -------------------------------------------------------- HeadingTracker
def test_tracker_on_a_straight_is_byte_identical_to_todays_behaviour():
    """radius None => advance_target is never called => what we hand the parent
    is exactly start_heading on every step, which is what abs_env_incar sets
    once today and never touches again."""
    t = HeadingTracker(0.7)
    for raw in (0.7, 0.71, 0.69, 0.70):
        t.observe(raw)
        assert t.parent_target() == pytest.approx(0.7, abs=1e-12)
    assert t.target == 0.7


def test_tracker_error_is_zero_when_the_car_tracks_the_arc_perfectly():
    t = HeadingTracker(0.0)
    dt = 0.005
    heading = 0.0
    for _ in range(200):
        t.advance_target(0.5, dt)
        heading += 0.5 * dt
        t.observe(wrap_pi(heading))
        assert t.error == pytest.approx(0.0, abs=1e-9)


def test_tracker_measures_true_error_when_the_car_lags_the_arc():
    t = HeadingTracker(0.0)
    dt = 0.005
    heading = 0.0
    for _ in range(200):
        t.advance_target(0.5, dt)
        heading += 0.4 * dt          # car rotates slower than the arc asks
        t.observe(wrap_pi(heading))
    assert t.error == pytest.approx(-0.1, abs=1e-6)   # 0.1 rad/s deficit over 1 s


def test_a_corner_across_the_branch_cut_really_would_fake_a_crash():
    """Why heading_branch_is_safe exists, demonstrated rather than asserted.
    The parent subtracts raw headings; on the single step where the reading
    jumps +pi -> -pi that subtraction reads ~2pi, far past CRASH_HEADING. The
    tracker cannot repair it from the target side (it does not know which
    branch the NEXT reading will land on), so the geometry is refused up front."""
    crash_heading = _crash_heading()
    t = HeadingTracker(math.pi - 0.05)
    heading, dt, worst = math.pi - 0.05, 0.005, 0.0
    for _ in range(100):
        t.advance_target(0.5, dt)
        parent_target = t.parent_target()
        heading += 0.5 * dt
        raw = wrap_pi(heading)
        t.observe(raw)
        worst = max(worst, abs(raw - parent_target))
    assert worst > crash_heading                      # the fictional crash
    assert not heading_branch_is_safe(math.pi - 0.05, t.arc_rad, LEFT)


def test_tracker_is_exact_for_every_geometry_the_guard_admits():
    """Sweep start headings and both directions; wherever the guard says yes,
    the parent's own arithmetic must equal the true error at every step."""
    crash_heading = _crash_heading()
    arc = 0.5
    for start in [i * 0.25 - math.pi for i in range(26)]:
        for direction in (LEFT, RIGHT):
            if not heading_branch_is_safe(start, arc, direction):
                continue
            t = HeadingTracker(start)
            heading, dt = start, 0.005
            for _ in range(200):
                t.advance_target(direction * 0.5, dt)
                parent_target = t.parent_target()
                heading += direction * 0.4 * dt       # car lags the arc
                raw = wrap_pi(heading)
                t.observe(raw)
                assert abs(raw - parent_target) == pytest.approx(abs(t.error), abs=1e-9)
                assert abs(raw - parent_target) < crash_heading


def test_parent_arithmetic_reproduces_the_true_error():
    """Exactly what abs_env_incar.py:374 computes, fed the tracker's target."""
    t = HeadingTracker(0.0)
    dt = 0.005
    heading = 0.0
    for i in range(300):
        t.advance_target(0.5, dt)
        parent_target = t.parent_target()
        heading += 0.3 * dt
        raw = wrap_pi(heading)
        parent_error = abs(raw - parent_target)
        t.observe(raw)
        assert parent_error == pytest.approx(abs(t.error), abs=1e-9)


def test_tracker_records_how_far_around_the_arc_it_swept():
    t = HeadingTracker(0.0)
    for _ in range(100):
        t.advance_target(0.5, 0.005)
    assert t.arc_rad == pytest.approx(0.25)


# ------------------------------------------------------------ CornerSpec
def test_corner_spec_rejects_radii_outside_the_supported_band():
    with pytest.raises(ValueError):
        CornerSpec(RADIUS_MIN - 1)
    with pytest.raises(ValueError):
        CornerSpec(RADIUS_MAX + 1)


def test_corner_spec_refuses_to_hand_out_a_guessed_steering_angle():
    """Training a corner with an unmeasured angle would silently run some other
    radius than the one the calibration row was measured at."""
    spec = CornerSpec(50.0)
    with pytest.raises(ValueError, match="steering seek"):
        spec.signed_steering


def test_corner_spec_signs_steering_by_direction():
    assert CornerSpec(50.0, LEFT, 0.25).signed_steering == pytest.approx(0.25)
    assert CornerSpec(50.0, RIGHT, 0.25).signed_steering == pytest.approx(-0.25)


def test_corner_spec_radius_is_rounded_to_the_calibration_key_precision():
    from calibration import config_key
    spec = CornerSpec(50.06)
    assert "radius=50.1" in config_key(1.0, 60.0, spec.radius_m)


@pytest.mark.parametrize("text", ["", "off", "none", "straight", "  STRAIGHT "])
def test_parse_corner_spec_none_means_straight(text):
    assert parse_corner_spec(text) is None


def test_parse_corner_spec_directions():
    assert parse_corner_spec("50") == CornerSpec(50.0, LEFT)
    assert parse_corner_spec("50L") == CornerSpec(50.0, LEFT)
    assert parse_corner_spec("50r") == CornerSpec(50.0, RIGHT)


def test_parse_corner_spec_rejects_garbage():
    with pytest.raises(ValueError):
        parse_corner_spec("wide")


# --------------------------------------------------------- steering seek
def test_seek_grows_the_angle_when_the_measured_corner_was_too_wide():
    assert steering_seek_update(0.20, measured_radius=70.0, target_radius=50.0) > 0.20


def test_seek_shrinks_the_angle_when_the_measured_corner_was_too_tight():
    assert steering_seek_update(0.20, measured_radius=35.0, target_radius=50.0) < 0.20


def test_seek_is_damped_not_a_full_ratio_step():
    """A full ratio step overshoots into understeer/slide near the grip limit."""
    full = 0.20 * (70.0 / 50.0)
    damped = steering_seek_update(0.20, 70.0, 50.0)
    assert damped < full


def test_seek_never_exceeds_full_lock():
    assert steering_seek_update(0.9, measured_radius=500.0, target_radius=20.0) == STEERING_MAX


def test_seek_converges_on_a_linear_plant():
    target = 50.0
    steering = initial_steering_guess(target)
    for _ in range(12):
        measured = 12.0 / steering          # toy plant: R = k / steering
        if steering_seek_converged(measured, target):
            break
        steering = steering_seek_update(steering, measured, target)
    assert steering_seek_converged(12.0 / steering, target)


def test_seek_rejects_a_non_yawing_measurement():
    with pytest.raises(ValueError):
        steering_seek_update(0.2, measured_radius=None, target_radius=50.0)


def test_convergence_tolerance_is_tight_enough_to_matter():
    assert steering_seek_converged(51.0, 50.0)
    assert not steering_seek_converged(60.0, 50.0)


def test_initial_guess_is_tighter_steering_for_tighter_corners():
    assert initial_steering_guess(25.0) > initial_steering_guess(100.0)
    assert 0.0 < initial_steering_guess(50.0) <= STEERING_MAX


# ------------------------------------------------- picking a corner by grip
def test_radius_and_lateral_g_round_trip():
    from corner import lateral_g_for_radius, radius_for_lateral_g
    r = radius_for_lateral_g(26.8, 0.4)
    assert lateral_g_for_radius(26.8, r) == pytest.approx(0.4)


def test_a_measured_saturation_matches_the_cars_lateral_limit():
    """The live seek saturated at ~64 m at 60 mph (etk800, dry). That is 1.04 g
    lateral -- the same limit its straight-line slam produces longitudinally,
    which is what saturation SHOULD mean."""
    from corner import lateral_g_for_radius
    assert lateral_g_for_radius(25.5, 63.7) == pytest.approx(1.04, abs=0.02)


def test_a_braking_corner_leaves_grip_to_brake_with():
    """0.4 g lateral at 60 mph is a ~150 m radius, not 50."""
    from corner import radius_for_lateral_g
    assert radius_for_lateral_g(26.8, 0.4) == pytest.approx(183, abs=5)
    assert radius_for_lateral_g(26.8, 0.5) == pytest.approx(146, abs=5)


# ------------------------------------------------------ seek saturation
def test_saturation_is_not_called_before_there_is_evidence():
    from corner import seek_is_saturated
    assert not seek_is_saturated([(0.1, 89.2)])
    assert not seek_is_saturated([(0.1, 89.2), (0.18, 71.5)])


def test_saturation_is_not_called_while_radius_is_still_falling():
    from corner import seek_is_saturated
    assert not seek_is_saturated([(0.1, 89.2), (0.18, 71.5), (0.24, 66.3)])


def test_the_live_etk800_seek_is_detected_as_saturated():
    """The real probe history from 2026-08-29. Detection must fire before the
    8-probe ceiling -- the remaining probes cost minutes and cannot succeed."""
    from corner import seek_is_saturated
    history = [(0.1077, 89.23), (0.1753, 71.49), (0.2356, 66.34),
               (0.2972, 63.74), (0.3625, 63.99), (0.4436, 63.76),
               (0.5413, 64.09), (0.6633, 66.76)]
    fired = next(i for i in range(1, len(history) + 1)
                 if seek_is_saturated(history[:i]))
    assert fired <= 5                     # caught within 5 probes, not 8
    assert min(r for _, r in history[:fired]) == pytest.approx(63.74, abs=0.1)


def test_a_converging_seek_is_never_called_saturated():
    from corner import seek_is_saturated, initial_steering_guess, steering_seek_update
    target, steering, history = 150.0, initial_steering_guess(150.0), []
    for _ in range(10):
        measured = 20.0 / steering        # toy plant, no limit
        history.append((steering, measured))
        assert not seek_is_saturated(history)
        if steering_seek_converged(measured, target):
            break
        steering = steering_seek_update(steering, measured, target)
