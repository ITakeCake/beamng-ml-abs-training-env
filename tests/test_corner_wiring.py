"""Corner wiring inside the env: the steering injector, the per-step yaw target,
and -- the one that guards everything already shipped -- that a straight-line
episode is completely untouched by any of it."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import abs_env_residual
from abs_env_residual import ARC_DECEL_ESTIMATE_MS2, CornerSteerInjector
from corner import LEFT, RIGHT, CornerSpec


class FakeVehicle:
    def __init__(self):
        self.sent = []
        self.controls = []

    def queue_lua_command(self, cmd):
        self.sent.append(cmd)

    def control(self, **kwargs):
        self.controls.append(kwargs)


# --------------------------------------------------- CornerSteerInjector
def test_wheel_goes_on_right_after_the_slam_is_armed():
    """armBrakeSlam is the parent's last mention of steering; everything after
    it is the coast-down, so this is the turn-in point."""
    veh = FakeVehicle()
    inj = CornerSteerInjector(veh, steering=0.25)
    with inj:
        veh.queue_lua_command("extensions.load('abstelemetry')")
        assert veh.controls == []                       # straight until then
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8224)")
        veh.queue_lua_command("something.else()")

    assert veh.controls == [{"steering": 0.25}]
    assert inj.applied is True


def test_steering_is_applied_exactly_once():
    veh = FakeVehicle()
    with CornerSteerInjector(veh, steering=0.25):
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
    assert len(veh.controls) == 1


def test_a_right_hand_corner_turns_the_other_way():
    veh = FakeVehicle()
    with CornerSteerInjector(veh, steering=CornerSpec(50.0, RIGHT, 0.25).signed_steering):
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
    assert veh.controls == [{"steering": -0.25}]


def test_no_corner_means_the_wheel_is_never_touched():
    veh = FakeVehicle()
    with CornerSteerInjector(veh, steering=None):
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
    assert veh.controls == []
    assert veh.sent == ["extensions.abstelemetry.armBrakeSlam(26.8)"]


def test_a_callable_angle_is_resolved_at_the_arm_point_not_before():
    """The angle can depend on the episode's target speed, which the parent
    only picks partway through its own reset."""
    calls = []

    def late_angle():
        calls.append("resolved")
        return 0.31

    veh = FakeVehicle()
    with CornerSteerInjector(veh, steering=late_angle) as inj:
        veh.queue_lua_command("extensions.load('abstelemetry')")
        assert calls == []                              # not resolved yet
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
    assert calls == ["resolved"]
    assert veh.controls == [{"steering": 0.31}]
    assert inj.angle == pytest.approx(0.31)


def test_restores_queue_lua_command_even_if_reset_raises():
    veh = FakeVehicle()
    original = veh.queue_lua_command
    with pytest.raises(RuntimeError):
        with CornerSteerInjector(veh, steering=0.25):
            raise RuntimeError("reset blew up")
    assert veh.queue_lua_command == original


# ------------------------------------------------ per-step target wiring
class StubEnv(abs_env_residual.ABSLearningEnvResidual):
    """Exercises _start_corner_tracking / _advance_corner_target without a
    game: everything they touch is plain attributes on self."""

    def __init__(self, corner_spec, start_heading=0.0, start_speed_ms=26.8):
        self._corner = corner_spec
        self.radius_m = corner_spec.radius_m if corner_spec else None
        self._heading = None
        self.start_heading = start_heading
        self.start_speed_ms = start_speed_ms
        self.current_heading = start_heading
        self.target_heading = start_heading
        self.target_yaw_rate = 0.0
        # What abs_env_incar.reset() actually leaves behind: a sentinel that is
        # only replaced partway through the first step().
        self._last_gps_speed = 999.0
        self._dt = 1.0 / 200.0


def test_straight_line_leaves_the_parents_target_exactly_where_it_was():
    """The regression guard for everything already shipped: with no corner,
    target_yaw_rate stays a hard 0.0 and target_heading stays the start
    heading, which is what abs_env_incar sets and never changes today."""
    env = StubEnv(None, start_heading=0.7)
    env._start_corner_tracking()
    for _ in range(50):
        env._advance_corner_target()
        assert env.target_yaw_rate == 0.0
        assert env.target_heading == pytest.approx(0.7, abs=1e-12)
        env.current_heading = 0.7 + 0.001    # a little real-world wander
        env._heading.observe(env.current_heading)


def test_the_first_step_uses_the_start_speed_not_the_parents_sentinel():
    """abs_env_incar.reset() leaves _last_gps_speed at 999.0 and only fills it
    in partway through step(). Reading it before the first step would demand
    999/R rad/s -- one step of that exhausts the whole terminal yaw budget and
    fires the catastrophic backstop on every single corner episode."""
    env = StubEnv(CornerSpec(50.0, LEFT, 0.25), start_speed_ms=26.8)
    env._start_corner_tracking()
    env._advance_corner_target()
    assert env.target_yaw_rate == pytest.approx(26.8 / 50.0)
    assert env._heading.arc_rad < 0.01


def test_yaw_target_is_v_over_r_and_falls_with_speed():
    env = StubEnv(CornerSpec(50.0, LEFT, 0.25))
    env._start_corner_tracking()
    env._advance_corner_target()
    entry = env.target_yaw_rate
    assert entry == pytest.approx(26.8 / 50.0)

    env._corner_speed = 5.0                  # late in the stop
    env._advance_corner_target()
    assert env.target_yaw_rate == pytest.approx(0.1)
    assert env.target_yaw_rate < entry


def test_the_parents_heading_error_is_zero_for_a_car_that_holds_the_arc():
    """Drive the arc exactly and step the env's own bookkeeping; the value
    abs_env_incar.py:374 computes must stay ~0, not grow toward CRASH."""
    env = StubEnv(CornerSpec(50.0, LEFT, 0.25))
    env._start_corner_tracking()
    heading, speed = 0.0, 26.8
    for _ in range(400):
        env._corner_speed = speed
        env._advance_corner_target()
        heading += (speed / 50.0) * env._dt          # perfect tracking
        speed = max(0.0, speed - 9.0 * env._dt)      # ~0.9 g stop
        env.current_heading = heading
        parent_error = abs(env.current_heading - env.target_heading)
        env._heading.observe(env.current_heading)
        assert parent_error == pytest.approx(0.0, abs=1e-9)


def test_a_car_that_runs_wide_accumulates_a_real_heading_error():
    env = StubEnv(CornerSpec(50.0, LEFT, 0.25))
    env._start_corner_tracking()
    heading, speed = 0.0, 26.8
    for _ in range(400):
        env._corner_speed = speed
        env._advance_corner_target()
        heading += 0.5 * (speed / 50.0) * env._dt    # only half the rotation
        speed = max(0.0, speed - 9.0 * env._dt)
        env.current_heading = heading
        env._heading.observe(env.current_heading)
    assert abs(env.current_heading - env.target_heading) > 0.1


def test_the_arc_sweep_matches_the_geometry_the_guard_was_given():
    """Guard and reality must use the same arc, or the guard is decoration."""
    env = StubEnv(CornerSpec(50.0, LEFT, 0.25))
    env._start_corner_tracking()
    heading, speed = 0.0, 26.8
    while speed > 0.0:
        env._corner_speed = speed
        env._advance_corner_target()
        heading += (speed / 50.0) * env._dt
        speed = max(0.0, speed - 9.0 * env._dt)
        env.current_heading = heading
        env._heading.observe(env.current_heading)
    from corner import arc_radians
    predicted = arc_radians(26.8, 50.0, ARC_DECEL_ESTIMATE_MS2)
    assert env._heading.arc_rad < predicted        # 5.0 m/s^2 is the pessimistic bound
    assert env._heading.arc_rad == pytest.approx(arc_radians(26.8, 50.0, 9.0), rel=0.02)


def test_a_corner_that_would_cross_the_wrap_is_refused_at_reset():
    env = StubEnv(CornerSpec(50.0, LEFT, 0.25), start_heading=math.pi - 0.05)
    with pytest.raises(RuntimeError, match="CRASH that never happened"):
        env._start_corner_tracking()


# ------------------------------------------- per-episode steering lookup
class FakeTable:
    def __init__(self, steering):
        self.steering = steering

    def steering_for(self, key):
        if key not in self.steering:
            raise KeyError(f"no steering angle for {key!r}")
        return self.steering[key]["steering"]


def test_steering_comes_from_the_row_that_will_also_score_the_episode():
    from calibration import config_key
    key = config_key(grip=0.5, speed_mph=60, radius_m=50.0)
    env = StubEnv(CornerSpec(50.0, LEFT))
    env._calibration = FakeTable({key: {"steering": 0.27}})
    env.grip, env.target_mph = 0.5, 60
    assert env._corner_steering() == pytest.approx(0.27)


def test_direction_comes_from_the_spec_not_the_stored_sign():
    """config_key carries no turn direction, so a row measured on a right-hander
    is the only row a left-hander can find. Taking the stored sign would steer
    right while target_yaw_rate demanded left -- 2v/R of yaw error all episode
    and a guaranteed 'crash' that is pure bookkeeping."""
    from calibration import config_key
    key = config_key(grip=1.0, speed_mph=60, radius_m=50.0)
    table = FakeTable({key: {"steering": -0.27}})     # measured turning right

    left = StubEnv(CornerSpec(50.0, LEFT))
    left._calibration, left.grip, left.target_mph = table, 1.0, 60
    assert left._corner_steering() == pytest.approx(+0.27)

    right = StubEnv(CornerSpec(50.0, RIGHT))
    right._calibration, right.grip, right.target_mph = table, 1.0, 60
    assert right._corner_steering() == pytest.approx(-0.27)


def test_an_explicit_spec_angle_wins_over_the_table():
    env = StubEnv(CornerSpec(50.0, LEFT, 0.25))
    env._calibration = FakeTable({})
    env.grip, env.target_mph = 1.0, 60
    assert env._corner_steering() == pytest.approx(0.25)


def test_an_uncalibrated_corner_config_refuses_rather_than_guessing():
    env = StubEnv(CornerSpec(50.0, LEFT))
    env._calibration = FakeTable({})
    env.grip, env.target_mph = 1.0, 120
    with pytest.raises(KeyError):
        env._corner_steering()


def test_a_straight_line_episode_asks_for_no_steering_at_all():
    env = StubEnv(None)
    assert env._corner_steering() is None


def test_a_straight_line_run_never_hits_the_wrap_guard():
    """The guard is corner-only; a straight episode at any spawn heading must
    still start, exactly as it does today."""
    env = StubEnv(None, start_heading=math.pi - 0.01)
    env._start_corner_tracking()                    # must not raise
