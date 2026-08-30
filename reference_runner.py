"""Measures the slam and stock-ABS reference stops that the normalized reward
anchors on (PLAN_V2.md sections 2-3).

Why this exists instead of reusing the training env: abs_env_incar REQUIRES the
MTB-ML-ABS controller to engage before an episode counts, so it structurally
cannot measure a car whose ABS slot holds the *stock* part. This is the same
drive-brake-measure sequence with the controller handshake removed.

Critical invariant -- this runner NEVER calls abstelemetry.setBrakes(). That is
what latches abstelemetry's `perWheelMode` and makes it scale per-wheel
brakeTorque every physics tick; with it off (and releaseBrakes() called
defensively at each reset) BeamNG's own brake pipeline owns the brakes
completely, which is the whole point: we are measuring what the STOCK system
does, not what our actuation layer does to it.

The measurement itself is the project's standard 2 kHz brake-event metric, armed
exactly as training arms it (same measure-target offset, same 2 kHz in-Lua slam
latch), because these numbers are only meaningful if they were produced the same
way the numbers they will be compared against were.
"""
import argparse
import json
import os
import random
import sys
import time

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from calibration import (CalibrationTable, config_key, regime_name,
                         summarize, STRAIGHT)
from corner import (LEFT, STEERING_SEEK_MAX_ITERS, initial_steering_guess,
                    lateral_g_for_radius, parse_corner_spec, radius_for_lateral_g,
                    radius_from_yaw, seek_is_saturated, steering_seek_converged,
                    steering_seek_update)
from residual_log import setup_logging, get_logger
from sim_config import (
    SimConfig, load as load_sim_config, resolved_userpath, content_userpath,
    detect_game_version,
)
from compat import check_compat

log = get_logger("refrun")

# Matched to the training env so the two are directly comparable.
DETERM_HZ = 200
MAP_NAME = "smallgrid"
START_POS = (0, 0, 0.5)
START_ROT = (0, 0, 0)
MEASURE_OFFSET_MS = 1.0     # abs_env: setTargetSpeed(target_ms - 1.0)
ACCEL_OVERSHOOT_MPH = 2.0   # abs_env: accelerate to target+2, then coast in gear
STOP_SPEED_MS = 0.05
STOP_FRAMES = 15
MAX_STOP_STEPS = 6000       # 30 s at 200 Hz -- a stop that long has gone wrong
PROBE_STEPS = 600           # 3 s of steady-state cornering at 200 Hz
PROBE_SETTLE_STEPS = 300    # first 1.5 s is turn-in transient, not a radius

# The two reference cars: identical except for the ABS slot (verified by diffing
# their parts blocks -- ESC and TC are empty in BOTH, so ABS is the only variable).
REFERENCE_CARS = {
    "slam": "vehicles/etk800/Machine-Trainer-Boy-V2.pc",             # etk_DSE_ABS = "" (none)
    "stock": "vehicles/etk800/Machine-Trainer-Boy-V2-STOCKABS.pc",   # etk_DSE_ABS = stock part
}
# "realistic" = the vehicle's own configured ABS. Set explicitly because BeamNG's
# own built-in brake test defaults to "arcade" (the idealized, cheating mode) --
# calibrating against that would anchor the whole reward on a fiction.
ABS_BEHAVIOR = {"slam": "off", "stock": "realistic"}


def check_supported(grip, radius_m, steering=None):
    """Guard for dimensions the runner cannot measure honestly. Refusing loudly
    beats silently measuring a straight line and writing the result under a key
    that claims otherwise -- a wrong calibration row is worse than a missing
    one, because training will happily consume it."""
    if radius_m is not STRAIGHT and steering is None:
        raise ValueError(
            f"radius_m={radius_m} was asked for with no steering angle -- the car "
            f"would brake in a straight line and the result would be written under "
            f"a key claiming a corner. Run the steering seek first.")


def get_quat(x, y, z):
    import math
    roll, pitch, yaw = math.radians(x), math.radians(y), math.radians(z)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class ReferenceRunner:
    """Measures reference stops.

    `speed_factor` fast-forwards the ENGINE (be:setPhysicsSpeedFactor, the same
    mechanism BeamNG's own ESC-calibration harness uses at 2x and its FFB
    calibration at 100x). `live` drops deterministic stepping for the measured
    stop and lets the game free-run instead.

    Both change the regime the number was produced in, and the project's own
    notes record deterministic-vs-live moving braking g by 0.04-0.09 -- a third
    of the stock-over-slam margin. So neither is a silent default: they are
    flags, and a table measured with them is only comparable to other tables
    measured the same way."""

    def __init__(self, sim_cfg, reference, port=None, speed_factor=1.0, live=False):
        if reference not in REFERENCE_CARS:
            raise ValueError(f"reference must be one of {sorted(REFERENCE_CARS)}")
        self.reference = reference
        self.cfg = sim_cfg
        self.port = port or sim_cfg.port
        self.speed_factor = float(speed_factor)
        self.live = bool(live)

        kwargs = dict(host="localhost", port=self.port, home=sim_cfg.game_folder)
        user = resolved_userpath(sim_cfg)
        if user:
            kwargs["user"] = user
        self.bng = BeamNGpy(**kwargs)

        try:
            self.bng.open(launch=False)
            log.info("attached to a running BeamNG on port %d", self.port)
        except Exception:
            if sim_cfg.headless:
                self.bng.open(None, "-headless", "-gfx", "null", "-no-sound", launch=True)
            else:
                self.bng.open(None, "-no-sound", launch=True)
            log.info("launched BeamNG (headless=%s) on port %d", sim_cfg.headless, self.port)

        self._install_telemetry()

        part_config = REFERENCE_CARS[reference]
        self.vehicle = Vehicle("ref_0", model="etk800", partConfig=part_config)
        self.scenario = Scenario(MAP_NAME, f"ref_{random.randint(10000, 99999)}")
        self.start_quat = get_quat(*START_ROT)
        self.scenario.add_vehicle(self.vehicle, pos=START_POS, rot_quat=self.start_quat)
        self.scenario.make(self.bng)
        self.bng.scenario.load(self.scenario)
        self.bng.scenario.start()
        log.info("spawned reference car %s (%s)", part_config, reference)

        self.bng.control.queue_lua_command(
            'core_gamestate.setGameState("scenario", "freeroam", "freeroam")')
        self.bng.control.queue_lua_command("ui_visibility.set(false)")
        time.sleep(3)

        self.vehicle.sensors.attach("electrics", Electrics())
        self.vehicle.switch()
        self.vehicle.focus()
        self.vehicle.queue_lua_command("extensions.load('abstelemetry')")
        time.sleep(3)

        self.bng.control.pause()
        self.bng.settings.set_deterministic(DETERM_HZ)
        self.vehicle.control(gear=2)
        self.bng.step(30)

        status = "NOT_FOUND"
        for _ in range(10):
            self.vehicle.sensors.poll()
            status = self.vehicle.sensors["electrics"].get("tel_status", "NOT_FOUND")
            if "OK" in str(status):
                break
            self.bng.step(30)
        log.info("telemetry: %s", status)
        if "OK" not in str(status):
            raise RuntimeError(f"abstelemetry failed to initialize: {status}")

    def _await_live_result(self, speed_mph, grip, pedal, radius_m, steering,
                           prev_g=0.0):
        """Free-running measurement: wait for Lua to publish a completed brake
        event, then read it. No stepping, no per-step round trips."""
        # The Lua latch decides the exact TICK the pedal goes down, but
        # vehicle.control is what actually applies it: the game's ~60 Hz input
        # update re-propagates whatever vehicle.control last said, so leaving it
        # at 0 means the latch's write is overwritten 60 times a second and the
        # car coasts on engine braking alone (~0.6 m/s^2, measured). Stepped
        # mode re-sent this every step and the file's own comment says why --
        # "the car simply never stopped" -- which is exactly what happened here.
        self.vehicle.control(brake=float(pedal), throttle=0.0)
        deadline = time.monotonic() + 90.0
        result_g = 0.0
        neutral_dropped = False
        while time.monotonic() < deadline:
            time.sleep(0.05)
            self.vehicle.sensors.poll()
            e = self.vehicle.sensors["electrics"]
            spd = float(e.get("tel_inst_speed", 999.0))
            if not neutral_dropped and spd < 1.118:      # 2.5 mph, as stepped mode
                self.vehicle.control(brake=float(pedal), gear=0)
                neutral_dropped = True
            g = float(e.get("tel_last_brake_avg_g_arc", 0.0))
            # A NEW value, not merely a non-zero one: this channel holds the
            # PREVIOUS stop's result until the next event completes, so
            # "g > 0" is already true the instant stop 2 begins.
            if g > 0.0 and g != prev_g:
                result_g = g
                break

        self.vehicle.queue_lua_command("extensions.abstelemetry.disarmBrakeSlam()")
        self.vehicle.control(brake=0.0, throttle=0.0)
        # Leave the engine at normal speed but do NOT restore stepping: the
        # next stop's run-up resumes and re-applies its own factor, so pausing
        # here only costs a round trip.
        self._set_speed_factor(0)
        self.bng.control.pause()
        self.bng.settings.set_deterministic(DETERM_HZ)
        self.bng.step(5)
        self.vehicle.sensors.poll()
        e = self.vehicle.sensors["electrics"]

        result = {
            "reference": self.reference,
            "speed_mph": speed_mph,
            "grip": float(grip),
            "pedal": float(pedal),
            "radius_m": radius_m,
            "steering": None if steering is None else float(steering),
            "stopped": result_g > 0.0,
            "avg_g_arc": float(e.get("tel_last_brake_avg_g_arc", 0.0)),
            "dist_arc": float(e.get("tel_last_brake_dist_arc", 0.0)),
            "avg_g_chord": float(e.get("tel_last_brake_avg_g", 0.0)),
            "dist_chord": float(e.get("tel_last_brake_dist", 0.0)),
            "grip_applied": float(e.get("tel_grip_mult", 1.0)),
            "grip_nodes": int(e.get("tel_grip_nodes", 0)),
        }
        log.info("%s @ %s mph [live x%g]: arc_g=%.4f arc_dist=%.2fm",
                 self.reference, speed_mph, self.speed_factor,
                 result["avg_g_arc"], result["dist_arc"])
        if result["avg_g_arc"] <= 0.0:
            raise RuntimeError(
                f"no brake event completed within 90s ({self.reference}, "
                f"{speed_mph} mph, live x{self.speed_factor}) -- the 2 kHz latch "
                f"never fired or the car never stopped.")
        return result

    def _set_speed_factor(self, value):
        """0 = normal real-time non-deterministic, -1 = deterministic/stepped,
        N>1 = N times faster than wall clock. Queued on the GAME ENGINE Lua
        (not the vehicle), which is where be: lives."""
        self.bng.control.queue_lua_command(f"be:setPhysicsSpeedFactor({value})")

    def _install_telemetry(self):
        """abstelemetry.lua must be resolvable by extensions.load() in the live
        userpath -- same mechanism abs_env uses, done explicitly here."""
        import shutil
        from pathlib import Path
        try:
            paths = self.bng.system.get_environment_paths()
            user_root = Path(paths["user"])
        except Exception:
            user_root = Path(content_userpath(self.cfg))
        ext_dir = user_root / "lua" / "vehicle" / "extensions"
        ext_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(HERE, "abstelemetry.lua"), ext_dir / "abstelemetry.lua")
        veh_dir = user_root / "vehicles" / "etk800"
        veh_dir.mkdir(parents=True, exist_ok=True)
        for pc in ("Machine-Trainer-Boy-V2.pc", "Machine-Trainer-Boy-V2-STOCKABS.pc"):
            src = os.path.join(HERE, "assets", "cars", "etk800", pc)
            if os.path.exists(src):
                shutil.copy2(src, veh_dir / pc)
        log.info("installed telemetry + reference cars -> %s", user_root)

    def measure_stop(self, speed_mph, grip=1.0, radius_m=STRAIGHT, lead_seconds=0.0,
                     steering=None, pedal=1.0):
        """One reference stop. Returns the standard 2 kHz brake-event result.

        `steering` is the signed open-loop angle for a corner (None = straight).
        It goes on at the same moment training's CornerSteerInjector applies it
        -- immediately after the slam is armed -- and is then left alone, since
        vehicle.control only sends the inputs it is given."""
        check_supported(grip, radius_m, steering)

        target_ms = speed_mph * 0.44704
        measure_target = target_ms - MEASURE_OFFSET_MS

        self.vehicle.teleport(pos=START_POS, rot_quat=self.start_quat, reset=True)
        self.vehicle.focus()
        self.vehicle.control(throttle=0, steering=0, gear=0, parkingbrake=0, brake=0)
        self.vehicle.queue_lua_command("extensions.load('abstelemetry')")
        # NEVER setBrakes here: releaseBrakes unlatches perWheelMode so the stock
        # pipeline (and the stock ABS, when present) owns brake torque entirely.
        self.vehicle.queue_lua_command("extensions.abstelemetry.releaseBrakes()")
        # Always start from stock tire grip: the previous episode's multiplier
        # must not leak into this one's approach or its measurement.
        self.vehicle.queue_lua_command("extensions.abstelemetry.restoreGrip()")
        self.bng.step(15)
        self.vehicle.queue_lua_command(
            f'wheels.setABSBehavior("{ABS_BEHAVIOR[self.reference]}")')
        self.bng.step(2)

        self.vehicle.queue_lua_command("extensions.abstelemetry.resetAccum()")
        self.vehicle.queue_lua_command(
            f"extensions.abstelemetry.setTargetSpeed({measure_target})")
        self.bng.step(1)

        # --- accelerate (non-deterministic for speed), then coast IN GEAR ---
        self.bng.settings.set_nondeterministic()
        self.bng.control.resume()
        # The run-up is not measured, so fast-forwarding it is free. This is
        # the only wall-clock-bound part of a stop: a sleep-poll loop with a
        # 25 s deadline.
        if self.speed_factor > 1.0:
            self._set_speed_factor(self.speed_factor)
        accel_target_ms = (speed_mph + ACCEL_OVERSHOOT_MPH) * 0.44704
        self.vehicle.control(gear=2, throttle=1.0, steering=0, brake=0)
        deadline = time.monotonic() + 25.0
        reached = False
        while time.monotonic() < deadline:
            time.sleep(0.02 / max(1.0, self.speed_factor))
            self.vehicle.sensors.poll()
            e = self.vehicle.sensors["electrics"]
            # tel_inst_speed is computed in onPhysicsStep; electrics.airspeed is
            # GFX-rate and LAGS when physics outruns graphics -- exactly what
            # abstelemetry v3.3's own comment warns about at speed_factor > 1.
            # Reading the laggy one at 10x overshoots the target badly.
            spd = float(e.get("tel_inst_speed", e.get("airspeed", 0.0)))
            if spd >= accel_target_ms:
                reached = True
                break
        if not reached:
            raise RuntimeError(
                f"never reached {accel_target_ms:.2f} m/s within 25s "
                f"(reference={self.reference}, {speed_mph} mph)")

        self.vehicle.control(throttle=0.0, steering=0)
        if self.live:
            # Free-running: the 2 kHz latch and the brake-event accumulator are
            # entirely in Lua, so the measurement does not need Python in the
            # loop at all -- stepping only ever paced the stop-detection poll.
            self._set_speed_factor(self.speed_factor if self.speed_factor > 1.0 else 0)
        else:
            self.bng.control.pause()
            self.bng.settings.set_deterministic(DETERM_HZ)

        # --- 2 kHz-exact brake onset (identical to training) ---
        # Pedal goes to the LATCH, not just to vehicle.control: the latch
        # re-asserts input.brake every 0.5 ms tick and would otherwise stamp
        # full pedal over the level being measured, writing a full-pedal result
        # under a part-pedal key.
        self.vehicle.queue_lua_command(
            f"extensions.abstelemetry.armBrakeSlam({target_ms}, {float(pedal)})")
        # Grip is armed the SAME way training arms it (at brake onset, after the
        # slam target is set) -- a reference measured with different timing is
        # not comparable to the runs it is meant to be the ruler for.
        if float(grip) != 1.0:
            self.vehicle.queue_lua_command(
                f"extensions.abstelemetry.armGripChange({float(grip)}, {lead_seconds})")
        if steering is not None:
            self.vehicle.control(steering=float(steering))
        fired = False
        if self.live:
            fired = True      # the latch fires in Lua; the result wait proves it
        for _ in range(0 if self.live else 600):
            self.bng.step(20)
            self.vehicle.sensors.poll()
            if int(self.vehicle.sensors["electrics"].get("tel_slam_fired", 0)) == 1:
                fired = True
                break
        if not fired:
            raise RuntimeError(
                f"brake slam never fired ({self.reference}, {speed_mph} mph, "
                f"speed_factor={self.speed_factor}, live={self.live}). The latch "
                f"compares physics-rate instSpeed against the target every tick, "
                f"so this means the car never crossed it -- usually the coast "
                f"never started (throttle still on) or the run-up overshot so far "
                f"the target was already passed when the slam was armed.")

        # --- ride the stop out (identical regime to abs_env_incar.step) ---
        # The pedal must be re-sent from the Python side every step, not just
        # latched in Lua: the game's own input update runs at ~60 Hz and will
        # otherwise write input.brake back down from the last vehicle.control
        # value, fighting the 2 kHz latch (observed live -- the car simply never
        # stopped). Training sends brake=1.0 every step for exactly this reason.
        # The one-shot neutral drop below ~2.5 mph is also training's regime:
        # the automatic box creeps against the brakes at walking pace otherwise.
        stop_timer = 0
        stopped = False
        neutral_dropped = False
        if self.live:
            # Nothing here paces the physics, so tracking speed from Python is
            # hopeless: at ~5x realtime a 50 ms wall poll sees the car once per
            # 250 ms of sim time, which is several mph of coast per sample --
            # easily enough to miss the target crossing entirely.
            #
            # It does not need to. The 2 kHz latch, the pedal hold and the whole
            # brake-event accumulator run in Lua; the ONLY thing Python needs is
            # the finished result, which Lua publishes as tel_last_brake_avg_g_arc
            # when the event completes. So wait for that, and let the game get on
            # with it.
            self.vehicle.sensors.poll()
            prev_g = float(self.vehicle.sensors["electrics"]
                           .get("tel_last_brake_avg_g_arc", 0.0))
            return self._await_live_result(speed_mph, grip, pedal, radius_m,
                                           steering, prev_g)

        # In stepped mode the pedal is re-sent every step: the game's ~60 Hz
        # input update otherwise writes input.brake back down, fighting the
        # 2 kHz latch (observed live -- the car simply never stopped).
        stop_deadline = time.monotonic() + 60.0
        for _ in range(MAX_STOP_STEPS):
            self.vehicle.sensors.poll()
            spd = float(self.vehicle.sensors["electrics"].get("airspeed", 999.0))
            if not neutral_dropped and spd < 1.118:      # 2.5 mph
                self.vehicle.control(brake=float(pedal), gear=0)
                neutral_dropped = True
            elif not self.live:
                self.vehicle.control(brake=float(pedal))
            if self.live:
                time.sleep(0.02)
                if time.monotonic() > stop_deadline:
                    break
            else:
                self.bng.step(1)
            self.vehicle.sensors.poll()
            spd = float(self.vehicle.sensors["electrics"].get("airspeed", 999.0))
            stop_timer = stop_timer + 1 if spd < STOP_SPEED_MS else 0
            if stop_timer >= STOP_FRAMES:
                stopped = True
                break
        self.vehicle.queue_lua_command("extensions.abstelemetry.disarmBrakeSlam()")
        self.vehicle.control(brake=0.0)
        # Every first-party call site pairs a set with a reset; a leaked factor
        # would silently change every stop measured after this one.
        if self.speed_factor > 1.0 or self.live:
            self._set_speed_factor(-1 if not self.live else 0)
            self.bng.control.pause()
            self.bng.settings.set_deterministic(DETERM_HZ)
        self.bng.step(5)
        self.vehicle.sensors.poll()
        e = self.vehicle.sensors["electrics"]

        result = {
            "reference": self.reference,
            "speed_mph": speed_mph,
            "grip": float(grip),
            "pedal": float(pedal),
            "radius_m": radius_m,
            "steering": None if steering is None else float(steering),
            "grip_applied": float(e.get("tel_grip_mult", 1.0)),
            "grip_nodes": int(e.get("tel_grip_nodes", 0)),
            "stopped": stopped,
            "avg_g_arc": float(e.get("tel_last_brake_avg_g_arc", 0.0)),
            "dist_arc": float(e.get("tel_last_brake_dist_arc", 0.0)),
            "avg_g_chord": float(e.get("tel_last_brake_avg_g", 0.0)),
            "dist_chord": float(e.get("tel_last_brake_dist", 0.0)),
        }
        log.info("%s @ %s mph: arc_g=%.4f arc_dist=%.2fm (chord_g=%.4f chord_dist=%.2fm) stopped=%s",
                 self.reference, speed_mph, result["avg_g_arc"], result["dist_arc"],
                 result["avg_g_chord"], result["dist_chord"], stopped)
        if not stopped:
            raise RuntimeError(f"car never came to a stop ({self.reference}, {speed_mph} mph)")
        if result["avg_g_arc"] <= 0.0:
            raise RuntimeError(
                f"brake event produced no arc-g ({self.reference}, {speed_mph} mph) -- "
                f"the 2 kHz state machine never completed a measurement")
        return result

    def probe_radius(self, speed_mph, steering, grip=1.0):
        """Hold `steering` at roughly constant speed and read back the radius
        the car actually describes: R = v / yaw_rate, averaged over the settled
        half of the hold. No braking -- this measures geometry only.

        Speed is held by a crude proportional throttle rather than a fixed
        pedal: understeer scrubs speed off, and a car that is decelerating
        through the sample gives a yaw rate that belongs to no single radius."""
        target_ms = speed_mph * 0.44704

        self.vehicle.teleport(pos=START_POS, rot_quat=self.start_quat, reset=True)
        self.vehicle.focus()
        self.vehicle.control(throttle=0, steering=0, gear=0, parkingbrake=0, brake=0)
        self.vehicle.queue_lua_command("extensions.load('abstelemetry')")
        self.vehicle.queue_lua_command("extensions.abstelemetry.releaseBrakes()")
        self.vehicle.queue_lua_command("extensions.abstelemetry.restoreGrip()")
        self.bng.step(15)
        if float(grip) != 1.0:
            # The probe must run on the surface the corner will be measured on:
            # grip sets how far the car understeers, i.e. the whole answer.
            self.vehicle.queue_lua_command(
                f"extensions.abstelemetry.setGripMultiplier({float(grip)})")
        self.bng.step(2)

        self.bng.settings.set_nondeterministic()
        self.bng.control.resume()
        self.vehicle.control(gear=2, throttle=1.0, steering=0, brake=0)
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            time.sleep(0.02)
            self.vehicle.sensors.poll()
            if self.vehicle.sensors["electrics"].get("airspeed", 0.0) >= target_ms:
                break
        else:
            raise RuntimeError(f"probe never reached {target_ms:.2f} m/s in 25 s")

        self.bng.control.pause()
        self.bng.settings.set_deterministic(DETERM_HZ)
        self.vehicle.control(steering=float(steering), throttle=0.0)

        samples = []
        for i in range(PROBE_STEPS):
            self.bng.step(1)
            self.vehicle.sensors.poll()
            e = self.vehicle.sensors["electrics"]
            spd = float(e.get("airspeed", 0.0))
            self.vehicle.control(
                throttle=min(1.0, max(0.0, 0.15 * (target_ms - spd))),
                steering=float(steering))
            if i >= PROBE_SETTLE_STEPS:
                samples.append((spd, float(e.get("tel_yaw_rate_inst", 0.0))))

        self.vehicle.control(throttle=0.0, steering=0.0, brake=1.0)
        self.bng.step(5)
        self.vehicle.control(brake=0.0)
        self.vehicle.queue_lua_command("extensions.abstelemetry.restoreGrip()")

        speeds = [s for s, _ in samples]
        yaws = [y for _, y in samples]
        mean_speed = sum(speeds) / len(speeds)
        mean_yaw = sum(yaws) / len(yaws)
        radius = radius_from_yaw(mean_speed, mean_yaw)
        log.info("probe steering=%+.4f -> speed=%.2f m/s yaw=%.4f rad/s R=%s",
                 steering, mean_speed, mean_yaw,
                 "n/a" if radius is None else f"{radius:.1f} m")
        return radius, mean_speed, mean_yaw

    def seek_steering(self, speed_mph, radius_m, grip=1.0, direction=LEFT):
        """Find the open-loop angle that holds `radius_m` at this speed and
        grip. Iterative because the angle->radius map is not known a priori
        (wheelbase, understeer, grip all move it) and is not linear near the
        limit -- see corner.steering_seek_update for the damping."""
        steering = initial_steering_guess(radius_m)
        history = []
        for _ in range(STEERING_SEEK_MAX_ITERS):
            measured, _, mean_yaw = self.probe_radius(speed_mph, direction * steering,
                                                       grip=grip)
            # The seek itself only uses |yaw|, so an inverted sign convention
            # between beamngpy steering and tel_yaw_rate_inst would pass here
            # and only surface in training, as a car steering one way while the
            # reward demands the other. Catch it on the first probe instead.
            if mean_yaw * direction <= 0.0:
                raise RuntimeError(
                    f"steering {direction * steering:+.4f} produced yaw "
                    f"{mean_yaw:+.4f} rad/s -- opposite signs. The steering and "
                    f"yaw-rate sign conventions disagree; every corner would be "
                    f"driven against its own yaw target.")
            history.append((steering, measured))
            if seek_is_saturated(history):
                best_steer, best_r = min(history, key=lambda h: h[1])
                speed_ms = speed_mph * 0.44704
                raise RuntimeError(
                    f"R={radius_m} m is unreachable for this car at {speed_mph} mph "
                    f"on grip={grip}: more steering stopped buying radius at "
                    f"{best_r:.1f} m (steering {best_steer:+.4f}), which is already "
                    f"{lateral_g_for_radius(speed_ms, best_r):.2f} g of lateral load. "
                    f"R={radius_m} m would need "
                    f"{lateral_g_for_radius(speed_ms, radius_m):.2f} g.\n\n"
                    f"Pick a corner by grip budget instead -- braking in a turn only "
                    f"tests ABS if there is grip left to brake with. At {speed_mph} "
                    f"mph: {radius_for_lateral_g(speed_ms, 0.3):.0f} m = 0.3 g, "
                    f"{radius_for_lateral_g(speed_ms, 0.4):.0f} m = 0.4 g, "
                    f"{radius_for_lateral_g(speed_ms, 0.5):.0f} m = 0.5 g lateral.")
            if steering_seek_converged(measured, radius_m):
                log.info("steering seek converged: R=%.1fm steering=%+.4f "
                         "(measured %.1fm, %d probes)", radius_m,
                         direction * steering, measured, len(history))
                return direction * steering, measured
            steering = steering_seek_update(steering, measured, radius_m)
        raise RuntimeError(
            f"steering seek did not converge on R={radius_m} m at {speed_mph} mph "
            f"grip={grip} after {STEERING_SEEK_MAX_ITERS} probes: {history}. "
            f"The radius may be unreachable on this surface (the car understeers "
            f"wide however far the wheel is turned).")

    def close(self):
        try:
            self.bng.close()
        except Exception:
            pass


def run_calibration(sim_cfg, car, speeds, reps, grips=(1.0,), radius_m=STRAIGHT,
                    out_dir=None, port=None, lead_seconds=0.0, direction=LEFT,
                    pedals=(1.0,), speed_factor=1.0, live=False):
    """Measures both references at every (speed, grip) and writes the table.

    For a corner, the steering angle is sought FIRST -- once per (speed, grip),
    on the slam car -- and the same angle then drives every reference stop and,
    later, every training episode on that row. One procedure for all three is
    the only thing that makes "stock's advantage in this corner" mean what it
    says."""
    out_dir = out_dir or os.path.join(HERE, "calibration")
    path = os.path.join(out_dir, f"{car}.json")
    table = CalibrationTable.load(path) if os.path.exists(path) else CalibrationTable(car=car)

    if radius_m is not STRAIGHT:
        seeker = ReferenceRunner(sim_cfg, "slam", port=port,
                                 speed_factor=speed_factor, live=live)
        try:
            for mph in speeds:
                for grip in grips:
                    key = config_key(grip=grip, speed_mph=mph, radius_m=radius_m)
                    if key in table.steering:
                        log.info("steering already known for %s: %+.4f", key,
                                 table.steering[key]["steering"])
                        continue
                    signed, measured = seeker.seek_steering(mph, radius_m, grip=grip,
                                                            direction=direction)
                    table.put_steering(key, signed, measured)
                    table.save(path)      # a seek costs minutes; never lose one
        finally:
            seeker.close()
        time.sleep(2)

    for reference in ("slam", "stock"):
        runner = ReferenceRunner(sim_cfg, reference, port=port,
                                 speed_factor=speed_factor, live=live)
        try:
            for mph in speeds:
                for grip in grips:
                    # Steering belongs to the geometry, so it is sought and
                    # stored once per (grip, speed, radius) -- at the full-pedal
                    # key -- and reused for every pedal level below.
                    geom_key = config_key(grip=grip, speed_mph=mph, radius_m=radius_m)
                    steering = (None if radius_m is STRAIGHT
                                else table.steering_for(geom_key))
                    for pedal in pedals:
                        key = config_key(grip=grip, speed_mph=mph,
                                         radius_m=radius_m, pedal=pedal)
                        values = []
                        for rep in range(reps):
                            r = runner.measure_stop(mph, grip=grip, radius_m=radius_m,
                                                    lead_seconds=lead_seconds,
                                                    steering=steering, pedal=pedal)
                            values.append(r["avg_g_arc"])
                            time.sleep(0.5)
                        table.put(key, reference,
                                  summarize(values, regime_name(
                                      speed_factor, live)))
                        log.info("calibrated %s %s: %s", reference, key,
                                 summarize(values))
        finally:
            runner.close()
        time.sleep(2)

    table.save(path)
    log.info("wrote %s", path)
    return table, path


def parse_args():
    p = argparse.ArgumentParser(description="Measure slam/stock ABS reference stops")
    p.add_argument("--settings", default=os.path.join(HERE, "settings.json"))
    p.add_argument("--game", choices=["tech", "drive"], default=None)
    p.add_argument("--game-folder", default=None)
    p.add_argument("--userpath", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--windowed", action="store_true")
    p.add_argument("--car", default="etk800")
    p.add_argument("--speeds", default="60", help='e.g. "60,90,120"')
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--grips", default="1.0",
                   help='tire grip levels to calibrate, e.g. "1.0,0.75,0.5" '
                        "(1.0 = stock tires)")
    p.add_argument("--grip-lead", type=float, default=0.0,
                   help="apply grip this many seconds before brake onset "
                        "(0 = same physics tick); must match training")
    p.add_argument("--pedals", default="1.0",
                   help='driver pedal positions to calibrate, e.g. "1.0,0.75,0.5". '
                        "Each needs its own references, since a half-pedal stop "
                        "cannot reach the full-pedal lockup floor.")
    p.add_argument("--speed-factor", type=float, default=1.0,
                   help="fast-forward the engine by this factor "
                        "(be:setPhysicsSpeedFactor -- BeamNG's own ESC calibration "
                        "uses 2). 1 = off. Changes the regime the number was "
                        "measured in, so a table is only comparable to others "
                        "measured the same way.")
    p.add_argument("--live", action="store_true",
                   help="measure the stop free-running instead of stepped. The "
                        "2 kHz measurement is entirely in Lua, so stepping only "
                        "ever paced the stop-detection poll -- but live and "
                        "deterministic give different g (0.04-0.09 apart).")
    p.add_argument("--corner", default="straight",
                   help='corner radius in metres, "50" / "50L" / "50R"; '
                        '"straight" (default) = no corner')
    p.add_argument("--force", action="store_true",
                   help="proceed even if beamngpy doesn't match the detected game version")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(os.path.join(HERE, "logs", "calibration.log"), component="refrun")
    cfg = load_sim_config(args.settings)
    if args.game is not None:
        cfg.game = args.game
    if args.game_folder is not None:
        cfg.game_folder = args.game_folder
    if args.userpath is not None:
        cfg.userpath = args.userpath
    if args.windowed:
        cfg.headless = False

    import beamngpy
    compat = check_compat(detect_game_version(cfg.game, cfg.game_folder),
                          beamngpy.__version__.strip())
    log.info("compat: %s", compat.message)
    if compat.ok is False and not args.force:
        raise SystemExit(f"{compat.message}\n\nFix: {compat.fix_command}\n\n"
                        f"Or pass --force to proceed anyway.")

    speeds = [int(s.strip()) for s in args.speeds.split(",") if s.strip()]
    grips = [round(float(g.strip()), 3) for g in args.grips.split(",") if g.strip()]
    corner = parse_corner_spec(args.corner)
    pedals = [round(float(x.strip()), 2) for x in args.pedals.split(",") if x.strip()]
    table, path = run_calibration(cfg, args.car, speeds, args.reps, grips=grips,
                                  pedals=pedals,
                                  radius_m=STRAIGHT if corner is None else corner.radius_m,
                                  direction=LEFT if corner is None else corner.direction,
                                  port=args.port, lead_seconds=args.grip_lead,
                                  speed_factor=args.speed_factor, live=args.live)

    print("\n--- CALIBRATION ---")
    for key in sorted(table.rows):
        row = table.rows[key]
        slam = row.get("slam", {}).get("median")
        stock = row.get("stock", {}).get("median")
        if slam is not None and stock is not None:
            steer = table.steering.get(key, {}).get("steering")
            extra = "" if steer is None else f" steering={steer:+.4f}"
            print(f"{key}: slam={slam:.4f}g stock={stock:.4f}g "
                  f"gap={stock - slam:+.4f}g{extra}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
