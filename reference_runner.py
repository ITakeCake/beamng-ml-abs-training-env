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

from calibration import CalibrationTable, config_key, summarize, STRAIGHT
from corner import (LEFT, STEERING_SEEK_MAX_ITERS, initial_steering_guess,
                    parse_corner_spec, radius_from_yaw, steering_seek_converged,
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
    def __init__(self, sim_cfg, reference, port=None):
        if reference not in REFERENCE_CARS:
            raise ValueError(f"reference must be one of {sorted(REFERENCE_CARS)}")
        self.reference = reference
        self.cfg = sim_cfg
        self.port = port or sim_cfg.port

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
                     steering=None):
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
        accel_target_ms = (speed_mph + ACCEL_OVERSHOOT_MPH) * 0.44704
        self.vehicle.control(gear=2, throttle=1.0, steering=0, brake=0)
        deadline = time.monotonic() + 25.0
        reached = False
        while time.monotonic() < deadline:
            time.sleep(0.02)
            self.vehicle.sensors.poll()
            if self.vehicle.sensors["electrics"].get("airspeed", 0.0) >= accel_target_ms:
                reached = True
                break
        if not reached:
            raise RuntimeError(
                f"never reached {accel_target_ms:.2f} m/s within 25s "
                f"(reference={self.reference}, {speed_mph} mph)")

        self.vehicle.control(throttle=0.0, steering=0)
        self.bng.control.pause()
        self.bng.settings.set_deterministic(DETERM_HZ)

        # --- 2 kHz-exact brake onset (identical to training) ---
        self.vehicle.queue_lua_command(
            f"extensions.abstelemetry.armBrakeSlam({target_ms})")
        # Grip is armed the SAME way training arms it (at brake onset, after the
        # slam target is set) -- a reference measured with different timing is
        # not comparable to the runs it is meant to be the ruler for.
        if float(grip) != 1.0:
            self.vehicle.queue_lua_command(
                f"extensions.abstelemetry.armGripChange({float(grip)}, {lead_seconds})")
        if steering is not None:
            self.vehicle.control(steering=float(steering))
        fired = False
        for _ in range(600):
            self.bng.step(20)
            self.vehicle.sensors.poll()
            if int(self.vehicle.sensors["electrics"].get("tel_slam_fired", 0)) == 1:
                fired = True
                break
        if not fired:
            raise RuntimeError(f"brake slam never fired ({self.reference}, {speed_mph} mph)")

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
        for _ in range(MAX_STOP_STEPS):
            self.vehicle.sensors.poll()
            spd = float(self.vehicle.sensors["electrics"].get("airspeed", 999.0))
            if not neutral_dropped and spd < 1.118:      # 2.5 mph
                self.vehicle.control(brake=1.0, gear=0)
                neutral_dropped = True
            else:
                self.vehicle.control(brake=1.0)
            self.bng.step(1)
            self.vehicle.sensors.poll()
            spd = float(self.vehicle.sensors["electrics"].get("airspeed", 999.0))
            stop_timer = stop_timer + 1 if spd < STOP_SPEED_MS else 0
            if stop_timer >= STOP_FRAMES:
                stopped = True
                break
        self.vehicle.queue_lua_command("extensions.abstelemetry.disarmBrakeSlam()")
        self.vehicle.control(brake=0.0)
        self.bng.step(5)
        self.vehicle.sensors.poll()
        e = self.vehicle.sensors["electrics"]

        result = {
            "reference": self.reference,
            "speed_mph": speed_mph,
            "grip": float(grip),
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
                    out_dir=None, port=None, lead_seconds=0.0, direction=LEFT):
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
        seeker = ReferenceRunner(sim_cfg, "slam", port=port)
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
        runner = ReferenceRunner(sim_cfg, reference, port=port)
        try:
            for mph in speeds:
                for grip in grips:
                    key = config_key(grip=grip, speed_mph=mph, radius_m=radius_m)
                    steering = (None if radius_m is STRAIGHT
                                else table.steering_for(key))
                    values = []
                    for rep in range(reps):
                        r = runner.measure_stop(mph, grip=grip, radius_m=radius_m,
                                                lead_seconds=lead_seconds,
                                                steering=steering)
                        values.append(r["avg_g_arc"])
                        time.sleep(0.5)
                    table.put(key, reference, summarize(values))
                    log.info("calibrated %s %s: %s", reference, key, summarize(values))
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
    table, path = run_calibration(cfg, args.car, speeds, args.reps, grips=grips,
                                  radius_m=STRAIGHT if corner is None else corner.radius_m,
                                  direction=LEFT if corner is None else corner.direction,
                                  port=args.port, lead_seconds=args.grip_lead)

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
