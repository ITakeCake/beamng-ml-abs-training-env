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


def check_supported(grip, radius_m):
    """Guard for dimensions the runner does not implement yet. Refusing loudly
    beats silently measuring dry asphalt / a straight line and writing the
    result under a key that claims otherwise -- a wrong calibration row is
    worse than a missing one, because training will happily consume it."""
    if float(grip) != 1.0:
        raise NotImplementedError(
            f"grip={grip} needs the runtime ground-model change (PLAN_V2 section 5); "
            f"not implemented yet -- refusing rather than silently measuring dry asphalt.")
    if radius_m is not STRAIGHT:
        raise NotImplementedError(
            f"radius_m={radius_m} needs the arc reference path (PLAN_V2 section 4); "
            f"not implemented yet -- refusing rather than silently measuring a straight line.")


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

    def measure_stop(self, speed_mph, grip=1.0, radius_m=STRAIGHT):
        """One reference stop. Returns the standard 2 kHz brake-event result."""
        check_supported(grip, radius_m)

        target_ms = speed_mph * 0.44704
        measure_target = target_ms - MEASURE_OFFSET_MS

        self.vehicle.teleport(pos=START_POS, rot_quat=self.start_quat, reset=True)
        self.vehicle.focus()
        self.vehicle.control(throttle=0, steering=0, gear=0, parkingbrake=0, brake=0)
        self.vehicle.queue_lua_command("extensions.load('abstelemetry')")
        # NEVER setBrakes here: releaseBrakes unlatches perWheelMode so the stock
        # pipeline (and the stock ABS, when present) owns brake torque entirely.
        self.vehicle.queue_lua_command("extensions.abstelemetry.releaseBrakes()")
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

    def close(self):
        try:
            self.bng.close()
        except Exception:
            pass


def run_calibration(sim_cfg, car, speeds, reps, grip=1.0, radius_m=STRAIGHT,
                    out_dir=None, port=None):
    """Measures both references at every requested speed and writes the table."""
    out_dir = out_dir or os.path.join(HERE, "calibration")
    path = os.path.join(out_dir, f"{car}.json")
    table = CalibrationTable.load(path) if os.path.exists(path) else CalibrationTable(car=car)

    for reference in ("slam", "stock"):
        runner = ReferenceRunner(sim_cfg, reference, port=port)
        try:
            for mph in speeds:
                values = []
                for rep in range(reps):
                    r = runner.measure_stop(mph, grip=grip, radius_m=radius_m)
                    values.append(r["avg_g_arc"])
                    time.sleep(0.5)
                key = config_key(grip=grip, speed_mph=mph, radius_m=radius_m)
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
    table, path = run_calibration(cfg, args.car, speeds, args.reps, port=args.port)

    print("\n--- CALIBRATION ---")
    for key in sorted(table.rows):
        row = table.rows[key]
        slam = row.get("slam", {}).get("median")
        stock = row.get("stock", {}).get("median")
        if slam is not None and stock is not None:
            print(f"{key}: slam={slam:.4f}g stock={stock:.4f}g "
                  f"gap={stock - slam:+.4f}g")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
