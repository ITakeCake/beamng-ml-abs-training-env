"""Straight-line ABS training env over BeamNG co-simulation (v1).

Scope (deliberately fixed for v1): one car (Machine-Trainer-Boy-V2-MLABS), one
grip level, straight line, one algorithm downstream (PPO). Everything not braking
is pinned to a constant. Later versions widen it.

Action: [front_release, rear_release] in [0,1], axle-locked (FL==FR, RL==RR).
Zero action = full pedal = the lockup baseline; the policy learns WHEN to release.
Reward: selected RewardSpec preset; new runs select v6.0 in train_cosim.py while
v5.0 remains available for exact historical comparisons.
Actuation + obs both ride co-sim at physics rate; abstelemetry supplies the 2 kHz
brake metric (last_brake_dist / avg_g), read back through co-sim's Electrics group.
"""
import os
import shutil
import time
import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics

from cosim_link import CoSimLink, configure_diag_file, diag_write
from residual_core import (residual_to_brakes, wheel_release_to_brakes,
                           axle_view, WHEEL_MODES, ACT_DIM_FOR)
from reward_spec import RewardSpec

MPH_TO_MS = 0.44704
GRAV = 9.81
DT = 0.01                        # ~2x compute (4.9ms measured); window=20 ticks, safe margin
ARM_MARGIN_MS = 3.0              # brake from above target; measure at the crossing
RUNUP_HANDOFF_LEAD_MS = 1.5      # final 1.5 m/s is accelerated at real-time speed
RUNUP_TIMEOUT_S = 30.0
GEAR_MODES = ("neutral", "in_gear")   # per-episode coin flip at run-up handoff
NEUTRAL_SHIFT_MS = 5.0 * MPH_TO_MS    # in_gear: drop to neutral at/below 5 mph
MAX_BRAKE_NM = 3000.0            # full pedal == wheel lock (etk800, measured)
STOP_MS = 0.5                    # the game's 2 kHz metric closes at |v| <= 1.0
                                 # m/s, so below 0.5 it is final and the crawl to
                                 # a true standstill is dead episode time
STOP_FRAMES = 8                  # ~0.08 s of near-zero speed confirms the stop
MAX_STEPS = 4000                 # ~40 s. The scored stop is ~6 s, but the env hands
                                 # over above the trigger and an unbraked approach
                                 # coasts on drag for ~19 s, so a 24 s cap timed out
                                 # policies that could actually stop the car.
SPIN_YAW_LIMIT = 1.0             # integrated |yaw rate|; a clean stop stays well under
TRACE_STEPS = 2000               # per-wheel torque trace: ep1, first ~20 s only
DEFAULT_PC = "vehicles/etk800/Machine-Trainer-Boy-V2-MLABS.pc"


def set_control_dt(dt):
    """Change the policy's control period, with every step-count that means a
    duration moved with it.

    BeamNG derives the coupling rate from the value passed as time3rdParty:
    ``sendSkips = ceil(time3rdParty / physicsDt) - 1`` (tech/cosimulationCoupling
    .lua), and physics is 2 kHz, so DT is literally the control period. 0.01 is
    100 Hz, 0.005 is 200 Hz. Closed-loop braking is delay-limited at 100 Hz --
    see docs/CONTROL_CEILING.md -- so this is a first-class training knob.

    Must be called BEFORE constructing the env: the link takes DT at build time.
    """
    global DT, STOP_FRAMES, MAX_STEPS, TRACE_STEPS
    dt = float(dt)
    if not (0.0005 <= dt <= 0.05):
        raise ValueError("control dt must be from 0.0005 (2 kHz, one physics "
                         "tick) to 0.05 (20 Hz); got %r" % (dt,))
    scale = 0.01 / dt
    DT = dt
    STOP_FRAMES = max(1, int(round(8 * scale)))      # hold ~0.08 s below STOP_MS
    MAX_STEPS = int(round(4000 * scale))             # keep the ~40 s timeout
    TRACE_STEPS = int(round(2000 * scale))           # keep the ~20 s trace

# Ordered co-sim contract. Obs the policy sees is a SUBSET (built in _obs); the
# metric + slam channels are for reward/terminal, not fed to the network.
SIG_TO = [
    ("Kinematics", "vehicleGroundSpeed"),
    ("Wheels", "FRwheelSpeed"), ("Wheels", "FLwheelSpeed"),
    ("Wheels", "RRwheelSpeed"), ("Wheels", "RLwheelSpeed"),
    ("Electrics", "tel_gy_inst"), ("Electrics", "tel_gx_inst"),
    ("Electrics", "tel_yaw_rate_inst"), ("Electrics", "tel_slam_fired"),
    ("Electrics", "tel_slam_fire_speed"), ("Electrics", "tel_inst_speed"),
    ("Electrics", "tel_last_brake_dist"), ("Electrics", "tel_last_brake_avg_g"),
    ("Electrics", "tel_fused_speed"), ("Electrics", "tel_brake_active"),
    ("Driver", "brake"),             # input.brake as the metric SM actually reads it
    # --- MachineTrainerBoy frame sources (all published by abstelemetry every
    # physics tick from the PAST WIN_N ticks; nothing here looks ahead) ---
    ("Electrics", "tel_win_gy_avg"), ("Electrics", "tel_win_gy_min"),
    ("Electrics", "tel_win_gy_max"), ("Electrics", "tel_win_yaw_avg"),
    ("Electrics", "tel_rpm"), ("Electrics", "tel_gear"), ("Electrics", "tel_steer"),
    ("Electrics", "tel_att_pitch"), ("Electrics", "tel_att_roll"),
    ("Electrics", "tel_throttle_in"), ("Electrics", "tel_gz_inst"),
    ("Electrics", "tel_abs_speed"),  # wheels.lua virtualAirspeed: what stock ABS uses
    ("Electrics", "tel_brk_applied_fr"), ("Electrics", "tel_brk_applied_fl"),
    ("Electrics", "tel_brk_applied_rr"), ("Electrics", "tel_brk_applied_rl"),
]
SIG_FROM = [("Wheels", "FRbrakingTorque"), ("Wheels", "FLbrakingTorque"),
            ("Wheels", "RRbrakingTorque"), ("Wheels", "RLbrakingTorque"),
            ("Driver", "brake")]        # pin input.brake=1 so the metric SM stays armed

# _obs index map into a decoded "To" packet.
I_GS, I_WS0, I_GY, I_GX, I_YAW = 0, 1, 5, 6, 7
I_FIRED, I_FIRE_SPEED, I_INST_SPEED = 8, 9, 10
I_DIST, I_AVGG, I_FUSED, I_BACT, I_BRK = 11, 12, 13, 14, 15
I_WGY, I_WGYMIN, I_WGYMAX, I_WYAW = 16, 17, 18, 19
I_RPM, I_GEAR, I_STEER, I_PITCH, I_ROLL, I_THR, I_GZ = 20, 21, 22, 23, 24, 25, 26
I_ABSSPD = 27
I_BRKA0 = 28                     # applied brake torque fr, fl, rr, rl (Nm)
# Observation = the MachineTrainerBoy (SAC, 2026-06) 27-value frame plus four
# slips referenced to the stock ABS speed estimate (never ground truth) plus the
# four applied (pressure-delayed) brake torques, stacked
# N_STACK deep, oldest first / newest last, zero history at reset. Every frame
# is built from the packet just received plus EARLIER frames only: the rates
# (pitch/roll/wheel accel) are (current - previous) / DT. No channel can
# contain information from a later tick. Same for every wheel mode.
FRAME_DIM = 35
N_STACK = 64                     # 640 ms of history at 100 Hz
OBS_DIM = FRAME_DIM * N_STACK    # 2240
FRAME_LOW = np.array([
    0.0, 0.0, 0.0, 0.0,                 # 0-3   wheel speeds (m/s)
    -50.0, -50.0, -10.0,                # 4-6   gy_avg, gx_inst, yaw_avg
    0.0, 0.0, 0.0, 0.0,                 # 7-10  prev brake fractions fr, fl, rr, rl
    -50.0, -50.0,                       # 11-12 gy window min / max
    0.0, -2.0, -2.0,                    # 13-15 rpm, gear, steering
    -np.pi, -np.pi,                     # 16-17 pitch, roll (rad)
    0.0, 0.0,                           # 18-19 input brake, input throttle
    -50.0, -10.0, -10.0,                # 20-22 gz, pitch rate, roll rate
    -200.0, -200.0, -200.0, -200.0,     # 23-26 wheel accels (m/s^2)
    0.0, 0.0, 0.0, 0.0,                 # 27-30 slips vs stock-ABS speed estimate
    0.0, 0.0, 0.0, 0.0,                 # 31-34 applied brake torque fr, fl, rr, rl (Nm)
], dtype=np.float32)
FRAME_HIGH = np.array([
    100.0, 100.0, 100.0, 100.0,
    50.0, 50.0, 10.0,
    1.0, 1.0, 1.0, 1.0,
    50.0, 50.0,
    15000.0, 12.0, 2.0,
    np.pi, np.pi,
    1.0, 1.0,
    50.0, 10.0, 10.0,
    200.0, 200.0, 200.0, 200.0,
    1.0, 1.0, 1.0, 1.0,
    5000.0, 5000.0, 5000.0, 5000.0,
], dtype=np.float32)


def obs_dim_for(wheel_mode):
    ACT_DIM_FOR[wheel_mode]          # validates the mode; obs is mode-independent
    return OBS_DIM


def act_dim_for(wheel_mode):
    return ACT_DIM_FOR[wheel_mode]
WHEEL_KEYS = ("maxbrk_FR", "maxbrk_FL", "maxbrk_RR", "maxbrk_RL")  # FR/FL/RR/RL order


class ABSCoSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, speeds=(80,), vehicle_pc=DEFAULT_PC, port=64280,
                 game_folder=None, user_path=None, headless=True,
                 reward_spec=None, runup_speed_factor=4.0, log=None,
                 wheel_mode="axle",
                 artifact_dir=None):
        super().__init__()
        self._torque_trace_path = None
        if artifact_dir:
            artifact_dir = os.path.abspath(os.fspath(artifact_dir))
            os.makedirs(artifact_dir, exist_ok=True)
            configure_diag_file(os.path.join(artifact_dir, "cosim_diag.log"))
            self._torque_trace_path = os.path.join(
                artifact_dir, "torque_trace.csv")
        self.speeds = list(speeds)
        self.vehicle_pc = vehicle_pc
        self._reward = reward_spec or RewardSpec.v5()
        self._g_reward_tracker = self._reward.make_step_tracker(DT)
        self.runup_speed_factor = float(runup_speed_factor)
        if not np.isfinite(self.runup_speed_factor) or not (1.0 <= self.runup_speed_factor <= 1000.0):
            raise ValueError("runup_speed_factor must be from 1 to 1000")
        self._log = log
        self._configure_wheel_mode(wheel_mode)

        self._install_telemetry(user_path)
        self.bng = BeamNGpy("localhost", port, home=game_folder, user=user_path)
        args = ["-headless", "-gfx", "null", "-no-sound"] if headless else []
        self.bng.open(None, *args, launch=True)
        self.veh = (Vehicle("cosim", model="etk800", part_config=vehicle_pc)
                    if vehicle_pc else Vehicle("cosim", model="etk800"))
        self.veh.sensors.attach("electrics", Electrics())
        sc = Scenario("smallgrid", "abs_cosim")
        sc.add_vehicle(self.veh, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))
        sc.make(self.bng)
        self.bng.scenario.load(sc)
        self.bng.scenario.start()

        # Rated per-wheel brake torque, read ONCE on the fresh vehicle before any
        # setBrakes(0,0,0,0) zeros the live brakeTorque. Static for the whole run.
        time.sleep(2.5)
        self.veh.queue_lua_command(
            "for _,w in pairs(wheels.wheelRotators) do "
            "electrics.values['maxbrk_'..tostring(w.name)]=w.brakeTorque or 0 end")
        time.sleep(0.4)
        self.veh.sensors.poll()
        e = self.veh.sensors["electrics"]
        mb = [e.get(k) for k in WHEEL_KEYS]
        self._max_brake = np.array(
            [float(x) if x else MAX_BRAKE_NM for x in mb], dtype=np.float64)
        diag_write("MAXBRAKE(init) raw=%s -> %s" % (mb, self._max_brake.tolist()))

        self.link = CoSimLink(SIG_TO, SIG_FROM, time_3rd_party=DT)
        self.link.open()
        self._coupling_up = False
        self._trace_f = None
        self.episode = 0
        self.last_info = {}

    # ---------------------------------------------------------------- helpers
    def _configure_wheel_mode(self, wheel_mode):
        """axle: 2 releases (FL=FR, RL=RR). independent: 4 per-wheel releases."""
        if wheel_mode not in WHEEL_MODES:
            raise ValueError("wheel_mode must be one of %s" % (WHEEL_MODES,))
        self.wheel_mode = wheel_mode
        self.act_dim = act_dim_for(wheel_mode)
        self.obs_dim = obs_dim_for(wheel_mode)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, (self.act_dim,), np.float32)
        self._prev_rel = np.zeros(self.act_dim, dtype=np.float32)
        self._reset_frame_state()

    def _reset_frame_state(self):
        """Zero history: identical to SB3 VecFrameStack right after reset."""
        self._stack = [np.zeros(FRAME_DIM, dtype=np.float32) for _ in range(N_STACK)]
        self._prev_brakes = (1.0, 1.0, 1.0, 1.0)     # release 0 = full brake
        self._prev_ws = None
        self._prev_pitch = 0.0
        self._prev_roll = 0.0

    def _frame(self, v):
        """One MachineTrainerBoy frame from the packet just received."""
        ws = np.abs(np.array(v[I_WS0:I_WS0 + 4], dtype=np.float32))
        pitch = float(v[I_PITCH])
        roll = float(v[I_ROLL])
        if self._prev_ws is None:
            wa = np.zeros(4, dtype=np.float32)
            pitch_rate = roll_rate = 0.0
        else:
            wa = (ws - self._prev_ws) / DT
            pitch_rate = (pitch - self._prev_pitch) / DT
            roll_rate = (roll - self._prev_roll) / DT
        self._prev_ws = ws
        self._prev_pitch = pitch
        self._prev_roll = roll
        pb = self._prev_brakes
        abs_ref = max(float(v[I_ABSSPD]), 0.5)
        slip = np.clip(1.0 - ws / abs_ref, 0.0, 1.0)
        raw = np.array([
            ws[0], ws[1], ws[2], ws[3],
            float(v[I_WGY]), float(v[I_GX]), float(v[I_WYAW]),
            pb[0], pb[1], pb[2], pb[3],
            float(v[I_WGYMIN]), float(v[I_WGYMAX]),
            float(v[I_RPM]), float(v[I_GEAR]), float(v[I_STEER]),
            pitch, roll,
            float(v[I_BRK]), float(v[I_THR]),
            float(v[I_GZ]), pitch_rate, roll_rate,
            wa[0], wa[1], wa[2], wa[3],
            slip[0], slip[1], slip[2], slip[3],
            float(v[I_BRKA0]), float(v[I_BRKA0 + 1]),
            float(v[I_BRKA0 + 2]), float(v[I_BRKA0 + 3]),
        ], dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(raw, FRAME_LOW, FRAME_HIGH).astype(np.float32)

    def _action_to_brakes(self, action):
        if self.wheel_mode == "independent":
            return wheel_release_to_brakes(action, pedal=1.0)
        return residual_to_brakes(action, pedal=1.0)   # axle-locked

    def _q(self, cmd):
        self.veh.queue_lua_command(cmd)

    def _dense_g(self, v, braking_g):
        """G sample for the dense reward: instantaneous (v6) or 10 ms window (v8)."""
        if getattr(self._reward, "dense_g_source", "inst") == "window":
            return max(0.0, float(v[I_WGY]) / GRAV)
        return braking_g

    def _handoff_gear(self):
        """Neutral coasts; in_gear keeps the run-up drive gear until 5 mph."""
        if self.gear_mode == "neutral":
            self.veh.control(throttle=0.0, steering=0, gear=0)
        else:
            self.veh.control(throttle=0.0, steering=0)
        self._diag("GEAR_MODE ep=%d mode=%s" % (self.episode, self.gear_mode))

    def _maybe_shift_neutral(self, gs):
        if (self._neutral_shifted or not self._seen_moving
                or gs > NEUTRAL_SHIFT_MS):
            return
        self.veh.control(gear=0)     # automatics creep below this; stop needs neutral
        self._neutral_shifted = True
        self._diag("NEUTRAL_SHIFT ep=%d gs=%.3f" % (self.episode, gs))

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _copy_with_rollback(cls, source, destination):
        """Install one project asset while retaining any different predecessor."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = cls._file_sha256(source)
        if destination.is_file():
            existing_hash = cls._file_sha256(destination)
            if existing_hash == source_hash:
                return "unchanged"
            backup = destination.with_name(
                destination.name + ".backup-" + existing_hash[:12])
            if not backup.exists():
                shutil.copy2(destination, backup)
        shutil.copy2(source, destination)
        return "installed"

    @classmethod
    def _install_telemetry(cls, user_path):
        """Keep the live extension in sync with the 2 kHz run-up gate.

        The older in-car environment already deploys this project-owned file at
        startup.  Co-sim needs the same guarantee because the handoff must happen
        inside vehicle Lua, not in a delayed Python sensor poll.
        """
        if not user_path:
            return
        user_root = Path(user_path)
        if user_root.name.lower() != "current":
            user_root /= "current"
        source = Path(__file__).with_name("abstelemetry.lua")
        destinations = (
            user_root / "lua" / "vehicle" / "extensions" / source.name,
            user_root / "mods" / "unpacked" / "mtb_ml_abs" / "lua" /
            "vehicle" / "extensions" / source.name,
        )
        for destination in destinations:
            cls._copy_with_rollback(source, destination)

    def _set_physics_speed_factor(self, factor):
        """Set the GE-owned factor and wait until the Lua command has executed."""
        return self.bng.control.queue_lua_command(
            "be:setPhysicsSpeedFactor(%g); return true" % float(factor),
            response=True)

    def _poll_electrics(self):
        self.veh.sensors.poll()
        return self.veh.sensors["electrics"]

    @staticmethod
    def _sensor_speed(e):
        return float(e.get("tel_inst_speed", e.get("airspeed", 0.0)) or 0.0)

    @staticmethod
    def _lua_true(value):
        return value is True or value == 1 or str(value).strip().lower() == "true"

    def _accelerated_runup(self, arm_ms):
        """Fast-forward most of the launch, then hand back to real time safely.

        A 2 kHz vehicle-Lua latch cuts throttle at the handoff threshold.  This
        makes arbitrary speed factors safe even though Python only sees sensor
        data at graphics/poll rate.
        """
        handoff_ms = max(0.5, arm_ms - RUNUP_HANDOFF_LEAD_MS)
        armed = self.veh.queue_lua_command(
            "if not extensions.abstelemetry or "
            "not extensions.abstelemetry.armRunupHandoff then return false end; "
            "extensions.abstelemetry.armRunupHandoff(%f); return true" % handoff_ms,
            response=True)
        if not self._lua_true(armed):
            raise RuntimeError(
                "live abstelemetry.lua is missing the accelerated run-up handoff")
        try:
            self._set_physics_speed_factor(self.runup_speed_factor)
            self.veh.control(gear=2, throttle=1.0, steering=0, brake=0)
            deadline = time.monotonic() + RUNUP_TIMEOUT_S
            while time.monotonic() < deadline:
                time.sleep(0.01)
                e = self._poll_electrics()
                if float(e.get("tel_runup_handoff_fired", 0.0) or 0.0) > 0.5:
                    fire_speed = float(e.get(
                        "tel_runup_handoff_fire_speed", self._sensor_speed(e)) or 0.0)
                    self._diag(
                        "RUNUP_HANDOFF ep=%d factor=%g target_ms=%.4f speed_ms=%.4f"
                        % (self.episode, self.runup_speed_factor,
                           handoff_ms, fire_speed))
                    break
            else:
                raise RuntimeError(
                    "accelerated run-up never reached %.3f m/s within %.1f s"
                    % (handoff_ms, RUNUP_TIMEOUT_S))
        finally:
            # response=True is the acknowledgement boundary: when this returns,
            # the GameEngine Lua VM has executed the return-to-real-time command.
            try:
                ack = self._set_physics_speed_factor(0)
                e = self._poll_electrics()
                self._diag(
                    "RUNUP_REALTIME ep=%d ack=%s speed_ms=%.4f"
                    % (self.episode, self._lua_true(ack), self._sensor_speed(e)))
            finally:
                self._q("extensions.abstelemetry.disarmRunupHandoff()")

        # Release the Lua throttle latch and cover only the small final band at
        # normal speed. Brake activation still happens later under tight coupling.
        self.veh.control(gear=2, throttle=1.0, steering=0, brake=0)
        deadline = time.monotonic() + RUNUP_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(0.01)
            e = self._poll_electrics()
            if self._sensor_speed(e) >= arm_ms:
                return
        raise RuntimeError(
            "real-time run-up never reached brake-arm speed %.3f m/s" % arm_ms)

    def _diag(self, msg):
        diag_write(msg)
        if self._log:
            self._log.info(msg)

    def _trace_torque(self, v, torque):
        # Per-wheel APPLIED braking torque (Nm, positive) over ep1's first ~20 s.
        if not self._torque_trace_path:
            return
        if self.episode != 1 or self._steps > TRACE_STEPS:
            if self._trace_f is not None:
                self._trace_f.close()
                self._trace_f = None
            return
        if self._trace_f is None:
            self._trace_f = open(self._torque_trace_path, "x", newline="")
            self._trace_f.write("step,t_s,fused_ms,gs_ms,fr_nm,fl_nm,rr_nm,rl_nm\n")
        self._trace_f.write("%d,%.3f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f\n" % (
            self._steps, self._steps * DT, float(v[I_FUSED]), float(v[I_GS]),
            -torque[0], -torque[1], -torque[2], -torque[3]))
        self._trace_f.flush()

    def _stop_coupling(self):
        if self._coupling_up:
            self._q(self.link.stop_cmd())
            self._q("extensions.abstelemetry.disarmBrakeSlam()")
            time.sleep(0.4)
            self._coupling_up = False

    @staticmethod
    def _true_slips(v, gs):
        """Ground-speed-referenced slip for reward only (never in obs)."""
        if gs < 1.0:
            return ()   # below walking pace slip is undefined; term contributes 0
        ws = np.abs(np.array(v[I_WS0:I_WS0 + 4], dtype=np.float64))
        return np.clip(1.0 - ws / gs, 0.0, 1.0)

    def _obs(self, v):
        """Push the newest frame and return the 16-frame past window, flat."""
        self._stack.pop(0)
        self._stack.append(self._frame(v))
        return np.concatenate(self._stack).astype(np.float32)

    def _slips(self, v):
        """Fused-referenced slips for the reward/diagnostics (not in the obs)."""
        ws = np.abs(np.array(v[I_WS0:I_WS0 + 4], dtype=np.float32))
        ref = max(float(v[I_FUSED]), 0.5)
        return np.clip(1.0 - ws / ref, 0.0, 1.0).astype(np.float32)

    def _step_diagnostics(self, v, rel, torque, obs):
        """Physics-aligned values for the trainer's automatic trace recorder.

        These stay out of the policy observation and reward. Brake torque is the
        positive command sent over co-sim; it is not a second torque sensor.
        """
        front, rear = axle_view(rel)
        prev = self._prev_rel
        prev_front, prev_rear = axle_view(prev)
        per_wheel = (tuple(float(r) for r in rel) if len(rel) == 4
                     else (float(rel[0]), float(rel[0]), float(rel[1]), float(rel[1])))
        prev_wheel = (tuple(float(p) for p in prev) if len(prev) == 4
                      else (float(prev[0]), float(prev[0]),
                            float(prev[1]), float(prev[1])))
        slip = self._slips(v)
        return {
            "episode": self.episode,
            "episode_step": self._steps,
            "action_front_release": front,
            "action_rear_release": rear,
            "previous_action_front": prev_front,
            "previous_action_rear": prev_rear,
            "action_fr_release": per_wheel[0], "action_fl_release": per_wheel[1],
            "action_rr_release": per_wheel[2], "action_rl_release": per_wheel[3],
            "previous_action_fr": prev_wheel[0], "previous_action_fl": prev_wheel[1],
            "previous_action_rr": prev_wheel[2], "previous_action_rl": prev_wheel[3],
            "ground_speed_ms": float(v[I_GS]),
            "fused_speed_ms": float(v[I_FUSED]),
            "braking_g": max(0.0, float(v[I_GY]) / GRAV),
            "yaw_rate_rad_s": float(v[I_YAW]),
            "yaw_abs_sum": float(getattr(self, "_yaw_abs_sum", 0.0)),
            "wheel_speed_fr_ms": abs(float(v[I_WS0])),
            "wheel_speed_fl_ms": abs(float(v[I_WS0 + 1])),
            "wheel_speed_rr_ms": abs(float(v[I_WS0 + 2])),
            "wheel_speed_rl_ms": abs(float(v[I_WS0 + 3])),
            "slip_fr": float(slip[0]),
            "slip_fl": float(slip[1]),
            "slip_rr": float(slip[2]),
            "slip_rl": float(slip[3]),
            "brake_command_fr_nm": -float(torque[0]),
            "brake_command_fl_nm": -float(torque[1]),
            "brake_command_rr_nm": -float(torque[2]),
            "brake_command_rl_nm": -float(torque[3]),
            "brake_active": float(v[I_BACT]),
        }

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._stop_coupling()
        # Un-spin, un-damage, zero velocity: a prior spin-out leaves the car
        # sideways/wrecked and it can never re-accelerate for the next episode.
        self.veh.teleport((0.0, 0.0, 0.0), rot_quat=(0.0, 0.0, 0.0, 1.0), reset=True)
        time.sleep(0.3)
        self.episode += 1
        target_mph = float(self.np_random.choice(self.speeds))
        self.target_ms = target_mph * MPH_TO_MS
        arm_ms = self.target_ms + ARM_MARGIN_MS
        self.target_mph = target_mph
        self.gear_mode = GEAR_MODES[int(self.np_random.integers(len(GEAR_MODES)))]
        self._neutral_shifted = self.gear_mode == "neutral"

        # abstelemetry as metric only: zero stock capacity, ABS/ESC/TC off.
        self._q("extensions.load('abstelemetry')")
        self._q("extensions.abstelemetry.setBrakes(0,0,0,0)")
        self._q('wheels.setABSBehavior("off")')
        self._q("extensions.abstelemetry.resetAccum()")
        self._q("extensions.abstelemetry.setTargetSpeed(%f)" % self.target_ms)
        time.sleep(0.25)

        # Co-sim leaves its last torque latched on the rotators; clear it or the
        # wheels stay braked and the car can't launch the next episode.
        self._q("for _,w in pairs(wheels.wheelRotators) do "
                "w.desiredBrakingTorque=0 end")
        self._q("input.brake=0")
        self.bng.settings.set_nondeterministic()
        self.bng.control.resume()
        self._accelerated_runup(arm_ms)
        self._handoff_gear()
        if self._max_brake is None:                 # rated, static -> read once
            # Read AFTER the car has run: brakeTorque is 0 until wheels re-init
            # post-teleport, so an early read defaults everything to MAX_BRAKE_NM.
            self._q("for _,w in pairs(wheels.wheelRotators) do "
                    "electrics.values['maxbrk_'..tostring(w.name)]=w.brakeTorque or 0 end")
            time.sleep(0.2)
            self.veh.sensors.poll()
            e = self.veh.sensors["electrics"]
            mb = [e.get(k) for k in WHEEL_KEYS]
            self._max_brake = np.array(
                [float(x) if x else MAX_BRAKE_NM for x in mb], dtype=np.float64)
            self._diag("MAXBRAKE raw=%s -> %s (None=read-failed, defaulted to %g)"
                       % (mb, self._max_brake.tolist(), MAX_BRAKE_NM))
        self._q("extensions.abstelemetry.armBrakeSlam(%f)" % arm_ms)
        self._q(self.link.load_cmd())
        self._coupling_up = True

        self._prev_rel[:] = 0.0
        self._reset_frame_state()
        self._steps = 0
        self._stop_timer = 0
        self._peak_g = 0.0
        self._min_binput = 1.0           # min input.brake seen; if <0.001 the metric SM resets
        self._seen_moving = False        # co-sim vehicleGroundSpeed reads ~0 for first ~20 packets
        self._measuring_seen = False
        self._metric_closed_frames = 0
        self._early_rel_sum = 0.0
        self._early_rel_n = 0
        self._yaw_abs_sum = 0.0
        self._yaw_sq_sum = 0.0
        self._heading = 0.0              # signed integral of yaw rate (rad)
        self._prev_heading_err = 0.0
        self._g_reward_tracker.reset()
        self._t0 = time.monotonic()
        self.start_speed_ms = 0.0
        self._brake_activation_logged = False
        self._coupled_moving_packet_logged = False

        dropped = self.link.drain()
        v = self._first_packet()
        if dropped:
            self._diag("COSIM_DRAINED ep=%d stale_packets=%d" % (self.episode, dropped))
        self.start_speed_ms = float(v[I_GS])
        self._diag(
            "COSIM_FIRST_PACKET ep=%d speed_ms=%.4f ground_ms=%.4f fused_ms=%.4f valid=%s"
            % (self.episode, float(v[I_INST_SPEED]), float(v[I_GS]),
               float(v[I_FUSED]), float(v[I_INST_SPEED]) > 5.0))
        self._log_first_moving_packet(v)
        self._log_brake_activation(v)
        self._obs_last = self._obs(v)
        return self._obs_last, {"target_mph": target_mph}

    def _first_packet(self):
        for _ in range(600):                 # tolerate the initial handshake
            v = self.link.exchange([0.0, 0.0, 0.0, 0.0, 1.0])   # no brake torque, brake=1
            if v is not None:
                return v
        raise RuntimeError("co-sim never sent a packet after reset")

    def _log_brake_activation(self, v):
        if self._brake_activation_logged or float(v[I_FIRED]) <= 0.5:
            return
        speed = float(v[I_FIRE_SPEED])
        if speed < 0:
            speed = float(v[I_INST_SPEED])
        self._diag(
            "BRAKE_ACTIVATION ep=%d target_ms=%.4f speed_ms=%.4f"
            % (self.episode, self.target_ms + ARM_MARGIN_MS, speed))
        self.start_speed_ms = speed
        self._brake_activation_logged = True

    def _log_first_moving_packet(self, v):
        speed = float(v[I_INST_SPEED])
        if self._coupled_moving_packet_logged or speed <= 5.0:
            return
        self._diag(
            "COSIM_FIRST_MOVING_PACKET ep=%d speed_ms=%.4f ground_ms=%.4f fused_ms=%.4f"
            % (self.episode, speed, float(v[I_GS]), float(v[I_FUSED])))
        self._coupled_moving_packet_logged = True

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size < self.act_dim or not np.all(np.isfinite(action[:self.act_dim])):
            raise FloatingPointError("co-sim action contains NaN or Inf")
        fr, fl, rr, rl = self._action_to_brakes(action)   # fractions
        self._prev_brakes = (float(fr), float(fl), float(rr), float(rl))
        # Scale each wheel by ITS rated torque; send negative (controller negates).
        torque = [-fr * self._max_brake[0], -fl * self._max_brake[1],
                  -rr * self._max_brake[2], -rl * self._max_brake[3], 1.0]  # +brake=1
        v = self.link.exchange(torque)
        if v is None:
            # Lost the coupling mid-episode: end it rather than hang.
            components = self._reward.crash_reward_components(self._yaw_sq_sum)
            info = self._episode_info("LOST_LINK", 0.0, 0.0)
            info["reward_components"] = {
                "dense_g": 0.0, "step_yaw": 0.0, "step_slip": 0.0,
                "step_heading": 0.0,
                "terminal_g": 0.0, "terminal_clean_yaw_bonus": 0.0,
                "terminal_accumulated_yaw": 0.0,
                "terminal_stability": 0.0,
                "failure_base": components["failure_base"],
                "failure_accumulated_yaw": components["failure_accumulated_yaw"],
                "total": components["total"],
            }
            return (self._obs_last, components["total"], True, False, info)

        self._log_first_moving_packet(v)
        self._log_brake_activation(v)
        self._steps += 1
        rel = np.clip(action[:self.act_dim], 0.0, 1.0).astype(np.float32)
        if self._steps <= 400:            # opening commands decide the whole stop
            self._early_rel_sum += float(rel.mean())
            self._early_rel_n += 1
        gs = float(v[I_GS])
        gy_g = float(v[I_GY]) / GRAV
        braking_g = max(0.0, gy_g)
        yaw = float(v[I_YAW])
        self._peak_g = max(self._peak_g, braking_g)
        if v[I_BACT] > 0.5:
            self._measuring_seen = True
            self._metric_closed_frames = 0
        elif self._measuring_seen:
            # The game's own 2 kHz brake metric has latched (it closes at
            # |v| <= 1.0 m/s). Everything after this is a coast to standstill
            # that the metric does not score and the reward barely sees --
            # 5 s of dead episode, because below 1 m/s rolling resistance
            # alone is about 0.1 m/s^2.
            self._metric_closed_frames += 1
        if self._seen_moving:
            self._min_binput = min(self._min_binput, float(v[I_BRK]))
        self._yaw_abs_sum += abs(yaw) * DT
        self._yaw_sq_sum += yaw * yaw * DT
        self._heading += yaw * DT
        heading_err = abs(self._heading)
        self._trace_torque(v, torque)

        measuring = float(v[I_BACT]) > 0.5
        if getattr(self._reward, "metric_window_only", False) and not measuring:
            # v11: outside the scored window the metres are not the metric's,
            # so they are not charged; a flat time cost keeps the approach brisk.
            dense_reward = self._g_reward_tracker.push(
                self._dense_g(v, braking_g), speed=0.0)
            dense_reward -= self._reward.approach_time_k * DT
        else:
            dense_reward = self._g_reward_tracker.push(
                self._dense_g(v, braking_g), speed=gs)
        step_yaw_reward = self._reward.step_yaw_bonus(braking_g, abs(yaw))
        step_slip_reward = self._reward.step_slip_reward(
            self._true_slips(v, gs), DT)
        step_heading_reward = self._reward.step_heading_penalty(
            heading_err, self._prev_heading_err)
        self._prev_heading_err = heading_err
        reward = (dense_reward + step_yaw_reward + step_slip_reward
                  + step_heading_reward)
        obs = self._obs(v)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("co-sim observation contains NaN or Inf")
        diagnostics = self._step_diagnostics(v, rel, torque, obs)
        self._obs_last = obs
        self._prev_rel[:] = rel

        terminated = truncated = False
        reward_components = {
            "dense_g": dense_reward, "step_yaw": step_yaw_reward,
            "step_slip": step_slip_reward,
            "step_heading": step_heading_reward,
            "terminal_g": 0.0, "terminal_clean_yaw_bonus": 0.0,
            "terminal_accumulated_yaw": 0.0, "terminal_stability": 0.0,
            "failure_base": 0.0, "failure_accumulated_yaw": 0.0,
        }
        info = {"diagnostics": diagnostics,
                "reward_components": reward_components}
        if gs > 5.0:
            self._seen_moving = True
        self._maybe_shift_neutral(gs)
        self._stop_timer = self._stop_timer + 1 if gs < STOP_MS else 0
        if self._seen_moving and self._yaw_abs_sum > SPIN_YAW_LIMIT:
            self._q("extensions.abstelemetry.disarmBrakeSlam()")
            self._diag("SPINOUT ep=%d steps=%d yaw_abs=%.3f peak_g=%.3f start_ms=%.2f" % (
                self.episode, self._steps, self._yaw_abs_sum, self._peak_g,
                self.start_speed_ms))
            failure = self._reward.crash_reward_components(self._yaw_sq_sum)
            reward += failure["total"]
            reward_components.update(failure_base=failure["failure_base"],
                                     failure_accumulated_yaw=failure[
                                         "failure_accumulated_yaw"])
            terminated = True
            info.update(self._episode_info("SPINOUT", 0.0, 0.0))
        elif ((self._stop_timer >= STOP_FRAMES
               or self._metric_closed_frames >= STOP_FRAMES)
              and self._seen_moving):
            self._q("extensions.abstelemetry.disarmBrakeSlam()")
            time.sleep(0.15)
            self.veh.sensors.poll()
            e = self.veh.sensors["electrics"]
            avg_g = float(e.get("tel_last_brake_avg_g", v[I_AVGG]))
            dist = float(e.get("tel_last_brake_dist", v[I_DIST]))
            avg_g_arc = float(e.get("tel_last_brake_avg_g_arc", avg_g))
            dist_arc = float(e.get("tel_last_brake_dist_arc", dist))
            metric_duration = float(e.get(
                "tel_last_brake_duration", self._steps * DT))
            yaw_penalty = self._reward.terminal_stability_penalty(
                self._yaw_abs_sum)
            self._diag(
                "STOP ep=%d steps=%d stop_gs=%.3f start_ms=%.2f | metric avg_g=%.3f "
                "early_rel=%.4f | dist=%.2f | brake_active=%s slam_fired=%.0f measuring_seen=%s "
                "min_binput=%.3f | peak_g=%.3f yaw_abs=%.3f yaw_penalty=%.2f "
                "airspeed=%.3f" % (
                    self.episode, self._steps, gs, self.start_speed_ms, avg_g,
                    (self._early_rel_sum / self._early_rel_n) if self._early_rel_n else -1.0,
                    dist,
                    e.get("tel_brake_active"), v[I_FIRED], self._measuring_seen,
                    self._min_binput, self._peak_g, self._yaw_abs_sum, yaw_penalty,
                    e.get("airspeed", -1.0)))
            terminal = self._reward.terminal_reward_components(
                avg_g, self._yaw_abs_sum, self._yaw_sq_sum)
            reward += terminal["total"]
            reward_components.update({key: value for key, value in terminal.items()
                                      if key != "total"})
            terminated = True
            info.update(self._episode_info(
                "STOP", avg_g, dist, avg_g_arc=avg_g_arc,
                dist_arc=dist_arc, metric_duration=metric_duration))
        elif self._steps >= MAX_STEPS:
            self._diag(
                "TIMEOUT ep=%d steps=%d last_gs=%.3f start_ms=%.2f measuring_seen=%s "
                "peak_g=%.3f yaw_abs=%.3f" % (
                    self.episode, self._steps, gs, self.start_speed_ms,
                    self._measuring_seen, self._peak_g, self._yaw_abs_sum))
            failure = self._reward.timeout_reward_components(self._yaw_sq_sum)
            reward += failure["total"]
            reward_components.update(failure_base=failure["failure_base"],
                                     failure_accumulated_yaw=failure[
                                         "failure_accumulated_yaw"])
            truncated = True
            info.update(self._episode_info("TIMEOUT", 0.0, 0.0))

        reward_components["total"] = float(sum(
            value for key, value in reward_components.items() if key != "total"))
        diagnostics.update({"reward_" + key: value
                            for key, value in reward_components.items()})
        if (not np.isfinite(reward) or
                not all(np.isfinite(value)
                        for value in reward_components.values())):
            raise FloatingPointError("co-sim reward contains NaN or Inf")
        if not np.isclose(reward_components["total"], reward, rtol=1e-9,
                          atol=1e-9):
            raise RuntimeError("logged reward components do not reproduce reward")
        if "outcome" in info:
            self.last_info = info
        return obs, float(reward), terminated, truncated, info

    def _episode_info(self, outcome, avg_g, dist, *, avg_g_arc=0.0,
                      dist_arc=0.0, metric_duration=0.0):
        return {
            "ep_index": self.episode, "outcome": outcome,  # NOT "episode": SB3 reserves it
            "target_mph": int(round(self.target_mph)),
            "start_speed_ms": self.start_speed_ms,
            "steps": self._steps, "stop_time_s": self._steps * DT,
            "avg_g": avg_g, "stopping_dist_m": dist, "peak_g": self._peak_g,
            "avg_g_arc": avg_g_arc, "stopping_dist_arc_m": dist_arc,
            "brake_metric_duration_s": metric_duration,
            "yaw_abs_sum": self._yaw_abs_sum,
            "yaw_sq_sum": self._yaw_sq_sum,
            "gear_mode": getattr(self, "gear_mode", "neutral"),
            "wall_s": time.monotonic() - self._t0,
        }

    def close(self):
        if self._trace_f is not None:
            self._trace_f.close()
            self._trace_f = None
        try:
            self._stop_coupling()
        except Exception:
            pass
        try:
            self._set_physics_speed_factor(0)
        except Exception:
            pass
        self.link.close()
        try:
            self.bng.close()
        except Exception:
            pass
