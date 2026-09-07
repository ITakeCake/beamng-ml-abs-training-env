"""Residual (release-from-pedal) in-car ABS env, TRAINING_GUI_SPEC.md design.

Action (2 floats, Box[0,1]): [front_release, rear_release]. zero action = full-pedal
slam = the lockup baseline; the model learns WHEN/HOW MUCH to release. Axle-locked
(FL==FR, RL==RR) so brakes cannot yaw-spin the car (PPO_V2 lesson). Reward
presets are installed without editing the protected parent. Obs 27 -> 28: live
episode pedal appended (same pattern as
ABSLearningEnvAxle's split_cap channel, abs_env_axle.py:58-61).

Subclass of ABSLearningEnvIncar. DO NOT EDIT abs_env.py / abs_env_incar.py (copies
here are byte-identical to PPO_V3_AxleCurriculum's; verify with git diff --no-index).

Driver-pedal note (verified against controller_reference/MTB-ML-ABS.lua:694-712):
the in-car controller force-overwrites input.brake with maxBrake(extCmd) every
tick it applies brakes ("regardless of what the driver pedal is doing"), so the
literal value the parent's step()/reset() pass to vehicle.control(brake=...) has
NO effect on wheel torque, actual per-wheel torque is driven entirely by the
mailboxed setExtCmd values, which is exactly what residual_to_brakes()'s output
becomes after inversion below. The parent's hardcoded vehicle.control(brake=1.0)
calls only matter for the engage/disengage state machine (driverBrakeHeld), whose
"stay held" threshold is 0.05, comfortably satisfied by the parent's own 1.0
regardless of episode_pedal. Consequently no monkeypatch of vehicle.control is
needed to make "the episode pedal win": the pedal already wins, because it is the
sole input to the brake math that reaches the wheels. The explicit post-reset
vehicle.control(brake=episode_pedal) call below is kept for telemetry/debug
readability (so a game-side observer of the raw input pedal sees episode_pedal,
not a misleading 1.0) but is inert for physics and is not required for engagement
to hold once the parent's own reset() sequence has completed engagement.
"""
import os
import random
import time
import numpy as np

import sim_clock
import gymnasium as gym
import abs_env
import abs_env_incar
from abs_env_incar import ABSLearningEnvIncar, MAX_EPISODE_STEPS
from residual_core import residual_to_brakes
from residual_log import get_logger, StepRingBuffer, YawTrace, CornerDiag
from sim_config import SimConfig, resolved_userpath
from calibration import config_key
from corner import HeadingTracker, arc_radians, heading_branch_is_safe
from reward_spec import RewardSpec

PEDAL_OBS_INDEX = 27
OBS_DIM = 28
# Deliberately pessimistic: a LOWER assumed deceleration means a LONGER stop,
# a wider arc, and therefore a stricter heading-branch guard.
ARC_DECEL_ESTIMATE_MS2 = 5.0
STEP_RING_CAPACITY = 50   # per-step detail kept in memory, dumped only on bad episodes

log = get_logger("env")


def _resolve_launch_kwargs(cfg, kwargs):
    """Pure function (no game imports needed to call it, only to use the
    result): what to override on the abs_env module + what kwargs to hand to
    super().__init__(), given a SimConfig. Kept separate from __init__ so it's
    unit-testable without a live game connection.

    Reference-machine defaults are preserved when a field isn't explicitly
    set: bng_home is only overridden when cfg.game_folder is non-empty (an
    empty string would break BeamNGpy), so a caller who never touches the
    Simulator tab still launches exactly as abs_env.BNG_HOME says."""
    kwargs = dict(kwargs)
    kwargs.setdefault("port", cfg.port)
    if cfg.game_folder:
        kwargs.setdefault("bng_home", cfg.game_folder)
    kwargs.setdefault("user_path", resolved_userpath(cfg))
    return cfg.headless, cfg.map, kwargs


def _resolve_cpu_cores(cfg, total_cores):
    """None = pinning disabled (the new default, see sim_config.py).
    Explicit beamng_cores wins; otherwise every core not claimed by Python."""
    if not cfg.cpu_pinning:
        return None
    py_cores = list(cfg.python_cores)
    bng_cores = (list(cfg.beamng_cores) if cfg.beamng_cores
                else [c for c in range(total_cores) if c not in py_cores])
    return py_cores, bng_cores


# ---------------------------------------------------------------- reward spec
# abs_env_incar binds every reward constant and _terminal_g_shape into its OWN
# module namespace (`from abs_env import ...`) and reads them as globals inside
# step(). Rebinding those names therefore redirects the parent's own reward
# computation, the same seam already used for HEADLESS / MAP_NAME /
# VEHICLE_PC, and what makes duplicating ~150 lines of step() (with its drift
# risk) unnecessary. abs_env itself is never touched.
#
# LIMITATION: these are module globals, so one spec applies per PROCESS, not
# per env instance. Fine for this project (DummyVecEnv with a single env); a
# multi-env setup with differing specs would need the duplication instead.
_REWARD_NAMES = (
    "PER_STEP_K", "PER_STEP_G_GATE", "YAW_BONUS_K_STEP", "YAW_BONUS_ALPHA",
    "YAW_BONUS_K_TERMINAL", "YAW_BONUS_THRESHOLD", "YAW_PEN_K_TERMINAL",
    "YAW_RATE_DEADZONE_RAD_S", "CRASH_PENALTY",
)
_SPEC_FIELD_FOR = {
    "PER_STEP_K": "per_step_k",
    "PER_STEP_G_GATE": "per_step_g_gate",
    "YAW_BONUS_K_STEP": "yaw_bonus_k_step",
    "YAW_BONUS_ALPHA": "yaw_bonus_alpha",
    "YAW_BONUS_K_TERMINAL": "yaw_bonus_k_terminal",
    "YAW_BONUS_THRESHOLD": "yaw_bonus_threshold",
    "YAW_PEN_K_TERMINAL": "yaw_pen_k_terminal",
    "YAW_RATE_DEADZONE_RAD_S": "yaw_rate_deadzone",
    "CRASH_PENALTY": "crash_penalty",
}


def install_reward_spec(spec, refs_provider, episode_step_provider=None,
                        dt_provider=None, step_g_provider=None):
    """Point abs_env_incar's reward globals at `spec`. `refs_provider` is
    called at scoring time (not now) so a normalized spec picks up the
    calibration references for whatever configuration the CURRENT episode is
    running, speeds vary per episode, so refs cannot be bound once.

    The protected parent calls the same shape function once for every dense
    step and a second time on a successful terminal step. v6 uses that order
    to keep its rolling accumulator stateful without copying the parent's
    monolithic step(): a new (episode, step) key is dense scoring; a repeated
    key is terminal scoring.
    """
    if spec.step_mode == "rolling_consistency":
        if episode_step_provider is None or dt_provider is None:
            raise ValueError(
                "rolling reward wiring requires episode/step and dt providers")
        tracker = spec.make_step_tracker(dt_provider())
        last_key = [None]
        last_episode = [None]

        def shape(g):
            episode, step = episode_step_provider()
            key = (episode, step)
            if key != last_key[0]:
                if episode != last_episode[0]:
                    tracker.reset()
                    last_episode[0] = episode
                last_key[0] = key
                step_g = step_g_provider() if step_g_provider is not None else g
                return tracker.push(step_g, refs_provider())
            return spec.g_shape(g, refs_provider())
    else:
        def shape(g):
            return spec.g_shape(g, refs_provider())
    abs_env_incar._terminal_g_shape = shape
    for name in _REWARD_NAMES:
        setattr(abs_env_incar, name, getattr(spec, _SPEC_FIELD_FOR[name]))


def restore_reward_defaults():
    """Put the protected file's own values back."""
    abs_env_incar._terminal_g_shape = abs_env._terminal_g_shape
    for name in _REWARD_NAMES:
        setattr(abs_env_incar, name, getattr(abs_env, name))


def resolve_refs(spec, table, speed_mph, grip, radius_m, pedal=1.0):
    """(slam_g, stock_g) for the current configuration, or None when the spec
    doesn't normalize. Raises rather than returning None for a normalized spec
    with no matching row: training against absolute anchors is exactly the bug
    normalization exists to fix."""
    if not spec.normalize:
        return None
    if table is None:
        raise KeyError(
            "reward spec is normalized but no calibration table was loaded, "
            "run 'Calibrate baselines' for this car first (refusing to score "
            "against absolute anchors).")
    return table.references(config_key(grip=grip, speed_mph=speed_mph,
                                      radius_m=radius_m, pedal=pedal))


class GripArmInjector:
    """Arms the tire-grip change inside the parent's monolithic reset().

    There is no hook between the parent's `extensions.load('abstelemetry')`
    and its `armBrakeSlam(...)`, and the teleport earlier in that same reset
    reloads the vehicle Lua extension (so anything armed beforehand is lost).
    This wraps queue_lua_command for the duration of reset() and injects the
    grip arm immediately AFTER the slam arm passes through.

    Injecting after is safe: armBrakeSlam only sets the target speed: the slam
    FIRES later, on the physics tick where the coast-down crosses it. Lua
    commands execute in queue order, so the grip is armed before that tick.
    """

    def __init__(self, vehicle, grip, lead_seconds=0.0):
        self.vehicle = vehicle
        self.grip = grip
        self.lead_seconds = lead_seconds
        self.armed = False
        self._original = None

    def __enter__(self):
        if self.grip is None:
            return self
        self._original = self.vehicle.queue_lua_command

        def wrapped(cmd):
            self._original(cmd)
            if not self.armed and "armBrakeSlam" in cmd:
                self._original(
                    f"extensions.abstelemetry.armGripChange("
                    f"{self.grip}, {self.lead_seconds})")
                self.armed = True

        self.vehicle.queue_lua_command = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original is not None:
            self.vehicle.queue_lua_command = self._original
        return False


class CornerSteerInjector:
    """Turns the wheel inside the parent's reset(), at the same hook the grip
    change uses: the queued `armBrakeSlam`.

    That is the last moment the parent touches steering (it holds steering=0
    through acceleration so the car reaches speed in a straight line), and it
    is followed by the coast-down loop, so the wheel goes on at the start of
    the coast and the car turns in before the slam latches. Steering is not
    re-sent afterwards: the parent's per-step `vehicle.control(brake=1.0)`
    sends only the arguments it was given, so an input it never mentions keeps
    the value it was last set to.

    Open loop, one fixed angle, held to the stop. A closed-loop arc follower
    would react differently to each car, which would make a measured
    "stock ABS advantage in a corner" partly an advantage of the follower --
    the calibration only means anything if slam, stock and the model steer
    identically.

    `steering` may be a number or a zero-argument callable. It is a callable
    when the angle depends on values the parent only sets partway through its
    own reset (the episode's target speed): resolving it at the arm point means
    one corner can train across a speed list, each episode steering by the angle
    that row was actually calibrated at."""

    def __init__(self, vehicle, steering):
        self.vehicle = vehicle
        self.steering = steering
        self.applied = False
        self.angle = None
        self._original = None

    def __enter__(self):
        if self.steering is None:
            return self
        self._original = self.vehicle.queue_lua_command

        def wrapped(cmd):
            self._original(cmd)
            if not self.applied and "armBrakeSlam" in cmd:
                self.angle = float(self.steering() if callable(self.steering)
                                   else self.steering)
                self.vehicle.control(steering=self.angle)
                self.applied = True

        self.vehicle.queue_lua_command = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original is not None:
            self.vehicle.queue_lua_command = self._original
        return False


def _maybe_override_vehicle_pc(vehicle_pc):
    """Set BEFORE super().__init__(): abs_env_incar.py reads its own module
    global VEHICLE_PC_INCAR (not a constructor parameter) when forcing the
    training car onto the ego vehicle. None leaves the reference-machine
    default (Machine-Trainer-Boy-V2-MLABS.pc) untouched, same seam pattern
    as HEADLESS/MAP_NAME above, one module over."""
    if vehicle_pc:
        abs_env_incar.VEHICLE_PC_INCAR = vehicle_pc


class ABSLearningEnvResidual(ABSLearningEnvIncar):
    """In-car env with residual (release-from-pedal) action space. Action is
    [front_release, rear_release] in [0,1]; the episode's driver pedal position
    (constant per episode, optionally randomized) is appended to the parent's
    27-dim published obs as channel 28."""

    def __init__(self, *args, sim_config=None, vehicle_pc=None, pedal_range=None,
                 reward_spec=None, calibration_table=None, grip=1.0,
                 radius_m=None, grip_spec=None, grip_lead_seconds=0.0,
                 corner_spec=None, force_action=None, deterministic=True,
                 train_speed_factor=1.0, **kwargs):
        # self._sim_config must exist before super().__init__() runs, the
        # parent's __init__ calls self._apply_performance_tuning(...) (our
        # override below) partway through its own body.
        self._sim_config = sim_config or SimConfig()

        # Reward: default is the frozen v5.0 preset, which install_reward_spec
        # proves (184 parity tests) leaves the parent's numbers bit-identical.
        self._reward_spec = reward_spec or RewardSpec.v5()
        self._calibration = calibration_table
        self.grip = grip
        # A CornerSpec owns the radius once one is given; the bare radius_m
        # argument stays for callers that only want to key a calibration row.
        self._corner = corner_spec
        self.radius_m = corner_spec.radius_m if corner_spec else radius_m
        if (corner_spec is not None and corner_spec.steering is None
                and not (calibration_table and calibration_table.steering)):
            # Fail here, not 40 minutes into a run: with no angle the car would
            # brake in a straight line while every log said it was cornering.
            raise ValueError(
                f"corner R={corner_spec.radius_m} m has no steering angle and the "
                f"calibration table has none either, run the steering seek "
                f"(reference_runner.py --corner ...) before training this corner.")
        self._heading = None
        # Overrides the policy's action every step. force_action=(0,0) is the
        # pure-slam control: zero release means full pedal, the genuine
        # worst-case brake input, which is what a "floor" measurement needs --
        # a policy sampling releases uniformly on [0,1] brakes at roughly half
        # pedal and is nowhere near the grip limit.
        self._force_action = (None if force_action is None
                              else np.asarray(force_action, dtype=np.float64))
        # None => "stock": grip is never touched, and _draw_grip returns 1.0 so
        # the calibration key still says grip=1.000 (which is what stock IS).
        self._grip_spec = grip_spec
        self.grip_lead_seconds = grip_lead_seconds
        self._pending_grip = None
        # _current_refs() can be reached before reset() draws the first value.
        self.episode_pedal = 1.0
        install_reward_spec(
            self._reward_spec,
            self._current_refs,
            episode_step_provider=lambda: (
                getattr(self, "episode_count", 0),
                getattr(self, "steps_taken", 0),
            ),
            dt_provider=lambda: getattr(self, "_dt", 1.0 / 200.0),
            step_g_provider=lambda: getattr(self, "_reward_positive_g", 0.0),
        )
        log.info("reward spec: %s (hash=%s, normalize=%s, default=%s)",
                 self._reward_spec.name, self._reward_spec.hash(),
                 self._reward_spec.normalize, self._reward_spec.is_default())
        if sim_config is not None:
            headless, map_name, kwargs = _resolve_launch_kwargs(sim_config, kwargs)
            # HEADLESS/MAP_NAME are read as module globals inside abs_env's
            # __init__ body, not passed as parameters, same seam
            # abs_env_incar.py already uses for VEHICLE_PC. Only touched when
            # sim_config is explicitly given, so an existing caller that never
            # passes one launches exactly as before.
            abs_env.HEADLESS = headless
            abs_env.MAP_NAME = map_name
            log.info("sim_config: game=%s headless=%s map=%s port=%s "
                     "user_path=%s cpu_pinning=%s", sim_config.game, headless,
                     map_name, kwargs.get("port"), kwargs.get("user_path"),
                     sim_config.cpu_pinning)
        _maybe_override_vehicle_pc(vehicle_pc)
        if vehicle_pc:
            log.info("vehicle_pc override: %s", vehicle_pc)
        super().__init__(*args, **kwargs)
        # AFTER the parent built self.bng and ran its scenario setup: the setup
        # is a handful of calls and wants ordinary stepping, while the seam is
        # only about the per-step hot loop. sim_clock.wrap returns the handle
        # untouched when deterministic, so the default path is byte-identical.
        self.deterministic = bool(deterministic)
        self.train_speed_factor = float(train_speed_factor)
        self.bng = sim_clock.wrap(self.bng, deterministic=self.deterministic,
                                  speed_factor=self.train_speed_factor)
        # The frame limiter is deliberately LEFT ALONE. Removing it makes step()
        # return without advancing physics, 16.5 python steps per controller
        # tick, measured, so episodes never reach the stop and score zero.
        # See the block in sim_clock.py and `stage_probe.py --uncap on`.
        self.pedal_range = pedal_range          # None => constant 1.0
        self.episode_pedal = 1.0
        low = np.concatenate([self.observation_space.low, [0.0]]).astype(np.float32)
        high = np.concatenate([self.observation_space.high, [1.0]]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self._ring = StepRingBuffer(STEP_RING_CAPACITY)
        self._yaw_trace = YawTrace()
        self._corner_diag = CornerDiag()
        self._ep_rel_sum = np.zeros(2)
        self._ep_rel_max = np.zeros(2)
        self._ep_t0 = 0.0
        log.info("env ready: port=%s obs_dim=%d action=%s pedal_range=%s fixed_mph=%s",
                 getattr(self, "port", "?"), OBS_DIM, self.action_space.shape,
                 pedal_range, getattr(self, "fixed_mph", None))

    def _apply_performance_tuning(self, python_cores, beamng_cores):
        """Override, not an edit to the protected abs_env.py: that file calls
        this unconditionally with its own hardcoded core lists. CPU pinning is
        opt-in now (sim_config.cpu_pinning, default False), a config tuned
        for one specific CPU has no business running unasked on someone else's
        machine, and it was already a silent no-op here anyway (psutil isn't
        in this project's venv)."""
        cores = _resolve_cpu_cores(self._sim_config, os.cpu_count() or 16)
        if cores is None:
            log.info("cpu pinning disabled (sim_config.cpu_pinning=False)")
            return
        super()._apply_performance_tuning(*cores)

    def _draw_pedal(self):
        """One value per episode, held for the whole stop. Quantised to 2
        decimals by the spec, because the drawn value keys a calibration row."""
        if self.pedal_range is None:
            return 1.0
        if hasattr(self.pedal_range, "draw"):
            return self.pedal_range.draw()
        lo, hi = self.pedal_range          # legacy (lo, hi) callers
        return round(random.uniform(lo, hi), 2)

    def _draw_grip(self):
        """Returns the multiplier for this episode. With no spec (stock) the
        value is 1.0 AND nothing is armed, "don't touch grip at all" is
        different from "explicitly set grip to 1.0", even though both describe
        the same physics, because only the latter writes to the tire nodes."""
        if self._grip_spec is None:
            self._pending_grip = None
            return 1.0
        value = self._grip_spec.draw()
        self._pending_grip = value
        return value

    def _append_pedal(self, obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32),
                               [np.float32(self.episode_pedal)]])

    def _published_obs_vector(self, electrics):
        """Remember positive deceleration before the protected parent applies
        its historical ``abs(g)`` conversion.

        v5 still receives the parent's exact value. v6's dense tracker reads
        this side channel so acceleration in the opposite direction cannot
        earn braking reward.
        """
        obs = super()._published_obs_vector(electrics)
        self._reward_positive_g = max(0.0, float(obs[4]) / 9.81)
        return obs

    def _current_refs(self):
        """Calibration references for the episode currently running. Called at
        scoring time, not bound once, because the target speed (and later grip
        and radius) change per episode."""
        return resolve_refs(self._reward_spec, self._calibration,
                            getattr(self, "target_mph", None),
                            self.grip, self.radius_m, self.episode_pedal)

    def _outcome(self):
        """Derive the parent's terminal outcome (info is always {}): the parent
        sets ep_stopping_dist only on the STOP path; TIMEOUT is the step cap;
        anything else terminal is the heading CRASH."""
        if self.ep_stopping_dist > 0:
            return "STOP"
        if self.steps_taken >= MAX_EPISODE_STEPS:
            return "TIMEOUT"
        return "CRASH"

    def reset(self, seed=None):
        self.episode_pedal = self._draw_pedal()
        # Tire grip is drawn per episode and applied AT BRAKE ONSET (see
        # GripArmInjector): the acceleration and coast approach always run at
        # stock grip, so a low-grip episode still reaches its target speed.
        # `grip` stays 1.0 when no spec was given, "stock", never touched.
        self.grip = self._draw_grip()
        self._ring.clear()
        self._yaw_trace.clear()
        self._corner_diag.clear()
        self._ep_rel_sum[:] = 0.0
        self._ep_rel_max[:] = 0.0
        t0 = time.monotonic()
        log.info("reset: episode=%d pedal=%.3f grip=%s speeds=%s",
                 self.episode_count + 1, self.episode_pedal,
                 "stock" if self._grip_spec is None else f"{self.grip:.3f}",
                 self.fixed_mph)
        try:
            steer = None if self._corner is None else self._corner_steering
            with CornerSteerInjector(self.vehicle, steer) as steer_inj,                  GripArmInjector(self.vehicle, self._pending_grip,
                                 self.grip_lead_seconds) as inj:
                obs, info = super().reset(seed=seed)
            if self._corner is not None and not steer_inj.applied:
                raise RuntimeError(
                    "corner steering was never applied during reset(), the "
                    "parent's armBrakeSlam call was not seen, so this episode "
                    f"would have braked in a straight line while being logged "
                    f"as radius={self.radius_m} m. Refusing to continue.")
            if self._pending_grip is not None and not inj.armed:
                raise RuntimeError(
                    "grip change was never armed during reset(), the parent's "
                    "armBrakeSlam call was not seen, so this episode would have "
                    "run at stock grip while being logged as "
                    f"grip={self.grip}. Refusing to continue.")
        except Exception as e:
            # The parent's two RuntimeErrors name the failed phase ("brake slam
            # never fired" vs "never engaged/exited warmup") and embed the
            # electrics snapshot in the message, log verbatim, then re-raise.
            log.error("reset FAILED after %.1fs: %s: %s", time.monotonic() - t0,
                      type(e).__name__, e)
            raise
        self._ep_t0 = time.monotonic()
        self._start_corner_tracking()
        log.info("reset ok: episode=%d target=%smph start_speed=%.2fm/s took=%.1fs",
                 self.episode_count, self.target_mph, self.start_speed_ms,
                 self._ep_t0 - t0)
        # Parent's reset() has already engaged the controller with a hardcoded
        # brake=1.0 (abs_env_incar.py:254), required, since engage needs
        # driverBrake > threshold and episode_pedal may be well below that. This
        # call is INERT: abs_env_incar.py's step() re-asserts brake=1.0 every tick
        # (module docstring), so whatever is set here is overwritten before the
        # next physics step anyway. Kept only as a harmless statement of intent
        # for a reader of this method, not because it has any physics effect.
        self.vehicle.control(brake=float(self.episode_pedal))
        info["episode_pedal"] = self.episode_pedal
        return self._append_pedal(obs), info

    def _corner_steering(self):
        """The angle for the episode about to run. An explicit CornerSpec angle
        wins; otherwise it comes from the calibration row for this episode's
        (grip, speed, radius), the same row whose slam/stock references will
        score it, so the run and its ruler are driven identically."""
        if self._corner is None:
            return None
        if self._corner.steering is not None:
            return self._corner.signed_steering
        # Sign from the spec, magnitude from the table. config_key carries no
        # turn direction, so a row measured on a right-hander would otherwise
        # steer right while target_yaw_rate demanded left, yaw error of 2v/R
        # for the whole episode and a guaranteed "crash" that is pure
        # bookkeeping. Reusing the magnitude assumes the two directions are
        # symmetric, which holds on the flat, featureless calibration map.
        # Steering is a property of the corner geometry, not of how hard the
        # brakes are pressed, so it is looked up at the full-pedal key rather
        # than duplicated per pedal level.
        return self._corner.direction * abs(self._calibration.steering_for(
            config_key(grip=self.grip, speed_mph=self.target_mph,
                       radius_m=self.radius_m)))

    def _start_corner_tracking(self):
        """Arms the arc bookkeeping for the episode the parent just set up.
        Straight-line episodes get a tracker too: it then reports the start
        heading unchanged on every step, exactly what abs_env_incar assigns to
        target_heading once today, so nothing about a straight run moves."""
        self._heading = HeadingTracker(self.start_heading)
        self.target_yaw_rate = 0.0
        # NOT self._last_gps_speed: the parent resets that to 999.0 and only
        # fills it in partway through its own step(), so reading it before the
        # first step would ask for 999/R rad/s of yaw, one step of that
        # exhausts the entire terminal yaw budget and triggers the catastrophic
        # backstop on every corner episode.
        self._corner_speed = self.start_speed_ms
        self._entry_yaw_target = 0.0
        self._entry_yaw_actual = None    # filled on the first step
        if self._corner is None:
            return
        arc = arc_radians(self.start_speed_ms, self._corner.radius_m,
                          ARC_DECEL_ESTIMATE_MS2)
        if not heading_branch_is_safe(self.start_heading, arc,
                                      self._corner.direction):
            raise RuntimeError(
                f"corner would sweep {arc:.2f} rad from a start heading of "
                f"{self.start_heading:.2f} rad, crossing the +-pi wrap in the raw "
                "heading channel. The parent subtracts raw headings, so one step "
                "mid-corner would read a ~2pi deviation and terminate the episode "
                "as a CRASH that never happened. Spawn the car facing nearer 0.")
        # _corner_steering(), NOT the spec's own signed_steering: the angle
        # usually lives in the calibration table, and the spec property raises
        # when the spec itself has none.
        self._entry_yaw_target = self._corner.yaw_target(self.start_speed_ms)
        log.info("corner: R=%.1fm dir=%s steering=%+.3f arc=%.2frad "
                 "entry_yaw_target=%.3frad/s", self._corner.radius_m,
                 "L" if self._corner.direction > 0 else "R",
                 self._corner_steering(), arc,
                 self._corner.yaw_target(self.start_speed_ms))

    def _advance_corner_target(self):
        """Move the arc tangent one step, and hand the parent a target heading
        its own `abs(current - target)` will evaluate to the true deviation.

        target_yaw_rate is recomputed here from the CURRENT speed rather than
        latched at brake onset: the demand is v/R, and v is falling to zero
        over the stop. Speed is last step's reading (the parent has not polled
        yet), one step of lag at 200 Hz, i.e. under 0.3% of the entry speed."""
        if self._heading is None:
            return
        if self._corner is not None:
            self.target_yaw_rate = self._corner.yaw_target(self._corner_speed)
            self._heading.advance_target(self.target_yaw_rate, self._dt)
        self.target_heading = self._heading.parent_target()

    def step(self, action):
        self._advance_corner_target()
        if self._force_action is not None:
            action = self._force_action
        fr, fl, rr, rl = residual_to_brakes(np.asarray(action, dtype=np.float64),
                                            self.episode_pedal)
        # Parent step() maps its 4-float action via brakes = 0.01+0.99*a then
        # clip(0.01,1.0) (abs_env_incar.py:307-308) before mailboxing via setExtCmd.
        # Invert that mapping so the mailboxed per-wheel commands are EXACTLY our
        # (fr, fl, rr, rl). residual_to_brakes already floors each output at 0.01,
        # so the inverted value is guaranteed to land in [0,1].
        parent_action = (np.array([fr, fl, rr, rl]) - 0.01) / 0.99
        try:
            obs, reward, terminated, truncated, info = super().step(parent_action)
        except Exception as e:
            log.error("step FAILED at step=%d episode=%d: %s: %s",
                      self.steps_taken, self.episode_count, type(e).__name__, e)
            self._dump_ring("exception")
            raise

        if terminated or truncated:
            # The protected parent's TIMEOUT branch predates a fixed timeout
            # penalty, and its STOP branch predates v6's capped stability guard.
            # The logging overrides below apply the same adjustment to CSV and
            # audit output.
            reward += self._terminal_reward_adjustment(self._outcome())

        # The parent has just overwritten current_heading from telemetry, and
        # filled in _last_gps_speed for this step.
        self._heading.observe(self.current_heading)
        self._corner_speed = self._last_gps_speed
        # obs[6] is yaw_avg, the exact channel abs_env_incar's reward reads, and
        # target_yaw_rate is what it compared against on THIS step, so the
        # trace decomposes the same number the reward gated on, not a lookalike.
        yaw_error = float(obs[6]) - self.target_yaw_rate
        self._yaw_trace.push(yaw_error, self._dt)
        # obs[5] is lateral_g, how much of the tires' grip the corner itself is
        # using, which decides whether there is any left to brake with.
        self._corner_diag.push(yaw_error, obs[5], self._last_gps_speed, self._dt)
        if self._entry_yaw_actual is None:
            self._entry_yaw_actual = float(obs[6])

        rel = np.clip(np.asarray(action, dtype=np.float64)[:2], 0.0, 1.0)
        self._ep_rel_sum += rel
        self._ep_rel_max = np.maximum(self._ep_rel_max, rel)
        self._ring.push(self.steps_taken, rel, (fr, fl, rr, rl),
                        obs[4], obs[6], self._last_gps_speed)

        if terminated or truncated:
            self._log_episode_end()
        return self._append_pedal(obs), reward, terminated, truncated, info

    def _terminal_reward_adjustment(self, outcome):
        if outcome == "TIMEOUT":
            return self._reward_spec.timeout_penalty
        if outcome == "STOP":
            return self._reward_spec.terminal_stability_penalty(
                self.ep_yaw_abs_sum)
        return 0.0

    def _log_episode(self, outcome, avg_g, terminal_rew):
        super()._log_episode(
            outcome,
            avg_g,
            terminal_rew + self._terminal_reward_adjustment(outcome),
        )

    def _write_audit_terminal(self, avg_g, dist, terminal_rew, outcome):
        super()._write_audit_terminal(
            avg_g,
            dist,
            terminal_rew + self._terminal_reward_adjustment(outcome),
            outcome,
        )

    def _log_episode_end(self):
        outcome = self._outcome()
        n = max(1, self.steps_taken)
        mean_rel = self._ep_rel_sum / n
        wall = time.monotonic() - self._ep_t0
        msg = ("episode %d %s: target=%smph pedal=%.3f steps=%d wall=%.1fs "
               "peak_g=%.3f dist=%.1fm rel_mean=(%.3f,%.3f) rel_max=(%.3f,%.3f) "
               "yaw_int=%.3f")
        args = (self.episode_count, outcome, self.target_mph, self.episode_pedal,
                self.steps_taken, wall, self.ep_peak_g, self.ep_stopping_dist,
                mean_rel[0], mean_rel[1], self._ep_rel_max[0], self._ep_rel_max[1],
                self.ep_yaw_abs_sum)
        if outcome == "STOP":
            log.info(msg, *args)
        else:
            log.warning(msg, *args)
            self._dump_ring(outcome)
        if self._corner is not None:
            # Turn-in check: how much of the demanded rotation the car had
            # actually reached by the first braking step. Well under 1.0 means
            # the wheel went on too late for the turn to establish, and the
            # yaw-error integral is measuring the procedure, not the controller.
            entry = self._entry_yaw_actual or 0.0
            frac = (entry / self._entry_yaw_target) if self._entry_yaw_target else 0.0
            log.info("episode %d yaw trace: %s | entry_yaw actual=%.3f "
                     "target=%.3f (%.0f%% established)", self.episode_count,
                     self._yaw_trace.summary(), entry, self._entry_yaw_target,
                     frac * 100.0)
            # Where the car ENDED UP pointing versus where the arc says it
            # should. The reward only grades the yaw RATE error integrated over
            # time, which a car that lags and then over-rotates can pass while
            # finishing at the wrong heading, the two errors cancel in the
            # integral but not on the road. Diagnostic only; nothing scores it.
            import math as _math
            h = self._heading
            log.info("episode %d heading: swept=%.1fdeg intended=%.1fdeg "
                     "final_err=%+.1fdeg (%.4frad) | yaw_int=%.4f",
                     self.episode_count,
                     _math.degrees(h.current - h.start_heading),
                     _math.degrees(h.target - h.start_heading),
                     _math.degrees(h.error), h.error, self.ep_yaw_abs_sum)
            log.info("episode %d corner diag: %s", self.episode_count,
                     self._corner_diag.summary())

    def _dump_ring(self, why):
        rows = self._ring.dump()
        sep = chr(10) + "  "
        log.warning("last %d steps before %s (episode %d):%s%s",
                    len(rows), why, self.episode_count, sep, sep.join(rows))
