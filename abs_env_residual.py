"""Residual (release-from-pedal) in-car ABS env — TRAINING_GUI_SPEC.md design.

Action (2 floats, Box[0,1]): [front_release, rear_release]. zero action = full-pedal
slam = the lockup baseline; the model learns WHEN/HOW MUCH to release. Axle-locked
(FL==FR, RL==RR) so brakes cannot yaw-spin the car (PPO_V2 lesson). Reward v5.0
inherited untouched. Obs 27 -> 28: live episode pedal appended (same pattern as
ABSLearningEnvAxle's split_cap channel, abs_env_axle.py:58-61).

Subclass of ABSLearningEnvIncar. DO NOT EDIT abs_env.py / abs_env_incar.py (copies
here are byte-identical to PPO_V3_AxleCurriculum's; verify with git diff --no-index).

Driver-pedal note (verified against controller_reference/MTB-ML-ABS.lua:694-712):
the in-car controller force-overwrites input.brake with maxBrake(extCmd) every
tick it applies brakes ("regardless of what the driver pedal is doing"), so the
literal value the parent's step()/reset() pass to vehicle.control(brake=...) has
NO effect on wheel torque -- actual per-wheel torque is driven entirely by the
mailboxed setExtCmd values, which is exactly what residual_to_brakes()'s output
becomes after inversion below. The parent's hardcoded vehicle.control(brake=1.0)
calls only matter for the engage/disengage state machine (driverBrakeHeld), whose
"stay held" threshold is 0.05 -- comfortably satisfied by the parent's own 1.0
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
import gymnasium as gym
import abs_env
from abs_env_incar import ABSLearningEnvIncar, MAX_EPISODE_STEPS
from residual_core import residual_to_brakes
from residual_log import get_logger, StepRingBuffer
from sim_config import SimConfig, resolved_userpath

PEDAL_OBS_INDEX = 27
OBS_DIM = 28
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
    """None = pinning disabled (the new default -- see sim_config.py).
    Explicit beamng_cores wins; otherwise every core not claimed by Python."""
    if not cfg.cpu_pinning:
        return None
    py_cores = list(cfg.python_cores)
    bng_cores = (list(cfg.beamng_cores) if cfg.beamng_cores
                else [c for c in range(total_cores) if c not in py_cores])
    return py_cores, bng_cores


class ABSLearningEnvResidual(ABSLearningEnvIncar):
    """In-car env with residual (release-from-pedal) action space. Action is
    [front_release, rear_release] in [0,1]; the episode's driver pedal position
    (constant per episode, optionally randomized) is appended to the parent's
    27-dim published obs as channel 28."""

    def __init__(self, *args, sim_config=None, pedal_range=None, **kwargs):
        # self._sim_config must exist before super().__init__() runs -- the
        # parent's __init__ calls self._apply_performance_tuning(...) (our
        # override below) partway through its own body.
        self._sim_config = sim_config or SimConfig()
        if sim_config is not None:
            headless, map_name, kwargs = _resolve_launch_kwargs(sim_config, kwargs)
            # HEADLESS/MAP_NAME are read as module globals inside abs_env's
            # __init__ body, not passed as parameters -- same seam
            # abs_env_incar.py already uses for VEHICLE_PC. Only touched when
            # sim_config is explicitly given, so an existing caller that never
            # passes one launches exactly as before.
            abs_env.HEADLESS = headless
            abs_env.MAP_NAME = map_name
            log.info("sim_config: game=%s headless=%s map=%s port=%s "
                     "user_path=%s cpu_pinning=%s", sim_config.game, headless,
                     map_name, kwargs.get("port"), kwargs.get("user_path"),
                     sim_config.cpu_pinning)
        super().__init__(*args, **kwargs)
        self.pedal_range = pedal_range          # None => constant 1.0
        self.episode_pedal = 1.0
        low = np.concatenate([self.observation_space.low, [0.0]]).astype(np.float32)
        high = np.concatenate([self.observation_space.high, [1.0]]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self._ring = StepRingBuffer(STEP_RING_CAPACITY)
        self._ep_rel_sum = np.zeros(2)
        self._ep_rel_max = np.zeros(2)
        self._ep_t0 = 0.0
        log.info("env ready: port=%s obs_dim=%d action=%s pedal_range=%s fixed_mph=%s",
                 getattr(self, "port", "?"), OBS_DIM, self.action_space.shape,
                 pedal_range, getattr(self, "fixed_mph", None))

    def _apply_performance_tuning(self, python_cores, beamng_cores):
        """Override, not an edit to the protected abs_env.py: that file calls
        this unconditionally with its own hardcoded core lists. CPU pinning is
        opt-in now (sim_config.cpu_pinning, default False) -- a config tuned
        for one specific CPU has no business running unasked on someone else's
        machine, and it was already a silent no-op here anyway (psutil isn't
        in this project's venv)."""
        cores = _resolve_cpu_cores(self._sim_config, os.cpu_count() or 16)
        if cores is None:
            log.info("cpu pinning disabled (sim_config.cpu_pinning=False)")
            return
        super()._apply_performance_tuning(*cores)

    def _draw_pedal(self):
        if self.pedal_range is None:
            return 1.0
        lo, hi = self.pedal_range
        return round(random.uniform(lo, hi), 3)

    def _append_pedal(self, obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32),
                               [np.float32(self.episode_pedal)]])

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
        self._ring.clear()
        self._ep_rel_sum[:] = 0.0
        self._ep_rel_max[:] = 0.0
        t0 = time.monotonic()
        log.info("reset: episode=%d pedal=%.3f speeds=%s",
                 self.episode_count + 1, self.episode_pedal, self.fixed_mph)
        try:
            obs, info = super().reset(seed=seed)
        except Exception as e:
            # The parent's two RuntimeErrors name the failed phase ("brake slam
            # never fired" vs "never engaged/exited warmup") and embed the
            # electrics snapshot in the message -- log verbatim, then re-raise.
            log.error("reset FAILED after %.1fs: %s: %s", time.monotonic() - t0,
                      type(e).__name__, e)
            raise
        self._ep_t0 = time.monotonic()
        log.info("reset ok: episode=%d target=%smph start_speed=%.2fm/s took=%.1fs",
                 self.episode_count, self.target_mph, self.start_speed_ms,
                 self._ep_t0 - t0)
        # Parent's reset() has already engaged the controller with a hardcoded
        # brake=1.0 (abs_env_incar.py:254) -- required, since engage needs
        # driverBrake > threshold and episode_pedal may be well below that. This
        # call is INERT: abs_env_incar.py's step() re-asserts brake=1.0 every tick
        # (module docstring), so whatever is set here is overwritten before the
        # next physics step anyway. Kept only as a harmless statement of intent
        # for a reader of this method, not because it has any physics effect.
        self.vehicle.control(brake=float(self.episode_pedal))
        info["episode_pedal"] = self.episode_pedal
        return self._append_pedal(obs), info

    def step(self, action):
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

        rel = np.clip(np.asarray(action, dtype=np.float64)[:2], 0.0, 1.0)
        self._ep_rel_sum += rel
        self._ep_rel_max = np.maximum(self._ep_rel_max, rel)
        self._ring.push(self.steps_taken, rel, (fr, fl, rr, rl),
                        obs[4], obs[6], self._last_gps_speed)

        if terminated or truncated:
            self._log_episode_end()
        return self._append_pedal(obs), reward, terminated, truncated, info

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

    def _dump_ring(self, why):
        rows = self._ring.dump()
        sep = chr(10) + "  "
        log.warning("last %d steps before %s (episode %d):%s%s",
                    len(rows), why, self.episode_count, sep, sep.join(rows))
