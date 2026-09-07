"""PPO trainer for the co-sim straight-line ABS env (v1).

GUI-facing entrypoint. The GUI never imports this -- it writes a config.json,
spawns `python train_cosim.py --config <path>`, tails train.log / episode_log.csv,
and drops the stop file to end a run gracefully. See docs/GUI_CONTRACT.md.

Graceful stop only -- a hard kill (taskkill) can corrupt PPO's optimizer state.
"""
import argparse
import atexit
import csv
import faulthandler
import json
import math
import os
import re
import secrets
import signal
import sys
import time
import traceback

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sim_config
import abs_env_cosim
from abs_env_cosim import (ABSCoSimEnv, DEFAULT_PC, obs_dim_for, act_dim_for,
                           OBS_DIM as COSIM_OBS_DIM)
from residual_core import WHEEL_MODES
from reward_spec import PRESETS
from training_diagnostics import (
    LoggedCheckpointCallback,
    TrainingDiagnosticsCallback,
    parse_diagnostics_config,
)
from bounded_ppo import (
    ACTION_INTERFACE, DEFAULT_INITIAL_RELEASE, DEFAULT_LOG_STD_INIT,
    UnitIntervalActorCriticPolicy, policy_kwargs as bounded_policy_kwargs,
    has_bounded_action_interface,
)
from experiment_io import (
    all_tensors_finite, append_jsonl, atomic_write_json, paired_vecnormalize_path,
    save_checkpoint_pair, source_provenance, update_run_state,
    utc_now, validate_checkpoint_pair,
)
from simulator_guard import require_beamng_available

EPISODE_COLUMNS = [
    "episode", "target_mph", "start_speed_ms", "steps", "stop_time_s",
    "stopping_dist_m", "avg_g", "stopping_dist_arc_m", "avg_g_arc",
    "brake_metric_duration_s", "peak_g", "yaw_abs_sum", "outcome", "gear_mode",
    "reward_total", "reward_undiscounted", "reward_discounted",
    "reward_dense_g", "reward_step_yaw", "reward_step_slip", "reward_step_heading",
    "reward_terminal_g",
    "reward_terminal_clean_yaw_bonus", "reward_terminal_accumulated_yaw",
    "reward_terminal_stability", "reward_failure_base",
    "reward_failure_accumulated_yaw", "wall_s",
]

DEFAULT_PPO = dict(lr=1e-4, n_steps=8192, batch_size=512, n_epochs=4,
                   gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                   vf_coef=0.5, max_grad_norm=0.5, net_arch=[256, 256, 256],
                   initial_release=DEFAULT_INITIAL_RELEASE,
                   log_std_init=DEFAULT_LOG_STD_INIT,
                   target_kl=0.02)   # 0 disables; stops an update before KL runs away
OBS_DIM = COSIM_OBS_DIM   # 35-value frame (MachineTrainerBoy 27 + 4 ABS-speed slips + 4 applied brake Nm) x 64 past frames
ACTION_DIM = 2


class EpisodeCSV(BaseCallback):
    """One row per finished episode. `info` carries the env's episode summary."""

    def __init__(self, path, gamma=0.99):
        super().__init__()
        self.path = path
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=EPISODE_COLUMNS,
                                 extrasaction="ignore")
        self._w.writeheader()
        self._f.flush()
        self._ep_reward = 0.0
        self._ep_discounted_reward = 0.0
        self._ep_step = 0
        self._reward_components = {}
        self.gamma = float(gamma)

    def _on_step(self):
        rewards = self.locals.get("rewards")
        if rewards is not None:
            value = float(rewards[0])
            self._ep_reward += value
            self._ep_discounted_reward += self.gamma ** self._ep_step * value
            self._ep_step += 1
        for info in self.locals.get("infos", []):
            for key, value in (info.get("reward_components") or {}).items():
                if key != "total":
                    self._reward_components[key] = (
                        self._reward_components.get(key, 0.0) + float(value))
            if "outcome" in info:
                row = dict(info)
                row["episode"] = info.get("ep_index")
                row["reward_total"] = round(self._ep_reward, 2)
                row["reward_undiscounted"] = self._ep_reward
                row["reward_discounted"] = self._ep_discounted_reward
                row.update({"reward_" + key: value for key, value in
                            self._reward_components.items()})
                self._w.writerow(row)
                self._f.flush()
                self._ep_reward = 0.0
                self._ep_discounted_reward = 0.0
                self._ep_step = 0
                self._reward_components = {}
        return True

    def _on_training_end(self):
        self.close()

    def close(self):
        if self._f is not None and not self._f.closed:
            self._f.close()


class Heartbeat(BaseCallback):
    """A HEARTBEAT line in train.log every `every` steps: live g/speed + best avg_g."""

    def __init__(self, log_path, every=1000, state_path=None):
        super().__init__()
        self.log_path = log_path
        self.every = every
        self._t0 = time.monotonic()
        self._t_last = self._t0
        self._n_last = 0
        self.best_avg_g = 0.0
        self.last_avg_g = 0.0
        self.current_g = 0.0
        self.current_ws = 0.0
        self.current_yaw = 0.0
        self.ep = 0
        self.state_path = state_path

    def _on_training_start(self):
        # A resumed model retains its global timestep.
        self._n_last = int(self.num_timesteps)

    def _write(self, line):
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            diagnostic = info.get("diagnostics") or {}
            if diagnostic:
                # Use unnormalized physical values. `new_obs` has already passed
                # through VecNormalize and its slots are not display units.
                self.current_g = float(diagnostic.get("braking_g", self.current_g))
                self.current_ws = float(diagnostic.get(
                    "wheel_speed_fr_ms", self.current_ws))
                self.current_yaw = float(diagnostic.get(
                    "yaw_abs_sum", self.current_yaw))
            if "outcome" in info:
                self.ep = info.get("ep_index", self.ep)
                self.last_avg_g = float(info.get("avg_g", 0.0))
                self.best_avg_g = max(self.best_avg_g, self.last_avg_g)
        if self.num_timesteps - self._n_last >= self.every:
            now = time.monotonic()
            fps = (self.num_timesteps - self._n_last) / max(1e-6, now - self._t_last)
            self._write(
                "HEARTBEAT step=%d fps=%.1f ep=%d cur_g=%.3f cur_ws=%.2f "
                "last_avg_g=%.3f best_avg_g=%.3f cur_yaw=%.3f" % (
                    self.num_timesteps, fps, self.ep,
                    self.current_g, self.current_ws,
                    self.last_avg_g, self.best_avg_g, self.current_yaw))
            if self.state_path:
                update_run_state(
                    self.state_path, status="training",
                    step=int(self.num_timesteps), episode=int(self.ep),
                    fps=float(fps), current_g=float(self.current_g),
                    current_yaw=float(self.current_yaw),
                    last_avg_g=float(self.last_avg_g),
                    best_avg_g=float(self.best_avg_g),
                )
            self._t_last, self._n_last = now, self.num_timesteps
        return True


class StopFile(BaseCallback):
    """Graceful stop: learn() exits cleanly when the stop file appears."""

    def __init__(self, stop_file, log_path):
        super().__init__()
        self.stop_file = stop_file
        self.log_path = log_path
        self.requested = False

    def _on_step(self):
        if self.num_timesteps % 10 == 0 and os.path.exists(self.stop_file):
            self.requested = True
            with open(self.log_path, "a") as f:
                f.write("STOP file seen -> graceful stop at %d steps\n"
                        % self.num_timesteps)
            return False
        return True


class FiniteGuard(BaseCallback):
    """Fail loudly at the first non-finite policy/interface value."""

    @staticmethod
    def _require_finite(name, value):
        try:
            finite = np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
        except (TypeError, ValueError):
            return
        if not finite:
            raise FloatingPointError("NaN or Inf detected in %s" % name)

    def _on_step(self):
        for name in ("actions", "clipped_actions", "rewards", "new_obs"):
            if self.locals.get(name) is not None:
                self._require_finite(name, self.locals[name])
        for info in self.locals.get("infos", []):
            for key, value in (info.get("diagnostics") or {}).items():
                if isinstance(value, (int, float, np.number)):
                    self._require_finite("diagnostics.%s" % key, value)
        return True

    def _on_rollout_end(self):
        self._check_model()

    def _on_rollout_start(self):
        self._check_model()

    def _on_training_end(self):
        self._check_model()

    def _check_model(self):
        if not all_tensors_finite(self.model.policy.state_dict()):
            raise FloatingPointError("NaN or Inf detected in policy parameters")
        if not all_tensors_finite(self.model.policy.optimizer.state_dict()):
            raise FloatingPointError("NaN or Inf detected in optimizer state")


def _log(path, line):
    with open(path, "a") as f:
        f.write(line + "\n")


class RunLog:
    """Tiny logger adapter so environment diagnostics reach GUI train.log."""

    def __init__(self, path):
        self.path = path

    def info(self, line):
        _log(self.path, line)


INDEPENDENT_WHEEL_REWARDS = ("v3.3", "v6.1", "v7.0", "v8.0", "v9.0", "v10.0",
                             "v11.0")


def parse_wheel_mode(value, reward_name):
    """v6.1 and v7.0 default to independent wheels; older presets keep the axle lock."""
    if value in (None, ""):
        return "independent" if reward_name in INDEPENDENT_WHEEL_REWARDS else "axle"
    mode = str(value).strip().lower()
    if mode not in WHEEL_MODES:
        raise SystemExit("wheel_mode must be one of %s" % (WHEEL_MODES,))
    return mode


PHYSICS_DT = 0.0005              # BeamNG vehicle physics, 2 kHz


def parse_control_hz(value):
    """Policy control rate in Hz, which is also the co-sim coupling rate.

    BeamNG computes sendSkips = ceil(time3rdParty / physicsDt) - 1, so only
    rates that divide the 2 kHz physics tick are actually delivered; anything
    else silently rounds down to one that does. Closed-loop braking is
    delay-limited at 100 Hz (docs/CONTROL_CEILING.md: a P regulator scores
    1.098 g at 100 Hz, 1.132 at 200 Hz and 1.189 at 400 Hz), so this is a
    training knob, not a formality.
    """
    if value in (None, ""):
        return 100.0
    try:
        hz = float(value)
    except (TypeError, ValueError):
        raise SystemExit("control_hz must be a number, got %r" % (value,))
    if not 20.0 <= hz <= 2000.0:
        raise SystemExit("control_hz must be from 20 to 2000, got %g" % hz)
    ticks = 1.0 / (hz * PHYSICS_DT)
    if abs(ticks - round(ticks)) > 1e-9:
        near = [2000.0 / n for n in (round(ticks), max(1, round(ticks) - 1))]
        raise SystemExit(
            "control_hz %g does not divide the 2 kHz physics tick; BeamNG would "
            "round it down. Nearest usable: %s" % (hz, ", ".join("%g" % n for n in near)))
    return hz


def parse_runup_speed_factor(value):
    try:
        factor = float(value)
    except (TypeError, ValueError):
        raise SystemExit("runup_speed_factor must be a number from 1 to 1000")
    if not math.isfinite(factor) or not (1.0 <= factor <= 1000.0):
        raise SystemExit("runup_speed_factor must be a number from 1 to 1000")
    return factor


def parse_run_name(value):
    name = str(value or "").strip()
    if (not name or name in (".", "..")
            or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None):
        raise SystemExit(
            "run_name must contain only letters, numbers, dot, underscore, or dash")
    return name


def parse_seed(value):
    """Return a concrete recorded seed; missing/blank means securely generated."""
    if value is None or str(value).strip().lower() in ("", "auto", "random"):
        return secrets.randbelow(2 ** 32)
    try:
        seed = int(value)
    except (TypeError, ValueError):
        raise SystemExit("seed must be an integer from 0 to 4294967295, or auto")
    if not 0 <= seed < 2 ** 32:
        raise SystemExit("seed must be an integer from 0 to 4294967295, or auto")
    return seed


def _read_json_object(path):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("could not read resume metadata %s: %s" % (path, exc))
    if not isinstance(value, dict):
        raise SystemExit("resume metadata must be a JSON object: %s" % path)
    return value


def _resume_run_dir(checkpoint):
    directory = os.path.dirname(os.path.abspath(checkpoint))
    return os.path.dirname(directory) if os.path.basename(directory) == "checkpoints" else directory


def inherit_resume_contract(cfg):
    """Copy the immutable training contract from a resume checkpoint's run.

    A GUI resume should not depend on the values currently visible in unrelated
    widgets. Runtime placement (device, port, headless mode) and the new target
    length remain selectable; policy/reward/vehicle semantics are inherited.
    """
    checkpoint = cfg.get("resume_checkpoint") or cfg.get("resume")
    if not checkpoint or not cfg.get("resume_inherit_config", False):
        return cfg
    parent_dir = _resume_run_dir(os.path.abspath(os.fspath(checkpoint)))
    parent_cfg = _read_json_object(os.path.join(parent_dir, "config.json"))
    for key in ("reward", "vehicle_pc", "speeds", "runup_speed_factor",
                "action_interface", "deployment_interface", "control_hz"):
        if key in parent_cfg:
            cfg[key] = parent_cfg[key]
    if isinstance(parent_cfg.get("ppo"), dict):
        cfg["ppo"] = dict(parent_cfg["ppo"])
    return cfg


def resolve_resume(cfg):
    """Validate an immutable parent pair and its training-contract metadata."""
    checkpoint = cfg.get("resume_checkpoint") or cfg.get("resume")
    if not checkpoint:
        return None
    checkpoint = os.path.abspath(os.fspath(checkpoint))
    vec_path = os.path.abspath(os.fspath(
        cfg.get("resume_vecnormalize") or paired_vecnormalize_path(checkpoint)))
    wheel_mode = cfg.get("wheel_mode") or "axle"
    validation = validate_checkpoint_pair(
        checkpoint, vec_path, expected_obs_dim=obs_dim_for(wheel_mode),
        expected_action_dim=act_dim_for(wheel_mode), require_bounded=True)
    parent_dir = _resume_run_dir(checkpoint)
    parent_cfg = _read_json_object(os.path.join(parent_dir, "config.json"))
    mismatches = []
    for key in ("reward", "vehicle_pc", "speeds", "action_interface",
                "deployment_interface", "control_hz", "wheel_mode"):
        wanted = cfg.get(key)
        inherited = parent_cfg.get(key)
        if wanted is not None and inherited is not None and wanted != inherited:
            mismatches.append("%s parent=%r requested=%r" % (key, inherited, wanted))
    parent_ppo = parent_cfg.get("ppo") or {}
    requested_ppo = cfg.get("ppo") or {}
    for key, wanted in requested_ppo.items():
        if key in parent_ppo and wanted != parent_ppo[key]:
            mismatches.append("ppo.%s parent=%r requested=%r" %
                              (key, parent_ppo[key], wanted))
    if mismatches:
        raise SystemExit("resume contract mismatch: " + "; ".join(mismatches))
    if int(validation["timesteps"]) >= int(cfg["total_steps"]):
        raise SystemExit(
            "resume checkpoint already has %d steps, target is %d" %
            (validation["timesteps"], cfg["total_steps"]))
    return {
        "checkpoint": checkpoint,
        "vecnormalize": vec_path,
        "parent_run": os.path.basename(parent_dir),
        "parent_config": parent_cfg,
        "validation": validation,
    }


def parse_ppo_config(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise SystemExit("ppo must be an object")
    ppo = dict(DEFAULT_PPO, **value)
    integer_keys = ("n_steps", "batch_size", "n_epochs")
    float_keys = ("lr", "gamma", "gae_lambda", "clip_range", "ent_coef",
                  "vf_coef", "max_grad_norm", "initial_release", "log_std_init",
                  "target_kl")
    try:
        for key in integer_keys:
            ppo[key] = int(ppo[key])
        for key in float_keys:
            ppo[key] = float(ppo[key])
    except (TypeError, ValueError) as exc:
        raise SystemExit("PPO hyperparameters must be numeric: %s" % exc)
    if ppo["n_steps"] < 2 or ppo["batch_size"] < 1 or ppo["n_epochs"] < 1:
        raise SystemExit("ppo n_steps >=2, batch_size >=1, and n_epochs >=1 are required")
    if ppo["batch_size"] > ppo["n_steps"]:
        raise SystemExit("ppo.batch_size cannot exceed ppo.n_steps")
    for key in float_keys:
        if not math.isfinite(ppo[key]):
            raise SystemExit("ppo.%s must be finite" % key)
    if ppo["lr"] <= 0 or ppo["clip_range"] <= 0 or ppo["max_grad_norm"] <= 0:
        raise SystemExit("ppo lr, clip_range, and max_grad_norm must be > 0")
    if not 0 < ppo["gamma"] <= 1 or not 0 < ppo["gae_lambda"] <= 1:
        raise SystemExit("ppo gamma and gae_lambda must be in (0,1]")
    if ppo["ent_coef"] < 0 or ppo["vf_coef"] < 0:
        raise SystemExit("ppo ent_coef and vf_coef must be >= 0")
    if ppo["target_kl"] < 0:
        raise SystemExit("ppo.target_kl must be >= 0 (0 disables)")
    if not 0 < ppo["initial_release"] < 1:
        raise SystemExit("ppo.initial_release must be strictly between 0 and 1")
    arch = ppo.get("net_arch")
    if (not isinstance(arch, list) or not arch or len(arch) > 24
            or any(type(width) is not int or width < 1 or width > 2048
                   for width in arch)):
        raise SystemExit(
            "ppo.net_arch must be a non-empty list of 1..24 integer widths, each 1..2048")
    return ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.json")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise SystemExit("config root must be an object")
    cfg["run_name"] = parse_run_name(cfg.get("run_name"))
    cfg["seed"] = parse_seed(cfg.get("seed"))
    cfg = inherit_resume_contract(cfg)
    try:
        cfg["total_steps"] = int(cfg["total_steps"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit("total_steps must be a positive integer")
    if cfg["total_steps"] <= 0:
        raise SystemExit("total_steps must be a positive integer")
    speeds = cfg.get("speeds", [80])
    if not isinstance(speeds, list) or not speeds:
        raise SystemExit("speeds must be a non-empty list of positive mph values")
    try:
        speeds = [float(speed) for speed in speeds]
    except (TypeError, ValueError):
        raise SystemExit("speeds must be a non-empty list of positive mph values")
    if any(not math.isfinite(speed) or speed <= 0 for speed in speeds):
        raise SystemExit("speeds must be a non-empty list of positive mph values")
    cfg["speeds"] = speeds
    try:
        cfg["port"] = int(cfg.get("port", 64280))
    except (TypeError, ValueError):
        raise SystemExit("port must be an integer from 1 to 65535")
    if not 1 <= cfg["port"] <= 65535:
        raise SystemExit("port must be an integer from 1 to 65535")

    reward_name = cfg.get("reward", "v6.0")
    if reward_name not in PRESETS:
        raise SystemExit(
            "unknown reward preset %r (choose one of: %s)"
            % (reward_name, ", ".join(sorted(PRESETS))))
    reward_spec = PRESETS[reward_name]()
    if reward_spec.normalize:
        raise SystemExit(
            "co-sim has no calibration-reference input yet; choose v6.0 or v5.0")
    cfg["reward"] = reward_name
    cfg["wheel_mode"] = parse_wheel_mode(cfg.get("wheel_mode"), reward_name)
    runup_speed_factor = parse_runup_speed_factor(
        cfg.get("runup_speed_factor", 4.0))
    cfg["runup_speed_factor"] = runup_speed_factor
    diagnostics = parse_diagnostics_config(cfg.get("diagnostics"))
    cfg["diagnostics"] = diagnostics
    cfg["action_interface"] = ACTION_INTERFACE
    cfg["deployment_interface"] = ("cosim_wheel_release_v2"
                                   if cfg["wheel_mode"] == "independent"
                                   else "cosim_axle_release_v2")
    cfg["control_hz"] = parse_control_hz(cfg.get("control_hz"))
    ppo = parse_ppo_config(cfg.get("ppo"))
    cfg["ppo"] = ppo
    resume = resolve_resume(cfg)
    here = os.path.dirname(os.path.abspath(__file__))
    sc = sim_config.load(os.path.join(here, "settings.json"))
    try:
        require_beamng_available(game=sc.game)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    run_name = cfg["run_name"]
    run_dir = os.path.join(here, "runs", run_name)
    if os.path.exists(run_dir):
        raise SystemExit("run dir already exists: %s (GUI picks a unique name)"
                         % run_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir)
    log_path = os.path.join(run_dir, "train.log")
    csv_path = os.path.join(run_dir, "episode_log.csv")
    state_path = os.path.join(run_dir, "run_state.json")
    event_path = os.path.join(run_dir, "run_events.jsonl")
    try:
        import beamngpy
        raw_beamngpy_version = getattr(beamngpy, "__version__", None)
        beamngpy_version = (str(raw_beamngpy_version).strip()
                            if raw_beamngpy_version is not None else None)
    except ImportError:
        beamngpy_version = None
    try:
        beamng_version = sim_config.detect_game_version(sc.game, sc.game_folder)
    except Exception:
        beamng_version = None
    cfg["reward_hash"] = reward_spec.hash()
    cfg["reward_implementation_hash"] = reward_spec.implementation_hash()
    cfg["provenance"] = source_provenance(here)
    cfg["provenance"].update({
        "beamng_version": beamng_version,
        "beamngpy_version": beamngpy_version,
    })
    if resume:
        cfg["resume"] = {
            "checkpoint": resume["checkpoint"],
            "vecnormalize": resume["vecnormalize"],
            "parent_run": resume["parent_run"],
            "parent_timesteps": resume["validation"]["timesteps"],
            "model_sha256": resume["validation"]["model_sha256"],
            "vecnormalize_sha256": resume["validation"]["vecnormalize_sha256"],
        }
        cfg.pop("resume_checkpoint", None)
        cfg.pop("resume_vecnormalize", None)
        cfg.pop("resume_inherit_config", None)
    atomic_write_json(os.path.join(run_dir, "config.json"), cfg)
    pid_path = os.path.join(run_dir, "pid.txt")
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))

    update_run_state(
        state_path, status="starting", pid=os.getpid(), run_name=run_name,
        target_steps=cfg["total_steps"], seed=cfg["seed"],
        reward=reward_spec.name, reward_hash=reward_spec.hash(),
        reward_implementation_hash=reward_spec.implementation_hash(),
        nan_inf_detected=False,
        resumed_from=(resume["checkpoint"] if resume else None),
    )
    append_jsonl(event_path, {
        "event": "process_start", "utc": utc_now(), "pid": os.getpid(),
        "seed": cfg["seed"], "target_steps": cfg["total_steps"],
        "resume": cfg.get("resume"),
    })

    def _remove_pid():
        try:
            if os.path.exists(pid_path):
                os.remove(pid_path)
        except OSError:
            pass
    atexit.register(_remove_pid)

    # A 2240x256x256x256x4 forward pass costs 0.236 ms across torch's default 10
    # threads and 0.114 ms on one: the network is far too small to pay for thread
    # synchronization, and those threads also contend with the simulator for
    # cores. At 400 Hz the whole control period is 2.5 ms, so this is latency
    # that directly becomes late brake commands.
    th.set_num_threads(1)

    stop_file = os.path.join(run_dir, cfg.get("stop_file", "STOP_TRAINING.txt"))
    control_hz = float(cfg.get("control_hz") or 0)
    if control_hz:
        abs_env_cosim.set_control_dt(1.0 / control_hz)
    _log(log_path, "START run=%s total_steps=%d speeds=%s pc=%s device=%s "
         "reward=%s reward_hash=%s runup_speed_factor=%g wheel_mode=%s "
         "control_hz=%g"
         % (run_name, cfg["total_steps"], cfg.get("speeds", [80]),
            cfg.get("vehicle_pc", DEFAULT_PC), cfg.get("device", "auto"),
            reward_spec.name, reward_spec.hash(), runup_speed_factor,
            cfg["wheel_mode"], 1.0 / abs_env_cosim.DT))

    def _make():
        return ABSCoSimEnv(
            speeds=cfg.get("speeds", [80]),
            vehicle_pc=cfg.get("vehicle_pc", DEFAULT_PC),
            port=cfg.get("port", 64280),
            game_folder=sc.game_folder,
            user_path=sim_config.resolved_userpath(sc),
            headless=cfg.get("headless", True),
            reward_spec=reward_spec,
            runup_speed_factor=runup_speed_factor,
            wheel_mode=cfg["wheel_mode"],
            log=RunLog(log_path), artifact_dir=run_dir)

    device = cfg.get("device") or ("cuda" if th.cuda.is_available() else "cpu")
    model = None
    venv = None
    episode_callback = None
    diagnostics_callback = None
    checkpoint_callback = None
    stop_callback = None
    fatal_stream = open(os.path.join(run_dir, "fatal.log"), "a", buffering=1)
    faulthandler.enable(file=fatal_stream, all_threads=True)

    class _ShutdownSignal(Exception):
        pass

    received_signal = {"number": None}

    def _signal_handler(signum, _frame):
        received_signal["number"] = signum
        raise _ShutdownSignal("received signal %d" % signum)

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        handled_signals.append(signal.SIGBREAK)
    previous_handlers = {}
    for handled in handled_signals:
        previous_handlers[handled] = signal.signal(handled, _signal_handler)

    try:
        raw_venv = DummyVecEnv([_make])
        if resume:
            venv = VecNormalize.load(resume["vecnormalize"], raw_venv)
            # A policy distilled against FIXED observation statistics is scored
            # with those same statistics, and it is sensitive to them: PPO-66
            # resumed a 1.195 g checkpoint and fell to 0.5 g within four
            # episodes as the running mean drifted under it. freeze_obs_norm
            # keeps the parent's statistics exactly as trained.
            venv.training = not bool(cfg.get("freeze_obs_norm", False))
            venv.norm_reward = False
            model = PPO.load(resume["checkpoint"], env=venv, device=device)
            # PPO.load takes its hyperparameters from the SAVED model, so a
            # checkpoint built elsewhere (a distilled one, say) silently imposes
            # whatever it was constructed with -- SB3 defaults of lr 3e-4,
            # gamma 0.99, lambda 0.95, no target_kl. Every resume before this
            # ran at 30x the configured learning rate while the log claimed
            # otherwise. Apply the run's configuration explicitly.
            model.learning_rate = ppo["lr"]
            model.lr_schedule = get_linear_fn(ppo["lr"], ppo["lr"], 1.0)
            model.n_steps = ppo["n_steps"]
            model.batch_size = ppo["batch_size"]
            model.n_epochs = ppo["n_epochs"]
            model.gamma = ppo["gamma"]
            model.gae_lambda = ppo["gae_lambda"]
            model.ent_coef = ppo["ent_coef"]
            model.vf_coef = ppo["vf_coef"]
            model.max_grad_norm = ppo["max_grad_norm"]
            model.target_kl = ppo["target_kl"] or None
            model.clip_range = get_linear_fn(ppo["clip_range"], ppo["clip_range"], 1.0)
            model.rollout_buffer.gamma = ppo["gamma"]
            model.rollout_buffer.gae_lambda = ppo["gae_lambda"]
            model.rollout_buffer.buffer_size = ppo["n_steps"]
            model.rollout_buffer.reset()
            for group in model.policy.optimizer.param_groups:
                group["lr"] = ppo["lr"]
            if not has_bounded_action_interface(model):
                raise RuntimeError("resumed model lost bounded action interface")
            _log(log_path, "RESUME parent=%s step=%d checkpoint=%s" % (
                resume["parent_run"], model.num_timesteps,
                resume["checkpoint"]))
        else:
            venv = VecNormalize(raw_venv, norm_obs=True, norm_reward=False)
            model = PPO(
                UnitIntervalActorCriticPolicy, venv,
                learning_rate=ppo["lr"], n_steps=ppo["n_steps"],
                batch_size=ppo["batch_size"], n_epochs=ppo["n_epochs"],
                gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"],
                clip_range=ppo["clip_range"], ent_coef=ppo["ent_coef"],
                vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"],
                target_kl=ppo["target_kl"] or None,
                policy_kwargs=bounded_policy_kwargs(
                    dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=ppo["net_arch"], vf=ppo["net_arch"])),
                    initial_release=ppo["initial_release"],
                    log_std_init=ppo["log_std_init"]),
                seed=cfg["seed"], device=device, verbose=0)
        _log(log_path,
             "MODEL device=%s net_arch=%s lr=%s action_interface=%s "
             "initial_release=%g log_std_init=%g ent_coef=%g target_kl=%g "
             "latent_bound=%g seed=%d" %
             (model.device, ppo["net_arch"], ppo["lr"], ACTION_INTERFACE,
              ppo["initial_release"], ppo["log_std_init"], ppo["ent_coef"],
              ppo["target_kl"], getattr(model.policy, "latent_bound", 0.0),
              cfg["seed"]))

        class DrainAfterUpdate(BaseCallback):
            """Empty the co-sim queue after each optimizer stall.

            PPO trains between rollouts while BeamNG keeps transmitting, so the
            update's ~17 ms (much more with big batches) arrives as a backlog
            that the next steps would consume as live telemetry -- mid-episode,
            where reset's drain cannot help.
            """

            def _on_rollout_start(self):
                venv = self.training_env
                try:
                    env = venv.venv.envs[0] if hasattr(venv, "venv") else venv.envs[0]
                except (AttributeError, IndexError):
                    return
                dropped = env.link.drain() if getattr(env, "link", None) else 0
                if dropped:
                    _log(log_path, "DRAIN_AFTER_UPDATE stale_packets=%d" % dropped)

            def _on_step(self):
                return True

        episode_callback = EpisodeCSV(csv_path, gamma=ppo["gamma"])
        heartbeat_callback = Heartbeat(log_path, state_path=state_path)
        callbacks = [FiniteGuard(), DrainAfterUpdate(), episode_callback,
                     heartbeat_callback]
        if diagnostics["enabled"]:
            # Native SB3 output is kept as a lossless source alongside the easier to
            # interpret, physically aligned diagnostic CSVs.
            sb3_dir = os.path.join(run_dir, "diagnostics", "sb3")
            model.set_logger(configure(sb3_dir, ["csv"]))
            diagnostics_callback = TrainingDiagnosticsCallback(
                run_dir, log_path, diagnostics)
            checkpoint_callback = LoggedCheckpointCallback(
                run_name, checkpoint_dir, log_path,
                diagnostics["checkpoint_every_steps"],
                state_path=state_path, event_path=event_path)
            callbacks.extend([diagnostics_callback, checkpoint_callback])
            _log(log_path,
                 "DIAGNOSTICS enabled trace_every_steps=%d checkpoint_every_steps=%d "
                 "near_lock_slip=%.2f lock_slip=%.2f lock_min_speed_ms=%.1f"
                 % (diagnostics["trace_every_steps"],
                    diagnostics["checkpoint_every_steps"],
                    diagnostics["near_lock_slip"], diagnostics["lock_slip"],
                    diagnostics["lock_min_speed_ms"]))
        stop_callback = StopFile(stop_file, log_path)
        callbacks.append(stop_callback)
        starting_step = int(model.num_timesteps)
        remaining_steps = cfg["total_steps"] - starting_step
        update_run_state(
            state_path, status="training", step=starting_step,
            remaining_steps=remaining_steps)
        model.learn(
            total_timesteps=remaining_steps, callback=callbacks,
            reset_num_timesteps=not bool(resume))

        final_path = os.path.join(run_dir, "final.zip")
        final_vec_path = os.path.join(run_dir, "vecnormalize.pkl")
        last_trained_step = (checkpoint_callback.last_trained_step
                             if checkpoint_callback is not None
                             else int(model.num_timesteps))
        if (checkpoint_callback is not None and
                checkpoint_callback._rollout_completed):
            last_trained_step = int(model.num_timesteps)
        final_manifest = save_checkpoint_pair(
            model, venv, final_path, final_vec_path,
            metadata={"kind": "final", "run_name": run_name,
                      "timesteps": int(model.num_timesteps),
                      "last_trained_step": int(last_trained_step),
                      "graceful_stop": bool(stop_callback.requested)},
        )
        final_status = "stopped" if stop_callback.requested else "completed"
        shutdown_reason = "stop_file" if stop_callback.requested else "target_reached"
        update_run_state(
            state_path, status=final_status, step=int(model.num_timesteps),
            exit_code=0, shutdown_reason=shutdown_reason,
            final_checkpoint=final_path,
            final_checkpoint_sha256=final_manifest["model_sha256"],
            last_valid_checkpoint=final_path,
            last_valid_vecnormalize=final_vec_path,
            last_completed_rollout=int(last_trained_step),
        )
        append_jsonl(event_path, {
            "event": final_status, "utc": utc_now(),
            "step": int(model.num_timesteps),
            "reason": shutdown_reason, "exit_code": 0,
        })
        _log(log_path, "DONE committed final.zip + vecnormalize.pkl status=%s" %
             final_status)
    except BaseException as exc:
        is_signal = isinstance(exc, _ShutdownSignal)
        reason = ("signal_%s" % received_signal["number"] if is_signal
                  else "python_exception")
        current_step = int(getattr(model, "num_timesteps", 0))
        trace_text = "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))
        with open(os.path.join(run_dir, "crash.log"), "a", encoding="utf-8") as f:
            f.write(trace_text)
            f.flush()
            os.fsync(f.fileno())
        update_run_state(
            state_path, status="failed", step=current_step, exit_code=1,
            shutdown_reason=reason, exception_type=type(exc).__name__,
            exception_message=str(exc), traceback_path="crash.log",
            nan_inf_detected=isinstance(exc, FloatingPointError))
        append_jsonl(event_path, {
            "event": "failed", "utc": utc_now(), "step": current_step,
            "reason": reason, "exception_type": type(exc).__name__,
            "exception_message": str(exc), "exit_code": 1,
        })
        _log(log_path, "FAILED step=%d reason=%s exception=%s: %s" % (
            current_step, reason, type(exc).__name__, exc))
        raise
    finally:
        if episode_callback is not None:
            episode_callback.close()
        if diagnostics_callback is not None:
            diagnostics_callback.close()
        if venv is not None:
            venv.close()
        for handled, previous in previous_handlers.items():
            signal.signal(handled, previous)
        faulthandler.disable()
        fatal_stream.close()


if __name__ == "__main__":
    main()
