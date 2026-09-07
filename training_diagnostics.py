"""Automatic policy, physics, and PPO diagnostics for co-sim training.

The diagnostic files are deliberately separate from episode_log.csv: that file
is the stable GUI contract, while these files may grow as experiments teach us
which measurements are useful.
"""
import csv
import gzip
import json
import math
import os

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from experiment_io import save_checkpoint_pair


DEFAULT_DIAGNOSTICS = {
    "enabled": True,
    "trace_every_steps": 10_000,
    "checkpoint_every_steps": 50_000,
    "saturation_margin": 0.01,
    "near_lock_slip": 0.80,
    "lock_slip": 0.95,
    "lock_min_speed_ms": 5.0,
}

from residual_core import axle_view

WHEELS = ("fr", "fl", "rr", "rl")
AXLES = ("front", "rear")
DT = 0.01
REWARD_COMPONENTS = (
    "dense_g", "step_yaw", "step_slip", "step_heading", "terminal_g",
    "terminal_clean_yaw_bonus",
    "terminal_accumulated_yaw", "terminal_stability", "failure_base",
    "failure_accumulated_yaw",
)


def parse_diagnostics_config(value):
    """Return a validated, JSON-serializable diagnostics configuration."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise SystemExit("diagnostics must be an object")
    cfg = dict(DEFAULT_DIAGNOSTICS, **value)
    cfg["enabled"] = bool(cfg["enabled"])

    for key in ("trace_every_steps", "checkpoint_every_steps"):
        try:
            cfg[key] = int(cfg[key])
        except (TypeError, ValueError):
            raise SystemExit("diagnostics.%s must be a positive integer" % key)
        if cfg[key] <= 0:
            raise SystemExit("diagnostics.%s must be a positive integer" % key)

    for key in ("saturation_margin", "near_lock_slip", "lock_slip",
                "lock_min_speed_ms"):
        try:
            cfg[key] = float(cfg[key])
        except (TypeError, ValueError):
            raise SystemExit("diagnostics.%s must be a finite number" % key)
        if not math.isfinite(cfg[key]):
            raise SystemExit("diagnostics.%s must be a finite number" % key)

    if not (0.0 <= cfg["saturation_margin"] < 0.5):
        raise SystemExit("diagnostics.saturation_margin must be from 0 to <0.5")
    if not (0.0 < cfg["near_lock_slip"] < cfg["lock_slip"] <= 1.0):
        raise SystemExit(
            "diagnostics slip thresholds must satisfy 0 < near_lock_slip "
            "< lock_slip <= 1")
    if cfg["lock_min_speed_ms"] < 0.0:
        raise SystemExit("diagnostics.lock_min_speed_ms must be >= 0")
    return cfg


def _number(value, default=float("nan")):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def _series(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return {key: float("nan") for key in
                ("mean", "std", "min", "p05", "p50", "p95", "max")}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "p05": float(np.quantile(a, 0.05)),
        "p50": float(np.quantile(a, 0.50)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
    }


def dominant_oscillation(values, dt=DT, minimum_hz=0.5):
    """Return (dominant Hz, peak-power fraction) for a non-flat action trace.

    A Hann window reduces episode-boundary leakage. The strength value prevents
    random exploration noise from being mistaken for a meaningful pulse rate.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 32 or float(np.std(x)) < 1e-4:
        return float("nan"), float("nan")
    centered = x - np.mean(x)
    power = np.abs(np.fft.rfft(centered * np.hanning(x.size))) ** 2
    frequencies = np.fft.rfftfreq(x.size, d=dt)
    mask = (frequencies >= minimum_hz) & (frequencies <= 0.8 * (0.5 / dt))
    if not np.any(mask):
        return float("nan"), float("nan")
    band_power = power[mask]
    total = float(np.sum(band_power))
    if total <= 1e-12:
        return float("nan"), float("nan")
    band_frequencies = frequencies[mask]
    peak = int(np.argmax(band_power))
    return float(band_frequencies[peak]), float(band_power[peak] / total)


def _variation_per_second(values, dt=DT):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.sum(np.abs(np.diff(x))) / ((x.size - 1) * dt))


def _reversal_hz(values, dt=DT, minimum_delta=0.02):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    delta = np.diff(x)
    significant = delta[np.abs(delta) >= minimum_delta]
    if significant.size < 2:
        return 0.0
    reversals = np.sum(np.sign(significant[1:]) != np.sign(significant[:-1]))
    return float(reversals / ((x.size - 1) * dt))


def _csv_value(value):
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return "%.9g" % float(value)
    return value


class PhysicalAccumulator:
    """Bounded per-episode/per-rollout storage plus derived physical metrics."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = 0
        self.reward_sum = 0.0
        self.applied = {axle: [] for axle in AXLES}
        self.raw = {axle: [] for axle in AXLES}
        self.mode = {axle: [] for axle in AXLES}
        self.slip = {wheel: [] for wheel in WHEELS}
        self.torque = {wheel: [] for wheel in WHEELS}
        self.valid_lock_steps = 0
        self.front_near = self.front_lock = 0
        self.rear_near = self.rear_lock = 0
        self.episode_avg_g = []
        self.outcomes = {}
        self.reward_components = {key: 0.0 for key in REWARD_COMPONENTS}

    def push(self, raw_action, applied_action, diagnostic, reward=0.0,
             deterministic_action=None):
        self.samples += 1
        self.reward_sum += _number(reward, 0.0)
        raw_axles = axle_view(raw_action)
        applied_axles = axle_view(applied_action)
        mode_axles = (axle_view(deterministic_action)
                      if deterministic_action is not None else (float("nan"),) * 2)
        for index, axle in enumerate(AXLES):
            self.raw[axle].append(_number(raw_axles[index]))
            self.applied[axle].append(_number(applied_axles[index]))
            self.mode[axle].append(_number(mode_axles[index]))

        if not diagnostic:
            return
        for key in REWARD_COMPONENTS:
            self.reward_components[key] += _number(
                diagnostic.get("reward_" + key), 0.0)
        for wheel in WHEELS:
            self.slip[wheel].append(_number(diagnostic.get("slip_" + wheel)))
            self.torque[wheel].append(
                _number(diagnostic.get("brake_command_" + wheel + "_nm")))

        if _number(diagnostic.get("fused_speed_ms"), 0.0) >= self.cfg["lock_min_speed_ms"]:
            self.valid_lock_steps += 1
            front_slip = max(_number(diagnostic.get("slip_fr"), 0.0),
                             _number(diagnostic.get("slip_fl"), 0.0))
            rear_slip = max(_number(diagnostic.get("slip_rr"), 0.0),
                            _number(diagnostic.get("slip_rl"), 0.0))
            self.front_near += front_slip >= self.cfg["near_lock_slip"]
            self.front_lock += front_slip >= self.cfg["lock_slip"]
            self.rear_near += rear_slip >= self.cfg["near_lock_slip"]
            self.rear_lock += rear_slip >= self.cfg["lock_slip"]

    def finish_episode(self, info):
        outcome = str(info.get("outcome", "UNKNOWN"))
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        if outcome == "STOP":
            avg_g = _number(info.get("avg_g"))
            if math.isfinite(avg_g):
                self.episode_avg_g.append(avg_g)

    def summary(self):
        row = {
            "samples": self.samples,
            "reward_sum": self.reward_sum,
            "episodes_completed": sum(self.outcomes.values()),
            "stop_episodes": self.outcomes.get("STOP", 0),
            "spinout_episodes": self.outcomes.get("SPINOUT", 0),
            "timeout_episodes": self.outcomes.get("TIMEOUT", 0),
            "lost_link_episodes": self.outcomes.get("LOST_LINK", 0),
        }
        row.update({"reward_" + key: value
                    for key, value in self.reward_components.items()})
        g_stats = _series(self.episode_avg_g)
        row.update({"episode_avg_g_" + key: value
                    for key, value in g_stats.items()})

        margin = self.cfg["saturation_margin"]
        for axle in AXLES:
            values = np.asarray(self.applied[axle], dtype=np.float64)
            raw = np.asarray(self.raw[axle], dtype=np.float64)
            mode = np.asarray(self.mode[axle], dtype=np.float64)
            row.update({"action_%s_%s" % (axle, key): value
                        for key, value in _series(values).items()})
            row["action_%s_saturated_zero_fraction" % axle] = (
                float(np.mean(values <= margin)) if values.size else float("nan"))
            row["action_%s_saturated_one_fraction" % axle] = (
                float(np.mean(values >= 1.0 - margin)) if values.size else float("nan"))
            row["raw_action_%s_mean" % axle] = _mean(raw)
            row["raw_action_%s_std" % axle] = _series(raw)["std"]
            row["raw_action_%s_clipped_low_fraction" % axle] = (
                float(np.mean(raw < 0.0)) if raw.size else float("nan"))
            row["raw_action_%s_clipped_high_fraction" % axle] = (
                float(np.mean(raw > 1.0)) if raw.size else float("nan"))
            clip_delta = np.abs(raw - values)
            row["sb3_action_clip_delta_%s_mean" % axle] = _mean(clip_delta)
            row["sb3_action_clip_delta_%s_max" % axle] = (
                float(np.nanmax(clip_delta)) if clip_delta.size else float("nan"))
            row["deterministic_action_%s_mean" % axle] = _mean(mode)
            row["deterministic_action_%s_std" % axle] = _series(mode)["std"]
            exploration = np.abs(raw - mode)
            exploration_stats = _series(exploration)
            row["exploration_delta_%s_mean" % axle] = exploration_stats["mean"]
            row["exploration_delta_%s_p95" % axle] = exploration_stats["p95"]
            hz, strength = dominant_oscillation(values)
            row["action_%s_dominant_hz" % axle] = hz
            row["action_%s_oscillation_strength" % axle] = strength
            row["action_%s_variation_per_s" % axle] = _variation_per_second(values)
            row["action_%s_reversal_hz" % axle] = _reversal_hz(values)

        for wheel in WHEELS:
            stats = _series(self.slip[wheel])
            row["slip_%s_mean" % wheel] = stats["mean"]
            row["slip_%s_p95" % wheel] = stats["p95"]
            row["slip_%s_max" % wheel] = stats["max"]
            torque_stats = _series(self.torque[wheel])
            row["brake_command_%s_mean_nm" % wheel] = torque_stats["mean"]
            row["brake_command_%s_p95_nm" % wheel] = torque_stats["p95"]
            row["brake_command_%s_max_nm" % wheel] = torque_stats["max"]

        denominator = self.valid_lock_steps
        for axle, near, locked in (
                ("front", self.front_near, self.front_lock),
                ("rear", self.rear_near, self.rear_lock)):
            row[axle + "_near_lock_fraction"] = (
                near / denominator if denominator else float("nan"))
            row[axle + "_lock_fraction"] = (
                locked / denominator if denominator else float("nan"))
            row[axle + "_near_lock_time_s"] = near * DT
            row[axle + "_lock_time_s"] = locked * DT

        difference = row["front_near_lock_fraction"] - row["rear_near_lock_fraction"]
        if not math.isfinite(difference) or abs(difference) < 0.05:
            row["bottleneck_axle"] = "balanced"
        else:
            row["bottleneck_axle"] = "front" if difference > 0 else "rear"
        return row


ACTION_STATS = ("mean", "std", "min", "p05", "p50", "p95", "max")
PHYSICAL_COLUMNS = [
    "samples", "reward_sum", "episodes_completed", "stop_episodes",
    "spinout_episodes", "timeout_episodes", "lost_link_episodes",
] + ["episode_avg_g_" + key for key in ACTION_STATS]
for _axle in AXLES:
    PHYSICAL_COLUMNS += ["action_%s_%s" % (_axle, key) for key in ACTION_STATS]
    PHYSICAL_COLUMNS += [
        "action_%s_saturated_zero_fraction" % _axle,
        "action_%s_saturated_one_fraction" % _axle,
        "raw_action_%s_mean" % _axle, "raw_action_%s_std" % _axle,
        "raw_action_%s_clipped_low_fraction" % _axle,
        "raw_action_%s_clipped_high_fraction" % _axle,
        "sb3_action_clip_delta_%s_mean" % _axle,
        "sb3_action_clip_delta_%s_max" % _axle,
        "deterministic_action_%s_mean" % _axle,
        "deterministic_action_%s_std" % _axle,
        "exploration_delta_%s_mean" % _axle,
        "exploration_delta_%s_p95" % _axle,
        "action_%s_dominant_hz" % _axle,
        "action_%s_oscillation_strength" % _axle,
        "action_%s_variation_per_s" % _axle,
        "action_%s_reversal_hz" % _axle,
    ]
for _wheel in WHEELS:
    PHYSICAL_COLUMNS += [
        "slip_%s_mean" % _wheel, "slip_%s_p95" % _wheel,
        "slip_%s_max" % _wheel,
        "brake_command_%s_mean_nm" % _wheel,
        "brake_command_%s_p95_nm" % _wheel,
        "brake_command_%s_max_nm" % _wheel,
    ]
PHYSICAL_COLUMNS += [
    "front_near_lock_fraction", "front_lock_fraction",
    "front_near_lock_time_s", "front_lock_time_s",
    "rear_near_lock_fraction", "rear_lock_fraction",
    "rear_near_lock_time_s", "rear_lock_time_s", "bottleneck_axle",
]
PHYSICAL_COLUMNS += ["reward_" + key for key in REWARD_COMPONENTS]

PPO_COLUMNS = [
    "latent_policy_std_front", "latent_policy_std_rear",
    "transformed_policy_entropy", "latent_gaussian_entropy",
    "latent_mean_abs_max", "latent_mean_abs_p99", "latent_raw_abs_max",
    "action_saturated_fraction",
    # Compatibility aliases retained for analysis scripts written before the
    # bounded policy made the latent/physical distinction important.
    "policy_std_front", "policy_std_rear", "policy_entropy",
    "ppo_approx_kl", "ppo_clip_fraction", "ppo_explained_variance",
    "ppo_policy_loss", "ppo_value_loss", "ppo_total_loss",
    "ppo_learning_rate", "ppo_updates",
]

ROLLOUT_COLUMNS = ["rollout", "end_step", "rollout_complete"] + \
                  PHYSICAL_COLUMNS + PPO_COLUMNS
EPISODE_COLUMNS = ["episode", "end_step", "outcome", "target_mph",
                   "avg_g", "stopping_dist_m"] + PHYSICAL_COLUMNS
TRACE_COLUMNS = [
    "milestone_step", "global_step", "episode", "episode_step", "reward",
    "outcome", "raw_action_front", "raw_action_rear", "action_front_release",
    "action_rear_release", "previous_action_front", "previous_action_rear",
    "deterministic_action_front", "deterministic_action_rear",
    "raw_action_fr", "raw_action_fl", "raw_action_rr", "raw_action_rl",
    "action_fr_release", "action_fl_release", "action_rr_release",
    "action_rl_release", "previous_action_fr", "previous_action_fl",
    "previous_action_rr", "previous_action_rl",
    "ground_speed_ms", "fused_speed_ms", "braking_g", "yaw_rate_rad_s",
    "wheel_speed_fr_ms", "wheel_speed_fl_ms", "wheel_speed_rr_ms",
    "wheel_speed_rl_ms", "slip_fr", "slip_fl", "slip_rr", "slip_rl",
    "brake_command_fr_nm", "brake_command_fl_nm", "brake_command_rr_nm",
    "brake_command_rl_nm", "brake_active",
    "reward_dense_g", "reward_step_yaw", "reward_step_slip", "reward_terminal_g",
    "reward_terminal_clean_yaw_bonus", "reward_terminal_accumulated_yaw",
    "reward_terminal_stability", "reward_failure_base",
    "reward_failure_accumulated_yaw", "reward_total",
]


README_TEXT = """Co-Sim ABS automatic diagnostics

rollout_metrics.csv
  One row per PPO rollout (normally n_steps=2048). It aligns empirical action
  and physics statistics with the PPO update performed from that rollout.

episode_metrics.csv
  One row per completed braking episode, including action distribution, wheel
  slip, lock time, torque command, oscillation, and bottleneck summaries.

sample_traces.csv.gz
  Full 100 Hz histories for one complete episode at each trace milestone. Open
  with Python/pandas, 7-Zip, or any tool that reads gzip-compressed CSV.

sb3/progress.csv
  Stable-Baselines3's native PPO log. rollout_metrics.csv carries the important
  fields again, aligned with the physical rollout that produced the update.

Lock definitions apply only while fused speed is at least lock_min_speed_ms.
Near-lock means the configured near_lock_slip or greater; lock means lock_slip
or greater. A bottleneck axle has at least 5 percentage points more near-lock
time than the other axle. Actions are release fractions: 0 is full braking and
1 is full release. brake_command_* is the torque command sent over co-sim, not
an independently measured torque sensor.

dominant_hz is an FFT peak from 0.5 Hz to 80% of Nyquist. Always inspect its
oscillation_strength: a weak peak is usually broad exploration noise, not a
stable pulse strategy. reversal_hz and variation_per_s are non-spectral checks.

raw_action_* is PPO's sampled physical release. action_* is what SB3 passed to
the environment. For the bounded policy they should be identical and the
sb3_action_clip_delta fields should be zero. deterministic_action_* is policy
mode; exploration_delta is the distance between that mode and the sample.
latent_policy_std is the pre-tanh Gaussian standard deviation.
latent_mean_abs_max / latent_mean_abs_p99 are the bounded pre-tanh policy
means over the rollout (bound is 3.0); latent_raw_abs_max is the unbounded
head output before the bound. action_saturated_fraction is the share of steps
where any wheel's sampled release was exactly 0.0 or 1.0 in float32. A p99
near the bound or any saturation is the early warning that preceded the
PPO-52 NaN divergence by ~1M steps.
transformed_policy_entropy is the Monte-Carlo entropy of physical [0,1]
releases. ppo_clip_fraction is PPO's probability-ratio clipping statistic; it
is unrelated to action clipping.
"""


class TrainingDiagnosticsCallback(BaseCallback):
    """Write automatically aligned rollout, episode, trace, and PPO diagnostics."""

    def __init__(self, run_dir, log_path, cfg):
        super().__init__()
        self.run_dir = run_dir
        self.log_path = log_path
        self.cfg = cfg
        self.directory = os.path.join(run_dir, "diagnostics")
        self._rollout_file = self._episode_file = self._trace_file = None
        self._rollout_writer = self._episode_writer = self._trace_writer = None
        self._rollout = None
        self._episode = None
        self._episode_index = None
        self._rollout_index = 0
        self._pending_rollout = None
        self._next_trace_step = 1
        self._trace_active = False
        self._trace_milestone = None
        self._reset_latent_stats()

    def _reset_latent_stats(self):
        self._latent_abs = []
        self._latent_raw_max = 0.0
        self._sat_steps = 0
        self._sat_total = 0

    def _latent_summary(self):
        arr = np.asarray(self._latent_abs, dtype=np.float64)
        return {
            "latent_mean_abs_max": float(arr.max()) if arr.size else float("nan"),
            "latent_mean_abs_p99": (float(np.quantile(arr, 0.99))
                                    if arr.size else float("nan")),
            "latent_raw_abs_max": float(self._latent_raw_max),
            "action_saturated_fraction": (self._sat_steps / self._sat_total
                                          if self._sat_total else float("nan")),
        }

    def _on_training_start(self):
        os.makedirs(self.directory, exist_ok=True)
        with open(os.path.join(self.directory, "README.txt"), "w",
                  encoding="utf-8") as f:
            f.write(README_TEXT)
        with open(os.path.join(self.directory, "config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

        self._rollout_file, self._rollout_writer = self._open_csv(
            "rollout_metrics.csv", ROLLOUT_COLUMNS)
        self._episode_file, self._episode_writer = self._open_csv(
            "episode_metrics.csv", EPISODE_COLUMNS)
        self._trace_file = gzip.open(
            os.path.join(self.directory, "sample_traces.csv.gz"), "wt",
            encoding="utf-8", newline="", compresslevel=6)
        self._trace_writer = csv.DictWriter(self._trace_file,
                                            fieldnames=TRACE_COLUMNS)
        self._trace_writer.writeheader()

    def _open_csv(self, name, columns):
        handle = open(os.path.join(self.directory, name), "w", newline="",
                      encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        handle.flush()
        return handle, writer

    def _on_rollout_start(self):
        self._flush_pending_rollout(with_ppo=True)
        self._rollout = PhysicalAccumulator(self.cfg)
        self._reset_latent_stats()

    def _on_step(self):
        raw = np.asarray(self.locals.get("actions", [[float("nan")] * 2]))[0]
        applied = np.asarray(self.locals.get("clipped_actions", [raw]))[0]
        deterministic = self._deterministic_action()
        rewards = self.locals.get("rewards", [0.0])
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        diagnostic = info.get("diagnostics") or {}
        reward = _number(rewards[0], 0.0)

        if self._rollout is None:
            self._rollout = PhysicalAccumulator(self.cfg)
        finite = np.asarray(raw, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            self._sat_total += 1
            if np.any(finite <= 0.0) or np.any(finite >= 1.0):
                self._sat_steps += 1
        self._rollout.push(raw, applied, diagnostic, reward, deterministic)
        self._push_episode(raw, applied, diagnostic, reward, info, deterministic)
        self._write_trace(raw, applied, diagnostic, reward, info, deterministic)

        if "outcome" in info:
            self._rollout.finish_episode(info)
            self._finish_episode(info)
            if self._trace_active:
                self._finish_trace()
        return True

    def _deterministic_action(self):
        obs = self.locals.get("obs_tensor")
        if obs is None:
            return None
        try:
            with th.no_grad():
                distribution = self.model.policy.get_distribution(obs)
                mode = distribution.mode()
                mean = getattr(distribution.distribution, "mean", None)
                if mean is not None:
                    self._latent_abs.append(
                        float(mean.abs().max().cpu().item()))
                raw = getattr(self.model.policy, "_last_raw_mean", None)
                if raw is not None:
                    self._latent_raw_max = max(
                        self._latent_raw_max, float(raw.abs().max().cpu().item()))
            return mode.detach().cpu().numpy()[0]
        except (AttributeError, RuntimeError, ValueError):
            return None

    def _push_episode(self, raw, applied, diagnostic, reward, info,
                      deterministic):
        episode = diagnostic.get("episode")
        if episode is None:
            episode = info.get("ep_index")
        if episode is None:
            return
        episode = int(episode)
        if self._episode_index != episode:
            if self._episode is not None and self._episode.samples:
                self._finish_episode({"outcome": "RESET_WITHOUT_TERMINAL"})
            self._episode_index = episode
            self._episode = PhysicalAccumulator(self.cfg)
            self._trace_active = self.num_timesteps >= self._next_trace_step
            self._trace_milestone = (self._next_trace_step
                                     if self._trace_active else None)
        self._episode.push(raw, applied, diagnostic, reward, deterministic)

    def _finish_episode(self, info):
        if self._episode is None:
            return
        self._episode.finish_episode(info)
        row = {
            "episode": self._episode_index,
            "end_step": self.num_timesteps,
            "outcome": info.get("outcome", "UNKNOWN"),
            "target_mph": info.get("target_mph", ""),
            "avg_g": info.get("avg_g", ""),
            "stopping_dist_m": info.get("stopping_dist_m", ""),
        }
        row.update(self._episode.summary())
        self._write_row(self._episode_writer, self._episode_file, row)
        self._episode = None
        self._episode_index = None

    def _write_trace(self, raw, applied, diagnostic, reward, info,
                     deterministic):
        if not self._trace_active or not diagnostic:
            return
        raw_axles = axle_view(raw)
        applied_axles = axle_view(applied)
        det_axles = axle_view(deterministic) if deterministic is not None else ("", "")
        row = {
            "milestone_step": self._trace_milestone,
            "global_step": self.num_timesteps,
            "episode": diagnostic.get("episode"),
            "episode_step": diagnostic.get("episode_step"),
            "reward": reward,
            "outcome": info.get("outcome", ""),
            "raw_action_front": raw_axles[0], "raw_action_rear": raw_axles[1],
            "action_front_release": applied_axles[0],
            "action_rear_release": applied_axles[1],
            "deterministic_action_front": det_axles[0],
            "deterministic_action_rear": det_axles[1],
        }
        if len(raw) >= 4:
            row.update({"raw_action_" + w: raw[i] for i, w in enumerate(WHEELS)})
        row.update(diagnostic)
        self._trace_writer.writerow({key: _csv_value(row.get(key, ""))
                                     for key in TRACE_COLUMNS})

    def _finish_trace(self):
        self._trace_file.flush()
        while self._next_trace_step <= self.num_timesteps:
            self._next_trace_step += self.cfg["trace_every_steps"]
        self._trace_active = False
        self._trace_milestone = None

    def _on_rollout_end(self):
        self._rollout_index += 1
        row = {
            "rollout": self._rollout_index,
            "end_step": self.num_timesteps,
            "rollout_complete": True,
        }
        row.update(self._rollout.summary())
        row.update(self._latent_summary())
        self._pending_rollout = row
        self._rollout = None

    def _ppo_values(self):
        values = getattr(self.logger, "name_to_value", {})
        get = lambda key: _number(values.get(key))
        row = {
            "transformed_policy_entropy": -get("train/entropy_loss"),
            "ppo_approx_kl": get("train/approx_kl"),
            "ppo_clip_fraction": get("train/clip_fraction"),
            "ppo_explained_variance": get("train/explained_variance"),
            "ppo_policy_loss": get("train/policy_gradient_loss"),
            "ppo_value_loss": get("train/value_loss"),
            "ppo_total_loss": get("train/loss"),
            "ppo_learning_rate": get("train/learning_rate"),
            "ppo_updates": get("train/n_updates"),
            "latent_policy_std_front": float("nan"),
            "latent_policy_std_rear": float("nan"),
            "latent_gaussian_entropy": float("nan"),
        }
        log_std = getattr(self.model.policy, "log_std", None)
        if log_std is not None:
            std = np.exp(log_std.detach().cpu().numpy()).reshape(-1)
            if std.size:
                std_front, std_rear = axle_view(std) if std.size in (2, 4) else (std[0], std[-1])
                row["latent_policy_std_front"] = float(std_front)
                row["latent_policy_std_rear"] = float(std_rear)
                row["latent_gaussian_entropy"] = float(np.sum(
                    0.5 * np.log(2.0 * math.pi * math.e * std ** 2)))
        row["policy_std_front"] = row["latent_policy_std_front"]
        row["policy_std_rear"] = row["latent_policy_std_rear"]
        row["policy_entropy"] = row["transformed_policy_entropy"]
        return row

    def _flush_pending_rollout(self, with_ppo):
        if self._pending_rollout is None:
            return
        if with_ppo:
            self._pending_rollout.update(self._ppo_values())
        self._write_row(self._rollout_writer, self._rollout_file,
                        self._pending_rollout)
        row = self._pending_rollout
        self._append_log(
            "DIAGNOSTICS step=%s sat0_front=%s sat1_front=%s "
            "sat0_rear=%s sat1_rear=%s nearlock_front=%s nearlock_rear=%s "
            "osc_front_hz=%s osc_rear_hz=%s kl=%s clip=%s explained_var=%s "
            "latent_max=%s latent_p99=%s latent_raw_max=%s sat=%s"
            % tuple(_short(row.get(key)) for key in (
                "end_step", "action_front_saturated_zero_fraction",
                "action_front_saturated_one_fraction",
                "action_rear_saturated_zero_fraction",
                "action_rear_saturated_one_fraction",
                "front_near_lock_fraction", "rear_near_lock_fraction",
                "action_front_dominant_hz", "action_rear_dominant_hz",
                "ppo_approx_kl", "ppo_clip_fraction",
                "ppo_explained_variance", "latent_mean_abs_max",
                "latent_mean_abs_p99", "latent_raw_abs_max",
                "action_saturated_fraction")))
        p99 = _number(row.get("latent_mean_abs_p99"))
        sat = _number(row.get("action_saturated_fraction"))
        if (p99 is not None and p99 > 2.7) or (sat is not None and sat > 0.0):
            self._append_log(
                "WARNING step=%s policy latent p99=%s (bound 3.0) saturated=%s: "
                "actions are pinning at the endpoints; expect frozen wheels"
                % (_short(row.get("end_step")), _short(p99), _short(sat)))
        self._pending_rollout = None

    @staticmethod
    def _write_row(writer, handle, row):
        writer.writerow({key: _csv_value(row.get(key, ""))
                         for key in writer.fieldnames})
        handle.flush()

    def _append_log(self, line):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _on_training_end(self):
        # A completed rollout has just been trained. A stop-file request can also
        # leave a partial rollout, which is useful physically but has no PPO update.
        self._flush_pending_rollout(with_ppo=True)
        if self._rollout is not None and self._rollout.samples:
            self._rollout_index += 1
            row = {
                "rollout": self._rollout_index,
                "end_step": self.num_timesteps,
                "rollout_complete": False,
            }
            row.update(self._rollout.summary())
            row.update(self._latent_summary())
            self._pending_rollout = row
            self._flush_pending_rollout(with_ppo=False)
        if self._episode is not None and self._episode.samples:
            self._finish_episode({"outcome": "TRAINING_ENDED"})
        self.close()

    def close(self):
        """Close and finalize CSV/gzip streams after clean or exceptional exit."""
        for handle in (self._rollout_file, self._episode_file, self._trace_file):
            if handle is not None and not handle.closed:
                handle.close()


def _short(value):
    number = _number(value)
    return "na" if not math.isfinite(number) else "%.4g" % number


class LoggedCheckpointCallback(BaseCallback):
    """Save paired snapshots at trained PPO-rollout boundaries.

    Saving on an arbitrary environment step would label weights that still only
    contain the previous rollout's update. Waiting for the next rollout boundary
    makes every checkpoint an honest, resumable trained-policy snapshot.  The
    effective interval is rounded *down* to a whole number of PPO rollouts so
    consecutive checkpoints never exceed the requested maximum spacing.
    """

    def __init__(self, run_name, checkpoint_dir, log_path, every_steps,
                 state_path=None, event_path=None):
        super().__init__()
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.log_path = log_path
        self.every_steps = every_steps
        self.effective_every_steps = None
        self._next_step = None
        self._last_saved_step = None
        self.state_path = state_path
        self.event_path = event_path
        self.last_trained_step = 0
        self._rollout_completed = False

    def _on_step(self):
        return True

    def _on_training_start(self):
        rollout_steps = int(getattr(self.model, "n_steps", 0))
        if rollout_steps <= 0:
            raise RuntimeError("cannot schedule checkpoints without PPO n_steps")
        if self.every_steps < rollout_steps:
            raise ValueError(
                "checkpoint_every_steps must be at least PPO n_steps (%d)" %
                rollout_steps)
        rollout_count = max(1, self.every_steps // rollout_steps)
        self.effective_every_steps = rollout_count * rollout_steps
        # Start one complete effective interval after the current trained policy.
        # A resumed run therefore never duplicates its parent checkpoint.
        self._next_step = int(self.num_timesteps) + self.effective_every_steps
        self.last_trained_step = int(self.num_timesteps)

    def _on_rollout_start(self):
        # This callback occurs after the preceding rollout's PPO update.
        self.last_trained_step = int(self.num_timesteps)
        self._rollout_completed = False
        if self.num_timesteps >= self._next_step:
            self._save()

    def _on_rollout_end(self):
        self._rollout_completed = True

    def _on_training_end(self):
        # There is no next rollout-start after the final update.
        if (self._rollout_completed and
                self.num_timesteps >= self._next_step and
                self._last_saved_step != self.num_timesteps):
            self.last_trained_step = int(self.num_timesteps)
            self._save()

    def _save(self):
        stem = "%s_%d_steps" % (self.run_name, self.num_timesteps)
        model_path = os.path.join(self.checkpoint_dir, stem + ".zip")
        vec_path = os.path.join(
            self.checkpoint_dir, stem + "_vecnormalize.pkl")
        vec = self.model.get_vec_normalize_env()
        if vec is None:
            raise RuntimeError("cannot checkpoint without paired VecNormalize")
        manifest = save_checkpoint_pair(
            self.model, vec, model_path, vec_path,
            metadata={"kind": "training_checkpoint",
                      "run_name": self.run_name,
                      "timesteps": int(self.num_timesteps)},
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("CHECKPOINT step=%d model=%s vecnormalize=%s\n" % (
                self.num_timesteps, os.path.basename(model_path),
                os.path.basename(vec_path)))
        if self.state_path:
            from experiment_io import update_run_state
            update_run_state(
                self.state_path, last_valid_checkpoint=model_path,
                last_valid_vecnormalize=vec_path,
                last_completed_rollout=int(self.num_timesteps),
                last_checkpoint_sha256=manifest["model_sha256"],
            )
        if self.event_path:
            from experiment_io import append_jsonl, utc_now
            append_jsonl(self.event_path, {
                "event": "checkpoint_committed", "utc": utc_now(),
                "step": int(self.num_timesteps), "model": model_path,
                "vecnormalize": vec_path,
                "model_sha256": manifest["model_sha256"],
            })
        self._last_saved_step = self.num_timesteps
        while self._next_step <= self.num_timesteps:
            self._next_step += self.effective_every_steps
