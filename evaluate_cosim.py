"""Frozen deterministic evaluation for paired co-sim PPO checkpoints.

All checkpoint pairs are loaded and validated before BeamNG is opened.  The
policy never learns, VecNormalize statistics remain frozen, and the simulator
physics remains free-running.  Evaluation order is interleaved by default so
slow session drift is not confounded with checkpoint age.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

import numpy as np

from residual_core import axle_view
from stable_baselines3 import PPO

import sim_config
from abs_env_cosim import ABSCoSimEnv, DEFAULT_PC
from experiment_io import (
    PAIR_SCHEMA, append_jsonl, atomic_write_json, paired_vecnormalize_path,
    pair_manifest_path, sha256_file, source_provenance, update_run_state,
    utc_now, validate_checkpoint_pair,
)
from reward_spec import PRESETS
from training_diagnostics import PhysicalAccumulator, parse_diagnostics_config
from simulator_guard import require_beamng_available


STEP_PATTERN = re.compile(r"_(\d+)_steps\.zip$", re.IGNORECASE)
OUTCOME_FAILURES = {"SPINOUT", "TIMEOUT", "LOST_LINK", "FAILED"}
TRACE_COLUMNS = [
    "checkpoint", "checkpoint_step", "evaluation_episode", "episode_step",
    "action_front_release", "action_rear_release", "reward",
    "action_fr_release", "action_fl_release", "action_rr_release",
    "action_rl_release",
    "ground_speed_ms", "fused_speed_ms", "braking_g", "yaw_rate_rad_s",
    "wheel_speed_fr_ms", "wheel_speed_fl_ms", "wheel_speed_rr_ms",
    "wheel_speed_rl_ms", "slip_fr", "slip_fl", "slip_rr", "slip_rl",
    "brake_command_fr_nm", "brake_command_fl_nm", "brake_command_rr_nm",
    "brake_command_rl_nm", "brake_active", "reward_dense_g",
    "reward_step_yaw", "reward_step_slip", "reward_terminal_g",
    "reward_terminal_clean_yaw_bonus", "reward_terminal_accumulated_yaw",
    "reward_terminal_stability", "reward_failure_base",
    "reward_failure_accumulated_yaw", "reward_total", "outcome",
]


class FrozenObservationNormalizer:
    """Read-only copy of the observation part of an SB3 VecNormalize."""

    def __init__(self, vec):
        self.norm_obs = bool(vec.norm_obs)
        self.mean = np.asarray(vec.obs_rms.mean, dtype=np.float64).copy()
        self.var = np.asarray(vec.obs_rms.var, dtype=np.float64).copy()
        self.epsilon = float(vec.epsilon)
        self.clip_obs = float(vec.clip_obs)

    def __call__(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if not self.norm_obs:
            return obs.copy()
        normalized = (obs.astype(np.float64) - self.mean) / np.sqrt(
            self.var + self.epsilon)
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(
            np.float32)


def checkpoint_step(path):
    match = STEP_PATTERN.search(os.path.basename(os.fspath(path)))
    if match:
        return int(match.group(1))
    return None


def discover_checkpoints(run_dir):
    """Return immutable checkpoint paths in policy-step order."""
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    paths = []
    if os.path.isdir(checkpoint_dir):
        paths.extend(os.path.join(checkpoint_dir, name)
                     for name in os.listdir(checkpoint_dir)
                     if STEP_PATTERN.search(name))
    final = os.path.join(run_dir, "final.zip")
    if os.path.isfile(final):
        paths.append(final)
    return sorted(paths, key=lambda path: (
        checkpoint_step(path) is None,
        checkpoint_step(path) if checkpoint_step(path) is not None else math.inf,
        path,
    ))


def make_schedule(checkpoints, episodes_per_checkpoint, seed):
    """Round-robin with a fresh checkpoint order in every evaluation block."""
    rng = random.Random(int(seed))
    schedule = []
    for repetition in range(int(episodes_per_checkpoint)):
        block = list(checkpoints)
        rng.shuffle(block)
        schedule.extend((path, repetition + 1) for path in block)
    return schedule


def _t95_critical(df):
    """Accurate two-sided 95% Student-t critical value without SciPy.

    The Cornish-Fisher expansion is substantially closer than silently using
    1.96 for the 50-stop checkpoint comparisons.  At df=49 it returns
    2.00955 (the tabulated value is 2.00958).
    """
    df = int(df)
    if df < 1:
        raise ValueError("Student-t degrees of freedom must be positive")
    exact_small = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df in exact_small:
        return exact_small[df]
    z = 1.959963984540054
    d = float(df)
    return (z + (z ** 3 + z) / (4.0 * d)
            + (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / (96.0 * d ** 2)
            + (3.0 * z ** 7 + 19.0 * z ** 5 + 17.0 * z ** 3 - 15.0 * z)
            / (384.0 * d ** 3))


def _ci95(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    critical = _t95_critical(len(values) - 1)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half, mean + half


def _finite_mean(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return statistics.fmean(values) if values else float("nan")


def summarize_checkpoint(rows, block_size=5):
    """Confidence and drift statistics without hiding failed stops."""
    rows = list(rows)
    all_g = [float(row["avg_g"]) if row["outcome"] == "STOP" else 0.0
             for row in rows]
    successful_g = [float(row["avg_g"]) for row in rows
                    if row["outcome"] == "STOP"]
    ci_low, ci_high = _ci95(all_g)
    success_ci_low, success_ci_high = _ci95(successful_g)
    blocks = [all_g[index:index + block_size]
              for index in range(0, len(all_g), block_size)]
    block_means = [statistics.fmean(block) for block in blocks if block]
    failures = sum(row["outcome"] != "STOP" for row in rows)
    summary = {
        "episodes": len(rows),
        "successful_stops": len(successful_g),
        "failures": failures,
        "failure_fraction": failures / len(rows) if rows else float("nan"),
        "avg_g_all": statistics.fmean(all_g) if all_g else float("nan"),
        "avg_g_all_ci95_low": ci_low,
        "avg_g_all_ci95_high": ci_high,
        "avg_g_success": (statistics.fmean(successful_g)
                          if successful_g else float("nan")),
        "avg_g_success_ci95_low": success_ci_low,
        "avg_g_success_ci95_high": success_ci_high,
        "block_size": block_size,
        "block_count": len(block_means),
        "block_avg_g_min": min(block_means) if block_means else float("nan"),
        "block_avg_g_max": max(block_means) if block_means else float("nan"),
        "block_avg_g_std": (statistics.stdev(block_means)
                            if len(block_means) > 1 else 0.0),
    }
    for key in (
        "stopping_dist_m", "stopping_dist_arc_m", "avg_g_arc",
        "brake_metric_duration_s", "yaw_abs_sum", "yaw_sq_sum", "peak_g",
        "reward_dense_g", "reward_terminal_g", "reward_terminal_stability",
        "reward_undiscounted", "reward_discounted",
        "action_front_mean", "action_rear_mean",
        "action_front_std", "action_rear_std",
        "action_front_saturated_zero_fraction",
        "action_front_saturated_one_fraction",
        "action_rear_saturated_zero_fraction",
        "action_rear_saturated_one_fraction",
        "front_near_lock_fraction", "front_lock_fraction",
        "rear_near_lock_fraction", "rear_lock_fraction",
        "front_near_lock_time_s", "front_lock_time_s",
        "rear_near_lock_time_s", "rear_lock_time_s",
        "action_front_dominant_hz", "action_rear_dominant_hz",
        "action_front_variation_per_s", "action_rear_variation_per_s",
    ):
        summary[key + "_mean"] = _finite_mean(rows, key)
    bottlenecks = [row.get("bottleneck_axle", "balanced") for row in rows]
    summary["front_bottleneck_fraction"] = (
        bottlenecks.count("front") / len(bottlenecks) if bottlenecks else 0.0)
    summary["rear_bottleneck_fraction"] = (
        bottlenecks.count("rear") / len(bottlenecks) if bottlenecks else 0.0)
    return summary


def _load_frozen_pair(path, device):
    validation = validate_checkpoint_pair(
        path, expected_obs_dim=13, expected_action_dim=2, require_bounded=True)
    model = PPO.load(path, device=device)
    with open(validation["vecnormalize_path"], "rb") as handle:
        vec = pickle.load(handle)
    return {
        "path": os.path.abspath(path), "name": os.path.basename(path),
        "step": checkpoint_step(path) or int(model.num_timesteps),
        "validation": validation, "model": model,
        "normalizer": FrozenObservationNormalizer(vec),
    }


def _episode_row(pair, repetition, info, accumulator, reward_sum,
                 discounted_sum, component_sums):
    row = {
        "checkpoint": pair["name"], "checkpoint_path": pair["path"],
        "checkpoint_step": pair["step"],
        "evaluation_episode": repetition,
        "outcome": info.get("outcome", "FAILED"),
        "avg_g": float(info.get("avg_g", 0.0)),
        "avg_g_arc": float(info.get("avg_g_arc", info.get("avg_g", 0.0))),
        "stopping_dist_m": float(info.get("stopping_dist_m", 0.0)),
        "stopping_dist_arc_m": float(info.get(
            "stopping_dist_arc_m", info.get("stopping_dist_m", 0.0))),
        "brake_metric_duration_s": float(info.get(
            "brake_metric_duration_s", info.get("stop_time_s", 0.0))),
        "peak_g": float(info.get("peak_g", 0.0)),
        "yaw_abs_sum": float(info.get("yaw_abs_sum", 0.0)),
        "yaw_sq_sum": float(info.get("yaw_sq_sum", 0.0)),
        "steps": int(info.get("steps", accumulator.samples)),
        "stop_time_s": float(info.get("stop_time_s", accumulator.samples * .01)),
        "wall_s": float(info.get("wall_s", 0.0)),
        "reward_undiscounted": float(reward_sum),
        "reward_discounted": float(discounted_sum),
    }
    row.update(accumulator.summary())
    row.update({"reward_" + key: value for key, value in component_sums.items()})
    return row


def evaluate_episode(env, pair, repetition, gamma, trace_writer,
                     episode_seed=None):
    obs, _ = env.reset(seed=episode_seed)
    accumulator = PhysicalAccumulator(parse_diagnostics_config(None))
    reward_sum = discounted_sum = 0.0
    component_sums = {}
    final_info = {"outcome": "FAILED"}
    step = 0
    while True:
        normalized = pair["normalizer"](obs)
        action, _ = pair["model"].predict(normalized, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        diagnostics = info.get("diagnostics") or {}
        accumulator.push(action, action, diagnostics, reward, action)
        reward_sum += float(reward)
        discounted_sum += gamma ** step * float(reward)
        for key, value in (info.get("reward_components") or {}).items():
            if key != "total":
                component_sums[key] = component_sums.get(key, 0.0) + float(value)
        trace = {
            "checkpoint": pair["name"], "checkpoint_step": pair["step"],
            "evaluation_episode": repetition, "episode_step": step + 1,
            "action_front_release": axle_view(action)[0],
            "action_rear_release": axle_view(action)[1],
            "reward": reward, "outcome": info.get("outcome", ""),
        }
        trace.update(diagnostics)
        trace_writer.writerow({key: trace.get(key, "") for key in TRACE_COLUMNS})
        step += 1
        if terminated or truncated:
            final_info = info
            break
    accumulator.finish_episode(final_info)
    row = _episode_row(pair, repetition, final_info, accumulator, reward_sum,
                       discounted_sum, component_sums)
    logged = sum(component_sums.values())
    if not math.isclose(logged, reward_sum, rel_tol=1e-8, abs_tol=1e-7):
        raise RuntimeError(
            "evaluation reward components %.12g do not equal return %.12g" %
            (logged, reward_sum))
    return row


def _write_csv(path, rows):
    rows = list(rows)
    columns = sorted({key for row in rows for key in row})
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix=".episodes.", suffix=".tmp",
                                     dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def _correlation(rows, left, right):
    values = []
    for row in rows:
        try:
            pair = (float(row[left]), float(row[right]))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in pair):
            values.append(pair)
    if len(values) < 2:
        return float("nan")
    a, b = np.asarray(values, dtype=np.float64).T
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def checkpoint_analysis(summaries):
    """Facts used by the experiment decision rule; no causal guesswork."""
    summaries = list(summaries)
    if not summaries:
        return {}
    best_g = max(summaries, key=lambda row: row["avg_g_all"])
    best_return = max(summaries, key=lambda row: row["reward_discounted_mean"])
    final = max(summaries, key=lambda row: row["checkpoint_step"])
    result = {
        "checkpoint_count": len(summaries),
        "best_g_step": best_g["checkpoint_step"],
        "best_g": best_g["avg_g_all"],
        "best_g_ci95_low": best_g["avg_g_all_ci95_low"],
        "best_discounted_return_step": best_return["checkpoint_step"],
        "best_discounted_return": best_return["reward_discounted_mean"],
        "latest_step": final["checkpoint_step"],
        "latest_g": final["avg_g_all"],
        "latest_discounted_return": final["reward_discounted_mean"],
        "g_drop_from_best": best_g["avg_g_all"] - final["avg_g_all"],
        "discounted_return_drop_from_best_g": (
            best_g["reward_discounted_mean"] - final["reward_discounted_mean"]),
        "checkpoint_g_vs_discounted_return_correlation": _correlation(
            summaries, "avg_g_all", "reward_discounted_mean"),
        "checkpoint_g_vs_dense_reward_correlation": _correlation(
            summaries, "avg_g_all", "reward_dense_g_mean"),
        "checkpoint_g_vs_terminal_reward_correlation": _correlation(
            summaries, "avg_g_all", "reward_terminal_g_mean"),
        "checkpoint_g_vs_front_near_lock_correlation": _correlation(
            summaries, "avg_g_all", "front_near_lock_fraction_mean"),
        "checkpoint_g_vs_rear_near_lock_correlation": _correlation(
            summaries, "avg_g_all", "rear_near_lock_fraction_mean"),
        "checkpoint_g_vs_front_action_hz_correlation": _correlation(
            summaries, "avg_g_all", "action_front_dominant_hz_mean"),
        "checkpoint_g_vs_yaw_correlation": _correlation(
            summaries, "avg_g_all", "yaw_abs_sum_mean"),
        "best_front_near_lock": best_g["front_near_lock_fraction_mean"],
        "latest_front_near_lock": final["front_near_lock_fraction_mean"],
        "best_front_action_hz": best_g["action_front_dominant_hz_mean"],
        "latest_front_action_hz": final["action_front_dominant_hz_mean"],
        "best_dense_reward": best_g["reward_dense_g_mean"],
        "latest_dense_reward": final["reward_dense_g_mean"],
        "best_terminal_stability": best_g["reward_terminal_stability_mean"],
        "latest_terminal_stability": final["reward_terminal_stability_mean"],
    }
    g_declined = result["g_drop_from_best"] > 0.0
    discounted_declined = result["discounted_return_drop_from_best_g"] > 0.0
    if g_declined and discounted_declined:
        result["decision_classification"] = "ppo_optimization_drift"
    elif g_declined:
        result["decision_classification"] = "reward_misalignment"
    else:
        result["decision_classification"] = "no_frozen_regression"
    result["dense_reward_prefers_latest_lower_g"] = (
        g_declined and result["latest_dense_reward"] > result["best_dense_reward"])
    return result


def attach_training_diagnostics(run_dir, summaries):
    """Attach nearest preceding rollout/PPO metrics to each frozen checkpoint."""
    path = os.path.join(run_dir, "diagnostics", "rollout_metrics.csv")
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            training_rows = list(csv.DictReader(handle))
    except OSError:
        return summaries
    parsed = []
    for row in training_rows:
        try:
            parsed.append((int(float(row["end_step"])), row))
        except (KeyError, TypeError, ValueError):
            continue
    fields = (
        "latent_policy_std_front", "latent_policy_std_rear",
        "transformed_policy_entropy", "ppo_approx_kl", "ppo_clip_fraction",
        "ppo_explained_variance", "ppo_policy_loss", "ppo_value_loss",
        "ppo_learning_rate", "ppo_updates",
    )
    for summary in summaries:
        eligible = [item for item in parsed
                    if item[0] <= summary["checkpoint_step"]]
        if not eligible:
            continue
        source_step, source = max(eligible, key=lambda item: item[0])
        summary["training_diagnostic_step"] = source_step
        for field in fields:
            try:
                summary["training_" + field] = float(source[field])
            except (KeyError, TypeError, ValueError):
                summary["training_" + field] = float("nan")
    return summaries


def verify_deployment_export(pair, output_dir, project_root,
                             interface="cosim_axle_release_v2", control_hz=100):
    """Export the exact selected pair and require SB3/blob numerical parity."""
    output = os.path.join(output_dir, "best_checkpoint_weights.lua")
    command = [
        sys.executable, os.path.join(project_root, "export_policy_weights.py"),
        "--ckpt", pair["path"],
        "--vecnorm", pair["validation"]["vecnormalize_path"],
        "--algo", "ppo", "--head", "ppo_tanh_release01",
        "--interface", interface,
        "--control-hz", "%g" % float(control_hz),
        "--out", output, "--n-parity", "1024",
    ]
    result = subprocess.run(command, cwd=project_root, capture_output=True,
                            text=True)
    log = result.stdout + result.stderr
    with open(os.path.join(output_dir, "deployment_parity.log"), "w",
              encoding="utf-8") as handle:
        handle.write(log)
    if result.returncode != 0 or "PASS" not in result.stdout:
        raise RuntimeError("selected checkpoint failed deployment export parity: "
                           + log[-1000:])
    match = re.search(r"max abs diff = ([\d.eE+-]+)", result.stdout)
    return {"weights_path": output,
            "weights_sha256": sha256_file(output),
            "samples": 1024,
            "max_abs_difference": (float(match.group(1)) if match else None),
            "interface": interface,
            "control_hz": 100}


def _report_markdown(run_name, summaries, analysis=None):
    lines = ["# Frozen checkpoint evaluation: %s" % run_name, "",
             "Deterministic policy actions; frozen VecNormalize statistics; "
             "free-running BeamNG physics.", "",
             "| step | stops | failures | avg G all (95% CI) | avg G success | "
             "arc avg G | discounted return | front near-lock | rear near-lock |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(summaries, key=lambda value: value["checkpoint_step"]):
        lines.append(
            "| {checkpoint_step} | {successful_stops}/{episodes} | {failures} | "
            "{avg_g_all:.4f} [{avg_g_all_ci95_low:.4f}, "
            "{avg_g_all_ci95_high:.4f}] | {avg_g_success:.4f} | "
            "{avg_g_arc_mean:.4f} | {reward_discounted_mean:.4f} | "
            "{front_near_lock_fraction_mean:.3f} | "
            "{rear_near_lock_fraction_mean:.3f} |".format(**row))
    if analysis:
        lines.extend([
            "", "## Decision evidence", "",
            "- Best physical G: step {best_g_step}, {best_g:.4f} G.",
            "- Latest physical G: step {latest_step}, {latest_g:.4f} G "
            "(drop {g_drop_from_best:.4f} G).",
            "- Physical-G versus PPO-discounted-return checkpoint correlation: "
            "{checkpoint_g_vs_discounted_return_correlation:.4f}.",
            "- Best discounted return occurred at step "
            "{best_discounted_return_step}.",
            "- Front near-lock changed from {best_front_near_lock:.3f} at the "
            "best-G checkpoint to {latest_front_near_lock:.3f} at the latest; "
            "front action frequency changed from {best_front_action_hz:.2f} Hz "
            "to {latest_front_action_hz:.2f} Hz.",
            "- Dense G reward changed from {best_dense_reward:.4f} to "
            "{latest_dense_reward:.4f}; dense reward prefers the later lower-G "
            "checkpoint: {dense_reward_prefers_latest_lower_g}.",
            "- Decision-rule classification: {decision_classification}.",
        ])
        lines[-8:] = [line.format(**analysis) for line in lines[-8:]]
    return "\n".join(lines) + "\n"


def preserve_best(pair, summary, evaluation_dir, project_root):
    source_dir = os.path.dirname(pair["path"])
    source_run_dir = (os.path.dirname(source_dir)
                      if os.path.basename(source_dir) == "checkpoints"
                      else source_dir)
    destination = os.path.join(
        project_root, "best_models",
        os.path.basename(source_run_dir) or "run",
        os.path.basename(evaluation_dir))
    if os.path.exists(destination):
        raise FileExistsError("refusing to overwrite preserved best model: %s" %
                              destination)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    partial = tempfile.mkdtemp(prefix=".partial-best-", dir=parent)
    model_target = os.path.join(partial, pair["name"])
    vec_target = os.path.join(
        partial, os.path.basename(pair["validation"]["vecnormalize_path"]))
    shutil.copy2(pair["path"], model_target)
    shutil.copy2(pair["validation"]["vecnormalize_path"], vec_target)
    validation = validate_checkpoint_pair(
        model_target, vec_target, expected_obs_dim=13, expected_action_dim=2,
        require_bounded=True)
    committed = dict(validation)
    committed.update({
        "schema": PAIR_SCHEMA, "committed_utc": utc_now(),
        "model_path": os.path.basename(model_target),
        "vecnormalize_path": os.path.basename(vec_target),
        "metadata": {"kind": "preserved_best",
                     "source_evaluation": evaluation_dir},
    })
    atomic_write_json(pair_manifest_path(model_target), committed)
    atomic_write_json(os.path.join(partial, "selection.json"), {
        "selected_utc": utc_now(), "source_evaluation": evaluation_dir,
        "source_model": pair["path"], "source_vecnormalize": pair["validation"][
            "vecnormalize_path"], "model_sha256": sha256_file(model_target),
        "vecnormalize_sha256": sha256_file(vec_target), "summary": summary,
    })
    os.replace(partial, destination)
    return destination


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="explicit .zip paths; default is every run checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--reevaluate-top", type=int, default=3)
    parser.add_argument("--top-total-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.episodes < 1 or args.reevaluate_top < 0:
        raise SystemExit("episodes must be positive and reevaluate-top non-negative")
    if args.top_total_episodes < args.episodes:
        raise SystemExit("top-total-episodes cannot be smaller than episodes")
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.abspath(args.run_dir)
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    paths = [os.path.abspath(path) for path in (args.checkpoints or
                                                discover_checkpoints(run_dir))]
    if not paths:
        raise SystemExit("no checkpoints found")
    reward_name = config.get("reward", "v6.0")
    if reward_name not in PRESETS:
        raise SystemExit("run uses unknown reward preset %r" % reward_name)
    reward = PRESETS[reward_name]()
    if reward.normalize:
        raise SystemExit("co-sim evaluator cannot evaluate normalized reward")
    if config.get("reward_hash") not in (None, reward.hash()):
        raise SystemExit("run reward hash does not match current immutable preset")
    if config.get("reward_implementation_hash") not in (
            None, reward.implementation_hash()):
        raise SystemExit("run reward implementation hash does not match source")

    # Load and validate every artifact before opening BeamNG.
    pairs = {path: _load_frozen_pair(path, args.device) for path in paths}
    sc = sim_config.load(os.path.join(project_root, "settings.json"))
    try:
        require_beamng_available(game=sc.game)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        if os.path.exists(output_dir):
            raise SystemExit("evaluation output already exists: %s" % output_dir)
    else:
        stamp = utc_now().replace(":", "-").replace("+", "_")
        output_dir = os.path.join(project_root, "evaluations",
                                  "%s_%s" % (config["run_name"], stamp))
    os.makedirs(output_dir)
    state_path = os.path.join(output_dir, "evaluation_state.json")
    event_path = os.path.join(output_dir, "evaluation_events.jsonl")
    eval_config = {
        "created_utc": utc_now(), "run_dir": run_dir,
        "run_name": config.get("run_name"), "checkpoint_paths": paths,
        "episodes": args.episodes, "reevaluate_top": args.reevaluate_top,
        "top_total_episodes": args.top_total_episodes, "schedule_seed": args.seed,
        "policy_deterministic": True, "vecnormalize_training": False,
        "physics_mode": "free_running", "reward": reward.name,
        "reward_hash": reward.hash(),
        "reward_implementation_hash": reward.implementation_hash(),
        "gamma": config.get("ppo", {}).get(
            "gamma", 0.99), "source_provenance": source_provenance(project_root),
    }
    atomic_write_json(os.path.join(output_dir, "config.json"), eval_config)
    update_run_state(state_path, status="starting", completed_episodes=0,
                     total_planned=len(paths) * args.episodes)
    append_jsonl(event_path, {"event": "evaluation_start", "utc": utc_now()})

    env = None
    rows = []
    trace_path = os.path.join(output_dir, "step_traces.csv.gz")
    try:
        env = ABSCoSimEnv(
            speeds=config.get("speeds", [80]),
            vehicle_pc=config.get("vehicle_pc", DEFAULT_PC),
            port=int(config.get("port", sc.port)), game_folder=sc.game_folder,
            user_path=sim_config.resolved_userpath(sc),
            headless=bool(config.get("headless", True)), reward_spec=reward,
            runup_speed_factor=float(config.get("runup_speed_factor", 4.0)),
            artifact_dir=output_dir,
        )
        with gzip.open(trace_path, "wt", encoding="utf-8", newline="",
                       compresslevel=6) as trace_handle:
            trace_writer = csv.DictWriter(trace_handle, fieldnames=TRACE_COLUMNS)
            trace_writer.writeheader()
            schedule = make_schedule(paths, args.episodes, args.seed)
            for index, (path, repetition) in enumerate(schedule, 1):
                row = evaluate_episode(
                    env, pairs[path], repetition,
                    float(eval_config["gamma"]), trace_writer,
                    episode_seed=args.seed + repetition)
                rows.append(row)
                _write_csv(os.path.join(output_dir, "episodes.csv"), rows)
                update_run_state(
                    state_path, status="evaluating", completed_episodes=index,
                    total_planned=len(schedule), checkpoint=pairs[path]["name"],
                    checkpoint_step=pairs[path]["step"],
                    last_outcome=row["outcome"], last_avg_g=row["avg_g"])

            grouped = {path: [row for row in rows
                              if row["checkpoint_path"] == path] for path in paths}
            preliminary = {path: summarize_checkpoint(group)
                           for path, group in grouped.items()}
            strongest = sorted(paths, key=lambda path: (
                preliminary[path]["avg_g_all_ci95_low"],
                preliminary[path]["avg_g_all"]), reverse=True)[:args.reevaluate_top]
            extra = args.top_total_episodes - args.episodes
            extra_schedule = make_schedule(strongest, extra, args.seed + 1)
            total_planned = len(schedule) + len(extra_schedule)
            for offset, (path, repetition) in enumerate(extra_schedule, 1):
                actual_repetition = args.episodes + repetition
                row = evaluate_episode(
                    env, pairs[path], actual_repetition,
                    float(eval_config["gamma"]), trace_writer,
                    episode_seed=args.seed + actual_repetition)
                rows.append(row)
                _write_csv(os.path.join(output_dir, "episodes.csv"), rows)
                update_run_state(
                    state_path, status="reevaluating_top",
                    completed_episodes=len(schedule) + offset,
                    total_planned=total_planned, checkpoint=pairs[path]["name"],
                    checkpoint_step=pairs[path]["step"],
                    last_outcome=row["outcome"], last_avg_g=row["avg_g"])

        summaries = []
        for path in paths:
            summary = summarize_checkpoint(
                row for row in rows if row["checkpoint_path"] == path)
            summary.update({"checkpoint": pairs[path]["name"],
                            "checkpoint_path": path,
                            "checkpoint_step": pairs[path]["step"]})
            summaries.append(summary)
        summaries.sort(key=lambda row: row["checkpoint_step"])
        attach_training_diagnostics(run_dir, summaries)
        analysis = checkpoint_analysis(summaries)
        _write_csv(os.path.join(output_dir, "checkpoint_summary.csv"), summaries)
        atomic_write_json(os.path.join(output_dir, "checkpoint_summary.json"),
                          summaries)
        atomic_write_json(os.path.join(output_dir, "analysis.json"), analysis)
        with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(_report_markdown(config.get("run_name"), summaries, analysis))
        best_summary = max(summaries, key=lambda row: (
            row["avg_g_all_ci95_low"], row["avg_g_all"]))
        best_pair = next(pair for pair in pairs.values()
                         if pair["path"] == best_summary["checkpoint_path"])
        parity = verify_deployment_export(
            best_pair, output_dir, project_root,
            config.get("deployment_interface", "cosim_axle_release_v2"),
            config.get("control_hz", 100))
        atomic_write_json(os.path.join(output_dir, "deployment_parity.json"), parity)
        preserved = preserve_best(best_pair, best_summary, output_dir, project_root)
        update_run_state(
            state_path, status="completed", completed_episodes=len(rows),
            total_planned=len(rows), best_checkpoint=best_pair["path"],
            best_preserved_path=preserved, exit_code=0)
        append_jsonl(event_path, {
            "event": "evaluation_completed", "utc": utc_now(),
            "episodes": len(rows), "best_checkpoint": best_pair["path"],
            "best_avg_g": best_summary["avg_g_all"],
            "best_ci95_low": best_summary["avg_g_all_ci95_low"],
            "preserved": preserved, "deployment_parity": parity,
        })
    except BaseException as exc:
        update_run_state(state_path, status="failed", exit_code=1,
                         exception_type=type(exc).__name__,
                         exception_message=str(exc))
        append_jsonl(event_path, {
            "event": "evaluation_failed", "utc": utc_now(),
            "exception_type": type(exc).__name__, "exception_message": str(exc),
        })
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
