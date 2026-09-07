import csv
import io

import numpy as np
import pytest

import evaluate_cosim
from evaluate_cosim import (
    FrozenObservationNormalizer,
    _ci95,
    _t95_critical,
    checkpoint_step,
    checkpoint_analysis,
    discover_checkpoints,
    evaluate_episode,
    make_schedule,
    preserve_best,
    summarize_checkpoint,
)


def test_checkpoint_discovery_orders_steps_and_keeps_final_last(tmp_path):
    run = tmp_path / "PPO-1"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    for name in ("PPO-1_100000_steps.zip", "PPO-1_50000_steps.zip", "noise.zip"):
        (checkpoints / name).write_bytes(b"x")
    (run / "final.zip").write_bytes(b"x")
    paths = discover_checkpoints(run)
    assert [checkpoint_step(path) for path in paths] == [50000, 100000, None]


def test_interleaved_schedule_balances_every_checkpoint_and_is_repeatable():
    checkpoints = ["a", "b", "c"]
    first = make_schedule(checkpoints, 5, 123)
    second = make_schedule(checkpoints, 5, 123)
    assert first == second
    assert {path: sum(item[0] == path for item in first)
            for path in checkpoints} == {"a": 5, "b": 5, "c": 5}
    assert all({path for path, _ in first[index:index + 3]} == set(checkpoints)
               for index in range(0, len(first), 3))


def test_frozen_normalizer_uses_saved_stats_without_mutating_them():
    class _Rms:
        mean = np.array([1.0, 2.0])
        var = np.array([4.0, 9.0])

    class _Vec:
        norm_obs = True
        obs_rms = _Rms()
        epsilon = 0.0
        clip_obs = 10.0

    normalizer = FrozenObservationNormalizer(_Vec())
    before = normalizer.mean.copy()
    assert normalizer(np.array([3.0, 5.0])).tolist() == pytest.approx([1.0, 1.0])
    assert normalizer.mean.tolist() == before.tolist()


def test_summary_counts_failures_as_zero_g_and_reports_blocks():
    rows = [
        {"outcome": "STOP", "avg_g": 1.1, "bottleneck_axle": "front"},
        {"outcome": "STOP", "avg_g": 0.9, "bottleneck_axle": "balanced"},
        {"outcome": "TIMEOUT", "avg_g": 0.0, "bottleneck_axle": "rear"},
        {"outcome": "STOP", "avg_g": 1.0, "bottleneck_axle": "front"},
    ]
    summary = summarize_checkpoint(rows, block_size=2)
    assert summary["avg_g_all"] == pytest.approx(0.75)
    assert summary["avg_g_success"] == pytest.approx(1.0)
    assert summary["failure_fraction"] == pytest.approx(0.25)
    assert summary["block_count"] == 2
    assert summary["front_bottleneck_fraction"] == 0.5


def test_fifty_stop_ci_uses_student_t_not_normal_critical_value():
    values = list(range(50))
    low, high = _ci95(values)
    expected_half = 2.009575 * np.std(values, ddof=1) / np.sqrt(50)
    assert _t95_critical(49) == pytest.approx(2.009575, rel=3e-5)
    assert high - np.mean(values) == pytest.approx(expected_half, rel=3e-5)
    assert np.mean(values) - low == pytest.approx(expected_half, rel=3e-5)


def test_episode_is_deterministic_frozen_and_reward_components_reproduce_return():
    class _Normalizer:
        def __init__(self):
            self.calls = 0

        def __call__(self, obs):
            self.calls += 1
            return np.asarray(obs) + 10

    class _Model:
        def __init__(self):
            self.flags = []

        def predict(self, obs, deterministic):
            self.flags.append((np.asarray(obs).copy(), deterministic))
            return np.array([0.2, 0.3], np.float32), None

    class _Env:
        def __init__(self):
            self.step_index = 0

        def reset(self, seed=None):
            self.step_index = 0
            return np.zeros(13, np.float32), {}

        def step(self, action):
            self.step_index += 1
            terminal = self.step_index == 3
            components = {
                "dense_g": 1.0, "step_yaw": 0.0,
                "terminal_g": 2.0 if terminal else 0.0,
                "terminal_clean_yaw_bonus": 0.0,
                "terminal_accumulated_yaw": 0.0, "terminal_stability": 0.0,
                "failure_base": 0.0, "failure_accumulated_yaw": 0.0,
                "total": 3.0 if terminal else 1.0,
            }
            diagnostic = {
                "episode": 1, "episode_step": self.step_index,
                "fused_speed_ms": 20.0, "slip_fr": 0.1, "slip_fl": 0.1,
                "slip_rr": 0.1, "slip_rl": 0.1,
            }
            diagnostic.update({"reward_" + key: value
                               for key, value in components.items()})
            info = {"diagnostics": diagnostic,
                    "reward_components": components}
            if terminal:
                info.update(outcome="STOP", avg_g=1.0, stopping_dist_m=30.0,
                            avg_g_arc=.99, stopping_dist_arc_m=30.3,
                            brake_metric_duration_s=2.5,
                            peak_g=1.2, yaw_abs_sum=0.01, yaw_sq_sum=0.001,
                            steps=3, stop_time_s=.03)
            return np.zeros(13, np.float32), components["total"], terminal, False, info

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=evaluate_cosim.TRACE_COLUMNS)
    writer.writeheader()
    pair = {"name": "p.zip", "path": "p.zip", "step": 50,
            "model": _Model(), "normalizer": _Normalizer()}
    row = evaluate_episode(_Env(), pair, 1, 0.5, writer)
    assert row["reward_undiscounted"] == 5.0
    assert row["reward_discounted"] == pytest.approx(2.25)
    assert row["reward_dense_g"] == 3.0
    assert row["reward_terminal_g"] == 2.0
    assert row["avg_g_arc"] == .99
    assert row["stopping_dist_arc_m"] == 30.3
    assert row["brake_metric_duration_s"] == 2.5
    assert pair["normalizer"].calls == 3
    assert all(flag is True for _, flag in pair["model"].flags)
    assert pair["model"].flags[0][0][0] == 10.0


def test_checkpoint_analysis_distinguishes_best_physics_and_best_return():
    rows = [
        {"checkpoint_step": 50, "avg_g_all": 1.0,
         "avg_g_all_ci95_low": .99, "reward_discounted_mean": 2.0,
         "reward_dense_g_mean": 1.0, "reward_terminal_g_mean": 25.0,
         "reward_terminal_stability_mean": -.1,
         "front_near_lock_fraction_mean": .1,
         "rear_near_lock_fraction_mean": .1,
         "action_front_dominant_hz_mean": 5.0,
         "yaw_abs_sum_mean": .01},
        {"checkpoint_step": 100, "avg_g_all": .9,
         "avg_g_all_ci95_low": .89, "reward_discounted_mean": 3.0,
         "reward_dense_g_mean": 1.1, "reward_terminal_g_mean": 22.5,
         "reward_terminal_stability_mean": -.2,
         "front_near_lock_fraction_mean": .3,
         "rear_near_lock_fraction_mean": .2,
         "action_front_dominant_hz_mean": 20.0,
         "yaw_abs_sum_mean": .02},
    ]
    analysis = checkpoint_analysis(rows)
    assert analysis["best_g_step"] == 50
    assert analysis["best_discounted_return_step"] == 100
    assert analysis["g_drop_from_best"] == pytest.approx(.1)
    assert analysis["checkpoint_g_vs_discounted_return_correlation"] == pytest.approx(-1.0)
    assert analysis["decision_classification"] == "reward_misalignment"
    assert analysis["dense_reward_prefers_latest_lower_g"] is True


def test_preserve_best_commits_copied_pair_before_publish(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "PPO-X" / "checkpoints"
    run.mkdir(parents=True)
    model = run / "PPO-X_50_steps.zip"
    vec = run / "PPO-X_50_steps_vecnormalize.pkl"
    model.write_bytes(b"model")
    vec.write_bytes(b"vec")
    pair = {
        "name": model.name, "path": str(model),
        "validation": {"vecnormalize_path": str(vec)},
    }

    def fake_validate(model_path, vec_path, **_kwargs):
        return {
            "model_path": str(model_path), "vecnormalize_path": str(vec_path),
            "timesteps": 50, "obs_dim": 13, "action_dim": 2,
            "model_sha256": evaluate_cosim.sha256_file(model_path),
            "vecnormalize_sha256": evaluate_cosim.sha256_file(vec_path),
            "model_bytes": 5, "vecnormalize_bytes": 3,
        }

    monkeypatch.setattr(evaluate_cosim, "validate_checkpoint_pair", fake_validate)
    evaluation = tmp_path / "evaluations" / "eval-1"
    evaluation.mkdir(parents=True)
    destination = preserve_best(pair, {"avg_g_all": 1.0}, str(evaluation),
                                str(tmp_path))
    commit = tmp_path / "best_models" / "PPO-X" / "eval-1" / \
        "PPO-X_50_steps.pair.json"
    assert destination == str(commit.parent)
    assert commit.is_file()
    assert (commit.parent / model.name).read_bytes() == b"model"
    assert (commit.parent / vec.name).read_bytes() == b"vec"
