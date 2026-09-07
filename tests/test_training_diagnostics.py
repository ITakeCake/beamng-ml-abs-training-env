import csv
import gzip
import math

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_cosim import FiniteGuard, Heartbeat
from bounded_ppo import UnitIntervalActorCriticPolicy, policy_kwargs
from training_diagnostics import (
    DEFAULT_DIAGNOSTICS,
    LoggedCheckpointCallback,
    PhysicalAccumulator,
    TrainingDiagnosticsCallback,
    dominant_oscillation,
    parse_diagnostics_config,
)


def test_diagnostics_config_defaults_and_threshold_validation():
    cfg = parse_diagnostics_config(None)
    assert cfg == DEFAULT_DIAGNOSTICS
    assert cfg is not DEFAULT_DIAGNOSTICS
    with pytest.raises(SystemExit, match="near_lock_slip"):
        parse_diagnostics_config({"near_lock_slip": 0.98, "lock_slip": 0.95})
    with pytest.raises(SystemExit, match="positive integer"):
        parse_diagnostics_config({"trace_every_steps": 0})


def test_dominant_oscillation_finds_a_known_pulse_rate():
    t = np.arange(500) * 0.01
    signal = 0.5 + 0.4 * np.sin(2 * np.pi * 5.0 * t)
    hz, strength = dominant_oscillation(signal)
    assert hz == pytest.approx(5.0, abs=0.21)
    assert strength > 0.5
    flat_hz, flat_strength = dominant_oscillation(np.full(500, 0.5))
    assert math.isnan(flat_hz) and math.isnan(flat_strength)


def test_accumulator_reports_saturation_lock_time_and_axle_bottleneck():
    cfg = parse_diagnostics_config(None)
    accumulator = PhysicalAccumulator(cfg)
    for step in range(100):
        diagnostic = {
            "fused_speed_ms": 20.0,
            "slip_fr": 0.96, "slip_fl": 0.96,
            "slip_rr": 0.20, "slip_rl": 0.20,
            "brake_command_fr_nm": 2000, "brake_command_fl_nm": 2000,
            "brake_command_rr_nm": 1500, "brake_command_rl_nm": 1500,
        }
        accumulator.push((-.2, 1.2), (0.0, 1.0), diagnostic, reward=1.0)
    summary = accumulator.summary()
    assert summary["action_front_saturated_zero_fraction"] == 1.0
    assert summary["action_rear_saturated_one_fraction"] == 1.0
    assert summary["raw_action_front_clipped_low_fraction"] == 1.0
    assert summary["raw_action_rear_clipped_high_fraction"] == 1.0
    assert summary["sb3_action_clip_delta_front_mean"] == pytest.approx(0.2)
    assert summary["front_lock_fraction"] == 1.0
    assert summary["front_lock_time_s"] == pytest.approx(1.0)
    assert summary["rear_near_lock_fraction"] == 0.0
    assert summary["bottleneck_axle"] == "front"


def test_heartbeat_uses_raw_physics_diagnostics_not_normalized_observation(tmp_path):
    path = tmp_path / "train.log"
    heartbeat = Heartbeat(path, every=1)
    heartbeat.num_timesteps = 1
    heartbeat.locals = {
        "new_obs": np.full((1, 13), -99.0, dtype=np.float32),
        "infos": [{"diagnostics": {
            "braking_g": 1.125,
            "wheel_speed_fr_ms": 23.5,
            "yaw_abs_sum": 0.0625,
        }}],
    }
    assert heartbeat._on_step()
    line = path.read_text(encoding="utf-8")
    assert "cur_g=1.125" in line
    assert "cur_ws=23.50" in line
    assert "cur_yaw=0.062" in line
    assert "-99" not in line


@pytest.mark.parametrize("field", ["actions", "clipped_actions", "rewards", "new_obs"])
def test_finite_guard_fails_on_nonfinite_learning_interface(field):
    guard = FiniteGuard()
    guard.locals = {field: np.array([float("nan")]), "infos": []}
    with pytest.raises(FloatingPointError, match=field):
        guard._on_step()


def test_finite_guard_checks_physics_diagnostics():
    guard = FiniteGuard()
    guard.locals = {"infos": [{"diagnostics": {"slip_fr": float("inf")}}]}
    with pytest.raises(FloatingPointError, match="diagnostics.slip_fr"):
        guard._on_step()


class _ToyDiagnosticEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10, 10, (3,), np.float32)
        self.action_space = gym.spaces.Box(0, 1, (2,), np.float32)
        self.episode = 0
        self.episode_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.episode_step = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.episode_step += 1
        slip_front = 0.9 if self.episode_step % 2 else 0.2
        diagnostic = {
            "episode": self.episode, "episode_step": self.episode_step,
            "previous_action_front": 0.2, "previous_action_rear": 0.3,
            "ground_speed_ms": 20.0, "fused_speed_ms": 20.0,
            "braking_g": 1.0, "yaw_rate_rad_s": 0.0,
            "wheel_speed_fr_ms": 10.0, "wheel_speed_fl_ms": 10.0,
            "wheel_speed_rr_ms": 15.0, "wheel_speed_rl_ms": 15.0,
            "slip_fr": slip_front, "slip_fl": slip_front,
            "slip_rr": 0.25, "slip_rl": 0.25,
            "brake_command_fr_nm": 2000, "brake_command_fl_nm": 2000,
            "brake_command_rr_nm": 1500, "brake_command_rl_nm": 1500,
            "brake_active": 1.0,
        }
        terminated = self.episode_step >= 20
        info = {"diagnostics": diagnostic}
        if terminated:
            info.update({
                "ep_index": self.episode, "outcome": "STOP",
                "target_mph": 80, "avg_g": 1.0,
                "stopping_dist_m": 30.0,
            })
        return np.zeros(3, np.float32), 1.0, terminated, False, info


class _StopAtStep(BaseCallback):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def _on_step(self):
        return self.num_timesteps < self.step


def test_callback_writes_aligned_metrics_traces_and_paired_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    log_path = run_dir / "train.log"
    log_path.write_text("", encoding="utf-8")
    cfg = parse_diagnostics_config({
        "trace_every_steps": 30,
        "checkpoint_every_steps": 50,
    })

    venv = VecNormalize(DummyVecEnv([_ToyDiagnosticEnv]),
                        norm_obs=True, norm_reward=False)
    model = PPO(UnitIntervalActorCriticPolicy, venv, n_steps=32,
                batch_size=16, n_epochs=1,
                policy_kwargs=policy_kwargs({"net_arch": [16]}),
                device="cpu", verbose=0)
    model.set_logger(configure(str(run_dir / "diagnostics" / "sb3"), ["csv"]))
    callbacks = [
        TrainingDiagnosticsCallback(str(run_dir), str(log_path), cfg),
        LoggedCheckpointCallback("TEST", str(checkpoint_dir), str(log_path), 50),
    ]
    model.learn(total_timesteps=64, callback=callbacks)
    venv.close()

    with open(run_dir / "diagnostics" / "rollout_metrics.csv", newline="") as f:
        rollouts = list(csv.DictReader(f))
    assert len(rollouts) == 2
    assert [row["end_step"] for row in rollouts] == ["32", "64"]
    assert all(row["ppo_approx_kl"] != "" for row in rollouts)
    assert all(row["policy_std_front"] != "" for row in rollouts)
    assert all(row["transformed_policy_entropy"] != "" for row in rollouts)
    assert all(float(row["sb3_action_clip_delta_front_max"]) == 0.0
               for row in rollouts)
    assert all(row["deterministic_action_front_mean"] != "" for row in rollouts)

    with open(run_dir / "diagnostics" / "episode_metrics.csv", newline="") as f:
        episodes = list(csv.DictReader(f))
    assert sum(row["outcome"] == "STOP" for row in episodes) == 3
    assert episodes[0]["front_near_lock_fraction"] != ""

    with gzip.open(run_dir / "diagnostics" / "sample_traces.csv.gz", "rt",
                   newline="") as f:
        traces = list(csv.DictReader(f))
    assert traces
    assert {row["milestone_step"] for row in traces} >= {"1", "31"}
    assert traces[0]["slip_fr"] != ""

    # The requested 50-step maximum is rounded down to one 32-step rollout,
    # keeping every saved policy honest while guaranteeing <=50-step gaps.
    assert callbacks[1].effective_every_steps == 32
    assert (checkpoint_dir / "TEST_32_steps.zip").is_file()
    assert (checkpoint_dir / "TEST_32_steps_vecnormalize.pkl").is_file()
    assert (checkpoint_dir / "TEST_32_steps.pair.json").is_file()
    assert (checkpoint_dir / "TEST_64_steps.zip").is_file()
    assert (checkpoint_dir / "TEST_64_steps_vecnormalize.pkl").is_file()
    assert (checkpoint_dir / "TEST_64_steps.pair.json").is_file()
    native_progress = run_dir / "diagnostics" / "sb3" / "progress.csv"
    assert native_progress.is_file()
    assert "DIAGNOSTICS step=64" in log_path.read_text(encoding="utf-8")
    checkpoint_log = log_path.read_text(encoding="utf-8")
    assert "CHECKPOINT step=32" in checkpoint_log
    assert "CHECKPOINT step=64" in checkpoint_log


def test_partial_rollout_never_gets_a_misleading_trained_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    log_path = tmp_path / "train.log"
    log_path.write_text("", encoding="utf-8")
    venv = VecNormalize(DummyVecEnv([_ToyDiagnosticEnv]), norm_reward=False)
    model = PPO(UnitIntervalActorCriticPolicy, venv, n_steps=32,
                batch_size=16, n_epochs=1,
                policy_kwargs=policy_kwargs({"net_arch": [16]}),
                device="cpu", verbose=0)
    checkpoint = LoggedCheckpointCallback(
        "TEST", str(checkpoint_dir), str(log_path), 50)
    try:
        model.learn(total_timesteps=100, callback=[checkpoint, _StopAtStep(55)])
        assert model.num_timesteps == 55
        assert checkpoint.last_trained_step == 32
        assert checkpoint.effective_every_steps == 32
        assert [path.name for path in checkpoint_dir.glob("*.zip")] == [
            "TEST_32_steps.zip"]
    finally:
        venv.close()


def test_checkpoint_interval_rejects_spacing_smaller_than_one_rollout(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    log_path = tmp_path / "train.log"
    log_path.write_text("", encoding="utf-8")
    venv = VecNormalize(DummyVecEnv([_ToyDiagnosticEnv]), norm_reward=False)
    model = PPO(UnitIntervalActorCriticPolicy, venv, n_steps=32,
                batch_size=16, n_epochs=1,
                policy_kwargs=policy_kwargs({"net_arch": [16]}),
                device="cpu", verbose=0)
    checkpoint = LoggedCheckpointCallback(
        "TEST", str(checkpoint_dir), str(log_path), 31)
    try:
        with pytest.raises(ValueError, match="at least PPO n_steps"):
            model.learn(total_timesteps=32, callback=[checkpoint])
        assert list(checkpoint_dir.glob("*")) == []
    finally:
        venv.close()
