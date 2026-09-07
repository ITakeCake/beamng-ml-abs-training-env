import json
import os

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bounded_ppo import UnitIntervalActorCriticPolicy, policy_kwargs
from experiment_io import (
    pair_manifest_path,
    paired_vecnormalize_path,
    save_checkpoint_pair,
    update_run_state,
    validate_checkpoint_pair,
)


class _ToyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10, 10, (3,), np.float32)
        self.action_space = gym.spaces.Box(0, 1, (2,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, np.float32), {}

    def step(self, action):
        return np.zeros(3, np.float32), 0.0, False, False, {}


def _model_and_vec():
    vec = VecNormalize(DummyVecEnv([_ToyEnv]), norm_reward=False)
    model = PPO(
        UnitIntervalActorCriticPolicy, vec, n_steps=4, batch_size=4,
        n_epochs=1, policy_kwargs=policy_kwargs({"net_arch": [8]}),
        device="cpu", verbose=0,
    )
    return model, vec


def test_atomic_pair_is_deeply_validated_and_committed_last(tmp_path):
    model, vec = _model_and_vec()
    model_path = tmp_path / "TEST_0_steps.zip"
    try:
        manifest = save_checkpoint_pair(
            model, vec, model_path, metadata={"kind": "test"})
        vec_path = tmp_path / "TEST_0_steps_vecnormalize.pkl"
        commit_path = tmp_path / "TEST_0_steps.pair.json"
        assert model_path.is_file()
        assert vec_path.is_file()
        assert commit_path.is_file()
        assert paired_vecnormalize_path(model_path) == str(vec_path)
        assert pair_manifest_path(model_path) == str(commit_path)
        loaded = json.loads(commit_path.read_text(encoding="utf-8"))
        assert loaded["model_sha256"] == manifest["model_sha256"]
        assert loaded["metadata"] == {"kind": "test"}
        validation = validate_checkpoint_pair(
            model_path, expected_obs_dim=3, expected_action_dim=2)
        assert validation["timesteps"] == 0
        assert validation["model_sha256"] == loaded["model_sha256"]
    finally:
        vec.close()


def test_pair_save_never_overwrites_committed_artifacts(tmp_path):
    model, vec = _model_and_vec()
    model_path = tmp_path / "checkpoint.zip"
    try:
        save_checkpoint_pair(model, vec, model_path)
        original = model_path.read_bytes()
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            save_checkpoint_pair(model, vec, model_path)
        assert model_path.read_bytes() == original
    finally:
        vec.close()


def test_corrupt_model_or_vecnormalize_is_rejected(tmp_path):
    model, vec = _model_and_vec()
    model_path = tmp_path / "checkpoint.zip"
    try:
        save_checkpoint_pair(model, vec, model_path)
        model_path.write_bytes(b"not a zip")
        with pytest.raises(ValueError, match="valid zip"):
            validate_checkpoint_pair(model_path)
    finally:
        vec.close()


def test_run_state_updates_merge_into_atomic_snapshot(tmp_path):
    state_path = tmp_path / "run_state.json"
    update_run_state(state_path, status="starting", pid=123)
    state = update_run_state(state_path, status="training", step=50)
    assert state["pid"] == 123
    assert state["status"] == "training"
    assert state["step"] == 50
    assert state["schema"] == 1
    assert state["updated_utc"].endswith("+00:00")


def test_atomic_json_is_strict_and_represents_missing_nonfinite_metrics_as_null(tmp_path):
    from experiment_io import atomic_write_json
    path = tmp_path / "metrics.json"
    atomic_write_json(path, {"finite": np.float32(1.25),
                             "missing": float("nan")})
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text) == {"finite": 1.25, "missing": None}


def test_atomic_json_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    import experiment_io

    path = tmp_path / "run_state.json"
    path.write_text('{"status": "old"}\n', encoding="utf-8")
    real_replace = os.replace
    calls = []
    sleeps = []

    def flaky_replace(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            raise PermissionError(5, "destination is temporarily in use")
        return real_replace(source, destination)

    monkeypatch.setattr(experiment_io.os, "replace", flaky_replace)
    monkeypatch.setattr(experiment_io.time, "sleep", sleeps.append)

    experiment_io.atomic_write_json(path, {"status": "training", "step": 10})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "training", "step": 10}
    assert len(calls) == 3
    assert sleeps == [0.01, 0.02]
    assert not list(tmp_path.glob(".run_state.json.*.tmp"))


def test_atomic_json_exhausted_retry_preserves_previous_file(tmp_path, monkeypatch):
    import experiment_io

    path = tmp_path / "run_state.json"
    original = '{"status": "old"}\n'
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(experiment_io, "ATOMIC_REPLACE_ATTEMPTS", 3)
    monkeypatch.setattr(experiment_io.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(
        experiment_io.os, "replace",
        lambda _source, _destination: (_ for _ in ()).throw(
            PermissionError(5, "destination remains in use")))

    with pytest.raises(PermissionError):
        experiment_io.atomic_write_json(path, {"status": "new"})

    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".run_state.json.*.tmp"))
