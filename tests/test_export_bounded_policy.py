import os
import subprocess
import sys

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bounded_ppo import UnitIntervalActorCriticPolicy, policy_kwargs


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _StackedObsEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2240,), np.float32)
    action_space = gym.spaces.Box(0, 1, (2,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2240, np.float32), {}

    def step(self, action):
        return np.zeros(2240, np.float32), 1.0, False, False, {}


def test_bounded_checkpoint_exports_with_exact_cosim_parity_and_metadata(tmp_path):
    vec = VecNormalize(DummyVecEnv([_StackedObsEnv]),
                       norm_obs=True, norm_reward=False)
    model = PPO(
        UnitIntervalActorCriticPolicy, vec, n_steps=16, batch_size=8,
        n_epochs=1, device="cpu", verbose=0,
        policy_kwargs=policy_kwargs({
            "activation_fn": th.nn.ReLU,
            "net_arch": {"pi": [16], "vf": [16]},
        }, initial_release=0.1, log_std_init=-0.7))
    model.learn(32)
    checkpoint = tmp_path / "model.zip"
    vecnorm = tmp_path / "vecnormalize.pkl"
    output = tmp_path / "weights.lua"
    model.save(checkpoint)
    vec.save(vecnorm)
    vec.close()

    result = subprocess.run([
        sys.executable, os.path.join(REPO, "export_policy_weights.py"),
        "--ckpt", str(checkpoint), "--vecnorm", str(vecnorm),
        "--algo", "ppo", "--head", "ppo_tanh_release01",
        "--interface", "cosim_axle_release_v2", "--control-hz", "100",
        "--out", str(output), "--n-parity", "64",
    ], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
    text = output.read_text(encoding="utf-8")
    assert 'M.interface = "cosim_axle_release_v2"' in text
    assert "M.obs_dim = 2240" in text
    assert "M.control_hz = 100" in text
    assert 'M.head = "ppo_tanh_release01"' in text
