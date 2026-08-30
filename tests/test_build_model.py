"""build_model must actually apply the kwargs the review found silently
missing (net_arch/activation, SAC train_freq, PPO ent_coef, device). Uses a
real tiny gym env (Pendulum-v1) so this exercises real SB3 construction, not
a mock -- these are exactly the fields that silently fall back to SB3
defaults if a kwarg is merely forgotten."""
import argparse
import os
import sys

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_residual import build_model, resolve_device


def _sac_ns(**over):
    base = dict(algo="sac", lr=1e-4, buffer_size=1000, tau=0.005,
                learning_starts=100, target_entropy=-2.0, train_freq=2,
                device=None,
                net_arch="3x256")
    base.update(over)
    return argparse.Namespace(**base)


def _ppo_ns(**over):
    base = dict(algo="ppo", lr=1e-4, n_steps=64, batch_size=32, n_epochs=2,
                clip_range=0.2, gae_lambda=0.95, ent_coef=0.005, device=None,
                net_arch="3x256")
    base.update(over)
    return argparse.Namespace(**base)


def test_resolve_device_explicit_wins():
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_is_cpu_or_cuda():
    assert resolve_device(None) in ("cpu", "cuda")


def test_sac_gets_3x256_relu_and_train_freq_2():
    env = gym.make("Pendulum-v1")
    model = build_model(_sac_ns(device="cpu"), env)
    assert model.policy.net_arch == {"pi": [256, 256, 256], "qf": [256, 256, 256]}
    assert model.policy.activation_fn.__name__ == "ReLU"
    assert model.train_freq.frequency == 2
    assert model.train_freq.unit.value == "step"


def test_ppo_gets_3x256_relu_and_ent_coef():
    env = gym.make("Pendulum-v1")
    model = build_model(_ppo_ns(device="cpu"), env)
    assert model.policy.net_arch == {"pi": [256, 256, 256], "vf": [256, 256, 256]}
    assert model.policy.activation_fn.__name__ == "ReLU"
    assert model.ent_coef == 0.005


def test_a_custom_network_shape_reaches_the_model():
    """The setting is new; without this the box could be ignored and nobody
    would notice, since a wrong-but-valid network still trains."""
    env = gym.make("Pendulum-v1")
    model = build_model(_ppo_ns(device="cpu", net_arch="4x64"), env)
    assert model.policy.net_arch == {"pi": [64] * 4, "vf": [64] * 4}


def test_sac_uses_qf_and_ppo_uses_vf_for_the_value_network():
    """SB3 names the value half differently per algorithm; getting it wrong
    silently falls back to SB3's own default instead of raising."""
    env = gym.make("Pendulum-v1")
    assert "qf" in build_model(_sac_ns(device="cpu", net_arch="2x32"), env).policy.net_arch
    assert "vf" in build_model(_ppo_ns(device="cpu", net_arch="2x32"), env).policy.net_arch
