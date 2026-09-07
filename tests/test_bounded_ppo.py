import numpy as np
import gymnasium as gym
import torch as th
from stable_baselines3 import PPO

from bounded_ppo import (
    UnitIntervalActorCriticPolicy, has_bounded_action_interface, policy_kwargs,
)


class _UnitActionEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
    action_space = gym.spaces.Box(0, 1, (2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, np.float32), {}

    def step(self, action):
        assert np.all(action >= 0.0) and np.all(action <= 1.0)
        reward = 1.0 - float(np.square(action - 0.2).mean())
        return np.zeros(3, np.float32), reward, False, False, {}


def _model():
    return PPO(
        UnitIntervalActorCriticPolicy, _UnitActionEnv(), n_steps=16,
        batch_size=8, n_epochs=1, device="cpu", verbose=0,
        policy_kwargs=policy_kwargs({"net_arch": [16]},
                                    initial_release=0.1,
                                    log_std_init=-0.7))


def test_samples_and_mode_are_physical_unit_interval_releases():
    model = _model()
    obs = th.zeros((4096, 3))
    distribution = model.policy.get_distribution(obs)
    actions = distribution.sample()
    assert bool(th.all(actions >= 0.0))
    assert bool(th.all(actions <= 1.0))
    assert bool(th.all(th.isfinite(distribution.log_prob(actions))))
    mode, _ = model.predict(np.zeros(3, np.float32), deterministic=True)
    assert np.allclose(mode, [0.1, 0.1], atol=1e-6)


def test_log_probability_includes_tanh_and_half_interval_jacobians():
    model = _model()
    distribution = model.policy.get_distribution(th.zeros((4, 3)))
    latent = th.tensor([[-2.0, -0.5], [0.0, 0.5], [1.0, 2.0], [-1.0, 1.0]])
    physical = (th.tanh(latent) + 1.0) * 0.5
    actual = distribution.log_prob(physical, gaussian_actions=latent)
    expected = th.sum(distribution.distribution.log_prob(latent), dim=1)
    expected -= th.sum(th.log(1.0 - th.tanh(latent) ** 2
                              + distribution.epsilon), dim=1)
    expected += 2 * np.log(2.0)
    assert th.allclose(actual, expected, atol=1e-6)


def test_short_training_and_save_load_preserve_bounded_interface(tmp_path):
    model = _model()
    model.learn(32)
    path = tmp_path / "bounded.zip"
    model.save(path)
    loaded = PPO.load(path, device="cpu")
    assert has_bounded_action_interface(loaded)
    for deterministic in (False, True):
        action, _ = loaded.predict(np.zeros(3, np.float32),
                                   deterministic=deterministic)
        assert np.all(action >= 0.0) and np.all(action <= 1.0)


def test_fresh_default_policy_is_centered_and_samples_both_sides_of_lock():
    """Default init must not sit inside the <0.5 release lock dead zone."""
    from bounded_ppo import DEFAULT_INITIAL_RELEASE, DEFAULT_LOG_STD_INIT
    assert DEFAULT_INITIAL_RELEASE == 0.5 and DEFAULT_LOG_STD_INIT == 0.0
    model = PPO(UnitIntervalActorCriticPolicy, _UnitActionEnv(), n_steps=16,
                batch_size=8, n_epochs=1, device="cpu", verbose=0, seed=0,
                policy_kwargs=policy_kwargs({"net_arch": [16]}))
    mode, _ = model.predict(np.zeros(3, np.float32), deterministic=True)
    assert np.allclose(mode, [0.5, 0.5], atol=0.05)
    samples = model.policy.get_distribution(th.zeros((4096, 3))).sample()
    unlocked = float((samples > 0.5).float().mean())
    assert 0.4 < unlocked < 0.6
    assert float(samples.max()) > 0.9 and float(samples.min()) < 0.1


def test_entropy_is_analytic_and_independent_of_the_mean():
    """PPO-52 root cause: -log_prob entropy grew without bound in the mean."""
    from bounded_ppo import UnitIntervalSquashedDiagGaussianDistribution as D
    d = D(4)
    ent = []
    for mu0 in (0.0, 3.0, 12.0):
        log_std = th.full((4,), float(np.log(0.75)), requires_grad=True)
        mu = th.full((64, 4), mu0, requires_grad=True)
        d.proba_distribution(mu, log_std)
        e = d.entropy()
        assert e is not None and e.shape == (64,)
        grads = th.autograd.grad(e.mean(), [mu, log_std], allow_unused=True)
        assert grads[0] is None or th.all(grads[0] == 0.0)
        assert grads[1] is not None and th.all(grads[1] > 0.0)
        ent.append(float(e.mean()))
    assert max(ent) - min(ent) < 1e-6
    expected = 4 * (0.5 * np.log(2 * np.pi * np.e * 0.75 ** 2))
    assert abs(ent[0] - expected) < 1e-5


def test_latent_mean_is_soft_bounded_so_tanh_never_saturates():
    from bounded_ppo import LATENT_MEAN_BOUND
    model = _model()
    policy = model.policy
    obs = th.full((8, 3), 1e4)                     # absurd inputs, huge raw head output
    with th.no_grad():
        distribution = policy.get_distribution(obs)
    mean = distribution.distribution.mean
    assert th.all(mean.abs() <= LATENT_MEAN_BOUND)
    assert LATENT_MEAN_BOUND == 3.0
    release = (th.tanh(mean) + 1.0) * 0.5
    assert th.all(release < 0.9976) and th.all(release > 0.0024)
    raw = policy._last_raw_mean
    assert raw.shape == mean.shape
