"""PPO policy whose physical actions are always axle releases in ``[0, 1]``.

Stable-Baselines3's standard PPO policy uses an unbounded diagonal Gaussian.
SB3 clips samples only immediately before calling the environment, while PPO
stores and optimizes the original samples.  That is a poor fit for brake
release commands, where the interval endpoints have physical meaning.

This module keeps the public environment action space unchanged and applies an
invertible affine-tanh transform inside the policy:

    latent ~ Normal(mean, std)
    release = (tanh(latent) + 1) / 2

The inverse transform and its Jacobian are included in ``log_prob`` so PPO's
importance ratios are computed for the same bounded action the simulator saw.
"""
import math

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution, TanhBijector,
)
from stable_baselines3.common.policies import ActorCriticPolicy


ACTION_INTERFACE = "ppo_unit_tanh_release_v1"
# NEVER bias these below 0.5 / -0.7 again. The front wheel stays locked below
# ~0.5 release, so a low-biased fresh policy cannot sample an unlock (2026-09-03
# probe: P(15-step streak) ~ 1e-28). Every v5/v6/v7 run before then locked.
DEFAULT_INITIAL_RELEASE = 0.50   # zero latent bias: plain centered PPO start
DEFAULT_LOG_STD_INIT = 0.0       # SB3 default spread; samples cover ~0.12..0.88
# Pre-tanh mean is soft-bounded to +-3 (tanh'(3) ~ 0.01, release 0.0025..0.9975):
# beyond ~8.7 float32 tanh rounds to an exact endpoint, the stored action loses
# the latent, and the -log_prob entropy estimate then grows without bound in the
LATENT_MEAN_BOUND = 3.0


def bound_latent_mean(raw):
    return LATENT_MEAN_BOUND * th.tanh(raw / LATENT_MEAN_BOUND)


class UnitIntervalSquashedDiagGaussianDistribution(
        SquashedDiagGaussianDistribution):
    """Tanh-squashed Gaussian affinely mapped from ``[-1, 1]`` to ``[0, 1]``."""

    def sample(self):
        self.gaussian_actions = self.distribution.rsample()
        return (th.tanh(self.gaussian_actions) + 1.0) * 0.5

    def mode(self):
        self.gaussian_actions = self.distribution.mean
        return (th.tanh(self.gaussian_actions) + 1.0) * 0.5

    def entropy(self):
        # Analytic base-Gaussian entropy (depends on log_std only). SB3's
        # fallback -log_prob.mean() is unbounded in the mean once samples
        # saturate, which is what drove the mean to infinity in PPO-52.
        return th.sum(self.distribution.entropy(), dim=1)

    def log_prob(self, actions, gaussian_actions=None):
        # Map physical [0,1] actions back to tanh-space.  The parent helper's
        # epsilon-stabilized inverse protects exact/near endpoints.
        unit_actions = actions * 2.0 - 1.0
        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(unit_actions)
        log_prob = th.sum(self.distribution.log_prob(gaussian_actions), dim=1)
        log_prob -= th.sum(
            th.log(1.0 - unit_actions ** 2 + self.epsilon), dim=1)
        # release=(unit+1)/2 has derivative 1/2 per action dimension.
        log_prob += self.action_dim * math.log(2.0)
        return log_prob


class UnitIntervalActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy implementing :data:`ACTION_INTERFACE`."""

    action_interface = ACTION_INTERFACE
    latent_bound = LATENT_MEAN_BOUND

    def _get_action_dist_from_latent(self, latent_pi):
        raw = self.action_net(latent_pi)
        self._last_raw_mean = raw.detach()
        return self.action_dist.proba_distribution(bound_latent_mean(raw),
                                                   self.log_std)

    def __init__(self, *args, initial_release=DEFAULT_INITIAL_RELEASE, **kwargs):
        initial_release = float(initial_release)
        if not np.isfinite(initial_release) or not 0.0 < initial_release < 1.0:
            raise ValueError("initial_release must be strictly between 0 and 1")
        self.initial_release = initial_release
        super().__init__(*args, **kwargs)

        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("bounded PPO requires a Box action space")
        if not (np.allclose(self.action_space.low, 0.0)
                and np.allclose(self.action_space.high, 1.0)):
            raise ValueError("bounded PPO requires an action space of [0, 1]")

        self.action_dist = UnitIntervalSquashedDiagGaussianDistribution(
            int(np.prod(self.action_space.shape)))
        # 0.5 is a zero bias; a checkpoint overwrites this via state_dict.
        # Invert the mean bound so the deterministic start is exactly the
        # requested release (clamped just inside the bound).
        latent_center = np.arctanh(2.0 * initial_release - 1.0)
        latent_center = float(np.clip(latent_center, -0.999 * LATENT_MEAN_BOUND,
                                      0.999 * LATENT_MEAN_BOUND))
        latent_center = LATENT_MEAN_BOUND * np.arctanh(
            latent_center / LATENT_MEAN_BOUND)
        if self.action_net.bias is not None:
            with th.no_grad():
                self.action_net.bias.fill_(float(latent_center))


def policy_kwargs(base_kwargs=None, *, initial_release=DEFAULT_INITIAL_RELEASE,
                  log_std_init=DEFAULT_LOG_STD_INIT):
    """Return SB3 policy kwargs for the bounded release policy."""
    out = dict(base_kwargs or {})
    out["initial_release"] = float(initial_release)
    out["log_std_init"] = float(log_std_init)
    return out


def has_bounded_action_interface(model_or_policy):
    """True only for checkpoints using the new physical-action contract."""
    policy = getattr(model_or_policy, "policy", model_or_policy)
    return getattr(policy, "action_interface", None) == ACTION_INTERFACE
