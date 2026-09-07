"""Warm-start a PPO policy on the scripted controller that already brakes at 1.189 g.

docs/CONTROL_CEILING.md measured what this channel can do without any learning: a
per-wheel proportional slip regulator at 400 Hz stops in 54.8 m for 1.189 g, past
the 1.180 g goal and level with stock ABS. PPO's problem is not that the policy
class cannot express that controller -- it is that white per-step exploration
noise, on an optimum only ~0.05 wide in release, rarely lands on it.

So hand PPO the answer as a starting point instead of a target: fit the policy by
regression to the teacher's actions, save it as an immutable checkpoint pair, and
let train_cosim resume from it. This is privileged-teacher distillation, not
cheating: the TEACHER reads ground-truth slip, the STUDENT only ever sees the
recorded observation vector, which contains no ground-truth speed. What the
student inherits is a behaviour, and it has to hold that behaviour up with honest
sensors alone or RL will immediately undo it.

    python slip_ceiling_probe.py --dt 0.0025 --reps 6 --releases "" \
        --targets "" --prop "0.12:6" --record teacher_400hz.npz
    python distill_teacher.py --data teacher_400hz.npz --config .gui-configs/PPO-63.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abs_env_cosim import obs_dim_for, act_dim_for
from bounded_ppo import (UnitIntervalActorCriticPolicy, bound_latent_mean,
                         policy_kwargs as bounded_policy_kwargs)

HERE = os.path.dirname(os.path.abspath(__file__))


class SpecOnlyEnv(gym.Env):
    """Spaces only. SB3 needs an env to build a policy; nothing steps this one."""

    def __init__(self, obs_dim, act_dim):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, (act_dim,), np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        raise RuntimeError("SpecOnlyEnv exists for its spaces, not for stepping")


def deterministic_release(policy, obs):
    """The policy's mode action, through the exact bounded-PPO transform."""
    features = policy.extract_features(obs)
    if policy.share_features_extractor:
        latent_pi, _ = policy.mlp_extractor(features)
    else:
        latent_pi, _ = policy.mlp_extractor(features[0], features[1])
    return (th.tanh(bound_latent_mean(policy.action_net(latent_pi))) + 1.0) * 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz from slip_ceiling_probe --record")
    ap.add_argument("--config", required=True, help="the child run's config json")
    ap.add_argument("--out", default=None, help="run dir to write (default runs/BC-<n>)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=float, default=0.1)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(HERE, args.config)
                         if not os.path.isabs(args.config) else args.config))
    wheel_mode = cfg.get("wheel_mode", "independent")
    obs_dim, act_dim = obs_dim_for(wheel_mode), act_dim_for(wheel_mode)

    data = np.load(os.path.join(HERE, args.data)
                   if not os.path.isabs(args.data) else args.data)
    obs, act = data["obs"].astype(np.float32), data["act"].astype(np.float32)
    if obs.shape[1] != obs_dim or act.shape[1] != act_dim:
        raise SystemExit("recorded shapes %s/%s do not match the configured "
                         "%d/%d -- the recording must come from the same "
                         "wheel_mode and observation contract"
                         % (obs.shape, act.shape, obs_dim, act_dim))
    print("teacher pairs: %d" % len(obs))

    venv = VecNormalize(DummyVecEnv([lambda: SpecOnlyEnv(obs_dim, act_dim)]),
                        norm_obs=True, norm_reward=False)
    venv.obs_rms.mean = obs.mean(axis=0).astype(np.float64)
    venv.obs_rms.var = obs.var(axis=0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))

    ppo_cfg = cfg.get("ppo", {})
    model = PPO(
        UnitIntervalActorCriticPolicy, venv, device=cfg.get("device") or "cpu",
        n_steps=ppo_cfg.get("n_steps", 2048), batch_size=ppo_cfg.get("batch_size", 512),
        policy_kwargs=bounded_policy_kwargs(
            # Must match train_cosim's fresh-model path exactly: ReLU and a
            # pi/vf net_arch dict. SB3 defaults to Tanh, which silently built a
            # different network from the one the trainer builds -- and
            # export_policy_weights only emits ReLU layers, so a Tanh
            # checkpoint cannot be deployed to the car at all.
            dict(activation_fn=th.nn.ReLU,
                 net_arch=dict(pi=ppo_cfg.get("net_arch", [256, 256, 256]),
                               vf=ppo_cfg.get("net_arch", [256, 256, 256]))),
            initial_release=ppo_cfg.get("initial_release", 0.5),
            log_std_init=ppo_cfg.get("log_std_init", 0.0)),
        verbose=0)

    normed = np.clip((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + venv.epsilon),
                     -venv.clip_obs, venv.clip_obs).astype(np.float32)
    cut = int(len(normed) * (1.0 - args.holdout))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(normed))
    train_idx, val_idx = order[:cut], order[cut:]
    x = th.as_tensor(normed, device=model.device)
    y = th.as_tensor(act, device=model.device)

    optimizer = th.optim.Adam(model.policy.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.policy.train()
        perm = train_idx[rng.permutation(len(train_idx))]
        total = 0.0
        for start in range(0, len(perm), args.batch):
            idx = perm[start:start + args.batch]
            loss = th.nn.functional.mse_loss(
                deterministic_release(model.policy, x[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.policy.eval()
        with th.no_grad():
            val = float(th.nn.functional.mse_loss(
                deterministic_release(model.policy, x[val_idx]), y[val_idx]))
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print("epoch %3d  train_mse=%.5f  val_mse=%.5f  (val rmse %.3f release)"
                  % (epoch, total / len(perm), val, val ** 0.5), flush=True)

    out_dir = args.out or os.path.join(HERE, "runs", "BC-" + cfg["run_name"])
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "final.zip"))
    venv.save(os.path.join(out_dir, "vecnormalize.pkl"))
    parent_cfg = dict(cfg)
    parent_cfg["run_name"] = os.path.basename(out_dir)
    parent_cfg["provenance"] = {"distilled_from": os.path.basename(args.data),
                                "pairs": int(len(obs))}
    with open(os.path.join(out_dir, "config.json"), "w") as handle:
        json.dump(parent_cfg, handle, indent=2)
    print("wrote %s (resume train_cosim from final.zip)" % out_dir)


if __name__ == "__main__":
    main()
