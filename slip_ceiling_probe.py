"""What can the co-sim action channel actually achieve, with no learning involved?

Every PPO run so far ceilings at ~1.05 g while stock ABS does 1.199 g on the same
car at 80 mph (calibration/etk800.json) and a fully locked wheel does 0.979 g.
That gap has two possible causes, and they need completely different fixes:

  learning   -- the channel is fine, the reward/exploration never finds the policy
  channel    -- 10 ms round-trip latency (or the release mapping) makes 1.05 g the
                physical best anything can do through this interface

This probe answers it without a policy. It drives the SAME env.step() path with
two scripted controllers and reports the game's own 2 kHz metric:

  const  -- one fixed release for the whole stop (the "best constant torque"
            solution, which is what a policy that ignores its observations gets)
  slip   -- per-wheel bang-bang on true slip with hysteresis, i.e. what stock ABS
            does, subject to the same one-step actuation delay the policy has

If the slip controller reaches ~1.15 g or better, the channel is not the ceiling
and the remaining work is reward and exploration. If it also sits at ~1.05 g, no
reward will ever fix it and the interface has to change.

    python slip_ceiling_probe.py --reps 2 --port 64280
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import abs_env_cosim as E
import sim_config

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_HI = 0.85          # how far off the brake a bang-bang release goes
MAX_STEPS = 9000


def true_slips(v):
    gs = float(v[E.I_GS])
    if gs < 1.0:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(min(1.0, max(0.0, 1.0 - abs(float(v[E.I_WS0 + i])) / gs))
                 for i in range(4))


class ConstantRelease:
    """One fixed release, every wheel, every step."""

    act_dim = 4

    def __init__(self, release):
        self.release = float(release)
        self.name = "const_%.2f" % self.release

    def reset(self):
        pass

    def __call__(self, v):
        return [self.release] * self.act_dim


class SlipBangBang:
    """Per-wheel release when slip runs past target, re-apply below target-band.

    Hysteresis is what keeps a bang-bang regulator from chattering on the
    one-step-delayed measurement; without it this degenerates into a constant.
    """

    act_dim = 4

    def __init__(self, target, band=0.06):
        self.target = float(target)
        self.band = float(band)
        self.name = "slip_%.2f" % self.target
        self.releasing = [False] * 4

    def reset(self):
        self.releasing = [False] * 4

    def __call__(self, v):
        slips = true_slips(v)
        if self.act_dim == 2:
            # Axle contract: one release per axle, driven by the WORSE wheel on
            # it. Releasing on the max keeps the locking wheel out of deep slip;
            # averaging would let one wheel sit locked behind a healthy partner.
            slips = (max(slips[0], slips[1]), max(slips[2], slips[3]))
        out = []
        for i, s in enumerate(slips):
            if self.releasing[i]:
                if s < self.target - self.band:
                    self.releasing[i] = False
            elif s > self.target:
                self.releasing[i] = True
            out.append(RELEASE_HI if self.releasing[i] else 0.0)
        return out


class SlipProportional:
    """release = Kp * (slip - target), clipped -- a plain P regulator per wheel.

    Bang-bang overshoots badly through a one-step-delayed measurement: by the
    time slip is seen past target the wheel is already deeper, and a full
    release then spins it back up. Proportional control degrades gracefully
    under that delay, which is why every production slip controller uses it.
    """

    act_dim = 4

    def __init__(self, target, kp=3.0, act_dim=4):
        self.target = float(target)
        self.kp = float(kp)
        self.act_dim = int(act_dim)
        self.name = "prop_%.2f_kp%.1f%s" % (self.target, self.kp,
                                            "" if act_dim == 4 else "_axle")

    def reset(self):
        pass

    def __call__(self, v):
        slips = true_slips(v)
        if self.act_dim == 2:
            slips = (max(slips[0], slips[1]), max(slips[2], slips[3]))
        return [min(1.0, max(0.0, self.kp * (s - self.target))) for s in slips]


class PolicyController:
    """A trained checkpoint, acting deterministically, through the same path.

    Used to check what a distilled policy actually brakes at before an hour of
    fine-tuning is committed to it. Observations come from the env, normalized
    by the checkpoint's own saved VecNormalize statistics -- never refitted, or
    the number would not be the one the policy was trained to produce.
    """

    def __init__(self, model_path, vecnorm_path, noise=0.0):
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        import bounded_ppo                                   # registers the policy
        self.np = np
        self.model = PPO.load(model_path, device="cpu")
        with open(vecnorm_path, "rb") as handle:
            import pickle
            self.venv = pickle.load(handle)
        self.name = "policy_" + os.path.basename(os.path.dirname(model_path))
        self.obs = None
        # Perturbing the student during DAgger collection is the point, not a
        # nuisance: the states it falls into under noise are exactly the ones it
        # cannot currently recover from, and those are what need teacher labels.
        self.noise = float(noise)
        self.stochastic = False
        self.rng = np.random.default_rng(0)

    def reset(self):
        self.obs = None

    def observe(self, obs):
        self.obs = obs

    def __call__(self, v):
        if self.obs is None:
            return [0.0] * 4
        normed = self.np.clip(
            (self.obs - self.venv.obs_rms.mean)
            / self.np.sqrt(self.venv.obs_rms.var + self.venv.epsilon),
            -self.venv.clip_obs, self.venv.clip_obs).astype(self.np.float32)
        action, _ = self.model.predict(normed, deterministic=not self.stochastic)
        action = self.np.asarray(action, dtype=float).reshape(-1)
        if self.noise > 0.0:
            action = self.np.clip(
                action + self.rng.normal(0.0, self.noise, action.shape), 0.0, 1.0)
        return list(action)


def run_episode(env, controller, packet, record=None, labeller=None):
    """Drive one stop. With `record`, append (obs, action) pairs to it.

    The obs appended is exactly what a policy would be handed at that step, and
    the action is what the teacher does in response, so the pairs are directly
    usable to warm-start a policy on a controller that already scores 1.189 g.
    The teacher reads ground-truth slip; the student would only ever see obs.
    """
    controller.reset()
    obs, _ = env.reset()
    action = [0.0] * getattr(controller, "act_dim", 4)   # full brake, like a slam
    info = {}
    for _ in range(MAX_STEPS):
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        if hasattr(controller, "observe"):
            controller.observe(obs)
        v = packet.get("v")
        action = (controller(v) if v is not None
                  else [0.0] * getattr(controller, "act_dim", 4))
        if record is not None:
            # DAgger: with a labeller the STUDENT drives and the TEACHER says what
            # it would have done, so the data covers the states the student
            # actually reaches rather than only the ones the teacher visits.
            label = labeller(v) if (labeller is not None and v is not None) else action
            record.append((obs.copy(), list(label)))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="slip_ceiling_probe.csv")
    ap.add_argument("--wheel-mode", choices=("independent", "axle"),
                    default="independent",
                    help="axle = the 2-action contract the deployed in-car "
                         "controller already accepts")
    ap.add_argument("--policy-stochastic", action="store_true",
                    help="sample from the policy exactly as PPO does while "
                         "training, instead of taking its deterministic mode")
    ap.add_argument("--policy-noise", type=float, default=0.0,
                    help="gaussian release noise on the student during collection")
    ap.add_argument("--dagger", default=None,
                    help="target:kp of the teacher that LABELS states the "
                         "--policy student visits (DAgger); needs --record")
    ap.add_argument("--policy", default=None,
                    help="run dir holding final.zip + vecnormalize.pkl, acted "
                         "deterministically instead of a scripted controller")
    ap.add_argument("--record", default=None,
                    help="npz path: save every (obs, action) pair for warm-starting")
    ap.add_argument("--releases", default="0.0,0.10,0.20,0.30")
    ap.add_argument("--targets", default="0.12,0.18,0.24")
    ap.add_argument("--prop", default="", help="target:kp pairs, e.g. 0.18:3,0.18:6")
    ap.add_argument("--dt", type=float, default=None,
                    help="control period in seconds (default 0.01 = 100 Hz). "
                         "BeamNG derives sendSkips = ceil(dt / 0.0005) - 1, so "
                         "0.005 is 200 Hz and 0.0025 is 400 Hz against 2 kHz physics.")
    args = ap.parse_args()
    if args.dt:
        E.set_control_dt(args.dt)
        print("control rate %.0f Hz (dt=%g)" % (1.0 / E.DT, E.DT), flush=True)

    sc = sim_config.load(os.path.join(HERE, "settings.json"))
    port = args.port or sc.port
    controllers = [ConstantRelease(float(r)) for r in args.releases.split(",") if r]
    act_dim = 2 if args.wheel_mode == "axle" else 4
    for c in controllers:
        c.act_dim = act_dim
    controllers += [SlipBangBang(float(t)) for t in args.targets.split(",") if t]
    for pair in args.prop.split(","):
        if not pair.strip():
            continue
        target, _, kp = pair.partition(":")
        controllers.append(SlipProportional(float(target), float(kp or 3.0),
                                            act_dim=act_dim))

    if args.policy:
        controllers.append(PolicyController(
            os.path.join(args.policy, "final.zip"),
            os.path.join(args.policy, "vecnormalize.pkl"),
            noise=args.policy_noise))
        controllers[-1].stochastic = bool(args.policy_stochastic)
        if args.policy_stochastic:
            controllers[-1].name += "_stochastic"

    env = E.ABSCoSimEnv(speeds=(80,), port=port, game_folder=sc.game_folder,
                        user_path=sim_config.resolved_userpath(sc), headless=True,
                        runup_speed_factor=16.0, wheel_mode=args.wheel_mode)
    packet = {}
    original_exchange = env.link.exchange

    def recording_exchange(values):
        v = original_exchange(values)
        if v is not None:
            packet["v"] = v
        return v

    env.link.exchange = recording_exchange

    rows = []
    record = [] if args.record else None
    labeller = None
    if args.dagger:
        target, _, kp = args.dagger.partition(":")
        labeller = SlipProportional(float(target), float(kp or 6.0),
                                    act_dim=act_dim)
        print("DAgger labelling with %s" % labeller.name, flush=True)
    try:
        for controller in controllers:
            for rep in range(args.reps):
                t0 = time.time()
                info = run_episode(env, controller, packet, record, labeller)
                row = {"controller": controller.name, "rep": rep,
                       "avg_g": info.get("avg_g"), "dist": info.get("stopping_dist_m"),
                       "outcome": info.get("outcome"), "peak_g": info.get("peak_g"),
                       "yaw_abs": info.get("yaw_abs_sum"),
                       "wall_s": round(time.time() - t0, 1)}
                rows.append(row)
                print("%-12s rep%d  avg_g=%s dist=%s %s" % (
                    controller.name, rep, row["avg_g"], row["dist"], row["outcome"]),
                    flush=True)
    finally:
        env.close()
        if record:
            import numpy as np
            np.savez_compressed(
                os.path.join(HERE, args.record),
                obs=np.asarray([o for o, _ in record], dtype=np.float32),
                act=np.asarray([a for _, a in record], dtype=np.float32))
            print("recorded %d (obs, action) pairs -> %s" % (len(record), args.record))
        if rows:
            with open(os.path.join(HERE, args.out), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        best = {}
        for r in rows:
            g = r.get("avg_g")
            if isinstance(g, (int, float)):
                best[r["controller"]] = max(best.get(r["controller"], 0.0), g)
        print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
