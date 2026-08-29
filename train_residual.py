"""Unified SAC/PPO trainer for the residual (release-from-pedal) ABS env.

Trains against the deployment-clock in-car loop (ABSLearningEnvResidual): the
controller is the sole obs-builder/actuator, this script only mailboxes actions
and runs the SB3 update. No curriculum, no algorithm-specific replay-buffer or
autocast optimizations -- plain SB3 SAC/PPO, matching the reference designs in
TRAINING_GUI_SPEC.md.

Sim guard: this project's beamngpy/BeamNG.tech pairing is pinned; a mismatched
beamngpy speaks a different wire protocol and fails the handshake outright.

Graceful stop: create STOP_TRAINING.txt next to this file (or pass --stop-file);
learn() exits cleanly and the finally block saves model + vecnorm + checkpoint.
Never taskkill a run -- a hard kill can corrupt the replay buffer/optimizer state.

Usage:
  python train_residual.py --algo sac --speeds "60,90,120" --pedal "0.4-1.0" \
      --total-steps 500000 --run-name residual_sac_r1
  python train_residual.py --algo ppo --speeds "60" --pedal off \
      --total-steps 200000 --run-name residual_ppo_smoke
"""
import argparse
import os
import sys
import time

import beamngpy

REQUIRED_BEAMNGPY_VERSION = "1.34.1"
_installed_version = beamngpy.__version__.strip()
if _installed_version != REQUIRED_BEAMNGPY_VERSION:
    raise RuntimeError(
        f"beamngpy {_installed_version!r} does not match the pinned "
        f"{REQUIRED_BEAMNGPY_VERSION!r} this project's BeamNG.tech install speaks. "
        f"A version mismatch fails the connection handshake outright -- fix the "
        f"environment before training, do not bypass this check.")

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from residual_core import parse_speeds, parse_pedal_spec
from residual_log import setup_logging

FRAME_STACK = 16
HEARTBEAT_STEPS = 2000
log = None   # set in main() once the run dir (and therefore the log path) is known

# Defaults taken from the project's existing in-car trainers, not invented here:
# SAC values match ml/SAC/MachineTrainerBoy/train_incar.py (the in-car fine-tune
# config); PPO values match ml/PPO/PPO_V3_AxleCurriculum/train_ppo_axle.py, whose
# LR was lowered 3e-4 -> 1e-4 and BATCH_SIZE raised 256 -> 512 after post-resume
# updates proved crash-heavy at the higher rate. Residual v1 inherits the
# already-tuned values rather than the plan's untuned first guesses.
SAC_DEFAULTS = dict(lr=1e-4, buffer_size=100_000, tau=0.005,
                    target_entropy=-2.0, learning_starts=5_000)
PPO_DEFAULTS = dict(lr=1e-4, n_steps=2048, batch_size=512, n_epochs=10,
                    clip_range=0.2, gae_lambda=0.95)


class StopFileCallback(BaseCallback):
    """Graceful stop: create the stop file and learn() exits cleanly on the next
    check, so the finally block below saves model + vecnorm + checkpoint. A hard
    kill (taskkill) skips that save and can corrupt SAC's replay buffer."""

    def __init__(self, stop_file, verbose=0):
        super().__init__(verbose)
        self.stop_file = stop_file

    def _on_step(self):
        if self.num_timesteps % 200 == 0 and os.path.exists(self.stop_file):
            log.info("STOP file %s detected -> graceful stop at %s steps",
                     self.stop_file, f"{self.num_timesteps:,}")
            return False
        return True


class HeartbeatCallback(BaseCallback):
    """One line every HEARTBEAT_STEPS so a hung sim/env is visible in train.log
    (timesteps, fps since last beat, wall time). Episode-level detail is logged
    by the env itself; avg_g lives in the episode CSV the GUI monitors."""

    def __init__(self, every=HEARTBEAT_STEPS, verbose=0):
        super().__init__(verbose)
        self.every = every
        self._next = every
        self._t_last = time.monotonic()
        self._n_last = 0
        self._t0 = self._t_last

    def _on_step(self):
        if self.num_timesteps >= self._next:
            now = time.monotonic()
            fps = (self.num_timesteps - self._n_last) / max(1e-6, now - self._t_last)
            log.info("heartbeat: timesteps=%s fps=%.1f elapsed=%.1fmin",
                     f"{self.num_timesteps:,}", fps, (now - self._t0) / 60.0)
            self._t_last, self._n_last = now, self.num_timesteps
            self._next += self.every
        return True


class LogMirrorCallback(BaseCallback):
    """abs_env.py's episode CSV logger writes to a fixed path next to that file
    (computed from its own __file__, not configurable without editing the
    byte-identical copy), APPENDING across every process that ever uses this env
    -- so the per-run log the GUI monitors is produced by periodically copying
    only the lines written since this run started, not the whole shared file
    (a whole-file copy would leak every prior run's/probe's episodes into this
    run's count and "best avg_g", which the monitor treats as ground truth)."""

    def __init__(self, src_path, dst_path, skip_lines=0, every_n_steps=500, verbose=0):
        super().__init__(verbose)
        self.src_path = src_path
        self.dst_path = dst_path
        self.skip_lines = skip_lines   # lines already in src_path before this run
        self.every_n = every_n_steps
        self._next = every_n_steps

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.every_n
            self._mirror()
        return True

    def _mirror(self):
        if not os.path.exists(self.src_path):
            return
        with open(self.src_path, encoding="utf-8") as fh:
            lines = fh.readlines()
        if not lines:
            return
        header = lines[0]
        this_run = lines[max(1, self.skip_lines):]
        os.makedirs(os.path.dirname(self.dst_path), exist_ok=True)
        with open(self.dst_path, "w", encoding="utf-8") as fh:
            fh.write(header)
            fh.writelines(this_run)


def make_venv(args):
    from abs_env_residual import ABSLearningEnvResidual

    def _make():
        pedal = parse_pedal_spec(args.pedal)
        env = ABSLearningEnvResidual(port=args.port, env_index=0, user_path=None,
                                     pedal_range=pedal)
        env.fixed_mph = parse_speeds(args.speeds)
        return env

    t0 = time.monotonic()
    log.info("creating env: port=%d speeds=%s pedal=%s", args.port, args.speeds, args.pedal)
    venv = DummyVecEnv([_make])
    venv = VecFrameStack(venv, n_stack=FRAME_STACK)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
    log.info("env ready in %.1fs: obs=%s action=%s frame_stack=%d",
             time.monotonic() - t0, venv.observation_space.shape,
             venv.action_space.shape, FRAME_STACK)
    return venv


def log_resolved_hyperparams(model, args):
    """What SB3 is ACTUALLY using, including everything that fell to a default
    because it was never passed (net_arch, activation, train_freq, ent_coef...)."""
    pol = model.policy
    try:
        arch = getattr(pol, "net_arch", None)
        act = getattr(pol, "activation_fn", None)
        act = act.__name__ if act is not None else None
    except Exception:
        arch, act = "?", "?"
    common = dict(algo=args.algo, device=str(model.device), lr=model.learning_rate,
                  gamma=model.gamma, net_arch=arch, activation=act,
                  obs_shape=model.observation_space.shape,
                  action_shape=model.action_space.shape)
    if args.algo == "sac":
        extra = dict(buffer_size=model.buffer_size, batch_size=model.batch_size,
                     tau=model.tau, train_freq=str(model.train_freq),
                     gradient_steps=model.gradient_steps,
                     learning_starts=model.learning_starts,
                     ent_coef=model.ent_coef, target_entropy=model.target_entropy)
    else:
        extra = dict(n_steps=model.n_steps, batch_size=model.batch_size,
                     n_epochs=model.n_epochs, gae_lambda=model.gae_lambda,
                     ent_coef=model.ent_coef, vf_coef=model.vf_coef,
                     max_grad_norm=model.max_grad_norm)
    log.info("resolved hyperparams: %s", {**common, **extra})


def build_model(args, venv):
    if args.algo == "sac":
        return SAC("MlpPolicy", venv, learning_rate=args.lr,
                   buffer_size=args.buffer_size, tau=args.tau,
                   learning_starts=args.learning_starts,
                   target_entropy=args.target_entropy, verbose=1, device="cuda")
    return PPO("MlpPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps,
               batch_size=args.batch_size, n_epochs=args.n_epochs,
               clip_range=args.clip_range, gae_lambda=args.gae_lambda,
               verbose=1, device="cuda")


def load_resume(args, venv):
    """Loads a checkpoint for --resume, refusing a shape mismatch loudly instead
    of letting SB3 fail deep inside the first predict() call."""
    algo_cls = SAC if args.algo == "sac" else PPO
    log.info("resume: loading %s", args.resume)
    model = algo_cls.load(args.resume, env=venv, device="cuda")
    log.info("resume: checkpoint obs=%s env obs=%s num_timesteps=%s",
             model.observation_space.shape, venv.observation_space.shape,
             f"{model.num_timesteps:,}")
    if model.observation_space.shape != venv.observation_space.shape:
        raise RuntimeError(
            f"--resume checkpoint observation shape {model.observation_space.shape} "
            f"does not match this run's env shape {venv.observation_space.shape} -- "
            f"refusing to resume onto a mismatched policy/env pair.")
    # This trainer's own save layout is <run_dir>/vecnormalize.pkl (see main()); the
    # "<stem>_vecnorm.pkl" convention is checked too for compatibility with the
    # project's other trainers, but a --resume onto one of THIS trainer's own
    # checkpoints only ever matches the first form.
    candidates = [
        os.path.join(os.path.dirname(args.resume), "vecnormalize.pkl"),
        os.path.splitext(args.resume)[0] + "_vecnorm.pkl",
    ]
    vecnorm_path = next((p for p in candidates if os.path.exists(p)), None)
    if vecnorm_path:
        venv = VecNormalize.load(vecnorm_path, venv.venv)
        model.set_env(venv)
        log.info("resume: VecNormalize stats loaded from %s", vecnorm_path)
    else:
        log.warning("resume: no VecNormalize stats found for %s (looked for: %s) "
                    "-- continuing with FRESH obs normalization, which does not "
                    "match the policy's training statistics and can degrade "
                    "performance until it re-adapts.", args.resume, candidates)
    return model, venv


def parse_args():
    p = argparse.ArgumentParser(description="Residual (release-from-pedal) ABS trainer")
    p.add_argument("--algo", choices=["sac", "ppo"], required=True)
    p.add_argument("--speeds", required=True, help='e.g. "70,120,150"')
    p.add_argument("--pedal", required=True, help='"off" or e.g. "0.4-1.0"')
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--port", type=int, default=64291)
    p.add_argument("--run-name", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--stop-file", default=os.path.join(HERE, "STOP_TRAINING.txt"))
    # shared
    p.add_argument("--lr", type=float, default=None)
    # SAC-only
    p.add_argument("--buffer-size", type=int, default=SAC_DEFAULTS["buffer_size"])
    p.add_argument("--tau", type=float, default=SAC_DEFAULTS["tau"])
    p.add_argument("--target-entropy", type=float, default=SAC_DEFAULTS["target_entropy"])
    p.add_argument("--learning-starts", type=int, default=SAC_DEFAULTS["learning_starts"])
    # PPO-only
    p.add_argument("--n-steps", type=int, default=PPO_DEFAULTS["n_steps"])
    p.add_argument("--batch-size", type=int, default=PPO_DEFAULTS["batch_size"])
    p.add_argument("--n-epochs", type=int, default=PPO_DEFAULTS["n_epochs"])
    p.add_argument("--clip-range", type=float, default=PPO_DEFAULTS["clip_range"])
    p.add_argument("--gae-lambda", type=float, default=PPO_DEFAULTS["gae_lambda"])
    args = p.parse_args()
    if args.lr is None:
        args.lr = SAC_DEFAULTS["lr"] if args.algo == "sac" else PPO_DEFAULTS["lr"]
    return args


def main():
    args = parse_args()

    run_dir = os.path.join(HERE, "runs", args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    global log
    log = setup_logging(os.path.join(run_dir, "train.log"), component="trainer")
    log.info("=== train_residual start: pid=%d python=%s beamngpy=%s cwd=%s",
             os.getpid(), sys.executable, _installed_version, os.getcwd())
    log.info("args: %s", vars(args))

    pid_path = os.path.join(run_dir, "pid.txt")
    with open(pid_path, "w") as fh:
        fh.write(str(os.getpid()))

    if os.path.exists(args.stop_file):
        os.remove(args.stop_file)
        log.info("removed stale stop file %s", args.stop_file)

    venv = make_venv(args)
    if args.resume:
        model, venv = load_resume(args, venv)
    else:
        model = build_model(args, venv)
    log_resolved_hyperparams(model, args)

    src_log = os.path.join(HERE, "logs", "episode_log_env0.csv")
    dst_log = os.path.join(run_dir, "episode_log_env0.csv")
    # Lines already in the shared log before this run starts must be excluded from
    # the per-run mirror -- the shared log is appended to by every process that
    # ever touches this env (other runs, baseline_probe.py, etc.).
    skip_lines = 0
    if os.path.exists(src_log):
        with open(src_log, encoding="utf-8") as fh:
            skip_lines = sum(1 for _ in fh)
    callbacks = [
        StopFileCallback(args.stop_file),
        LogMirrorCallback(src_log, dst_log, skip_lines=skip_lines),
        HeartbeatCallback(),
    ]
    log.info("episode log mirror: %s -> %s (skipping %d pre-existing lines)",
             src_log, dst_log, skip_lines)

    log.info("learn() start: run=%s total_steps=%s resume=%s",
             args.run_name, f"{args.total_steps:,}", args.resume)
    t0 = time.monotonic()
    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks,
                    reset_num_timesteps=(args.resume is None))
        log.info("learn() finished normally at %s steps in %.1fmin",
                 f"{model.num_timesteps:,}", (time.monotonic() - t0) / 60.0)
    except BaseException as e:   # KeyboardInterrupt included -- we want it in the log
        log.error("learn() aborted at %s steps after %.1fmin: %s: %s",
                  f"{model.num_timesteps:,}", (time.monotonic() - t0) / 60.0,
                  type(e).__name__, e, exc_info=True)
        raise
    finally:
        try:
            model.save(os.path.join(run_dir, "final.zip"))
            venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
            ts = int(time.time())
            ckpt = os.path.join(checkpoint_dir, f"{args.run_name}_{ts}.zip")
            model.save(ckpt)
            callbacks[1]._mirror()
            log.info("saved final.zip + vecnormalize.pkl + %s", os.path.basename(ckpt))
        except Exception as e:
            log.error("final save FAILED: %s: %s", type(e).__name__, e, exc_info=True)
            raise
        finally:
            if os.path.exists(pid_path):
                os.remove(pid_path)
            log.info("=== train_residual exit")


if __name__ == "__main__":
    main()
