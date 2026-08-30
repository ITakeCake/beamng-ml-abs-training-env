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

# Real per-target compat check (compat.py, against the ACTUAL game version
# pointed at by --game-folder) happens in make_venv() below, not here -- a
# fixed constant can't know which game a universal tool is running against.
_installed_version = beamngpy.__version__.strip()

import torch as th
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from residual_core import (parse_speeds, parse_pedal_spec, parse_grip_spec,
                           parse_net_arch, net_arch_repr)
from corner import parse_corner_spec
from residual_log import setup_logging
from sim_config import (
    SimConfig, load as load_sim_config, validate as validate_sim_config,
    detect_game_version,
)
from compat import check_compat
from reward_spec import RewardSpec, PRESETS
from calibration import CalibrationTable, config_key, table_hash

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
                    target_entropy=-2.0, learning_starts=5_000, train_freq=2)
PPO_DEFAULTS = dict(lr=1e-4, n_steps=2048, batch_size=512, n_epochs=10,
                    clip_range=0.2, gae_lambda=0.95, ent_coef=0.005)

# net_arch/activation match the reference trainers exactly (train.py's
# policy_kwargs for SAC, train_ppo_axle.py's for PPO) -- SB3's own defaults
# (64x64 tanh for PPO, 256x256 relu for SAC) are NOT these, and silently
# training on the wrong network makes every result incomparable to the
# reference runs this project's numbers are judged against.
POLICY_KWARGS_SAC = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))
POLICY_KWARGS_PPO = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))


def resolve_device(requested):
    """None = auto (cuda if available, else cpu) -- picks whatever GPU index
    0 is on this machine. An explicit "cuda:1"-style request always wins;
    there is no reference-machine-specific default here (that lived in the
    single-machine prototype, tuned to keep one particular GPU free)."""
    if requested:
        return requested
    return "cuda" if th.cuda.is_available() else "cpu"


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

    # 500 steps was ~half an episode when an episode was ~1084 steps. Under
    # free-running an episode is nearer 50, so 500 meant the per-run file did
    # not appear until episode 10 and then jumped ten at a time -- the monitor
    # looked frozen because there was genuinely nothing to read. Mirroring on
    # episode end makes the cadence follow episodes rather than a step count
    # that no longer means the same thing in both modes.
    def __init__(self, src_path, dst_path, skip_lines=0, every_n_steps=50, verbose=0):
        super().__init__(verbose)
        self.src_path = src_path
        self.dst_path = dst_path
        self.skip_lines = skip_lines   # lines already in src_path before this run
        self.every_n = every_n_steps
        self._next = every_n_steps

    def _on_step(self):
        # An episode just ended => a row was just appended => mirror it now.
        # locals["dones"] is the vec-env flag SB3 already has in hand, so this
        # costs a dict lookup rather than a stat() on every step.
        dones = self.locals.get("dones")
        if dones is not None and any(bool(d) for d in dones):
            self._next = self.num_timesteps + self.every_n
            self._mirror()
        elif self.num_timesteps >= self._next:
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


def build_sim_config(args):
    """Load settings.json, then apply any CLI flags the caller actually
    passed (None = "leave the loaded/default value alone")."""
    cfg = load_sim_config(args.settings)
    if args.game is not None:
        cfg.game = args.game
    if args.game_folder is not None:
        cfg.game_folder = args.game_folder
    if args.userpath is not None:
        cfg.userpath = args.userpath
    if args.windowed:
        cfg.headless = False
    if args.map is not None:
        cfg.map = args.map
    if args.cpu_pinning:
        cfg.cpu_pinning = True
    return cfg


def load_calibration(args):
    """(table, path). The car is taken from --vehicle-pc when given, else the
    reference car's model, so the table lines up with what is being trained."""
    car = args.calibration_car
    if not car and args.vehicle_pc:
        import re as _re
        m = _re.match(r"vehicles/([^/]+)/", args.vehicle_pc)
        car = m.group(1) if m else None
    car = car or "etk800"
    path = args.calibration or os.path.join(HERE, "calibration", f"{car}.json")
    if not os.path.exists(path):
        return None, path
    return CalibrationTable.load(path), path


def make_venv(args):
    from abs_env_residual import ABSLearningEnvResidual

    cfg = build_sim_config(args)
    problems = validate_sim_config(cfg)
    if problems:
        log.warning("sim_config has %d issue(s), proceeding anyway: %s",
                   len(problems), problems)

    game_version = detect_game_version(cfg.game, cfg.game_folder)
    compat = check_compat(game_version, _installed_version)
    log.info("compat check: game_version=%s beamngpy=%s ok=%s",
             game_version, _installed_version, compat.ok)
    if compat.ok is False and not args.force:
        log.error("%s", compat.message)
        raise SystemExit(
            f"{compat.message}\n\nFix: {compat.fix_command}\n\n"
            f"Or pass --force to proceed anyway (the connection will likely "
            f"fail outright on a real protocol mismatch).")
    elif compat.ok is not True:
        log.warning("proceeding despite compat check: %s", compat.message)

    spec = PRESETS[args.reward]()
    table, calib_path = load_calibration(args)
    log.info("reward=%s hash=%s default=%s | calibration=%s hash=%s",
             spec.name, spec.hash(), spec.is_default(),
             calib_path or "(none)", table_hash(table) if table else "(none)")
    if spec.normalize and table is None:
        raise SystemExit(
            f"--reward {args.reward} normalizes against measured references but "
            f"no calibration table was found at {calib_path!r}.\n\n"
            f'Run: python reference_runner.py --car <car> --speeds "60"')

    grip_spec = parse_grip_spec(args.grip)
    if spec.normalize and grip_spec is not None and grip_spec.needs_continuous_calibration:
        raise SystemExit(
            f"--grip {args.grip!r} draws continuously, so an episode can land on a "
            f"grip level nothing was calibrated at, and --reward normalized would "
            f"refuse mid-run.\n\n"
            f'Use a list of levels instead (e.g. --grip "0.5,0.75,1.0") and '
            f"calibrate each, or train with --reward v5.0.")
    log.info("grip: %s", "stock (untouched)" if grip_spec is None else grip_spec)

    corner_spec = parse_corner_spec(args.corner)
    if corner_spec is not None:
        # The angle is looked up per episode from the calibration row for that
        # episode's (grip, speed, radius); with no table there is nothing to
        # look up, and a corner with no angle brakes in a straight line.
        if table is None or not table.steering:
            raise SystemExit(
                f"--corner {args.corner!r} needs a measured steering angle, and "
                f"{calib_path or 'the calibration table'} has none.\n\n"
                f"Run: python reference_runner.py --car <car> "
                f'--corner {args.corner} --speeds "{args.speeds}"')
        if spec.normalize:
            missing = [config_key(grip=g, speed_mph=mph, radius_m=corner_spec.radius_m)
                       for mph in parse_speeds(args.speeds)
                       for g in (grip_spec.levels() or [] if grip_spec else [1.0])]
            absent = [k for k in missing if k not in table.rows]
            if absent:
                raise SystemExit(
                    f"--reward {args.reward} normalizes against measured references, "
                    f"but these corner configs have no calibration row:\n  "
                    + "\n  ".join(absent)
                    + f"\n\nRun: python reference_runner.py --corner {args.corner}")
    log.info("corner: %s", "straight" if corner_spec is None else corner_spec)

    # Pedal position keys a calibration row: the references are measured at a
    # specific pedal, and a half-pedal stop physically cannot reach the
    # full-pedal lockup floor. Scored against those anchors it lands far BELOW
    # "locked wheels" however well it modulates -- indistinguishable from
    # failing. So a normalized run needs a row per pedal level it can draw.
    pedal_spec = parse_pedal_spec(args.pedal)
    if spec.normalize and pedal_spec is not None:
        # A continuous range is no longer refused outright: fast calibration
        # measures all 51 levels of 0.5-1.0 in about twenty minutes, so the
        # question is simply whether the rows exist, which is checked below.
        if pedal_spec.needs_continuous_calibration and not table:
            raise SystemExit(
                f"--pedal {args.pedal!r} draws continuously, which is "
                f"{len(pedal_spec.levels())} distinct levels at 2 decimals -- each "
                f"needs its own measured references (~7 min), so this cannot be "
                f"calibrated.\n\n"
                f'Use a list instead (e.g. --pedal "0.5,0.75,1.0") and calibrate '
                f"those, or train with --reward v5.0.")
        levels = pedal_spec.levels()
        if table is not None:
            missing = [config_key(grip=g, speed_mph=mph,
                                  radius_m=None if corner_spec is None
                                  else corner_spec.radius_m, pedal=pd)
                       for mph in parse_speeds(args.speeds)
                       for g in ((grip_spec.levels() or [1.0]) if grip_spec else [1.0])
                       for pd in levels]
            absent = [k for k in missing if k not in table.rows]
            if absent:
                joined = "\n  ".join(absent[:10])
                more = (f"\n  ... and {len(absent) - 10} more"
                        if len(absent) > 10 else "")
                wanted = ",".join(str(p) for p in levels)
                raise SystemExit(
                    f"--reward {args.reward} normalizes against measured "
                    f"references, but these pedal configurations have no "
                    f"calibration row:\n  {joined}{more}\n\n"
                    f'Run: python reference_runner.py --pedals "{wanted}"')
    log.info("pedal: %s", "full (1.0)" if pedal_spec is None else pedal_spec)

    forced = None
    if args.force_action:
        forced = [float(x) for x in args.force_action.split(",")]
        if len(forced) != 2:
            raise SystemExit(f"--force-action needs 2 values, got {args.force_action!r}")
        log.warning("FORCED ACTION %s -- the policy is overridden every step; "
                    "this is a diagnostic control run, not training", forced)

    def _make():
        pedal = pedal_spec
        env = ABSLearningEnvResidual(port=args.port, env_index=0,
                                     sim_config=cfg, vehicle_pc=args.vehicle_pc,
                                     pedal_range=pedal, reward_spec=spec,
                                     calibration_table=table,
                                     grip_spec=grip_spec,
                                     grip_lead_seconds=args.grip_lead,
                                     corner_spec=corner_spec,
                                     force_action=forced,
                                     deterministic=args.deterministic,
                                     train_speed_factor=args.train_speed_factor)
        env.fixed_mph = parse_speeds(args.speeds)
        return env

    t0 = time.monotonic()
    log.info("creating env: port=%d speeds=%s pedal=%s game=%s headless=%s",
             args.port, args.speeds, args.pedal, cfg.game, cfg.headless)
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


def policy_kwargs_for(args):
    """net_arch from --net-arch, keeping the reference activation and the
    pi/qf vs pi/vf split each algorithm expects. Both halves get the same
    shape, which is what the reference trainers did."""
    layers = parse_net_arch(args.net_arch)
    value_key = "qf" if args.algo == "sac" else "vf"
    if log is not None:      # module-level `log` is only set inside main()
        log.info("network: %s (%d layers, %s)", net_arch_repr(layers), len(layers),
                 "default" if layers == parse_net_arch("") else "CUSTOM")
    return dict(activation_fn=th.nn.ReLU,
                net_arch={"pi": list(layers), value_key: list(layers)})


def build_model(args, venv):
    device = resolve_device(args.device)
    policy_kwargs = policy_kwargs_for(args)
    if args.algo == "sac":
        return SAC("MlpPolicy", venv, learning_rate=args.lr,
                   buffer_size=args.buffer_size, tau=args.tau,
                   learning_starts=args.learning_starts,
                   target_entropy=args.target_entropy,
                   train_freq=(args.train_freq, "step"), gradient_steps=1,
                   policy_kwargs=policy_kwargs, verbose=1, device=device)
    return PPO("MlpPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps,
               batch_size=args.batch_size, n_epochs=args.n_epochs,
               clip_range=args.clip_range, gae_lambda=args.gae_lambda,
               ent_coef=args.ent_coef, policy_kwargs=policy_kwargs,
               verbose=1, device=device)


def load_resume(args, venv):
    """Loads a checkpoint for --resume, refusing a shape mismatch loudly instead
    of letting SB3 fail deep inside the first predict() call."""
    algo_cls = SAC if args.algo == "sac" else PPO
    log.info("resume: loading %s", args.resume)
    model = algo_cls.load(args.resume, env=venv, device=resolve_device(args.device))
    log.info("resume: checkpoint obs=%s env obs=%s num_timesteps=%s",
             model.observation_space.shape, venv.observation_space.shape,
             f"{model.num_timesteps:,}")
    if model.observation_space.shape != venv.observation_space.shape:
        raise RuntimeError(
            f"--resume checkpoint observation shape {model.observation_space.shape} "
            f"does not match this run's env shape {venv.observation_space.shape} -- "
            f"refusing to resume onto a mismatched policy/env pair.")
    # --net-arch is ignored on resume (the checkpoint's own shape is loaded), so
    # a mismatch would silently train a different network than the box says.
    ckpt_arch = getattr(model.policy, "net_arch", None)
    wanted = parse_net_arch(args.net_arch)
    ckpt_layers = (ckpt_arch.get("pi") if isinstance(ckpt_arch, dict) else ckpt_arch)
    if ckpt_layers and list(ckpt_layers) != wanted:
        raise RuntimeError(
            f"--resume checkpoint was trained with network "
            f"{net_arch_repr(list(ckpt_layers))} but --net-arch says "
            f"{net_arch_repr(wanted)}. Resuming keeps the CHECKPOINT's shape, so "
            f"the run would not be what the setting claims. Set the network to "
            f"{net_arch_repr(list(ckpt_layers))}, or start a fresh run.")
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
    # simulator config -- see sim_config.py; a settings.json (GUI-written or
    # hand-edited) supplies defaults, these flags override individual fields
    p.add_argument("--settings", default=os.path.join(HERE, "settings.json"))
    p.add_argument("--game", choices=["tech", "drive"], default=None)
    p.add_argument("--game-folder", default=None,
                   help="BeamNG install root (contains BeamNG.<tech|drive>.exe)")
    p.add_argument("--userpath", default=None,
                   help="BeamNG userpath; default is the standard OS location for --game")
    p.add_argument("--windowed", action="store_true",
                   help="disable headless launch (BeamNG.tech only)")
    p.add_argument("--map", default=None)
    p.add_argument("--cpu-pinning", action="store_true",
                   help="pin Python/BeamNG to specific CPU cores (off by default)")
    p.add_argument("--grip", default="off",
                   help='tire grip: "off"/"stock" (never touched), a level like '
                        '"0.6", a list "0.5,0.75,1.0" (randomized among them), or '
                        'a range "0.4-1.0" (continuous random; incompatible with '
                        '--reward normalized)')
    p.add_argument("--force-action", default=None,
                   help='override the policy every step, e.g. "0,0" for pure '
                        "slam (zero release = full pedal). Diagnostic control "
                        "runs only -- the policy still trains on garbage.")
    p.add_argument("--net-arch", default="3x256",
                   help='hidden layers of the policy/value networks: "3x256" '
                        'or "512,256,128". Default 3x256 matches the reference '
                        "trainers. A resumed run must match its checkpoint.")
    p.add_argument("--no-deterministic", dest="deterministic",
                   action="store_false", default=True,
                   help="Free-run the engine instead of stepping it. Each policy "
                        "decision then costs one round trip instead of three, but "
                        "arrives once per round trip rather than once per 5 ms. "
                        "The 2 kHz brake metric is unaffected; 'steps' and "
                        "'stop_time_s' become meaningless. EXPERIMENTAL.")
    p.add_argument("--train-speed-factor", type=float, default=1.0,
                   help="Physics speed multiplier while free-running (needs "
                        "--no-deterministic). The engine saturates near 4-5x on "
                        "this machine, so larger values mainly cut how many "
                        "decisions the policy gets per stop.")
    p.add_argument("--corner", default="straight",
                   help='brake in a constant-radius turn: radius in metres, '
                        '"50" / "50L" / "50R". "straight" (default) = no corner. '
                        "Requires a steering angle measured by reference_runner.")
    p.add_argument("--grip-lead", type=float, default=0.0,
                   help="apply the grip change this many seconds BEFORE brake "
                        "onset (0 = same physics tick, exact)")
    p.add_argument("--reward", choices=sorted(PRESETS), default="v5.0",
                   help="reward preset: v5.0 (frozen default, absolute anchors) or "
                        "normalized (anchors on measured slam/stock references)")
    p.add_argument("--calibration", default=None,
                   help="path to a calibration table (default: calibration/<car>.json)")
    p.add_argument("--calibration-car", default=None,
                   help="car key for the calibration table (default: from --vehicle-pc)")
    p.add_argument("--force", action="store_true",
                   help="proceed even if beamngpy doesn't match the detected game version")
    p.add_argument("--vehicle-pc", default=None,
                   help='partConfig string, e.g. "vehicles/etk800/MyCar.pc" '
                        "(default: the reference Machine-Trainer-Boy-V2-MLABS.pc)")
    # shared
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None,
                   help='e.g. "cuda", "cuda:1", "cpu" (default: cuda if available, else cpu)')
    # SAC-only
    p.add_argument("--buffer-size", type=int, default=SAC_DEFAULTS["buffer_size"])
    p.add_argument("--tau", type=float, default=SAC_DEFAULTS["tau"])
    p.add_argument("--target-entropy", type=float, default=SAC_DEFAULTS["target_entropy"])
    p.add_argument("--learning-starts", type=int, default=SAC_DEFAULTS["learning_starts"])
    p.add_argument("--train-freq", type=int, default=SAC_DEFAULTS["train_freq"],
                   help="collect this many steps between gradient updates")
    # PPO-only
    p.add_argument("--n-steps", type=int, default=PPO_DEFAULTS["n_steps"])
    p.add_argument("--batch-size", type=int, default=PPO_DEFAULTS["batch_size"])
    p.add_argument("--n-epochs", type=int, default=PPO_DEFAULTS["n_epochs"])
    p.add_argument("--clip-range", type=float, default=PPO_DEFAULTS["clip_range"])
    p.add_argument("--gae-lambda", type=float, default=PPO_DEFAULTS["gae_lambda"])
    p.add_argument("--ent-coef", type=float, default=PPO_DEFAULTS["ent_coef"])
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
