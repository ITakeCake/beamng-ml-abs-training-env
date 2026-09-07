"""Settings dict <-> train_residual.py argv translation, plus pre-launch validation.
No game/GUI imports -- pure functions so the command builder is independently testable."""

import os
import sys

from residual_core import (parse_speeds, parse_pedal_spec, parse_grip_spec,
                           parse_net_arch)
from corner import parse_corner_spec

def _console_python():
    """sys.executable, but never pythonw.exe.

    The GUI is launched with pythonw so it has no console window, and a child
    started from it inherits that interpreter. pythonw sets sys.stdout to None,
    and Stable-Baselines3's verbose=1 logger writes to stdout unconditionally --
    it raises "Expected file or str, got None" the moment learn() starts, after
    the game has booted and driven a reset. The trainer runs in its own console
    window anyway, so it wants the console interpreter regardless of how the GUI
    itself was started."""
    exe = sys.executable or ""
    base = os.path.basename(exe).lower()
    if base.startswith("pythonw"):
        candidate = os.path.join(os.path.dirname(exe),
                                 base.replace("pythonw", "python", 1))
        if os.path.isfile(candidate):
            return candidate
    return exe


PYTHON = _console_python()

# Beyond this a calibration run stops being something a person starts
# and waits for. Time, not level count: fast mode changes the arithmetic.
MAX_CALIBRATION_SECONDS = 4 * 3600
REFERENCE_VEHICLE_PC = "vehicles/etk800/Machine-Trainer-Boy-V2-MLABS.pc"

# per-algo flags: dict key -> CLI flag (underscores become dashes)
SAC_KEYS = ["lr", "buffer_size", "tau", "target_entropy", "learning_starts", "train_freq"]
PPO_KEYS = ["lr", "n_steps", "batch_size", "n_epochs", "clip_range", "gae_lambda",
            "gamma", "ent_coef", "initial_release", "log_std_init", "target_kl"]


def build_cmd(settings):
    algo = settings["algo"]
    cmd = [PYTHON, "train_residual.py",
           "--algo", algo,
           "--speeds", str(settings["speeds"]),
           "--pedal", settings["pedal_spec"] if settings["pedal_random"] else "off",
           "--total-steps", str(settings["total_steps"]),
           "--run-name", str(settings["run_name"])]

    for key in (SAC_KEYS if algo == "sac" else PPO_KEYS):
        # Old saved GUI state predates newer algorithm knobs. Omitting an absent
        # key deliberately selects the trainer's own versioned default.
        if key in settings and str(settings[key]).strip() != "":
            cmd += ["--" + key.replace("_", "-"), str(settings[key])]

    if settings.get("resume"):
        cmd += ["--resume", settings["resume"]]
    if settings.get("stop_file"):
        cmd += ["--stop-file", settings["stop_file"]]

    if settings.get("vehicle_pc"):
        cmd += ["--vehicle-pc", settings["vehicle_pc"]]

    # Omitted keys keep train_residual.py's own defaults, so a settings dict
    # written before these existed still builds a valid command.
    if settings.get("reward"):
        cmd += ["--reward", str(settings["reward"])]
    if settings.get("grip"):
        cmd += ["--grip", str(settings["grip"])]
    if settings.get("corner"):
        cmd += ["--corner", str(settings["corner"])]
    if settings.get("net_arch"):
        cmd += ["--net-arch", str(settings["net_arch"])]

    # Absent key => deterministic, which is what every run before this switch
    # existed did. Only the opt-out is passed, so old settings files are safe.
    if not settings.get("deterministic", True):
        cmd += ["--no-deterministic",
                "--train-speed-factor", str(settings.get("train_speed_factor", 1))]

    return cmd


def build_cosim_cmd(config_path):
    """Argv for the JSON-configured co-sim PPO backend."""
    return [PYTHON, "train_cosim.py", "--config", os.path.abspath(config_path)]


def build_cosim_config(settings):
    """Translate validated GUI values to train_cosim.py's stable JSON contract."""
    ppo = {}
    integer_keys = {"n_steps", "batch_size", "n_epochs"}
    for key in PPO_KEYS:
        if key not in settings or str(settings[key]).strip() == "":
            continue
        ppo[key] = (int(settings[key]) if key in integer_keys
                    else float(settings[key]))
    ppo["net_arch"] = parse_net_arch(settings.get("net_arch") or "3x256")
    config = {
        "run_name": str(settings["run_name"]),
        "seed": settings.get("seed") or "auto",
        "total_steps": int(settings["total_steps"]),
        "speeds": parse_speeds(str(settings["speeds"])),
        "runup_speed_factor": float(settings.get("runup_speed_factor", 4.0)),
        "vehicle_pc": settings.get("vehicle_pc") or REFERENCE_VEHICLE_PC,
        "reward": settings.get("reward") or "v6.0",
        "device": settings.get("device") or None,
        "headless": bool(settings.get("headless", True)),
        "port": int(settings.get("port", 64280)),
        "stop_file": "STOP_TRAINING.txt",
        "diagnostics": {"enabled": True},
        "ppo": ppo,
    }
    if settings.get("resume"):
        config["resume_checkpoint"] = os.path.abspath(settings["resume"])
        config["resume_inherit_config"] = True
    return config


def build_calibration_cmd(settings, car, reps=3):
    """Argv for reference_runner.py over the SAME configuration the Training tab
    is set to. Calibration is only a ruler if it was measured on the exact
    configuration it will score, so speeds/grip/corner come from the training
    settings rather than being asked for twice -- a second set of fields is a
    second chance to measure the wrong thing.

    Grip levels: only the enumerable ones. A continuous range draws values
    nothing can be calibrated at, which the trainer already refuses to pair
    with a normalized reward."""
    cmd = [PYTHON, "reference_runner.py",
           "--car", str(car),
           "--speeds", str(settings["speeds"]),
           "--reps", str(reps)]
    grip = settings.get("grip")
    if grip:
        spec = parse_grip_spec(grip)
        levels = spec.levels() if spec else None
        if levels:
            cmd += ["--grips", ",".join(str(g) for g in levels)]
    if settings.get("corner"):
        cmd += ["--corner", str(settings["corner"])]
    # Pedal keys a calibration row too: references measured at full pedal do
    # not describe a half-pedal stop, which physically cannot reach the
    # full-pedal lockup floor. Only the levels the training run can actually
    # draw are measured -- "off" means full pedal, which is the default.
    if settings.get("pedal_random") and settings.get("pedal_spec"):
        spec = parse_pedal_spec(settings["pedal_spec"])
        levels = spec.levels() if spec else None
        if levels:
            cmd += ["--pedals", ",".join(str(p) for p in levels)]
    # Fast mode runs the measured stop free-running under a physics speed
    # factor instead of stepping it. Measured ~11x faster per stop at the same
    # result to within each regime's own noise (see calibration.regime_name).
    if settings.get("fast_calibration"):
        cmd += ["--live", "--speed-factor",
                str(settings.get("speed_factor") or 10)]
    return cmd


def validate_calibration_settings(settings):
    """Problems that would make a calibration run measure the wrong thing."""
    problems = validate_settings(settings)
    grip = settings.get("grip")
    if grip:
        try:
            spec = parse_grip_spec(grip)
        except ValueError:
            return problems          # already reported by validate_settings
        if spec and spec.needs_continuous_calibration:
            problems.append(
                f"grip {grip!r} draws continuously, so there is no finite set of "
                f"levels to calibrate. Use a list (e.g. \"0.5,0.75,1.0\") instead.")
    if settings.get("pedal_random") and settings.get("pedal_spec"):
        try:
            pedal = parse_pedal_spec(settings["pedal_spec"])
        except ValueError:
            return problems          # already reported by validate_settings
        if pedal:
            # Judge a range by how long it would actually TAKE, not by its level
            # count. Fast mode measures a stop in ~4 s instead of ~35 s, which
            # turns 0.5-1.0 (51 levels) from about six hours into about twenty
            # minutes -- so the old blanket refusal is now simply wrong.
            import calibration_progress as _cp
            from residual_core import parse_speeds as _ps, parse_grip_spec as _pg
            try:
                speeds = _ps(settings["speeds"])
            except ValueError:
                speeds = [60]
            grip = _pg(settings.get("grip") or "off")
            grips = (grip.levels() if grip else None) or [1.0]
            stops = _cp.planned_stops(speeds, grips, pedal.levels(), 3)
            secs = _cp.estimate_seconds(stops, bool(settings.get("fast_calibration")))
            if secs > MAX_CALIBRATION_SECONDS:
                problems.append(
                    f"pedal {settings['pedal_spec']!r} gives {len(pedal.levels())} "
                    f"levels, which with these speeds and grips is {stops:,} stops "
                    f"-- about {_cp.format_eta(secs)}. Tick 'Fast calibration', use "
                    f'a list (e.g. "0.5,0.75,1.0"), or cut a speed.')
    return problems


def validate_settings(settings):
    problems = []
    backend = settings.get("backend", "residual")
    if backend == "cosim":
        if settings.get("algo") != "ppo":
            problems.append("co-sim training supports PPO only")
        if settings.get("resume"):
            checkpoint = os.path.abspath(os.fspath(settings["resume"]))
            if not os.path.isfile(checkpoint):
                problems.append("resume checkpoint does not exist")
            elif not checkpoint.lower().endswith(".zip"):
                problems.append("resume checkpoint must be an SB3 .zip file")
            else:
                from experiment_io import paired_vecnormalize_path
                try:
                    vec_path = paired_vecnormalize_path(checkpoint)
                except ValueError as exc:
                    problems.append(str(exc))
                else:
                    if not os.path.isfile(vec_path):
                        problems.append(
                            "resume checkpoint is missing its paired VecNormalize file")
        if settings.get("reward") == "normalized":
            problems.append("co-sim supports reward v6.0 or v5.0, not normalized")
        raw = str(settings.get("runup_speed_factor", "4")).strip()
        try:
            factor = float(raw)
        except ValueError:
            problems.append(f"run-up speed must be a number, got {raw!r}")
        else:
            if not __import__("math").isfinite(factor) or not 1.0 <= factor <= 1000.0:
                problems.append("co-sim run-up speed must be from 1x to 1000x")
    elif not settings.get("deterministic", True):
        raw = str(settings.get("train_speed_factor", "1")).strip()
        try:
            factor = float(raw)
        except ValueError:
            problems.append(f"engine speed must be a number, got {raw!r}")
        else:
            if factor < 1.0:
                problems.append(
                    f"engine speed {factor:g}x is below real time; use 1 or more")
            elif factor > 50.0:
                problems.append(
                    f"engine speed {factor:g}x is beyond anything measured; the "
                    f"engine saturated near 4-5x on this machine")
    try:
        parse_speeds(settings["speeds"])
    except ValueError as e:
        problems.append(str(e))
    if backend != "cosim" and settings["pedal_random"]:
        try:
            parse_pedal_spec(settings["pedal_spec"])
        except ValueError as e:
            problems.append(str(e))
    if backend != "cosim" and settings.get("grip"):
        try:
            parse_grip_spec(settings["grip"])
        except ValueError as e:
            problems.append(str(e))
    if backend != "cosim" and settings.get("corner"):
        try:
            parse_corner_spec(settings["corner"])
        except ValueError as e:
            problems.append(str(e))
    if settings.get("net_arch"):
        try:
            parse_net_arch(settings["net_arch"])
        except ValueError as e:
            problems.append(str(e))
    run_name = str(settings.get("run_name", "")).strip()
    if (not run_name or run_name in (".", "..")
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                   for c in run_name)):
        problems.append(
            "run name must contain only letters, numbers, dot, underscore, or dash")
    # Validate the fields that would otherwise fail only after BeamNG boots.
    # (minimum, maximum, minimum-is-strict, maximum-is-strict)
    numeric = {
        "lr": (0.0, None, True, False),
        "total_steps": (1.0, None, False, False),
    }
    if settings.get("algo") == "ppo":
        numeric.update({
            "n_steps": (2.0, None, False, False),
            "batch_size": (1.0, None, False, False),
            "n_epochs": (1.0, None, False, False),
            "clip_range": (0.0, 1.0, True, False),
            "gae_lambda": (0.0, 1.0, True, False),
            "gamma": (0.0, 1.0, True, False),
            "ent_coef": (0.0, None, False, False),
            "initial_release": (0.0, 1.0, True, True),
            "log_std_init": (None, None, False, False),
        })
        if backend == "cosim":
            raw_seed = str(settings.get("seed", "auto")).strip().lower()
            if raw_seed not in ("", "auto", "random"):
                try:
                    seed = int(raw_seed)
                except ValueError:
                    problems.append("seed must be an integer or auto")
                else:
                    if not 0 <= seed < 2 ** 32:
                        problems.append("seed must be from 0 to 4294967295")
    for key, (low, high, low_open, high_open) in numeric.items():
        if key not in settings or str(settings.get(key, "")).strip() == "":
            continue
        raw = str(settings.get(key, "")).strip()
        try:
            value = float(raw)
        except ValueError:
            problems.append(f"{key.replace('_', ' ')} must be a number, got {raw!r}")
            continue
        if not __import__("math").isfinite(value):
            problems.append(f"{key.replace('_', ' ')} must be finite")
        elif low is not None and (value < low or (low_open and value == low)):
            word = "greater than" if low_open else "at least"
            problems.append(f"{key.replace('_', ' ')} must be {word} {low:g}")
        elif high is not None and (value > high or (high_open and value == high)):
            word = "less than" if high_open else "at most"
            problems.append(f"{key.replace('_', ' ')} must be {word} {high:g}")
    if settings.get("algo") == "ppo":
        try:
            n_steps = int(settings.get("n_steps", ""))
            batch_size = int(settings.get("batch_size", ""))
            if batch_size > n_steps:
                problems.append("PPO batch size cannot exceed n steps")
        except ValueError:
            pass
    return problems
