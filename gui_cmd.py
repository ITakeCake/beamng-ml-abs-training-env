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

# per-algo flags: dict key -> CLI flag (underscores become dashes)
SAC_KEYS = ["lr", "buffer_size", "tau", "target_entropy", "learning_starts", "train_freq"]
PPO_KEYS = ["lr", "n_steps", "batch_size", "n_epochs", "clip_range", "gae_lambda", "ent_coef"]


def build_cmd(settings):
    algo = settings["algo"]
    cmd = [PYTHON, "train_residual.py",
           "--algo", algo,
           "--speeds", str(settings["speeds"]),
           "--pedal", settings["pedal_spec"] if settings["pedal_random"] else "off",
           "--total-steps", str(settings["total_steps"]),
           "--run-name", str(settings["run_name"])]

    for key in (SAC_KEYS if algo == "sac" else PPO_KEYS):
        cmd += ["--" + key.replace("_", "-"), str(settings[key])]

    if settings.get("resume"):
        cmd += ["--resume", settings["resume"]]

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

    return cmd


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
        if pedal and pedal.needs_continuous_calibration:
            problems.append(
                f"pedal {settings['pedal_spec']!r} draws continuously, which is "
                f"{len(pedal.levels())} levels at 2 decimals -- roughly "
                f"{len(pedal.levels()) * 7 // 60} hours of calibration. Use a list "
                f'(e.g. "0.5,0.75,1.0") instead.')
    return problems


def validate_settings(settings):
    problems = []
    try:
        parse_speeds(settings["speeds"])
    except ValueError as e:
        problems.append(str(e))
    if settings["pedal_random"]:
        try:
            parse_pedal_spec(settings["pedal_spec"])
        except ValueError as e:
            problems.append(str(e))
    if settings.get("grip"):
        try:
            parse_grip_spec(settings["grip"])
        except ValueError as e:
            problems.append(str(e))
    if settings.get("corner"):
        try:
            parse_corner_spec(settings["corner"])
        except ValueError as e:
            problems.append(str(e))
    if settings.get("net_arch"):
        try:
            parse_net_arch(settings["net_arch"])
        except ValueError as e:
            problems.append(str(e))
    return problems
