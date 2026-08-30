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
