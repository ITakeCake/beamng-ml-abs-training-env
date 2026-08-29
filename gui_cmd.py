"""Settings dict <-> train_residual.py argv translation, plus pre-launch validation.
No game/GUI imports -- pure functions so the command builder is independently testable."""

import sys

from residual_core import parse_speeds, parse_pedal_spec

PYTHON = sys.executable

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

    return cmd


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
    return problems
