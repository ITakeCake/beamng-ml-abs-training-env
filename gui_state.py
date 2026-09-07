"""Remembers what the Training tab was set to, across app restarts and across
algorithm switches.

Two separate losses, one cause, nothing was ever written down:

  * Closing the app forgot every training box. settings.json exists but holds
    the SIMULATOR config (and is read by the trainer at launch), so the
    training fields have no home; this file gives them one rather than
    overloading a file another process parses.

  * Switching SAC -> PPO -> SAC rebuilt the per-algorithm panel from defaults,
    so a tuned learning rate was gone the moment you looked at the other
    algorithm. Each algorithm's values are kept under its own key, so the two
    sets never overwrite each other and switching back restores what was there.

Losing tuned values silently is worse than a crash: a run launched with numbers
the user did not choose still trains, and the result looks legitimate.

Pure JSON, no GUI imports, so the merge rules are testable without a display.
"""
import json
import os

STATE_VERSION = 1

# Written to the run fields (not per-algorithm). Anything not listed is not
# remembered, so adding a widget does not silently start persisting it.
RUN_KEYS = (
    "backend", "algo", "net_arch", "cosim_net_arch", "residual_net_arch",
    "cosim_reward", "residual_reward", "cosim_run_name", "residual_run_name",
    "speeds", "pedal_random", "pedal_spec",
    "grip", "corner", "reward", "run_name", "total_steps",
    "car_model", "car_trim", "car_custom", "runup_speed_factor", "seed",
)


def default_state():
    return {"version": STATE_VERSION, "run": {}, "algo": {}}


def load(path):
    """Missing or corrupt file -> empty state, never raises. This is a
    convenience restore on startup; a stack trace here would be worse than
    starting from defaults."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default_state()
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        # A future/older layout is not worth guessing at.
        return default_state()
    state = default_state()
    if isinstance(data.get("run"), dict):
        state["run"] = {k: v for k, v in data["run"].items() if k in RUN_KEYS}
    if isinstance(data.get("algo"), dict):
        state["algo"] = {a: dict(v) for a, v in data["algo"].items()
                         if isinstance(v, dict)}
    return state


def save(state, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)


def remember_run(state, values):
    """Record the run-level fields, keeping only the known keys."""
    state.setdefault("run", {}).update(
        {k: v for k, v in values.items() if k in RUN_KEYS})
    return state


def remember_algo(state, algo, values):
    """Record one algorithm's hyperparameters under its own key, so SAC's
    values and PPO's never collide."""
    state.setdefault("algo", {}).setdefault(algo, {}).update(
        {k: str(v) for k, v in values.items()})
    return state


def algo_values(state, algo, defaults):
    """What to put in the per-algorithm boxes: remembered values where they
    exist, defaults elsewhere.

    Merged per key rather than all-or-nothing, so a field added to the
    defaults after a state file was written still appears with its default
    instead of vanishing."""
    saved = state.get("algo", {}).get(algo, {})
    return {k: saved.get(k, str(v)) for k, v in defaults.items()}


def run_value(state, key, default):
    """One remembered run field, or `default` when it was never saved."""
    return state.get("run", {}).get(key, default)
