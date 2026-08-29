# BeamNG ML-ABS Training Environment

Train a machine-learned ABS controller (SAC / PPO) against BeamNG.tech or
BeamNG.drive, with a GUI that handles simulator setup, car selection, and
training runs.

## The idea

The trained action is a **release from the driver's pedal**, per axle, not the
brake level itself. Zero action = full-pedal slam = the known lockup baseline,
so training starts at the anti-lock boundary instead of having to discover
"brake hard" first. Reward is g-force only (per-step + terminal shape).

## Status

Early scaffold, migrated from a working single-machine prototype. Not yet
generalized to run on someone else's install -- see [docs/PLAN.md](docs/PLAN.md)
for what's done and what's left.

## Layout

- `abs_env.py`, `abs_env_incar.py` -- the training-loop environment (byte-identical
  reference copies; never hand-edit, only subclass)
- `abs_env_residual.py` -- the residual (release-from-pedal) env
- `residual_core.py` -- pure action-space math + CLI arg parsing (no game imports)
- `residual_log.py` -- shared logging (GUI / trainer / probe each get their own log file)
- `train_residual.py` -- SAC/PPO trainer (stable-baselines3)
- `gui_train.py`, `gui_cmd.py` -- tkinter launcher/monitor GUI
- `baseline_probe.py` -- sanity gates run before trusting a training config
- `abstelemetry.lua` -- in-game telemetry/actuation bridge
- `assets/` -- the BeamNG mod (ABS controller) + reference car configs this ships with
- `docs/PLAN.md` -- build plan and design notes

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

Requires a BeamNG.tech or BeamNG.drive install; `beamngpy` must match your
game version's wire protocol (see requirements.txt).

## Quick start (current, single-machine state)

```bash
python train_residual.py --algo sac --speeds "60" --pedal off \
    --total-steps 10000 --run-name smoke_sac
```

Or launch `gui_train.py` for the same flow with a live monitor.
