# Reinforcement Learning Based Anti-Lock Braking Systems

This is a collection of my PPO/SAC experiments on anti-lock braking systems inside of beamng.drive and beamng.tech (Tyvm beamng for the .tech license).
Currently this project is not suitable for full time use inside of beamng, just yet, as it does not meet or exceed stock ABS performance.
This project is still in active development and experimentation.

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
- `train_cosim.py` -- current 100 Hz co-sim PPO trainer
- `train_residual.py` -- legacy in-car SAC/PPO trainer
- `bounded_ppo.py` -- bounded PPO action distribution and checkpoint contract
- `experiment_io.py` -- atomic checkpoint pairs, validation, run state, provenance
- `evaluate_cosim.py` -- frozen checkpoint evaluation and best-pair preservation
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

Or launch `gui_train.py`, select `cosim`, set the run-up multiplier, and use the
live monitor. Graceful stop writes a run-local marker and remains usable after a
GUI restart while the trainer PID is present.

New bounded co-sim checkpoints export with the dedicated
`MTB-ML-ABS-CoSim.lua` 13-observation/two-axle controller. The Output tab blocks
legacy unbounded co-sim and 28x16 residual checkpoints instead of silently
routing them through the incompatible historical four-action controller.
