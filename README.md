# Reinforcement Learning Based Anti-Lock Braking Systems

This is a collection of my PPO/SAC experiments on anti-lock braking systems inside of beamng.drive and beamng.tech (Tyvm beamng for the .tech license).
Currently this project is not suitable for full time use inside of beamng, just yet, as it does not meet or exceed stock ABS performance.
This project is still in active development and experimentation.

## The idea

The trained action is a per-axle **release from the driver's brake pedal**, not the
brake level itself. Zero action = full pedal slam = the known lockup baseline.
Training starts at the anti-lock boundary instead of discovering "brake hard" first.
Reward is g-force only (per-step + terminal shape).

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

Requires a BeamNG.tech or BeamNG.drive install. `beamngpy` must match your game
version's wire protocol (see `requirements.txt`).

## Quick start: co-sim PPO (current default)

Launch `gui_train.py`, select `cosim`, configure the run, and press Start. The GUI
handles simulator setup, training, and live monitoring. Graceful stop writes a
marker file. Never kill the trainer process.

Or from the command line:

```bash
python train_cosim.py --config config.json
```

See [docs/REFERENCE.md](docs/REFERENCE.md) for the full config.json schema and
output file formats.

## Quick start: legacy residual backend

```bash
python train_residual.py --algo sac --speeds "60" --pedal off \
    --total-steps 10000 --run-name smoke_sac
```

Or launch `gui_train.py` and select the residual backend. Supports SAC and PPO.

## Layout

**Environments:**

- `abs_env.py`, `abs_env_incar.py`: base in-car training environment.
- `abs_env_cosim.py`: co-sim environment (current default).
- `abs_env_residual.py`: residual (release-from-pedal) env subclass.
- `cosim_link.py`: UDP co-simulation transport.

**Trainers:**

- `train_cosim.py`: co-sim PPO trainer (the GUI's default backend).
- `train_residual.py`: legacy in-car SAC/PPO trainer.
- `bounded_ppo.py`: affine-tanh action distribution and checkpoint contract.
- `distill_teacher.py`: supervised distillation from a privileged teacher.

**GUI:**

- `gui_train.py`: tkinter launcher and live monitor.
- `gui_cmd.py`: settings-to-argv translation and pre-launch validation.
- `gui_help.py`: inline help text.
- `gui_state.py`: field persistence.
- `train_monitor.py`, `train_progress.py`: live training stats.

**Evaluation and output:**

- `evaluate_cosim.py`: frozen checkpoint evaluation and best-pair preservation.
- `export_policy_weights.py`: exports trained weights to a deployable Lua blob.
- `mod_output.py`, `jbeam_generator.py`: generate the BeamNG mod for deployment.
- `model_registry.py`: lists finished runs for the GUI.

**Calibration and validation:**

- `calibration.py`, `calibration_progress.py`: per-config baseline measurement.
- `reference_runner.py`: automated slam/stock baseline runner.
- `baseline_probe.py`: sanity gates before trusting a training config.
- `simulator_guard.py`: pre-flight simulator checks.

**Reward and config:**

- `reward_spec.py`: reward presets (v5.0 through v11.0).
- `residual_core.py`: action-space math and CLI arg parsing.
- `sim_config.py`: simulator settings (game path, port, headless, map).
- `sim_clock.py`: frame limiter management.
- `corner.py`: arc geometry and steering seek for corner training.

**Shared:**

- `residual_log.py`: logging setup (GUI, trainer, probe each get their own file).
- `compat.py`: beamngpy version compatibility checks.
- `vehicle_scanner.py`: scans game zips for car models and trims.
- `asset_installer.py`: deploys mod files to the game userpath.
- `experiment_io.py`: atomic checkpoint pairs, validation, run state.
- `training_diagnostics.py`: per-rollout and per-episode diagnostic artifacts.

**In-game:**

- `abstelemetry.lua`: telemetry bridge and per-wheel brake control (2 kHz).
- `assets/`: the BeamNG mod (ABS controller Lua, jbeam parts, reference car configs).

**Tests:**

- `tests/`: 735 offline unit and integration tests.

**Docs:**

- `docs/REFERENCE.md`: complete technical reference (reward, GUI contract, control
  ceiling measurements, performance notes, the 1.19 g result).

## The metric

All performance numbers use the standardized 2 kHz brake metric
(`last_brake_avg_g` / `last_brake_dist`) computed by `abstelemetry.lua` inside the
game's physics loop. Never self-computed, never frame-rate.
