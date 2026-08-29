# BeamNG ML-ABS Training Environment — Build Plan

Goal: turn the ResidualABS experiment (Beamng_AI/ml/ResidualABS) into a standalone,
machine-independent tool anyone can clone, point at their BeamNG install, and train
an ML ABS controller with. Updates pushed to this repo at end of each working day.

## 0. Source of truth today (what gets migrated)

| Piece | Where it lives now | Ships? |
|---|---|---|
| Env stack (abs_env.py, abs_env_incar.py — protected byte-identical copies) | Beamng_AI/ml/ResidualABS | yes, as-is (edits only via subclass seams) |
| Residual env subclass, trainer, GUI, probe, logging helper, tests | Beamng_AI/ml/ResidualABS | yes |
| abstelemetry.lua (V3 base + arc channel, rebuilt 2026-08-28) | Beamng_AI/ml/ResidualABS | yes |
| ABS controller mod `mtb_ml_abs` (MTB-ML-ABS.lua, jbeam, info.json) | ONLY in the .tech userpath, in no repo | **yes — must ship or nobody can run this** |
| mtb_ml_weights.lua (3.1 MB deployed SAC weights) | .tech userpath | no — training runs the controller in ext mode, which skips the NN; ship a stub so the controller loads |
| Car configs (Machine-Trainer-Boy-V2-MLABS.pc + siblings) | .tech userpath / PPO_V3 folder | yes, under assets/ |

## 1. Simulator settings (new "Simulator" tab, persisted to settings.json)

Everything below is hardwired in abs_env.py today (BNG_HOME = one absolute path on
one PC, HEADLESS = True, userpath guessed, CPU pinning for one specific 12600K).

- **Game**: BeamNG.drive / BeamNG.tech radio.
- **Game folder**: folder picker; validated by locating the exe (BeamNG.drive.x64.exe
  / BeamNG.tech.x64.exe). Detected version shown next to it.
- **Userpath**: folder picker, auto-filled from the game choice
  (%LOCALAPPDATA%\BeamNG\BeamNG.<drive|tech>\current), overridable.
- **Headless**: checkbox; enabled only for .tech (.drive has no `-gfx null`).
  .drive toggle ships labeled "untested" until a real run proves it.
- **Port**, **Map** (default smallgrid).
- **CPU pinning**: off by default. On = explicit core lists. (Currently silently
  skipped anyway: psutil isn't installed in the venv.)
- **Attach vs launch**: "connect to already-running instance first" stays the default
  (matches the project law that .drive and .tech run side by side).

Implementation: `sim_config.py` (dataclass + load/save/validate + exe/version
detection, pure and unit-tested). `abs_env_residual.py` sets the parent module's
`BNG_HOME` / `HEADLESS` / `VEHICLE_PC` / `MAP_NAME` before `super().__init__()` (the
same seam abs_env_incar.py:79 already uses for the car) and overrides
`_apply_performance_tuning`. The protected copies stay byte-identical.

## 2. Car picker: Model -> Trim -> (Custom -> your .pc files)

Data sources, verified against the real install:
- **Models** = `content/vehicles/*.zip` in the game folder, filtered to
  `info.json "Type": "Car"` (121 zips, most are props/trailers).
- **Trims** = `vehicles/<model>/*.pc` inside that zip, display name from
  `info_<trim>.json "Configuration"` (etk800 has 29 stock trims).
- **Custom** = `<userpath>/vehicles/<model>/*.pc` (your 17 etk800 configs today).
  Custom entry appears only if that folder has any .pc files.

Three cascading comboboxes; selection resolves to the `partConfig` string beamngpy
needs. Scanning the zips is cached to `cache/vehicles_<gameversion>.json`.

Honesty note shown in the GUI: the reward, gates and controller were validated only
on the etk800 Machine-Trainer-Boy-V2-MLABS config. Any car works mechanically only
if it has the ML-ABS part slot (the mod's jbeam targets etk800); other models need
their own jbeam slot — surfaced as a warning, not silently allowed.

## 3. Asset installer ("Install game assets" button)

Copies from `assets/` into the chosen userpath: `mods/unpacked/mtb_ml_abs/`, the .pc
files, abstelemetry.lua. Status line shows installed / missing / stale (hash
compare). Same mechanism abs_env.py already uses for abstelemetry, just visible and
covering everything.

## 4. Compatibility check instead of a hard version refusal

Today: `beamngpy != 1.34.1` raises. Universal: detect game version from the exe
folder, look up the required beamngpy in a bundled table (from BeamNGpy's
COMPATIBILITY.md), show both, warn on mismatch with the fix command, allow
"proceed anyway".

## 5. Trainer correctness (from the 2026-08-28 review)

Add the kwargs that silently fell to SB3 defaults: `policy_kwargs` 3x256 ReLU for
both algos, SAC `train_freq=(2,"step")`, PPO `ent_coef=0.005`; GPU device becomes a
setting (default `cuda` if available, else `cpu`). The resolved-hyperparams log
line already shows what SB3 actually uses.

## 6. Repo hygiene

- `requirements.txt` pinned to what runs today: Python 3.10, torch 2.9.1+cu126,
  stable-baselines3 2.7.1, gymnasium 1.2.3, numpy 2.2.6, beamngpy (per compat table),
  psutil.
- `README.md`: what it is, install, first run (settings -> install assets -> START),
  what the metric is (the standard 2 kHz brake-distance method), known limits.
- `.gitignore`: runs/, logs/, cache/, settings.json (user-local), .venv.
- No AI attribution anywhere: commits, README, contributors.
- Private until told otherwise.

## 7. Still-open physics question (not a GUI item)

Pedal scaling has never been physics-tested — every live gate ran at pedal 1.0.
Before any long randomized-pedal run: slam at pedal 0.5 vs 1.0; g must drop well
below the ~1.03 g floor. If it doesn't, the slam latch is beating the controller's
`input.brake = maxBrake` write and pedal randomization is a no-op.

## Build order

1. Scaffold repo: copy ResidualABS + assets, requirements, README skeleton, tests green.
2. `sim_config.py` + Simulator tab + env seams (headless / game / paths / pinning).
3. Car picker (scanner module TDD'd against the real zips, then the three combos).
4. Asset installer + compatibility check.
5. Trainer kwargs fix.
6. Live smoke: .tech headless end-to-end, watch train.log; then the pedal probe.
7. .drive smoke (or ship the toggle labeled untested).
