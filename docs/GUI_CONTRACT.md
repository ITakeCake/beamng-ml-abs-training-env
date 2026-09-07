# GUI Contract — Co-Sim ABS Trainer (v1)

This is the file/process contract used by the co-sim backend in `gui_train.py`.
The GUI never imports the environment or touches a live BeamNG session itself;
it writes one JSON config, spawns the trainer, reads logs, and writes the
run-local graceful-stop marker.

---

## 1. What the GUI is for

1. **Configure** a PPO training run — all hyperparameters incl. network size.
2. **Launch** it (spawn the trainer subprocess).
3. **Monitor** it live in a pop-out — active run, current g / speed, graphs of
   avg-g and reward over episodes. Mirror the feel of the old trainer's monitor.
4. **Stop** it gracefully (write a stop file; never kill the process).

Fixed in v1: straight-line, full pedal, single grip level, and PPO. The shared
GUI also exposes the older residual backend. Export-to-game is enabled only
when the run metadata identifies a compatible controller contract.

---

## 2. How the GUI talks to the backend

```
GUI  ──writes──▶  config.json
GUI  ──spawns──▶  python train_cosim.py --config <path-to-config.json>
GUI  ──tails───▶  runs/<run_name>/train.log        (live status + heartbeat)
GUI  ──tails───▶  runs/<run_name>/episode_log.csv  (one row per finished stop)
GUI  ──writes──▶  runs/<run_name>/STOP_TRAINING.txt (graceful stop)
```

The GUI uses its current console Python interpreter (converting `pythonw.exe`
to its `python.exe` sibling when necessary).
Working directory: the repo root (`beamng-ml-abs-training-env`).

The trainer launches and manages BeamNG itself (attach-first-then-launch). The
GUI does **not** start BeamNG. Sim status is reported through `train.log`.

---

## 3. Entry point

```
python train_cosim.py --config <config.json>
```

One argument. Everything else is in the config file. Exit code 0 = clean finish
or graceful stop; non-zero = crash (surface `train.log`'s tail to the user).

---

## 4. config.json schema

The GUI writes this file. All fields shown with type and default. Unknown fields
are ignored; missing fields take the default.

```jsonc
{
  "run_name":    "PPO-01",     // string, REQUIRED. GUI owns naming (see §7).
  "seed":        390001,       // uint32 or "auto"; concrete value is recorded.
  "total_steps": 200000,       // int,   REQUIRED. PPO timesteps to train.
  "speeds":      [80],         // int[], mph brake-test speeds, sampled per episode.
  "runup_speed_factor": 4.0,   // float, 1..1000; 1 = real-time acceleration.
  "vehicle_pc":  "vehicles/etk800/Machine-Trainer-Boy-V2-MLABS.pc", // string
  "reward":      "v6.0",      // "v6.0" (default) | "v6.1" (v6 scoring, independent wheels) | "v7.0" | "v5.0"
  "wheel_mode":  "axle",      // optional: "axle" (2 releases, FL=FR/RL=RR) | "independent"
                              // (4 per-wheel releases). Default: "independent" for v6.1 and v7.0, else "axle".
  "device":      "cuda",       // "cuda" | "cpu" | null(auto). null => auto-pick.
  "headless":    true,         // bool. false shows the BeamNG window (slower).
  "port":        64280,        // int, BeamNG tech port.
  "stop_file":   "STOP_TRAINING.txt", // filename, relative to the run dir.
  "resume_checkpoint": null,   // optional paired PPO .zip; always into a NEW run.
  "resume_inherit_config": true, // GUI inherits parent training semantics.

  "diagnostics": {              // optional; automatic defaults shown.
    "enabled": true,
    "trace_every_steps": 10000, // sample one complete 100 Hz episode near each milestone
    "checkpoint_every_steps": 50000, // paired model + VecNormalize snapshots
    "saturation_margin": 0.01, // action <=.01 or >=.99 counts as saturated
    "near_lock_slip": 0.80,
    "lock_slip": 0.95,
    "lock_min_speed_ms": 5.0   // exclude near-zero-speed false lock readings
  },

  "ppo": {                     // all optional; defaults shown.
    "lr":            0.0001,   // float
    "n_steps":       8192,     // int, rollout length (single env: ~16 episodes per update)
    "batch_size":    512,      // int  (should divide n_steps cleanly)
    "n_epochs":      4,        // int (few epochs over a correlated single-env rollout)
    "gamma":         0.995,     // float
    "gae_lambda":    0.95,     // float
    "clip_range":    0.2,      // float
    "ent_coef":      0.0,      // float; bounded policy explores from its initial std
    "initial_release": 0.50,   // deterministic starting release, strictly 0..1 (0.5 = unbiased)
    "log_std_init":  0.0,      // latent Gaussian starting spread (SB3 default)
    "target_kl":     0.02,     // stop an update early once approx KL exceeds this; 0 disables
    "vf_coef":       0.5,      // float
    "max_grad_norm": 0.5,      // float
    "net_arch":      [256, 256, 256]  // int[], hidden layer widths (shared pi/vf)
  }
}
```

Recommended GUI form: **basic** (run_name, total_steps, speeds) always visible;
**advanced** (everything under `ppo`, plus device/headless/port) in a collapsible
section. `net_arch` is a list of layer widths — expose as an editable list or a
"N layers × W width" pair.

---

## 5. Output files (what the GUI reads)

All under `runs/<run_name>/`, created by the trainer at start.

```
runs/<run_name>/
  config.json          # snapshot of what was run
  run_state.json       # atomic latest status/heartbeat/exit/checkpoint snapshot
  run_events.jsonl     # append-only process and checkpoint event history
  train.log            # human-readable status + heartbeat (see §5.1)
  episode_log.csv      # one row per finished episode (see §5.2)
  checkpoints/         # paired model + VecNormalize saves at ~50k rollout boundaries
  diagnostics/
    rollout_metrics.csv # physical/action summary + PPO health each rollout
    episode_metrics.csv # detailed physical/action summary each episode
    sample_traces.csv.gz # sampled full 100 Hz action/slip/torque histories
    README.txt           # field semantics and threshold definitions
    config.json          # effective diagnostic settings
    sb3/progress.csv     # Stable-Baselines3's native training log
  final.zip            # final PPO model (written at end / on stop)
  vecnormalize.pkl     # obs normalization stats (paired with final.zip)
  final.pair.json      # commit record written after both final files validate
  fatal.log            # Python faulthandler output
  crash.log            # exception and traceback when Python catches a failure
  STOP_TRAINING.txt    # appears only if the GUI wrote it
```

### 5.1 train.log line formats

One line per event, plain text, append-only. Parse by the leading keyword:

```
START run=PPO-01 total_steps=200000 speeds=[80] pc=vehicles/... device=cuda reward=v6.0 reward_hash=...
MODEL device=cuda net_arch=[256, 256, 256] lr=0.0001
RUNUP_HANDOFF ep=1 factor=4 target_ms=37.2632 speed_ms=37.2634
RUNUP_REALTIME ep=1 ack=True speed_ms=37.2510
COSIM_FIRST_PACKET ep=1 speed_ms=38.7630 ground_ms=38.7600 fused_ms=38.7400 valid=True
COSIM_FIRST_MOVING_PACKET ep=1 speed_ms=38.7630 ground_ms=38.7600 fused_ms=38.7400
BRAKE_ACTIVATION ep=1 target_ms=38.7632 speed_ms=38.7630
HEARTBEAT step=1000 fps=182.4 ep=3 cur_g=0.912 cur_ws=18.30 last_avg_g=0.885 best_avg_g=0.901
DIAGNOSTICS step=2048 sat0_front=0.12 sat1_front=0.08 sat0_rear=0.10 sat1_rear=0.09 nearlock_front=0.18 nearlock_rear=0.07 osc_front_hz=4.9 osc_rear_hz=5.1 kl=0.008 clip=0.11 explained_var=0.42
CHECKPOINT step=51200 model=PPO-01_51200_steps.zip vecnormalize=PPO-01_51200_steps_vecnormalize.pkl
...
STOP file seen -> graceful stop at 42000 steps      # only if stopped early
DONE committed final.zip + vecnormalize.pkl status=completed
```

**HEARTBEAT** is the live sub-episode feed (~every 1000 steps):
- `step`        — timesteps done (numerator for progress; denominator = config `total_steps`)
- `fps`         — training steps/sec since last heartbeat
- `ep`          — episodes finished so far
- `cur_g`       — current braking g (live)
- `cur_ws`      — a current wheel speed, m/s (live)
- `last_avg_g`  — avg-g of the most recent finished stop
- `best_avg_g`  — best avg-g so far this run

Progress % = `step / total_steps * 100`. Run is finished when a `DONE` line
appears (or the process exits).

### 5.2 episode_log.csv columns

Header row then one row per finished episode. Column order:

```
episode, target_mph, start_speed_ms, steps, stop_time_s, stopping_dist_m,
avg_g, peak_g, yaw_abs_sum, outcome, reward_total, reward_undiscounted,
reward_discounted, reward_dense_g, reward_step_yaw, reward_terminal_g,
reward_terminal_clean_yaw_bonus, reward_terminal_accumulated_yaw,
reward_terminal_stability, reward_failure_base,
reward_failure_accumulated_yaw, wall_s
```

- `episode`         — 1-based index
- `target_mph`      — the brake-test speed for this episode
- `start_speed_ms`  — speed at brake onset, m/s
- `steps`           — env steps in the episode
- `stop_time_s`     — braking duration, s
- `stopping_dist_m` — **the benchmark**: 2 kHz metric stopping distance, m
- `avg_g`           — **the benchmark**: 2 kHz metric average braking g
- `peak_g`          — peak braking g seen
- `yaw_abs_sum`     — integrated |yaw rate| (straight-line stability; lower better)
- `outcome`         — `STOP` | `TIMEOUT` | `LOST_LINK`
- `reward_total`    — summed reward for the episode
- `wall_s`          — wall-clock seconds the episode took

### 5.3 automatic training diagnostics

Diagnostics are enabled by default and require no GUI interaction. The compact
`rollout_metrics.csv` is the primary plateau-analysis file. It records front and
rear sampled/applied/mode action distributions, SB3 action-clip deltas,
saturation at both limits, latent standard deviation, transformed entropy,
exploration distance, all four wheel-slip summaries,
near-lock/lock fractions and seconds, per-wheel brake-torque commands,
oscillation frequency/strength, an axle-bottleneck classification, and PPO's KL,
clip fraction, explained variance, policy loss, value loss, and learning rate.

`episode_metrics.csv` keeps the same physical/action measurements at episode
granularity, allowing rare good stops to be separated from mediocre averages.
`sample_traces.csv.gz` captures full 10 ms histories for one complete episode at
each configured milestone, so action timing can be compared directly with slip,
g, wheel speed, yaw, and torque. The torque fields are commands sent to co-sim,
not readings from an independent torque sensor.

---

## 6. Live monitor (the pop-out)

The monitor consumes two data sources:

- **CSV-primary** (episode granularity): plot `avg_g` and `reward_total` vs
  `episode`; show a table of recent episodes; show best/last avg_g. A minimal GUI
  can run on the CSV alone.
- **train.log HEARTBEAT** (sub-episode): live gauges for `cur_g`, `cur_ws`,
  `fps`, and a progress bar from `step / total_steps`. Smoother, optional.

Suggested "active run" panel (mirrors the old trainer's monitor): current
`target_mph`, live `cur_g`, live `cur_ws`, episodes done, best avg_g, progress %.
The two headline graphs are **avg-g per episode** and **reward per episode**.

Benchmark note for the user-facing display: `avg_g` and `stopping_dist_m` are the
**standardized 2 kHz brake metric** — the numbers that decide whether ML beats
DynamicABS. Treat them as the score; everything else is diagnostic.

---

## 7. Run naming (GUI owns it)

The GUI chooses `run_name`. The backend uses it verbatim as the run-dir name and
**refuses to start if `runs/<run_name>/` already exists** (exits non-zero with a
message on stderr and no run dir mutation). So the GUI must pick a unique name —
e.g. scan `runs/` for existing `PPO-NN` and pick the next, or timestamp-suffix.

---

## 8. Graceful stop (important)

To stop a run: **create the stop file** `runs/<run_name>/STOP_TRAINING.txt`
(any contents, even empty). Within ~10 steps the trainer exits cleanly, saving
`final.zip` + `vecnormalize.pkl`, and writes the `STOP` then `DONE` lines.

**Never kill the process** (no taskkill / SIGKILL). A hard kill skips the final
save, but the most recent committed periodic pair remains independently
validated. The GUI's stop button validates the recorded PID as a live trainer,
writes the marker, and shows "stopping…" until `DONE` appears or the process
exits. A stale or reused PID is not treated as an active trainer.

---

## 9. Calibration and deployment

- **Calibrate baselines** belongs to the residual backend and is disabled for
  co-sim v5/v6 runs.
- **Export to game** uses `cosim_axle_release_v2` (MachineTrainerBoy 27 values + 4 stock-ABS-speed slips + 4 applied brake torques, x 64 past frames = 2240 inputs) for new bounded co-sim
  runs. Older co-sim checkpoints lack that action stamp and are blocked. The
  28x16/two-action residual contract is also blocked until it has its own Lua
  controller; neither is ever guessed compatible with the four-action legacy
  controller.

---

## 10. Constraints / gotchas for the GUI author

- **One run at a time.** Don't offer concurrent runs; the sim port is single-use.
- Files are append-only text/CSV — tail them, don't hold exclusive locks.
- `train.log` and `episode_log.csv` may not exist for the first ~1-2 s after
  spawn (the trainer creates the run dir first). Poll until they appear.
- A run can end three ways: `DONE` line (finished or stopped), process exit with
  non-zero (crash — show `train.log` tail), or the user's stop file.
- `stopping_dist_m` / `avg_g` are 0 on a `TIMEOUT` or `LOST_LINK` row — filter
  those out of the benchmark graphs or they'll dip to zero.
