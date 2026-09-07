# Reference

Complete technical reference for the ML-ABS training environment. Covers the
action space, reward design, GUI contract, experiment workflow, measured control
ceiling, performance notes, the 1.19 g result, and the legacy residual backend.


## 1. Action space: release from pedal

The trained action is a per-axle **release** from the driver's brake pedal, not
the brake level itself. Zero action = full pedal slam = the known lockup baseline.
Training starts at the anti-lock boundary instead of discovering "brake hard" first.

The co-sim backend (`train_cosim.py`) uses `bounded_ppo.py`, an affine-tanh
distribution where every sampled and optimized action is a physical release in
[0, 1]. SB3 no longer learns on hidden out-of-range actions that were clipped only
at the environment boundary.

Wheel modes:

- **axle** (2 outputs): front and rear release, left equals right per axle.
- **independent** (4 outputs): one release per wheel.


## 2. Reward design

### 2.1 The problem with v5.0 (absolute anchors)

v5.0 uses one shape function `_terminal_g_shape(g)` for both the per-step and
terminal reward, with absolute anchors: -400 at 0.3 g, 0 at 0.5 g, +1000 at
1.05 g, then a gatekeeper at 1.06 g (+500 jump, quadratic 200000*d^2 above).

Measured 2026-08-29 (etk800, Machine-Trainer-Boy V2, smallgrid, 60 mph, dry):

| reference | arc avg_g (median of 3) | distance | v5.0 reward |
|---|---|---|---|
| slam / full lockup | **1.0315 g** | 33.0 m | **+966** |
| stock ABS | **1.1884 g** | 28.6 m | +5049 |
| ResidualABS best gated result | 1.043 g | . | +987 |

The gatekeeper sits below stock ABS (1.19 g), and only 0.03 g above lockup.
Locking the wheels scores +966. The best result the project produced (1.043 g)
scores within 2% of locking the wheels. The reward cannot distinguish them.

Re-anchored: slam = 0.000, stock = 1.000, best result = 0.073. The best result
captured 7% of the margin stock ABS has over locked wheels.

### 2.2 Reward normalization (v6.0+)

A **configuration** = (car .pc, surface grip, start speed, corner radius). For
each one, calibration measures two references with the standard 2 kHz metric:

- `slam_g`: zero-action full lockup (the measured floor).
- `stock_g`: the car's stock ABS, same trigger, same everything.

The shape keeps its curve and balance. Its landmarks become config-relative:

```
zero anchor  : stock_g
gatekeeper   : stock_g + margin  (default 0.03 g)
slope unit   : (stock_g - slam_g)
low anchor   : slam_g
```

Equivalently, feed the shape a normalized g:
`g_n = (g - slam_g) / (stock_g - slam_g)`, with anchors at g_n = 0 (slam),
1 (stock), 1 + margin_n (gatekeeper). The per-step reward calls the same
function, so per-step is fixed for free.

Fallback when a config has no calibration row: refuse to train.

### 2.3 The metric

`avg_g = (v_start^2 - v_end^2) / (2 * distance * gravity)`. Stopping distance
normalized by start speed, in speed-invariant units. Corners inflate g through
distance: the chord is shorter than the driven path. The arc-length channel
(`last_brake_avg_g_arc`) uses path length instead.

Rule: `avg_g_arc` is the metric for reward and calibration. Chord `avg_g` stays
a diagnostic column only. Straight-line runs produce identical values for both.

### 2.4 Reward presets

- **v5.0**: absolute anchors. Historical baseline only.
- **v6.0**: normalized, axle mode (2 releases).
- **v6.1**: normalized, independent wheels (4 releases).
- **v7.0**: terminal G + banded true-slip per-step cost, yaw guard.
- **v9.0**: dense term = negative distance integral (the signal IS the metric).
- **v11.0**: v9 but charges the distance integral only while `tel_brake_active`
  is set. Flat 3/s cost outside the scored window.

Every run stamps its reward hash + calibration hash into train.log and the run
directory.


## 3. GUI contract (co-sim backend, v1)

The GUI never imports the environment or touches a live BeamNG session.
It writes one JSON config, spawns the trainer, reads logs, and writes the
run-local graceful-stop marker.

### 3.1 Communication flow

```
GUI  writes   config.json
GUI  spawns   python train_cosim.py --config <path-to-config.json>
GUI  tails    runs/<run_name>/train.log        (live status + heartbeat)
GUI  tails    runs/<run_name>/episode_log.csv  (one row per finished stop)
GUI  writes   runs/<run_name>/STOP_TRAINING.txt (graceful stop)
```

The trainer launches and manages BeamNG itself. The GUI does not start BeamNG.
Working directory is the repo root.

### 3.2 config.json schema

```jsonc
{
  "run_name":    "PPO-01",
  "seed":        390001,
  "total_steps": 200000,
  "speeds":      [80],
  "runup_speed_factor": 4.0,
  "vehicle_pc":  "vehicles/etk800/Machine-Trainer-Boy-V2-MLABS.pc",
  "reward":      "v6.0",
  "wheel_mode":  "axle",
  "device":      "cuda",
  "headless":    true,
  "port":        64280,
  "stop_file":   "STOP_TRAINING.txt",
  "resume_checkpoint": null,
  "resume_inherit_config": true,

  "diagnostics": {
    "enabled": true,
    "trace_every_steps": 10000,
    "checkpoint_every_steps": 50000,
    "saturation_margin": 0.01,
    "near_lock_slip": 0.80,
    "lock_slip": 0.95,
    "lock_min_speed_ms": 5.0
  },

  "ppo": {
    "lr":            0.0001,
    "n_steps":       8192,
    "batch_size":    512,
    "n_epochs":      4,
    "gamma":         0.995,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.0,
    "initial_release": 0.50,
    "log_std_init":  0.0,
    "target_kl":     0.02,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "net_arch":      [256, 256, 256]
  }
}
```

All fields except `run_name` and `total_steps` have defaults. Unknown fields are
ignored. The `ppo` and `diagnostics` blocks are optional.

### 3.3 Output files

All under `runs/<run_name>/`, created by the trainer at start:

```
config.json           snapshot of what was run
run_state.json        atomic latest status, heartbeat, exit, checkpoint snapshot
run_events.jsonl      append-only process and checkpoint event history
train.log             human-readable status + heartbeat
episode_log.csv       one row per finished episode
checkpoints/          paired model + VecNormalize saves at ~50k rollout boundaries
diagnostics/          rollout metrics, episode metrics, sample traces, SB3 progress
final.zip             final PPO model
vecnormalize.pkl      obs normalization stats (paired with final.zip)
final.pair.json       commit record, written after both final files validate
STOP_TRAINING.txt     appears only if the GUI wrote it
```

### 3.4 train.log line formats

```
START run=PPO-01 total_steps=200000 speeds=[80] pc=... device=cuda reward=v6.0
MODEL device=cuda net_arch=[256, 256, 256] lr=0.0001
HEARTBEAT step=1000 fps=182.4 ep=3 cur_g=0.912 last_avg_g=0.885 best_avg_g=0.901
CHECKPOINT step=51200 model=PPO-01_51200_steps.zip vecnormalize=...
STOP file seen -> graceful stop at 42000 steps
DONE committed final.zip + vecnormalize.pkl status=completed
```

Progress = `step / total_steps * 100`. Run is finished when `DONE` appears or the
process exits.

### 3.5 episode_log.csv columns

```
episode, target_mph, start_speed_ms, steps, stop_time_s, stopping_dist_m,
avg_g, peak_g, yaw_abs_sum, outcome, reward_total, reward_undiscounted,
reward_discounted, reward_dense_g, reward_step_yaw, reward_terminal_g, ...
```

`avg_g` and `stopping_dist_m` are the standardized 2 kHz brake metric. These
are the score. Everything else is diagnostic.

### 3.6 Graceful stop

Create `runs/<run_name>/STOP_TRAINING.txt` (any contents, even empty). The
trainer exits cleanly within ~10 steps, saving `final.zip` + `vecnormalize.pkl`.

**Never kill the process.** A hard kill skips the final save. The most recent
periodic pair remains validated.

### 3.7 Constraints

- One run at a time. The sim port is single-use.
- `train.log` and `episode_log.csv` may not exist for the first ~1-2 seconds.
- `stopping_dist_m` and `avg_g` are 0 on a `TIMEOUT` or `LOST_LINK` row. Filter
  those out of graphs.
- Export to game is enabled only when the run metadata identifies a compatible
  controller contract. Older unbounded co-sim and 28x16 residual checkpoints
  are blocked.


## 4. Experiment workflow

The experiment record is append-only at `experiments/ledger.jsonl`. Existing run
directories and reward presets are evidence: never edit or replace them.

### 4.1 Frozen checkpoint evaluation

```powershell
py -3.10 evaluate_cosim.py --run-dir runs\PPO-38 --episodes 20 --reevaluate-top 3 --top-total-episodes 50
```

The evaluator deep-validates every model and VecNormalize pair before opening
BeamNG. Uses deterministic policy actions, frozen observation statistics, and
free-running physics. Results go to an immutable directory under `evaluations/`.
The selected pair is copied to `best_models/`. Failed stops count as zero G.

### 4.2 Crash-safe training and resume

Each run writes `run_state.json` and append-only `run_events.jsonl`. Periodic saves
consist of a model, its exact VecNormalize state, and a `.pair.json` commit manifest.
The manifest appears only after both files load, are finite, match the bounded
contract, and reach their final names.

Resume example:

```json
{
  "run_name": "PPO-39-resume",
  "total_steps": 1000000,
  "resume_checkpoint": "runs/PPO-39/checkpoints/PPO-39_501760_steps.zip",
  "resume_inherit_config": true,
  "seed": 390001
}
```

### 4.3 Decision discipline

Before every million-step run, append the hypothesis, single conceptual change,
expected result, and rejection criterion to the ledger. Evaluate frozen
checkpoints before deciding what changes next. A reward change is permitted only
after measured physical G and PPO-discounted return disagree.


## 5. Control ceiling measurements (2026-09-06)

Every PPO run through PPO-58 ceilinged near 1.05 g while stock ABS does 1.199 g on
the same car. These measurements determine whether the ceiling is a learning
failure or a property of the interface.

Car: `Machine-Trainer-Boy-V2-MLABS.pc` (etk800), smallgrid, dry, 80 mph, brake
torques FR/FL 3100 Nm, RR/RL 1700 Nm.

### 5.1 References

| controller | avg_g | source |
|---|---|---|
| full lockup (slam) | 0.979 | `calibration/etk800.json` |
| stock ABS | 1.199 | `calibration/etk800.json` |
| best PPO ever (PPO-51) | 1.063 | `runs/PPO-51` |
| goal | 1.180 | |

### 5.2 Constant release, 100 Hz

| release | avg_g | stopping distance |
|---:|---:|---:|
| 0.00 (full brake, locked) | 0.967 | 67.4 m |
| 0.10 | 0.964 | 67.6 m |
| 0.20 | 0.967 | 67.3 m |
| 0.30 | 1.007 | 64.7 m |
| **0.35** | **1.119** | **58.2 m** |
| 0.40 | 1.065 | 61.2 m |
| 0.45 | 1.002 | 65.0 m |
| 0.55 | 0.781 | 83.4 m |

A constant 0.35 release beat every policy this project had trained. The optimum is
roughly 0.05 wide. PPO-57's converged action distribution had a deterministic mean
of 0.277 and exploration std of 0.21, four times wider than the peak it needed.
That is why runs converged to ~1.05 g: noise smeared them across a sharp optimum.

### 5.3 Closed-loop slip control, 100 Hz

| controller | avg_g |
|---|---|
| bang-bang, target 0.12 | 0.909 |
| bang-bang, target 0.18 | 0.936 |
| bang-bang, target 0.24 | 0.963 |
| proportional, target 0.15, kp 3 | 1.098 |
| proportional, target 0.21, kp 3 | 1.069 |
| proportional, target 0.15, kp 6 | 0.999 |
| proportional, target 0.21, kp 6 | 0.968 |
| proportional, target 0.21, kp 10 | 0.973 |

At 100 Hz, closed-loop slip regulation is worse than the best constant. Gain makes
it worse still. Both are the signature of a delay-limited loop: by the time slip is
measured past target, the correction arrives a control period late.

### 5.4 Control rate

BeamNG's coupling decides the rate:

```
sendSkips = ceil(time3rdParty / physicsDt) - 1
```

`abs_env_cosim.DT` passes through as `time3rdParty`, physics runs at 2 kHz.
`DT = 0.01` = 100 Hz, `DT = 0.005` = 200 Hz, `DT = 0.0025` = 400 Hz. Stock ABS
and DynamicABS both run in-vehicle at the full 2 kHz.

### 5.5 Control rate sweep

| controller | 100 Hz | 200 Hz | 400 Hz |
|---|---:|---:|---:|
| constant 0.35 | 1.119 | 1.118 | 1.104 |
| proportional 0.21, kp 3 | 1.069 | 1.104 | |
| proportional 0.15, kp 3 | 1.098 | 1.132 | 1.144 |
| proportional 0.15, kp 6 | 0.999 | 1.132 | 1.173 |
| proportional 0.15, kp 10 | | | 1.128 |
| proportional 0.18, kp 6 | | | 1.151 |
| **proportional 0.12, kp 6** | | | **1.189** |

A three-line P regulator at 400 Hz stops at **1.189 g**, past the goal and within
noise of stock ABS (1.199). Control rate, not policy capacity, was gating
closed-loop braking.

The kp 6 regulator that was unstable at 100 Hz (0.999) matches the best at 200 Hz.
Doubling rate does nothing for the constant (correctly, it has no feedback to
delay) and lifts every closed-loop controller.

### 5.6 Conclusions

- The co-sim channel is not the ceiling. At 400 Hz a scripted P regulator reaches
  1.189 g through it.
- Exploration width is a first-class problem: the target is a narrow ridge.
- Control rate, not policy capacity, was the bottleneck for closed-loop controllers.


## 6. Performance notes

### 6.1 The frame limiter is load-bearing

Measured 2026-08-30 on BeamNG.tech 0.37.6.0, beamngpy 1.34.1, i5-12600K.

`step(N)` decrements `blocking.data` once per rendered frame
(`techCore.lua:559`), not per physics tick. With the limiter on, one frame
carries exactly one tick, making deterministic stepping meaningful. Uncapped,
frames outrun physics and a step returns on a frame where physics may not have
ticked.

```
limiter ON     702 steps -> 702 distinct controller ticks  1:1      PASS
limiter OFF   1500 steps ->  91 distinct controller ticks  16.5:1   FAIL
```

The apparent 45x speedup was step() skipping the work. **Do not remove the
frame limiter.** `sim_clock.py` keeps `uncap_frame_rate()` for reproducing the
experiment, and a test asserts the training path never calls it.

### 6.2 Throughput baselines

```
operation                       p50 ms   frames
bng.control.step(1)              31.14     2
bng.control.step(20)            327.85    21
vehicle.sensors.poll()           15.53     1
full env step (3 calls)          31.10     2
```

Deterministic: 13.1 steps/s. Free-running: 22.5 steps/s.

### 6.3 400 Hz trainer deadline

At 400 Hz the control period is 2.5 ms. The trainer spends 1.6 to 2.0 ms of it
in Python. Achieved g degrades monotonically with per-step Python cost:

| harness | per-step Python | avg_g, episode 2+ |
|---|---|---|
| probe, no learning | . | 1.187 to 1.196 |
| DummyVecEnv + VecNormalize | ~1.3 ms | 1.187 |
| model.learn(), no callbacks | 1.6 to 1.7 ms | 1.01 to 1.17 |
| model.learn() + diagnostics | 1.9 to 2.0 ms | 0.71 to 0.75 |
| full trainer | . | 0.60 to 0.64 |

This is a property of the trainer, not the policy. The identical weights hold
1.19 g indefinitely under lighter loops. Mitigations: train at 200 Hz (5 ms
budget), or disable per-step diagnostics at 400 Hz.

### 6.4 400 Hz episode log distortion

PPO-71 logged ~0.73 g while stopping at 1.19 g. This distortion is specific to
400 Hz. At 100 Hz the same overhead is a fifth of a 10 ms period and harmless.

Rule: judge any 400 Hz run by evaluating its checkpoint, never by its episode
log. Historical 100 Hz results stand.

### 6.5 Two framework bugs found (2026-09-06)

**1. The UDP link never drained.** The socket buffer fills during each optimizer
pause. The next episode consumed the backlog as if it were live, feeding up to
1618 stale packets (4 s at 400 Hz) into the 64-frame history at the start of
every episode. `CoSimLink.drain()` now empties the queue on reset.

**2. Resume ignored configured hyperparameters.** `PPO.load()` takes them from
the saved model, so a checkpoint built elsewhere imposes its original values.
Every fine-tune ran at 30x the configured learning rate while the log printed
the config's value. The resume path now assigns the run's configuration
explicitly.


## 7. Results: 1.19 g (2026-09-06)

| checkpoint | what it is | mean avg_g | 95% CI | >= 1.18 |
|---|---|---:|---|---:|
| **`runs/PPO-71/`** | RL-trained, 1.5M PPO steps, reward v11, 400 Hz | **1.1853** | [1.1825, 1.1882] | 13/16 |
| **`runs/BC-D3/`** | distilled starting point (no RL) | **1.1952** | [1.1931, 1.1973] | 16/16 |

### 7.1 What BC-D3 is

The same `UnitIntervalActorCriticPolicy` network `train_cosim.py` trains: 3x256,
four wheel-release outputs. Reads only the honest observation vector: wheel
speeds, stock-ABS-estimated slip, applied brake torques, IMU. No ground-truth
speed reaches the network at training time or at run time.

Weights come from supervised distillation. A scripted per-wheel proportional
slip regulator (`release = 6 * (slip - 0.10)`, running at 400 Hz, scoring
1.192 to 1.201 g) was recorded through the real env. The policy was fitted to
its actions. The teacher reads ground-truth slip. The student never does and
has to reproduce the behaviour from sensors alone.

Machine-Trainer-Boy, 80 mph, dry, straight. 16 consecutive stops:

```
mean 1.1952 g   sd 0.0043   95% CI [1.1931, 1.1973]
min 1.1861      max 1.2003      16/16 over the 1.180 goal
mean stopping distance 54.50 m
```

References: locked wheel 0.979 g, stock ABS 1.199 g, DynamicABS 1.205 g.

### 7.2 Distillation pipeline

| stage | pairs | avg_g in sim |
|---|---:|---|
| teacher (privileged slip) | . | 1.192 to 1.201 |
| BC iteration 0 | 23,676 | 0.62 to 0.83 (mean 0.72) |
| DAgger iteration 1 | 42,523 | 0.85 to 1.208 (mean 1.04) |
| DAgger iteration 2 | 63,836 | 0.99 to 1.130 (mean 1.04) |
| DAgger iteration 3 (perturbed) | . | 1.187 to 1.201 (mean 1.194) |

Perturbing the student during collection was the step that mattered.
Deterministic DAgger plateaued bimodal at ~1.04 g. Adding noise put the states
the policy fell off into the dataset with teacher labels.

### 7.3 Reproduce

```bash
# measure the checkpoint
python slip_ceiling_probe.py --reps 16 --dt 0.0025 --releases "" --targets "" \
    --policy runs/BC-D3

# rebuild from scratch: record the teacher
python slip_ceiling_probe.py --reps 8 --dt 0.0025 --releases "" --targets "" \
    --prop "0.10:6" --record teacher.npz

# distill
python distill_teacher.py --data teacher.npz --config .gui-configs/PPO-65.json \
    --out runs/BC-1 --epochs 80

# DAgger with the student perturbed
python slip_ceiling_probe.py --reps 14 --dt 0.0025 --releases "" --targets "" \
    --policy runs/BC-1 --policy-noise 0.04 --dagger "0.10:6" --record dagger.npz
```

Merge datasets, redistill, repeat.


## 8. Legacy residual backend

The residual backend (`train_residual.py`, `abs_env_residual.py`) uses the same
release-from-pedal action space but trains in-car via the `abs_env.py` base
environment. Supports both SAC and PPO.

### 8.1 Live gate results (2026-08-27, Machine-Trainer-Boy, smallgrid, 60 mph)

| Gate | Result | Verdict |
|---|---|---|
| Zero-action reproduces lockup baseline | 1.028 g mean | PASS |
| Scripted release beats slam mean | 1.043 g mean (+0.015 g) | PASS |

The release mechanism reaches the wheels. The margin is thin on dry asphalt,
where there is almost no headroom above a hard slam.

### 8.2 CLI examples

```bash
# short smoke test
python train_residual.py --algo sac --speeds "60" --pedal off \
    --total-steps 10000 --run-name smoke_sac

# real run with randomized pedal
python train_residual.py --algo sac --speeds "60,90,120" --pedal "0.4-1.0" \
    --total-steps 500000 --run-name residual_sac_r1

# PPO
python train_residual.py --algo ppo --speeds "60,120" --pedal "0.4-1.0" \
    --total-steps 500000 --run-name residual_ppo_r1

# graceful stop (never kill the process, can corrupt SAC replay buffer)
echo stop > STOP_TRAINING.txt
```

`--total-steps` is an absolute target, not an increment, when resuming. Resuming
a 200k checkpoint with `--total-steps 200000` does nothing. Pass 500000 to add
300k more steps on top of the existing 200k.


## 9. Calibration

`reference_runner.py` measures slam and stock baselines per configuration.
Teleport, accelerate, arm the 2 kHz slam latch, read `avg_g_arc`. Stores
median + spread to `calibration/<car>.json` keyed by (grip, speed, radius).

GUI: "Calibrate baselines" over the current training matrix. Cost: roughly
160 stops for 3 speeds x 3 grips x 3 radii x 2 refs x 3 reps. One-time per
car, cached.

The reward refuses to train a configuration that has no calibration row.


## 10. Corners

Single constant-radius corners only. `corner.py` provides arc geometry, heading
tracking, and steering seek math. The env computes `target_yaw_rate = v / R`
per step from current speed. The crash rule uses deviation from the arc's
tangent heading at the car's current position, not from the start heading.


## 11. Grip levels

`be:setGroundModel("ASPHALT", cdata)` changes static/sliding friction at runtime.
Grip is a per-episode setting (friction multiplier on the ground model, 1.0 = stock
asphalt), reset via `reloadGroundModels()`. Grip is not in the observation. The
model must infer it.
