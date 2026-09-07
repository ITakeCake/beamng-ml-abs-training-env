# What the co-sim channel can actually brake at (measured, 2026-09-06)

Every PPO run through PPO-58 ceilinged near 1.05 g while stock ABS does 1.199 g on
the same car, and nobody had measured whether that was a *learning* failure or a
property of the interface. `slip_ceiling_probe.py` measures it directly: scripted
controllers, no policy, driven through the identical `env.step()` path and scored
with the game's own 2 kHz metric.

Car: `Machine-Trainer-Boy-V2-MLABS.pc` (etk800), smallgrid, dry, 80 mph, brake
torques FR/FL 3100 Nm, RR/RL 1700 Nm.

## References for this configuration

| controller | avg_g | source |
|---|---|---|
| full lockup (slam) | 0.979 | `calibration/etk800.json` |
| stock ABS | 1.199 | `calibration/etk800.json` |
| best PPO ever (PPO-51) | 1.063 | `runs/PPO-51` |
| goal | 1.180 | |

## Constant release, 100 Hz (one fixed action for the whole stop)

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

**A constant 0.35 release beats every policy this project has ever trained.** The
optimum is roughly 0.05 wide; PPO-57's converged action distribution had a
deterministic mean of 0.277 and an exploration std of 0.21, which is four times
wider than the peak it needed to sit on. That is why runs converge to ~1.05: the
noise smears them across a sharp optimum they never actually locate.

## Closed-loop slip control, 100 Hz

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

At 100 Hz, closed-loop slip regulation is *worse* than the best constant, and gain
makes it worse still. Both are the signature of a delay-limited loop: by the time
slip is measured past target the wheel is already deeper, and the correction
arrives a control period late.

## The control rate is one constant

BeamNG's own coupling decides the rate:

    sendSkips = ceil(time3rdParty / physicsDt) - 1     -- cosimulationCoupling.lua:817

`abs_env_cosim.DT` is passed straight through as `time3rdParty`, and physics runs
at 2 kHz, so `DT = 0.01` means the policy sees every 20th physics tick: 100 Hz.
Python's own compute is 0.2 ms per exchange (`DIAG TIMING ... outside=0.2ms`), so
the budget is nowhere near spent. `--dt 0.005` gives 200 Hz, `--dt 0.0025` gives
400 Hz. Stock ABS and DynamicABS both run in-vehicle at the full 2 kHz.

## What this rules in and out

- The channel is **not** the ceiling: at 400 Hz a scripted P regulator reaches
  1.189 g through it. There is no reason to abandon co-sim for the in-car loop.
- Reward noise was real and is addressed by v9.0 (dense term = the distance
  integral, so the signal is the metric rather than a noisy proxy for it).
- Exploration width is a first-class problem, not a detail: the target is a narrow
  ridge, and entropy was being paid to stay off it.
- Whether >1.12 g needs a faster control rate is measured in the 200/400 Hz rows
  below.

## Control rate sweep (same controllers, `--dt`)

| controller | 100 Hz | 200 Hz | 400 Hz |
|---|---:|---:|---:|
| constant 0.35 | 1.119 | 1.118 | 1.104 |
| proportional 0.21, kp 3 | 1.069 | 1.104 | |
| proportional 0.15, kp 3 | 1.098 | 1.132 | 1.144 |
| proportional 0.15, kp 6 | 0.999 | 1.132 | 1.173 |
| proportional 0.15, kp 10 | | | 1.128 |
| proportional 0.18, kp 6 | | | 1.151 |
| **proportional 0.12, kp 6** | | | **1.189** |

A three-line P regulator at 400 Hz stops in 54.8 m for **1.189 g** -- past the
1.180 g goal and within noise of stock ABS (1.199). Nothing about the co-sim
interface prevents ABS-grade braking; it was running 20x slower than the physics
it was trying to control.

Doubling the rate does nothing for the constant (correctly -- it has no feedback to
delay) and lifts every closed-loop controller. The kp 6 regulator that was unstable
at 100 Hz (0.999) matches the best at 200 Hz. That is the clean signature of a
delay-limited loop, and it means **control rate, not policy capacity, was gating
closed-loop braking**.

Practical ceiling on rate: Python spends 0.2 ms per exchange in the probe, but a
trained policy also pays a torch forward pass (~0.3-1 ms on CPU for the 2240-wide
observation). 200 Hz leaves a 5 ms budget and is safe; 400 Hz leaves 2.5 ms and
needs the inference time measured before a long run is committed to it.

## The approach phase was most of the reward (2026-09-06)

At 400 Hz, PPO-62 episode 10 ran 23.2 s of which only **3.7 s was the scored brake
event**. The env hands the car over ~4.5 m/s above the 80 mph trigger, and a policy
that is not braking yet coasts down on drag alone. Three consequences, all bad:

- ~80% of every rollout is collected outside the task being scored.
- v9 charges its distance integral over the whole episode, so the approach
  contributed roughly -760 against the ~-55 the measured stop is worth. The dense
  signal was mostly grading an unscored phase.
- With `MAX_STEPS` capped at 24 s, a 19 s approach plus a real stop exceeded the
  episode budget, so policies that could stop the car timed out anyway and ate the
  -600 failure penalty. PPO-62 peaked at 0.952 g on episode 10, took two timeouts,
  and fell to 0.416.

Fixes: `MAX_STEPS` is now ~40 s at any control rate, and reward **v11.0** charges
the distance integral only while `tel_brake_active` is set -- the vehicle Lua
clears it in the same block that latches `avg_g` -- with a flat 3/s cost outside
that window so the approach still cannot be dawdled through. The dense sum is then
exactly minus the scored stopping distance.

Gating on that channel is safe: `brakeState` needs `brakeInput > 0.05` to arm, and
the env pins `input.brake = 1.0` for the whole episode (the policy modulates
per-wheel torque, never the pedal), so the window opens on the speed crossing alone.

## Distilling the teacher into the policy (2026-09-06)

The scripted regulator proves the channel can do it, but the policy has to do it
from honest observations: wheel speeds, the stock-ABS speed estimate, applied
torques and IMU, stacked 64 frames deep. At 400 Hz that stack is only 160 ms, too
short to integrate ground speed from a known start, so the student is strictly
less informed than the teacher it is copying.

Teacher quality (`--prop 0.10:6`, 8 reps): 1.1915 to 1.2008 g, mean 1.196.

| stage | pairs | avg_g in sim |
|---|---:|---|
| teacher (privileged slip) | -- | 1.192 - 1.201 |
| BC iteration 0 | 23,676 | 0.62 - 0.83 (mean 0.72) |
| DAgger iteration 1 | 42,523 | 0.85 - **1.208** (mean 1.04) |
| DAgger iteration 2 | 63,836 | 0.99 - 1.130 (mean 1.04) |

Held-out action RMSE was ~0.05 release at every stage, and the teacher's action is
`6 * (slip - 0.10)`, so the student infers slip to about 0.008 ON THE TEACHER'S
STATE DISTRIBUTION. The information is present in the honest observation; the
failure is compounding drift into states the demonstrations never covered, which
is what DAgger exists to fix, and it moved the mean from 0.72 to 1.04 g in one
iteration.

Iteration 2 traded the peak for consistency (bad mode 0.80 -> 0.99, good mode
1.208 -> 1.130): plain regression pulls toward the mean of whatever states it is
given, and it optimizes agreement with the teacher rather than the metric. That is
the point to stop imitating and let RL optimize the thing actually being scored.

## Result: 1.1936 g mean, 8/8 stops over target (BC-D3, 2026-09-06)

DAgger iteration 3 changed one thing: the student was PERTURBED during collection
(0.04 gaussian release noise) instead of driven deterministically. Fragility, not
observability, was the remaining problem -- the policy was bimodal, and the states
that noise knocked it into were exactly the ones it had no teacher label for.

| policy | stops | mean avg_g | min | max | >= 1.18 |
|---|---:|---:|---:|---:|---:|
| teacher (privileged slip) | 8 | 1.196 | 1.1915 | 1.2008 | 8/8 |
| BC iteration 0 | 5 | 0.72 | 0.616 | 0.830 | 0/5 |
| DAgger 1 | 5 | 1.04 | 0.848 | 1.208 | 1/5 |
| DAgger 2 | 6 | 1.044 | 0.987 | 1.130 | 0/6 |
| **DAgger 3 (perturbed)** | **8** | **1.1936** | **1.1868** | **1.2008** | **8/8** |

Imitation error at BC-D3: MAE 0.0118 release, RMSE 0.0226, spread evenly across
slip bands. The one systematic flaw left is a +0.0069 bias in the 172k samples
where the teacher commands full brake -- the student under-brakes slightly exactly
where maximum braking is wanted.

What this policy is, precisely: the same `UnitIntervalActorCriticPolicy` network
train_cosim trains (3x256, four wheel-release outputs), reading ONLY the honest
observation vector -- wheel speeds, stock-ABS-estimated slip, applied torques,
IMU. No ground-truth speed reaches it. Its weights come from supervised
distillation of a privileged teacher, NOT from RL reward optimization. The number
is real and honestly obtained; it is not evidence that PPO's own learning found it.

## Two framework bugs found while fine-tuning (2026-09-06)

Both silently affected every co-sim run this project has done, not just tonight's.

**1. The UDP link never drained.** BeamNG transmits every control period whether or
not Python is reading, so the socket buffer fills during each optimizer pause and
the next episode consumes the backlog as if it were live. Measured up to **1618
stale packets** (4 s of stale data at 400 Hz) at the start of an episode. The
64-frame history was therefore built from the PREVIOUS episode's dying moments,
exactly when the opening braking decisions that dominate the metric are made.
`CoSimLink.drain()` now empties the queue on reset and logs what it discarded.
Symptom in the diagnostics: `COSIM_FIRST_PACKET ... speed_ms=0.3174 valid=False`
on every episode after the first.

**2. Resume ignored the configured hyperparameters.** `PPO.load()` takes them from
the SAVED model, so a checkpoint built anywhere else imposes whatever it was
constructed with -- SB3 defaults of lr 3e-4, gamma 0.99, lambda 0.95, ent_coef 0,
no target_kl. Every fine-tune ran at 30x the configured learning rate while the
MODEL log line printed the config's 1e-5, which is why it stayed hidden. The
resume path now assigns the run's configuration explicitly.

Still open: with identical weights, frozen observation statistics, a drained queue
and sampling proven harmless (5/5 stochastic reps at 1.1925-1.1946 g), the policy
commands `early_rel` 0.27 on episode 1 and ~0.53 from episode 2 onward -- and 0.53
is about what an untrained action head outputs. Something makes the observation
uninformative after the first episode of a TRAINING run specifically; the probe
runs 16 consecutive episodes with no such effect. `early_rel` (mean commanded
release over the opening 400 steps) is now logged per episode to chase it.

## Why fine-tuning destroyed a 1.195 g policy: the trainer misses its deadline

Bisected against the same BC-D3 checkpoint, every step measured in the sim:

| harness | per-step Python | avg_g, episode 2+ |
|---|---|---|
| probe, no learning | -- | 1.187 - 1.196 |
| DummyVecEnv + VecNormalize, deterministic | ~1.3 ms | 1.187, 1.187 |
| the same, sampled as PPO samples | ~1.3 ms | 1.196, 1.192 |
| `model.learn()`, no callbacks | 1.6 - 1.7 ms | 1.10, 1.01, 1.17, 1.09 |
| `model.learn()` + TrainingDiagnosticsCallback | 1.9 - 2.0 ms | 0.75, 0.71, 0.75 |
| the full trainer (all callbacks) | -- | 0.60, 0.60, 0.64 |

Innocent, each disproved by measurement rather than argument: control rate, reward
shape, stale packets (fixed separately), observation-normalization drift (frozen,
still failed), additive action noise (1.145 - 1.193 at twice the trained noise),
stochastic sampling (5/5 at 1.19), and the vec wrapper itself.

**The control period at 400 Hz is 2.5 ms and the trainer spends 1.6 - 2.0 ms of it
in Python.** Every callback added per step eats the remaining margin, the reply
misses BeamNG's window, and the policy's commands land late -- which is fatal for a
controller whose whole job is reacting inside a wheel-lock transient. It degrades
smoothly with overhead, which is exactly the ordering above, and it explains why
episode 1 always looked fine: the queue is drained at reset, so the lag has to
rebuild each episode.

This is a property of the trainer, not of PPO or of the policy: the identical
weights sustain 1.19 g indefinitely when driven by a loop that answers in time.
Mitigations: run training with `diagnostics.enabled = false`, or train at 200 Hz
(5 ms budget) accepting that a policy distilled at 400 Hz would need redistilling,
or move the per-step diagnostic work off the control thread.

### Correction and current state of that hypothesis

Reducing per-step Python cost (torch to one thread, diagnostics off, a drain after
each optimizer stall) moved `outside` from 1.8 ms to 1.4 ms and the achieved g from
0.78-0.85 to 0.84-0.90. The direction supports a latency effect.

But the same run reports `recv_wait` of 1.2-1.3 ms, meaning the loop reaches the
socket BEFORE BeamNG has sent the next packet -- it is early, not late -- and the
period settles at 2.7 ms rather than the nominal 2.5 ms, so the sim, not Python, is
pacing it. A plain missed-deadline story does not fit that. What is established:

- Per-step Python cost and achieved g are monotonically related across five
  harnesses (1.3 ms -> 1.19 g, 1.6 ms -> 1.09, 1.9 ms -> 0.74, full trainer -> 0.60).
- The identical weights hold 1.187-1.196 g indefinitely under the lighter loops,
  so nothing is wrong with the policy.
- Episode 1 is always clean and degradation appears from episode 2, in every
  trainer variant, including before any gradient update.

`wrapper_bisect.py` reproduces the whole ladder in a few minutes and is the place to
continue: the remaining suspects are what the heavier loop changes about the timing
of when a command lands within the physics step, and whatever differs about episode
1 that the drain did not explain.

## The trainer's episode log understates g AT 400 Hz -- and the fine-tune worked

PPO-71 ran the full 1.5M steps from the distilled checkpoint and its episode log
settled around 0.73 g, which read as RL steadily destroying a 1.195 g policy. It
was not. Driven by the light probe loop, the SAME final weights stop at:

    mean 1.1853 g   sd 0.0058   95% CI [1.1825, 1.1882]
    min 1.1752      max 1.1941      13/16 stops over 1.180      54.95 m

So the fine-tune preserved the behaviour and the training-time numbers were the
thing that was wrong. Scope of that, measured rather than assumed: PPO-57 (trained
at 100 Hz, logged best 1.0487) re-measured in the same light loop scores 0.962 to
1.021 g -- no better than its log. **The distortion is specific to 400 Hz.** At
2.5 ms per control period, 1.6-2.0 ms of per-step Python is most of the budget; at
100 Hz the identical overhead is a fifth of a 10 ms period and harmless.

Consequences worth keeping:
- Historical 100 Hz results stand. PPO-58's collapse last night was real.
- Any 400 Hz run must be judged by evaluating its checkpoint, never by its episode
  log. Five runs were killed on that log before this was understood.
- The fix for the log itself is to make the 400 Hz control loop cheap enough that
  the measurement stops depending on the observer.
