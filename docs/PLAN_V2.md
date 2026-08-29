# Plan v2 — Reward normalization, turns, grip, user-editable reward

Follows PLAN.md (steps 1-8 complete 2026-08-29). Goal: make training and
evaluation fair across cars, surfaces, speeds and corners, and let users tune
the reward within guard rails without losing comparability to the default.

## 0. What v5.0 actually is (read 2026-08-29, abs_env.py:50-102)

One shape function `_terminal_g_shape(g)` drives BOTH the per-step reward
(`PER_STEP_K * shape(g_step)`) and the terminal reward (`shape(avg_g)`), with
absolute anchors: -400 @ 0.3 g, 0 @ 0.5 g, +1000 @ 1.05 g, then a "gatekeeper"
at 1.06 g (+500 jump, quadratic 200000*d^2 above). Yaw: per-step Gaussian bonus
on yaw ERROR vs `target_yaw_rate` (currently 0), terminal +300 if the yaw-error
integral stays under 0.1 rad, catastrophic backstop -5000*excess^2, crash -2000
at 90 deg heading deviation. Balanced ~1000/1000/1000 per-step/terminal/yaw.

The gatekeeper at 1.06 g is "beat the etk800 dry-asphalt slam floor (1.03 g) by
0.03". The whole curve is hand-drawn around ONE measured number. On ice a good
stop (~0.35 g) scores -400. Adding an offset can't fix it; the anchors have to
move per configuration.

## 1. The metric: arc-length avg_g, and why it's not "g vs distance"

`avg_g = (v_start^2 - v_end^2) / (2 * distance * gravity)` -- stopping distance
normalized by start speed. Same information, speed-invariant units. Corners fool
it only through `distance`: the chord (straight line start->stop) is shorter
than the driven path, inflating g for a car that cuts the curve. The arc-length
channel (`last_brake_avg_g_arc`, shipped in abstelemetry.lua) uses path length.

Rule: `avg_g_arc` is THE metric for reward and calibration. Chord `avg_g` stays a
diagnostic column only. Straight-line runs give identical values for both.

## 2. Reward normalization = re-anchor v5.0's landmarks per config

A **configuration** = (car .pc, surface grip, start speed, corner radius). For
each one, calibration measures two references with the standard metric:

- `slam_g`  : zero-action full lockup (the measured floor -- not an ABS, not
  claimed perfect, just physics as observed)
- `stock_g` : the car's stock ABS part, same trigger, same everything

No ceiling reference. BeamNG is emergent; no ABS mode (arcade included) is
trusted as a physical limit. Stock is a competitor used as a ruler, nothing more.

The shape keeps its curve and balance; its landmarks become config-relative:

```
zero anchor  : stock_g                          (was 0.5 g)
gatekeeper   : stock_g + margin                 (was 1.06 g; margin default 0.03 g)
slope unit   : (stock_g - slam_g)               (ramps scale with the room stock had)
low anchor   : slam_g  -> the old -400 point    (worse than lockup = strongly negative)
```

Equivalently: feed the shape a normalized g, `g_n = (g - slam_g)/(stock_g - slam_g)`,
with anchors at g_n = 0 (slam), 1 (stock), 1 + margin_n (gatekeeper). Because the
per-step reward calls the same function, per-step is fixed for free -- no
separate scaling scheme. Score is unbounded above: nobody knows the limit.

Zero-point sanity: at stock's performance the terminal reward is ~0; the +500
jump fires only when the model genuinely beats stock on that config.

Fallback when a config has no calibration row: refuse to train (loud), never
silently fall back to the dry-asphalt anchors. Every run is stamped with the
hash of the calibration table it trained against.

## 3. Calibration: a reference runner, built into the GUI

The in-car env REQUIRES the ML controller to engage before an episode counts,
so it cannot measure a stock-ABS car. New `reference_runner.py` (most of
baseline_probe.py minus the controller handshake): teleport, accelerate, arm
the 2 kHz slam latch, read `avg_g_arc` from the episode log. Runs slam and
stock (stock = a .pc with the car's stock ABS part) per config, N reps, stores
median + spread to `calibration/<car>.json` keyed by (grip, speed, radius).

GUI: "Calibrate baselines" over the current training matrix. Cost: 3 speeds x
3 grips x 3 radii x 2 refs x 3 reps ~= 160 stops ~= 2 h, one-time per car,
cached. Exact-config keys only in v2 (no interpolation between rows).

First task of this plan: stock etk800 straight-line 60 mph on Machine-Trainer-Boy
-- the number the whole reward is now anchored on, and one we do not have yet.

## 4. Turns: already plumbed, two fixes required

v5.0's yaw terms are all computed on yaw ERROR vs `target_yaw_rate`, so a
constant-radius corner is a target change, not a reward rewrite:

- `target_yaw_rate = v / R`, **recomputed every step from current speed** -- as
  the car slows, v/R shrinks; a fixed value set at brake onset would demand a
  yaw rate the car can't and shouldn't hold. Sign from turn direction.
- Crash rule becomes deviation from the **arc's tangent heading at the car's
  current position along the arc**, not from the start heading -- a correctly
  driven 90 deg corner legitimately rotates the car 90 deg. Threshold stays 90
  deg off the tangent.
- Terminal yaw bonus threshold (0.1 rad of integrated error) is re-validated on
  a corner: integrated error over a longer, slower stop may need a per-config
  scale. Measure before changing.

Scope: single constant-radius corner only. Real ABS certification (ISO 21994,
ECE R13) tests straight-line and braking-in-a-turn at one radius; S-curves test
the driver, not the ABS. Multi-faceted turns: not planned.

Path reference: arc from the brake-onset position along the initial heading.
Lateral deviation from it is logged as a diagnostic, not rewarded in v2 -- yaw
tracking + arc-g already cover it; add only if models learn to run wide.

## 5. Grip levels

No maps needed: `be:setGroundModel("ASPHALT", cdata)` changes static/sliding
friction at runtime (verified, BEAMNG skill notes). Grip is a per-episode
setting: a friction multiplier on the map's ground model (1.0 = stock asphalt),
reset via `reloadGroundModels()`. Grip is NOT in the observation -- the model
must infer it, same as reality. Applies identically to straight and corner
configs.

## 6. User-editable reward: presets, sliders, parity

Move the reward out of the protected abs_env.py into a `reward_spec.py` module
the residual env owns. Non-negotiable: **the v5.0 preset must reproduce the
protected file's numbers byte-for-byte** on a recorded episode (parity test in
the suite). Only after that passes does anything become editable.

`RewardSpec` fields: per-step gain, terminal shape (ramp slopes, gatekeeper
margin, quadratic k), yaw bonus/threshold/backstop, crash penalty, and
`normalize: bool` (v5.0 preset = False, absolute anchors; "v6 normalized"
preset = True, section 2). Sliders in the GUI with bounded ranges and **Reset
to default**. Every run stamps its reward hash + calibration hash into
train.log and the run dir; the Output tab shows a "custom reward" badge so a
result can never be mistaken for a default-reward result.

Alternative terminal signals (arc-g, chord-g, stopping distance) are a selector
on the terminal term -- same slot, no new machinery.

## 7. Inclines (future)

Terrain-generator heightmap pipeline is proven (BEAMNG skill), .tech-only.
Deferred; nothing above depends on it.

## Build order

1. Reference runner + stock/slam straight-line calibration (etk800, 60 mph) --
   the missing number, and the machinery for every later row.
2. reward_spec.py extraction + v5.0 byte-parity test (nothing editable yet).
3. Normalized preset (section 2) wired to calibration rows; refuse-if-missing.
4. Grip levels (cheapest new dimension; the pedal-gate result says there is
   headroom below dry asphalt).
5. Single-radius corner: v/R target per step, tangent-based crash rule,
   calibration rows for corners.
6. Sliders / presets UI + run stamping (last: easy once the plumbing exists).
7. Inclines, someday.
