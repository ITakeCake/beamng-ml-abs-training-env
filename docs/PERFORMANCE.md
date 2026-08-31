# Why training was slow, and what actually fixed it

Measured 2026-08-30 on BeamNG.tech 0.37.6.0 / beamngpy 1.34.1, i5-12600K.

## The finding, in one line

**BeamNG's frame limiter gated every beamngpy request. Removing it made a
simulation step 45x faster.**

```
limiter as found (fpsLimitEnabled=true,  background=false, cap=200)
    bng.control.step(1)   p50 = 31.14 ms   ->    32 Hz

limiter off      (fpsLimitEnabled=false, background=false, cap=2000)
    bng.control.step(1)   p50 =  0.69 ms   ->  1450 Hz
```

Both measured in one session, values read back from the engine to confirm they
applied. Applied automatically now: `sim_clock.uncap_frame_rate()`, called at
env startup in `abs_env_residual.__init__`.

## Two traps that cost most of the day

### 1. Editing settings.json does nothing

Changing `fpsLimitEnabled` in
`%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\settings\settings.json` while the
game was closed, then launching, measured **31.14 -> 31.03 ms**: no change at
all. The running process does not take the file's word for it.

**Only `settings.setValue(...)` at runtime works**, queued on the GameEngine VM:

```lua
settings.setValue('fpsLimitEnabled', false)
settings.setValue('fpsLimitBackgroundEnabled', false)
settings.setValue('fpsLimit', 2000)
```

Read the values back before believing a negative result -- that is what
distinguished the working fix from the broken one.

### 2. fpsLimitBackgroundEnabled matters as much as the main flag

A headless instance has no focused window, so it is a background window by any
usual test, and `fpsLimitBackground` defaults to **5 FPS**. Setting only
`fpsLimitEnabled` can leave the background limiter in charge.

## The mechanism (still true, and worth knowing)

`lua/ge/extensions/tech/techCore.lua`:

- **line 521** -- `M.onPreRender = function(dt)` is where the tech server
  services clients. Every beamngpy request waits for the next **rendered
  frame**, headless or not. `-gfx null` removes the drawing, not the loop.
- **line 559** -- while a step is blocking, `blocking.data` decrements **once
  per frame**, then `return`s without servicing anything else. So `step(N)`
  costs **N frames**, and no other request is answered mid-step.
- **line 590** -- `while tcom.checkMessages(M, clients) do end` drains every
  queued message in one frame, so non-blocking commands batch cheaply.

Confirmed by measurement before the fix: `step(20)` = 327.85 ms = 21 frames
(20 ticks + 1 for the call); `step(1)` = 2 frames; a full env step
(control + step + poll) = 2 frames.

## What was ruled out (do not re-litigate)

| suspect | verdict | evidence |
|---|---|---|
| Nagle / TCP | not it | beamngpy sets `TCP_NODELAY` and `settimeout(None)`; the recv is fully blocking, Python never sleeps |
| Windows timer granularity | not it | `NtQueryTimerResolution` reported the system already at **1.0 ms**, not 15.6 |
| CPU starvation / affinity | not it | BeamNG `Normal`->`AboveNormal`, trainer `High`->`Normal` and moved to the idle E-cores: **22.54 -> 22.51 steps/s**, no change |
| `step(count, wait=False)` pipelining | not it | 31.14 -> 31.09 ms. `vehicle.sensors.poll()` uses the *vehicle's own* connection (`techCore.lua:619`), so it is not in the queue drained on the step's completing frame |
| editing settings.json offline | not it | see above |

## Baseline numbers, for comparison later

Per-operation, deterministic, **before** the fix (`latency_probe.py`):

```
operation                       p50 ms   frames
bng.control.step(1)              31.14     2
bng.control.step(20)            327.85    21
vehicle.sensors.poll()           15.53     1
vehicle.control(brake=1)         15.51     1
vehicle.queue_lua_command        15.42     1
full env step (3 calls)          31.10     2
```

End-to-end training throughput, **before** the fix:

```
PPO-05  deterministic   13.9 steps/s   ~1084 steps/episode   ~80 s/episode
PPO-10  free-running    22.5 steps/s     ~50 steps/episode   ~2.2 s/episode
```

## What this retired

Before finding the limiter, the conclusion was that 64 Hz was a hard floor,
that 80 steps/s was unreachable over the socket, and that an in-Lua rollout
(policy runs in vlua, episode shipped to Python in one transfer) was the only
way past it. **None of that holds.** The ceiling was a setting.

The in-Lua rollout may still be worth building one day for the sim-to-deploy
argument -- training and deployment become literally the same loop -- but it is
no longer a performance necessity.

## Open question

The 45x is on an isolated `step(1)`. Real training also does torch inference,
obs assembly, reward and VecNormalize, none of which got faster. If the ~62 ms
of sim calls per PPO-05 step collapses to ~1.4 ms, the remaining ~10 ms of
Python work becomes the floor, which would land near 100 steps/s -- arithmetic,
not a measurement. **Run ~20k deterministic steps and read steps/s off the
monitor to find the real new ceiling.**

If deterministic now beats free-running, free-running is obsolete: deterministic
gives full 200 Hz decisions (~1200 per stop instead of ~49) and reproducibility.

## Tools

- `latency_probe.py` -- times each beamngpy operation separately. Attaches to a
  running instance if one is listening, else launches its own. Two instances
  cannot share a userpath (the launcher fails to rotate its log and dies).
- `fps_test.py` -- sets both limiter flags at runtime, reads them back through
  the BeamNG log, and times `step(1)` before and after in one session.
