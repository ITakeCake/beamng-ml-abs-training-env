# The frame limiter: a 45x speedup that was not real

Measured 2026-08-30 on BeamNG.tech 0.37.6.0 / beamngpy 1.34.1, i5-12600K.

## RETRACTED

An earlier version of this file reported that removing BeamNG's frame limiter
made stepping 45x faster. **It does not. It makes `step()` return without
advancing physics.** The corrected finding is below; the timing measurements
that led to the wrong conclusion are kept because they are accurate, and
because the trap is easy to fall into twice.

## The corrected finding

`step(N)` decrements `blocking.data` once per **rendered frame**
(`techCore.lua:559`), not per physics tick. With the limiter on, one frame
carries exactly one tick -- which is what makes deterministic stepping mean
anything at all. Uncapped, frames outrun physics and a step returns on a frame
where physics may not have ticked.

One episode each, `stage_probe.py`, nothing else differing:

```
limiter ON  (control)    702 python steps -> 702 distinct controller ticks  1:1
                         stopped at step 701, dist=62.45 m, avg_g=0.9855
                         13.1 steps/s                                  PASS

limiter OFF             1500 python steps ->  91 distinct controller ticks  16.5:1
                         never stopped, dist=0.0 m, avg_g=0.0
                         "150 steps/s"                                 FAIL
```

The apparent speedup was step() skipping the work. PPO-12's 259 steps/s was the
same illusion, which is exactly why every one of its episodes ended TIMEOUT with
`dist=0.0` while `peak_g` read 1.4-1.6: the car really was braking, and the
simulation was barely advancing.

**The frame limiter is load-bearing. Do not remove it.** `sim_clock.py` keeps
`uncap_frame_rate()` for reproducing the experiment, and a test asserts the
training path never calls it.

## How the wrong conclusion survived three checks

Worth recording, because each check looked like confirmation:

1. `step(1)` really did drop 31.14 ms -> 0.69 ms. True, and meaningless: it was
   timing a call that no longer did anything.
2. Editing `settings.json` offline changed nothing (31.14 -> 31.03 ms), which
   looked like "the fix did not apply" rather than "the fix is wrong".
3. PPO-12 hit 259 steps/s. Throughput went up exactly as predicted -- while
   every episode scored zero.

What finally settled it was counting **distinct `mlabs_tickseq` values per
python step**. Throughput cannot distinguish real work from skipped work; the
tick correspondence can. Prefer that check to any timing number.

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

## What still stands after the retraction

The earlier conclusion was that ~64 Hz is a hard floor, that 80 steps/s is
unreachable over the socket, and that the only way past it is to stop crossing
the socket per decision. **That conclusion survives.** The limiter looked like a
way around it and was not; it was a way to skip the work instead.

So the real options are unchanged:

1. **Deterministic, limiter on** -- 13.1 steps/s measured, 1:1 tick fidelity,
   full 200 Hz decisions, reproducible. Correct, and slow.
2. **Free-running** -- 22.5 steps/s, no `step()` at all so no frame gate, but
   the policy decides once per round trip (~49 decisions per stop instead of
   ~1200). Correct, faster, coarser.
3. **In-Lua rollout** -- the policy runs in vlua at 200 Hz and the episode ships
   to Python in one transfer. Still the only design that gets full decision rate
   AND speed, and now again the only route past the frame gate.

## Open question

Nothing measured today improved throughput. Deterministic is 13.1 steps/s and
free-running is 22.5 steps/s, both as they were before any of this.

Before building anything large, confirm free-running has the tick fidelity that
uncapping turned out to lack: run `stage_probe.py` against a free-running env
and check that distinct controller ticks track the sim time actually consumed.
Free-running never calls `step()`, so it should not have this failure mode --
but that is reasoning, not a measurement, and reasoning is what went wrong here.

## Tools

- `latency_probe.py` -- times each beamngpy operation separately. Attaches to a
  running instance if one is listening, else launches its own. Two instances
  cannot share a userpath (the launcher fails to rotate its log and dies).
- `fps_test.py` -- sets both limiter flags at runtime, reads them back through
  the BeamNG log, and times `step(1)` before and after in one session. Kept as
  the reproduction of the wrong result: it shows 31.14 -> 0.69 ms and proves
  nothing about whether the simulation advanced.
- `stage_probe.py` -- ONE episode, every stage of the brake event logged in
  order, plus the distinct-tick count that actually settled this. Run it with
  `--uncap off` (control) and `--uncap on` to reproduce both arms.
