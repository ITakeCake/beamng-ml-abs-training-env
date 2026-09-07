"""Free-running (non-deterministic) simulation clock for training.

Deterministic training advances physics one 5 ms tick per `bng.step(1)`, and
every one of those costs a TCP round trip. Measured on PPO-05: 62 ms of wall
clock to buy 5 ms of simulation, i.e. 1/14th of real time, with the CPU idle
between handshakes. The engine is not the bottleneck, the handshakes are.

The alternative is to stop stepping: let the engine free-run under
`be:setPhysicsSpeedFactor(N)` and have Python read state whenever it can. The
in-car controller already runs its own 200 Hz tick and actuates from the last
mailboxed action, so it does not care whether Python is keeping pace.

WHAT THIS TRADES
----------------
Deterministic: one policy decision per 5 ms of sim, exactly, forever.
Free-running:  one policy decision per round trip, whatever that costs today.

So the policy's effective decision rate falls as the speed factor rises, at
N=10 a six-second stop yields tens of decisions rather than ~1100. Episodes per
hour go up; decisions per episode go down. Which of those matters more is an
empirical question about this task, which is the point of putting it behind a
switch instead of guessing.

WHAT IT DOES NOT CHANGE
-----------------------
The brake metric. Onset is latched in Lua by armBrakeSlam at 2 kHz, and
last_brake_dist / last_brake_avg_g are accumulated in Lua at 2 kHz. Neither
depends on Python's cadence, so the standard metric stays valid and comparable
across modes.

`steps` and `stop_time_s` in the episode CSV do NOT stay comparable: they are
derived from step counts assuming DETERM_HZ, which no longer holds. Compare
runs by avg_g, not by step count.

The seam is a proxy rather than an edit: abs_env_incar is a protected parent
whose bng.step() calls carry the reward's timing semantics, so this stands in
front of the handle it calls instead of changing what it calls.
"""
import time

from abs_env import DETERM_HZ


class _ControlProxy:
    """`bng.control`, minus the pausing.

    The parent pauses before flipping to deterministic for the braking phase.
    Under free-running there is nothing to flip to and a pause would simply
    stop the world while Python sleeps against a clock that is no longer
    running, so pause() is dropped and everything else forwarded.
    """

    def __init__(self, control):
        self._control = control

    def pause(self):
        return None

    def __getattr__(self, name):
        return getattr(self._control, name)


class _SettingsProxy:
    """`bng.settings`, with set_deterministic() rerouted.

    The parent calls set_deterministic(DETERM_HZ) to enter the braking phase.
    Honouring that would undo the whole point, so it becomes "apply the speed
    factor" instead, the same transition (accel is over, measurement begins)
    expressed the way free-running mode expresses it.
    """

    def __init__(self, settings, clock):
        self._settings = settings
        self._clock = clock

    def set_deterministic(self, hz=None):
        self._clock.apply_speed_factor()
        return None

    def set_nondeterministic(self):
        # The parent calls this to start the acceleration run-up, which is timed
        # by polling airspeed against wall clock. Leaving the speed factor on
        # would run that loop N times faster than it can poll and blow straight
        # past the target speed, so the factor comes off here and goes back on
        # at set_deterministic() when the measured phase begins.
        self._clock.restore_realtime()
        return self._settings.set_nondeterministic()

    def __getattr__(self, name):
        return getattr(self._settings, name)


class FreeRunClock:
    """Stands in for the BeamNGpy handle so `bng.step(n)` waits instead of steps.

    Everything not named here forwards to the real handle, so callers that want
    scenarios, cameras or lua queues are unaffected.
    """

    def __init__(self, bng, speed_factor=1.0):
        self._bng = bng
        self.speed_factor = max(1.0, float(speed_factor))
        self.control = _ControlProxy(bng.control)
        self.settings = _SettingsProxy(bng.settings, self)
        self.sim_seconds = 0.0
        self.step_calls = 0

    #, the speed factor lives on the GameEngine Lua VM, where be: exists ----
    def apply_speed_factor(self):
        self._bng.control.queue_lua_command(
            f"be:setPhysicsSpeedFactor({self.speed_factor:g})")

    def restore_realtime(self):
        """Back to 1x free-running (0), NOT to stepped (-1).

        Leaving the engine at a speed factor would carry it into the next
        episode's acceleration run-up, which is timed by wall-clock polling and
        would overshoot the target speed.
        """
        self._bng.control.queue_lua_command("be:setPhysicsSpeedFactor(0)")

    def step(self, count=1):
        """Wait out the sim time the caller asked for, scaled by the factor.

        The sleep is a floor, not a guarantee: a round trip elsewhere in the
        caller's loop usually costs more than this, and at high speed factors
        the requested wait falls below the OS timer granularity. That is fine
      , the engine is running either way, and the caller's own latency is
        what actually paces the loop.
        """
        sim_s = float(count) / DETERM_HZ
        self.sim_seconds += sim_s
        self.step_calls += 1
        wall_s = sim_s / self.speed_factor
        if wall_s > 0.0:
            time.sleep(wall_s)
        return None

    def __getattr__(self, name):
        return getattr(self._bng, name)


def wrap(bng, deterministic=True, speed_factor=1.0):
    """Return the handle the env should use: the real one, or a free-run proxy."""
    if deterministic:
        return bng
    return FreeRunClock(bng, speed_factor=speed_factor)


# ---------------------------------------------------------------------------
# DO NOT UNCAP THE FRAME LIMITER. It is load-bearing.
# ---------------------------------------------------------------------------
# It looks like free speed and is not. BeamNG services beamngpy from
# onPreRender (techCore.lua:521) and step(N) decrements blocking.data once per
# RENDERED FRAME (line 559), not per physics tick. With the limiter on, one
# frame happens to carry exactly one tick, which is what makes deterministic
# stepping mean anything. Uncapped, frames outrun physics and step(1) returns on
# a frame where physics may not have ticked at all.
#
# Measured 2026-08-30, stage_probe.py, one episode each, nothing else differing:
#
#   limiter ON    702 python steps -> 702 distinct controller ticks   1:1
#                 stopped at step 701, dist=62.45 m, avg_g=0.9855, 13.1 steps/s
#                 VERDICT PASS
#
#   limiter OFF  1500 python steps ->  91 distinct controller ticks   16.5:1
#                 never stopped, dist=0.0, avg_g=0.0, "150 steps/s"
#                 VERDICT FAIL
#
# The apparent 45x (step(1) 31.14ms -> 0.69ms) was step() returning without
# doing the work, and PPO-12's 259 steps/s was the same illusion, which is
# exactly why every one of its episodes timed out with dist=0.0.
#
# Keep this function so the experiment is reproducible and nobody re-derives it
# from scratch, but nothing calls it. If you are tempted, run
# `python stage_probe.py --uncap on` and read the tick count first.
UNCAP_OK_MS = 5.0


def uncap_and_verify(bng, log=None):
    """Uncap with an acknowledged call, then measure what a step now costs.

    Two independent checks, because they can disagree and the disagreement is
    informative: the engine's own report of the settings, and the wall-clock
    cost of a step. The settings say what was applied; the timing says whether
    it mattered.
    """
    try:
        state = uncap_frame_rate(bng, verify=True)
    except Exception as e:
        if log:
            log.warning("frame limiter: uncap call failed (%s: %s)",
                        type(e).__name__, e)
        return None
    try:
        ms = measure_step_ms(bng)
    except Exception as e:
        if log:
            log.warning("frame limiter: settings=%s but could not time a step "
                        "(%s: %s)", state, type(e).__name__, e)
        return None
    if log:
        log.info("frame limiter: engine reports %s | step(1) = %.2f ms (%s)",
                 state, ms,
                 "uncapped" if ms < UNCAP_OK_MS else "STILL CAPPED")
    return ms


def measure_step_ms(bng, reps=15):
    """Median wall-clock ms of step(1). The honest check that the uncap worked.

    Reading the setting back proved unreliable (the queued log line never
    appeared in the instance's log), and the setting is not the point anyway --
    the per-step cost is. ~31 ms means the limiter is still in charge; ~1 ms or
    less means it is gone.
    """
    import time as _t
    for _ in range(3):
        bng.step(1)
    xs = []
    for _ in range(reps):
        t0 = _t.perf_counter()
        bng.step(1)
        xs.append((_t.perf_counter() - t0) * 1000.0)
    xs.sort()
    return xs[len(xs) // 2]


# ---------------------------------------------------------------------------
# The frame limiter, which is the whole ballgame.
# ---------------------------------------------------------------------------
# BeamNG services beamngpy requests from onPreRender (techCore.lua:521), so
# every call waits for the next frame and step(N) costs N frames. With the
# limiter on, frames arrive at ~64 Hz and a single step(1) measured 31.14 ms --
# 1/14th of real time, and the reason training crawled.
#
# Measured on 0.37.6.0, same session, read back to confirm it applied:
#
#     limiter as found (true|false|200)   step(1) p50 = 31.14 ms   32 Hz
#     limiter off      (false|false|2000) step(1) p50 =  0.69 ms 1450 Hz
#
# 45x. It MUST be set at runtime through settings.setValue: editing
# settings.json while the game is closed was measured to change nothing, so the
# running process does not take the file's word for it.
#
# fpsLimitBackgroundEnabled matters as much as the main flag, a headless
# instance has no focused window, so it is a background window by any usual
# test, and that limiter defaults to 5 FPS.
FPS_UNCAP_LUA = (
    "settings.setValue('fpsLimitEnabled', false); "
    "settings.setValue('fpsLimitBackgroundEnabled', false); "
    "settings.setValue('fpsLimit', 2000)"
)


def uncap_frame_rate(bng, verify=True):
    """Remove BeamNG's frame limiter and return what the engine says it now is.

    Uses queue_lua_command(response=True), which blocks for the engine's reply,
    rather than the fire-and-forget form. That distinction is not cosmetic: the
    async form created a real race, the env measured step(1) at 13.35 ms and
    logged STILL CAPPED while the command had only been queued, and the same
    run then reached 259 steps/s. An acknowledged call cannot report on a
    setting that has not been applied yet.

    Returns the engine's own "enabled|backgroundEnabled|limit" string, so a
    caller logs what is true rather than what was asked for.
    """
    if not verify:
        bng.control.queue_lua_command(FPS_UNCAP_LUA)
        return None
    chunk = FPS_UNCAP_LUA + (
        "; return tostring(settings.getValue('fpsLimitEnabled'))"
        "..'|'..tostring(settings.getValue('fpsLimitBackgroundEnabled'))"
        "..'|'..tostring(settings.getValue('fpsLimit'))")
    return bng.control.queue_lua_command(chunk, response=True)
