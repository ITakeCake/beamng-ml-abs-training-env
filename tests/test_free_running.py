"""Free-running (non-deterministic) training mode.

Stepping the simulation costs three TCP round trips per 5 ms of simulated time
-- 62 ms of wall clock, measured on PPO-05, with the CPU idle for most of it.
Free-running removes the stepping; these tests pin down what that must and must
not change, and that the default path is untouched.
"""
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------ the log volume
def test_beamngpy_chatter_is_silenced():
    """beamngpy logs one INFO line per bng.step(). At FRAME_SKIP=1 that was
    475,355 lines and 37 MB for a single run, burying the ~450 lines that say
    what actually happened."""
    import residual_log
    residual_log.quiet_beamngpy()
    assert logging.getLogger("beamngpy").level == logging.WARNING
    assert logging.getLogger("beamngpy.BeamNGpy").level == logging.WARNING


def test_silencing_keeps_real_problems_visible():
    """WARNING, not CRITICAL or disabled: a dropped connection must still be
    findable in the log."""
    import residual_log
    residual_log.quiet_beamngpy()
    assert logging.getLogger("beamngpy").isEnabledFor(logging.WARNING)
    assert not logging.getLogger("beamngpy").isEnabledFor(logging.INFO)


def test_the_trainers_own_logging_is_untouched():
    """Only the named beamngpy loggers are raised -- silencing the root would
    take the episode lines with it."""
    import residual_log
    residual_log.quiet_beamngpy()
    assert logging.getLogger("residual.env").isEnabledFor(logging.INFO)


# ---------------------------------------------------------------- the clock
class _FakeControl:
    def __init__(self):
        self.lua = []
        self.paused = 0
        self.resumed = 0

    def queue_lua_command(self, cmd):
        self.lua.append(cmd)

    def pause(self):
        self.paused += 1

    def resume(self):
        self.resumed += 1


class _FakeSettings:
    def __init__(self):
        self.determ_hz = None
        self.nondeterm = 0

    def set_deterministic(self, hz):
        self.determ_hz = hz

    def set_nondeterministic(self):
        self.nondeterm += 1


class _FakeBng:
    def __init__(self):
        self.control = _FakeControl()
        self.settings = _FakeSettings()
        self.stepped = []
        self.scenario = "sentinel"

    def step(self, n):
        self.stepped.append(n)


def test_deterministic_returns_the_handle_untouched():
    """The default path must be byte-identical to every run before this switch
    existed, so it cannot be a wrapper that merely forwards."""
    import sim_clock
    bng = _FakeBng()
    assert sim_clock.wrap(bng, deterministic=True) is bng


def test_free_running_never_steps_the_simulation():
    import sim_clock
    bng = _FakeBng()
    clock = sim_clock.wrap(bng, deterministic=False, speed_factor=10)
    clock.step(1)
    clock.step(20)
    assert bng.stepped == []


def test_step_accounts_the_sim_time_it_was_asked_for():
    """The parent's step counts still describe intended sim time even though
    nothing steps -- that is what makes the episode's timing legible."""
    import sim_clock
    clock = sim_clock.FreeRunClock(_FakeBng(), speed_factor=1000)
    clock.step(200)                       # 200 ticks @200Hz == 1.0 s of sim
    assert clock.sim_seconds == pytest.approx(1.0)


def test_a_high_factor_does_not_sleep_a_full_second():
    """The wait is sim time divided by the factor; at 1000x a one-second step
    must not block for one second."""
    import time
    import sim_clock
    clock = sim_clock.FreeRunClock(_FakeBng(), speed_factor=1000)
    t0 = time.monotonic()
    clock.step(200)
    assert time.monotonic() - t0 < 0.5


def test_the_speed_factor_goes_on_when_the_measured_phase_starts():
    """The parent marks the start of braking by calling set_deterministic();
    free-running mode has to hear that as 'apply the factor'."""
    import sim_clock
    bng = _FakeBng()
    clock = sim_clock.wrap(bng, deterministic=False, speed_factor=10)
    clock.settings.set_deterministic(200)
    assert "be:setPhysicsSpeedFactor(10)" in bng.control.lua
    assert bng.settings.determ_hz is None      # and never actually stepped


def test_the_factor_comes_off_for_the_acceleration_run_up():
    """The run-up polls airspeed against wall clock. Left at 10x it would
    overshoot the target speed before the next poll could see it."""
    import sim_clock
    bng = _FakeBng()
    clock = sim_clock.wrap(bng, deterministic=False, speed_factor=10)
    clock.settings.set_deterministic(200)
    bng.control.lua.clear()
    clock.settings.set_nondeterministic()
    assert "be:setPhysicsSpeedFactor(0)" in bng.control.lua


def test_pausing_is_dropped_while_free_running():
    """A pause would stop the world while step() sleeps against a clock that is
    no longer running -- the episode would simply hang."""
    import sim_clock
    bng = _FakeBng()
    clock = sim_clock.wrap(bng, deterministic=False, speed_factor=4)
    clock.control.pause()
    assert bng.control.paused == 0


def test_resume_still_reaches_the_engine():
    import sim_clock
    bng = _FakeBng()
    clock = sim_clock.wrap(bng, deterministic=False, speed_factor=4)
    clock.control.resume()
    assert bng.control.resumed == 1


def test_everything_else_forwards():
    """The env uses this handle for scenarios and lua queues too."""
    import sim_clock
    bng = _FakeBng()
    clock = sim_clock.wrap(bng, deterministic=False)
    assert clock.scenario == "sentinel"


def test_a_factor_below_one_cannot_slow_the_engine_down():
    import sim_clock
    assert sim_clock.FreeRunClock(_FakeBng(), speed_factor=0.1).speed_factor == 1.0


# ------------------------------------------------------------- the command
from gui_cmd import build_cmd, validate_settings


def _settings(**over):
    s = dict(algo="ppo", speeds="80", pedal_random=False, pedal_spec="0.4-1.0",
             grip="off", corner="straight", total_steps="1000", run_name="r",
             lr="2e-4", n_steps="2048", batch_size="1028", n_epochs="10",
             clip_range="0.2", gae_lambda="0.95", ent_coef="0.005")
    s.update(over)
    return s


def test_deterministic_passes_no_flags_at_all():
    """Absence is the default, so a settings file written before this switch
    existed still builds the command it always built."""
    cmd = build_cmd(_settings())
    assert "--no-deterministic" not in cmd
    assert "--train-speed-factor" not in cmd


def test_a_missing_key_reads_as_deterministic():
    s = _settings()
    s.pop("deterministic", None)
    assert "--no-deterministic" not in build_cmd(s)


def test_turning_it_off_carries_the_speed_factor():
    cmd = build_cmd(_settings(deterministic=False, train_speed_factor="10"))
    assert "--no-deterministic" in cmd
    assert cmd[cmd.index("--train-speed-factor") + 1] == "10"


def test_a_speed_factor_is_ignored_while_deterministic():
    """Stepping has no clock to multiply, so the number must not reach the
    trainer and imply that it did something."""
    cmd = build_cmd(_settings(deterministic=True, train_speed_factor="25"))
    assert "--train-speed-factor" not in cmd


# ------------------------------------------------------------- validation
def test_a_nonnumeric_speed_is_refused():
    assert any("number" in p for p in
               validate_settings(_settings(deterministic=False,
                                           train_speed_factor="fast")))


def test_slower_than_real_time_is_refused():
    assert any("real time" in p for p in
               validate_settings(_settings(deterministic=False,
                                           train_speed_factor="0.5")))


def test_an_absurd_factor_is_refused_with_the_measured_reason():
    problems = validate_settings(_settings(deterministic=False,
                                           train_speed_factor="500"))
    assert any("saturated" in p for p in problems)


def test_a_sane_factor_passes():
    assert validate_settings(_settings(deterministic=False,
                                       train_speed_factor="10")) == []


def test_a_bad_factor_is_not_checked_when_deterministic():
    """It is inert there, so refusing to start over it would be noise."""
    assert validate_settings(_settings(deterministic=True,
                                       train_speed_factor="nonsense")) == []


# ------------------------------------------------------------------ trainer
def test_the_trainer_accepts_both_flags():
    import train_residual
    p = train_residual.build_parser() if hasattr(train_residual, "build_parser") else None
    if p is None:
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "train_residual.py"),
            encoding="utf-8").read()
        assert '"--no-deterministic"' in src
        assert '"--train-speed-factor"' in src
        assert "deterministic=args.deterministic" in src
        assert "train_speed_factor=args.train_speed_factor" in src


# ------------------------------------------ silencing survives a reset
def test_silencing_survives_beamngpy_reconfiguring_itself(tmp_path):
    """BeamNGpy configures its own logging when it is CONSTRUCTED, long after
    setup_logging() ran, which resets whatever level was set beforehand. This
    was observed live: PPO-07 still logged 'Teleporting vehicle' at INFO. The
    handler filter is what actually holds."""
    import logging
    import residual_log
    p = tmp_path / "t.log"
    residual_log.setup_logging(str(p), "test", console=False)
    lg = logging.getLogger("beamngpy.BeamNGpy")
    lg.setLevel(logging.DEBUG)                       # what beamngpy does
    lg.info("Advancing the simulation by 1 steps.")
    lg.warning("connection lost")
    logging.getLogger("residual.trainer").info("episode 1 STOP")
    for h in logging.getLogger().handlers:
        h.flush()
    text = p.read_text(encoding="utf-8")
    assert "Advancing" not in text
    assert "connection lost" in text                 # real problems survive
    assert "episode 1 STOP" in text                  # trainer untouched


def test_the_filter_is_not_added_twice():
    """setup_logging is idempotent and gets called more than once."""
    import logging
    import residual_log
    residual_log.quiet_beamngpy()
    residual_log.quiet_beamngpy()
    for h in logging.getLogger().handlers:
        n = sum(1 for f in h.filters
                if isinstance(f, residual_log._QuietBeamngpy))
        assert n <= 1


# ------------------------------------------------ the mirror keeps up
def test_the_episode_mirror_follows_episodes_not_a_step_count():
    """At ~1084 steps/episode, mirroring every 500 steps was twice an episode.
    Free-running episodes are ~50 steps, so the same 500 meant the per-run file
    did not exist until episode 10 -- the monitor read as frozen because there
    was nothing to read."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "train_residual.py"), encoding="utf-8").read()
    assert 'dones = self.locals.get("dones")' in src
    assert "every_n_steps=50" in src
