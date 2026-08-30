"""The live training monitor's numbers.

Every field here is read while a run is in flight and acted on -- "is it
learning", "how much longer", "did something break". The two easiest to get
quietly wrong are the trend (which must not report the pedal draw as learning)
and the ETA (which must not be a constant multiplied by a step count), so those
carry the most tests.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_progress as tp


def _row(episode, avg_g, steps=1000, wall=80.0, outcome="STOP"):
    return {"episode": str(episode), "avg_g": str(avg_g), "steps": str(steps),
            "wall_clock_s": str(wall), "outcome": outcome}


def _rows(gs, **kw):
    return [_row(i + 1, g, **kw) for i, g in enumerate(gs)]


# ------------------------------------------------------------ empty / early
def test_no_episodes_is_not_an_error():
    """The CSV does not exist until episode 1 finishes; the window opens before
    that and must render something rather than raise."""
    s = tp.summarize([], total_steps=1000)
    assert s.episodes == 0 and s.best is None and s.percent == 0.0
    assert s.trend_word == "not enough episodes yet"


def test_rows_without_avg_g_are_skipped():
    s = tp.summarize([_row(1, 1.0), {"episode": "2", "avg_g": ""}])
    assert s.episodes == 1


def test_percent_never_exceeds_100():
    """A run that overshoots its step target must not render a 140% bar."""
    s = tp.summarize(_rows([1.0] * 3, steps=1000), total_steps=1000)
    assert s.percent == 100.0


# ------------------------------------------------------------- best / worst
def test_best_and_worst_carry_their_episode_numbers():
    s = tp.summarize([_row(10, 0.9), _row(11, 1.2), _row(12, 0.4)])
    assert s.best == (1.2, 11)
    assert s.worst == (0.4, 12)


def test_episode_numbers_come_from_the_log_not_the_row_index():
    """The per-run CSV is a mirror that can start mid-history, so position in
    the list is not the episode number."""
    s = tp.summarize([_row(430, 1.0), _row(431, 1.1)])
    assert s.best == (1.1, 431)


# ------------------------------------------------------------------- trend
def test_a_rising_run_trends_up():
    s = tp.summarize(_rows([0.90 + 0.01 * i for i in range(20)]))
    assert s.trend_word == "up"
    assert s.trend_per_100 == pytest.approx(1.0, rel=0.01)


def test_a_falling_run_trends_down():
    s = tp.summarize(_rows([1.10 - 0.01 * i for i in range(20)]))
    assert s.trend_word == "down"


def test_noise_around_a_constant_reads_as_flat():
    """PPO-05's avg_g swung 0.53-1.15 on the pedal draw alone while the policy
    did not improve. A monitor that calls that 'up' is worse than one with no
    trend at all."""
    noisy = [1.0, 0.6, 1.1, 0.7, 1.05, 0.65, 1.0, 0.75, 1.1, 0.6,
             1.0, 0.7, 1.05, 0.62, 1.08, 0.71, 1.0, 0.69, 1.02, 0.7]
    assert tp.summarize(_rows(noisy)).trend_word == "flat"


def test_the_trend_uses_a_window_not_the_whole_run():
    """A run that improved early then stalled should report the stall."""
    gs = [0.5 + 0.02 * i for i in range(30)] + [1.1] * tp.TREND_WINDOW
    assert tp.summarize(_rows(gs)).trend_word == "flat"


def test_two_episodes_are_not_enough_to_claim_a_direction():
    assert tp.summarize(_rows([0.5, 1.5])).trend_per_100 is None


def test_a_single_lucky_episode_cannot_invent_a_direction():
    """A least-squares fit gives +0.24 g per 100 episodes here, purely from
    the one outlier. The median of pairwise slopes gives zero."""
    gs = [1.0] * 30 + [1.4]
    assert tp.summarize(_rows(gs)).trend_word == "flat"


# --------------------------------------------------------------------- eta
def test_eta_uses_observed_wall_clock():
    # 20 episodes x 1000 steps in 100 s each -> 10 steps/s; 20,000 done,
    # 80,000 left -> 8000 s.
    s = tp.summarize(_rows([1.0] * 20, steps=1000, wall=100.0),
                     total_steps=100_000)
    assert s.steps_per_sec == pytest.approx(10.0)
    assert s.eta_seconds == pytest.approx(8000.0)


def test_eta_is_none_when_the_target_is_already_met():
    s = tp.summarize(_rows([1.0] * 5, steps=1000), total_steps=1000)
    assert s.eta_seconds is None


def test_eta_is_none_without_a_step_target():
    assert tp.summarize(_rows([1.0] * 5)).eta_seconds is None


def test_eta_ignores_the_first_episodes_boot_cost():
    """Episode 1 carries the game launch. Averaging it over a long run would
    understate the rate forever."""
    rows = [_row(1, 1.0, steps=1000, wall=600.0)]
    rows += [_row(i + 2, 1.0, steps=1000, wall=50.0) for i in range(tp.RATE_WINDOW)]
    s = tp.summarize(rows, total_steps=1_000_000)
    assert s.steps_per_sec == pytest.approx(20.0)


# ------------------------------------------------------------------ format
@pytest.mark.parametrize("secs,want", [
    (None, "--"), (0, "--"), (-5, "--"),
    (45, "45s"), (150, "2m 30s"), (3700, "1h 01m"), (100_000, "1d 3h"),
])
def test_eta_formatting(secs, want):
    assert tp.format_eta(secs) == want


def test_a_nan_eta_does_not_reach_the_window():
    assert tp.format_eta(float("nan")) == "--"
    assert tp.format_eta(float("inf")) == "--"


# ---------------------------------------------------------------- outcomes
def test_non_stop_outcomes_are_counted():
    rows = _rows([1.0] * 5)
    rows[2]["outcome"] = "CRASH"
    assert tp.summarize(rows).crashes == 1


def test_the_series_feeds_the_chart():
    s = tp.summarize([_row(3, 0.9), _row(4, 1.0)])
    assert s.series == [(3, 0.9), (4, 1.0)]


# ------------------------------------------------------------ real numbers
def test_against_the_real_ppo05_log():
    """The mirror that survived PPO-05: 476 episodes, best 1.1526 at ep 148."""
    import csv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "logs", "episode_log_env0.csv")
    if not os.path.exists(path):
        pytest.skip("no episode log on this machine")
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.DictReader(fh))
    s = tp.summarize(rows, total_steps=1_000_000, run_name="PPO-05")
    assert s.episodes > 100
    assert 0.5 < s.best[0] < 1.5
    assert s.steps_per_sec and s.steps_per_sec > 0
    assert tp.format_eta(s.eta_seconds) != ""


# ------------------------------------------------------------ data sources
def test_file_source_survives_a_missing_log(tmp_path):
    """The window opens before episode 1 exists."""
    import train_monitor as tm
    read = tm.file_source(str(tmp_path), total_steps=1000, run_name="X")
    s = read()
    assert s.episodes == 0 and s.run_name == "X"


def test_file_source_accepts_a_csv_path_directly(tmp_path):
    import train_monitor as tm
    p = tmp_path / "episode_log_env0.csv"
    p.write_text("episode,avg_g,steps,wall_clock_s,outcome\n1,1.0,10,5,STOP\n")
    assert tm.file_source(str(p))().episodes == 1


def test_replay_reveals_episodes_gradually(tmp_path):
    """The demo has to animate, or it shows a finished chart and proves nothing
    about how the window behaves during a run."""
    import train_monitor as tm
    p = tmp_path / "episode_log_env0.csv"
    lines = ["episode,avg_g,steps,wall_clock_s,outcome"]
    lines += [f"{i},1.0,10,5,STOP" for i in range(1, 41)]
    p.write_text("\n".join(lines))
    read = tm.replay_source(str(p), total_steps=1000, step=4)
    first = read().episodes
    for _ in range(3):
        later = read().episodes
    assert later > first


def test_replay_stops_at_the_end_of_the_file(tmp_path):
    import train_monitor as tm
    p = tmp_path / "episode_log_env0.csv"
    p.write_text("episode,avg_g,steps,wall_clock_s,outcome\n"
                 + "\n".join(f"{i},1.0,10,5,STOP" for i in range(1, 11)))
    read = tm.replay_source(str(p), step=50)
    for _ in range(10):
        s = read()
    assert s.episodes == 10


# ------------------------------------------------------------------- wiring
def test_the_gui_opens_a_monitor_when_training_starts():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "gui_train.py"), encoding="utf-8").read()
    assert "import train_monitor" in src
    assert "self._open_monitor(settings)" in src
    assert 'text="Show monitor"' in src


def test_the_monitors_stop_button_is_the_graceful_one():
    """Never a terminate: a killed trainer loses its replay buffer and its
    resumable checkpoint."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "gui_train.py"), encoding="utf-8").read()
    assert "on_stop=self.stop" in src
