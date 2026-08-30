"""YawTrace decomposes an episode's yaw-error integral. Its whole job is to
distinguish turn-in transient (a procedure problem) from evenly-spread error
(a threshold problem) -- two causes that produce the same total, which is why
the total alone could not settle the 0.1 rad question."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residual_log import YawTrace

DT = 1.0 / 200.0


def test_total_matches_the_reward_accumulator_arithmetic():
    """abs_env_incar does ep_yaw_abs_sum += yaw_error * dt. The trace must
    reproduce that exactly, or it is decomposing a different number than the
    one the reward gated on."""
    trace = YawTrace()
    expected = 0.0
    for i in range(1000):
        err = 0.05 + 0.01 * (i % 7)
        trace.push(err, DT)
        expected += abs(err) * DT
    assert trace.total == pytest.approx(expected)


def test_sign_of_the_error_does_not_matter():
    a, b = YawTrace(), YawTrace()
    for _ in range(100):
        a.push(0.08, DT)
        b.push(-0.08, DT)
    assert a.total == pytest.approx(b.total)


def test_buckets_sum_to_the_total():
    trace = YawTrace()
    for i in range(1200):
        trace.push(0.02 * (1 + i % 5), DT)
    assert sum(trace.buckets) == pytest.approx(trace.total)


def test_a_transient_episode_is_identified_by_its_head_fraction():
    """Big error for 0.4 s, near-nothing after: the signature of a car still
    turning in when the brakes hit."""
    trace = YawTrace()
    for _ in range(80):          # 0.4 s
        trace.push(0.5, DT)
    for _ in range(920):         # 4.6 s
        trace.push(0.002, DT)
    assert trace.head_fraction > 0.9
    assert trace.peak_t < 0.5


def test_an_evenly_spread_episode_has_a_low_head_fraction():
    """Same total, spread flat -- the threshold, not the procedure, is tight."""
    trace = YawTrace()
    for _ in range(1000):        # 5 s at constant error
        trace.push(0.0824, DT)
    assert trace.head_fraction == pytest.approx(0.1, abs=0.02)   # 0.5s of 5s
    assert max(trace.buckets) == pytest.approx(min(trace.buckets), rel=1e-6)


def test_the_two_cases_are_distinguishable_at_the_same_total():
    """The point of the whole class: identical totals, opposite diagnoses."""
    transient, spread = YawTrace(), YawTrace()
    for _ in range(80):
        transient.push(0.5, DT)
    for _ in range(920):
        transient.push(0.0022, DT)
    flat = transient.total / (1000 * DT)
    for _ in range(1000):
        spread.push(flat, DT)
    assert transient.total == pytest.approx(spread.total, rel=1e-6)
    assert transient.head_fraction > 4 * spread.head_fraction


def test_bucket_edges_do_not_drift_with_accumulated_float_time():
    """t += dt at 200 Hz over 5 s drifts enough to push samples across bucket
    boundaries, so an episode of perfectly constant error produced buckets of
    100 and 102 samples. Time is derived from the step count instead."""
    trace = YawTrace()
    for _ in range(1000):
        trace.push(0.05, DT)
    assert len(trace.buckets) == 10
    for b in trace.buckets:
        assert b == pytest.approx(trace.buckets[0], rel=1e-12)


def test_peak_records_when_the_worst_error_happened():
    trace = YawTrace()
    for _ in range(200):
        trace.push(0.01, DT)
    trace.push(0.9, DT)          # at t = 1.0 s
    for _ in range(200):
        trace.push(0.01, DT)
    assert trace.peak_error == pytest.approx(0.9)
    assert trace.peak_t == pytest.approx(1.0, abs=DT)


def test_an_empty_trace_reports_no_head_fraction_rather_than_dividing_by_zero():
    assert YawTrace().head_fraction == 0.0


def test_clear_resets_everything_between_episodes():
    trace = YawTrace()
    for _ in range(500):
        trace.push(0.3, DT)
    trace.clear()
    assert trace.total == 0.0 and trace.buckets == [] and trace.n == 0
    assert trace.peak_error == 0.0


def test_summary_reports_the_numbers_a_decision_needs():
    trace = YawTrace()
    for _ in range(1000):
        trace.push(0.0824, DT)
    out = trace.summary()
    assert "total=0.412" in out
    assert "head(0.5s)=" in out and "%" in out
    assert "peak=0.082rad/s" in out


def test_the_live_corner_total_is_reproduced():
    """The measured median from 2026-08-29 was 0.092 over a ~5.4 s stop, i.e.
    an average error of ~0.017 rad/s. Sanity-check the units line up."""
    trace = YawTrace()
    for _ in range(1080):
        trace.push(0.0170, DT)
    assert trace.total == pytest.approx(0.092, abs=0.002)
