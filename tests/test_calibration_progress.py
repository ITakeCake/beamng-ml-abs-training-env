"""Progress/ETA for a calibration run, parsed from the log the runner already
writes. A wrong estimate is worse than none -- "2 minutes left" that sits at 2
minutes for an hour destroys trust in the whole readout -- so the parsing is
pure and tested against real log text."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibration_progress as cp

REAL = """\
01:16:02.100 INFO  [refrun] compat: beamngpy 1.34.1 matches BeamNG 0.37.6.0.
01:17:05.200 INFO  [refrun] slam @ 60 mph: arc_g=1.0189 arc_dist=33.3m (chord_g=1.0 chord_dist=33.3m) stopped=True
01:18:08.300 INFO  [refrun] slam @ 60 mph: arc_g=1.0154 arc_dist=33.4m (chord_g=1.0 chord_dist=33.4m) stopped=True
01:19:11.400 INFO  [refrun] slam @ 60 mph: arc_g=1.0197 arc_dist=33.3m (chord_g=1.0 chord_dist=33.3m) stopped=True
01:19:12.000 INFO  [refrun] calibrated slam grip=1.000|speed=60.0|radius=straight: {'median': 1.0189}
"""


def test_planned_stops_counts_both_references_for_every_configuration():
    """Two anchors per config -- without the slam car there is no floor."""
    assert cp.planned_stops([60], [1.0], [1.0], 3) == 6
    assert cp.planned_stops([60, 120], [1.0], [0.6, 0.8, 1.0], 3) == 36


def test_progress_counts_stops_and_rows_from_real_log_text():
    p = cp.parse_progress(REAL, total_stops=6)
    assert p["stops_done"] == 3
    assert p["rows_done"] == 1
    assert p["last_stop"] == "slam @ 60 mph"


def test_eta_uses_observed_stop_duration_not_a_constant():
    """A 120 mph stop is ~150 m and takes far longer than a 60 mph one; a fixed
    seconds-per-stop would be wrong for every speed but the one it was tuned on."""
    p = cp.parse_progress(REAL, total_stops=6)
    assert p["seconds_per_stop"] == pytest.approx(63, abs=1)
    assert p["eta_seconds"] == pytest.approx(3 * 63, abs=3)


def test_fraction_is_bounded_even_if_more_stops_run_than_planned():
    p = cp.parse_progress(REAL, total_stops=2)
    assert p["fraction"] == 1.0


def test_an_earlier_runs_crash_is_not_reported_as_this_runs_status():
    """calibration.log is appended across runs. The bug this prevents: a live
    readout showing FAILED from a crash that happened hours ago."""
    log = ("01:00:00.000 CRITICAL [refrun] UNCAUGHT RuntimeError: old crash\n"
           + REAL)
    assert cp.parse_progress(log)["failed"] is None
    assert cp.parse_progress(log, whole_file=True)["failed"] is not None


def test_a_crash_in_the_current_run_is_reported():
    log = REAL + "01:20:00.000 CRITICAL [refrun] UNCAUGHT RuntimeError: car never stopped\n"
    kind, msg = cp.parse_progress(log)["failed"]
    assert kind == "RuntimeError" and "never stopped" in msg


def test_a_finished_run_is_recognised_by_its_output_file():
    log = REAL + r"01:20:00.000 INFO  [refrun] wrote C:\x\calibration\etk800.json" + "\n"
    p = cp.parse_progress(log, total_stops=6)
    assert p["finished"] and p["out_path"].endswith("etk800.json")
    assert "done" in cp.describe(p)


def test_a_corner_seek_is_distinguished_from_measuring():
    """Probes happen before any stop, so "0 of 36" would look stalled."""
    log = ("01:16:02.100 INFO  [refrun] compat: ok\n"
           "01:16:30.000 INFO  [refrun] probe steering=+0.1077 -> R=89.2 m\n")
    p = cp.parse_progress(log, total_stops=6)
    assert p["seeking"] and p["stops_done"] == 0
    assert "seeking" in cp.describe(p)


def test_a_converged_seek_stops_reporting_as_seeking():
    log = ("01:16:02.100 INFO  [refrun] compat: ok\n"
           "01:16:30.000 INFO  [refrun] probe steering=+0.05 -> R=153 m\n"
           "01:16:31.000 INFO  [refrun] steering seek converged: R=150.0m\n")
    assert not cp.parse_progress(log, total_stops=6)["seeking"]


def test_live_mode_stop_lines_are_counted_too():
    log = ("01:16:02.100 INFO  [refrun] compat: ok\n"
           "01:17:00.000 INFO  [refrun] slam @ 60 mph [live x4]: arc_g=1.01 arc_dist=33m\n")
    assert cp.parse_progress(log, total_stops=6)["stops_done"] == 1


def test_no_estimate_is_offered_before_there_is_evidence():
    """One stop gives no duration; claiming an ETA from nothing is the failure
    this whole module is trying to avoid."""
    log = ("01:16:02.100 INFO  [refrun] compat: ok\n"
           "01:17:00.000 INFO  [refrun] slam @ 60 mph: arc_g=1.01 arc_dist=33m\n")
    p = cp.parse_progress(log, total_stops=6)
    assert p["eta_seconds"] is None
    assert cp.format_eta(None) == "estimating..."


def test_an_interrupted_gap_does_not_poison_the_estimate():
    """A 40-minute gap means the run was paused (or a car respawned), not that a
    stop took 40 minutes. Such a gap is discarded rather than averaged in; here
    that leaves too little evidence to estimate from, and reporting nothing is
    the correct answer."""
    log = REAL.replace("01:18:08.300", "01:58:08.300")
    p = cp.parse_progress(log, total_stops=6)
    assert p["seconds_per_stop"] is None or p["seconds_per_stop"] < 300


def test_a_long_gap_is_dropped_but_good_samples_still_estimate():
    log = REAL + ("02:30:00.000 INFO  [refrun] slam @ 60 mph: arc_g=1.01 arc_dist=33m\n"
                  "02:31:04.000 INFO  [refrun] slam @ 60 mph: arc_g=1.01 arc_dist=33m\n")
    p = cp.parse_progress(log, total_stops=8)
    assert p["seconds_per_stop"] == pytest.approx(63, abs=2)


@pytest.mark.parametrize("secs,expect", [
    (45, "45s"), (63, "1m 03s"), (3600, "1h 00m"), (5400, "1h 30m")])
def test_eta_formatting(secs, expect):
    assert cp.format_eta(secs) == expect
