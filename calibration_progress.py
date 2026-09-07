"""Progress and time-remaining for a calibration run, derived from its log.

The runner already prints one line per completed stop and one per finished row,
so nothing about the measurement needs to change to report progress, this
parses what is already there. Keeping it a pure function of (log text, planned
matrix) means the estimate is testable without a game, which matters because a
wrong "2 minutes left" that sits at 2 minutes for an hour is worse than no
estimate at all.

Estimating from OBSERVED stop durations rather than a constant: a 120 mph stop
is ~150 m and takes far longer than a 60 mph one, corners add a steering seek
before any measurement, and a fast-forwarded run is several times quicker. A
fixed seconds-per-stop would be wrong in every one of those cases.
"""
import re

# "slam @ 60 mph: arc_g=1.0197 arc_dist=33.28m ...", one per completed stop.
STOP_RE = re.compile(
    r"^(\d\d:\d\d:\d\d)\.\d+\s+INFO\s+\[refrun\]\s+(slam|stock)\s+@\s+(\d+)\s+mph"
    r"(?:\s+\[live x[\d.]+\])?:", re.M)
# "calibrated slam grip=...|speed=...|radius=...: {...}", one per finished row.
ROW_RE = re.compile(r"\[refrun\]\s+calibrated\s+(slam|stock)\s+(\S+):", re.M)
# "probe steering=+0.1077 -> ...", corner seeks, which precede measurement.
PROBE_RE = re.compile(r"\[refrun\]\s+probe steering=", re.M)
SEEK_DONE_RE = re.compile(r"\[refrun\]\s+steering seek converged", re.M)
WROTE_RE = re.compile(r"\[refrun\]\s+wrote\s+(\S+)", re.M)
FAILED_RE = re.compile(r"\[refrun\]\s+UNCAUGHT\s+(\w+):\s*(.*)", re.M)


def planned_stops(speeds, grips, pedals, reps, references=2):
    """Total stops a run will perform. references=2 (slam and stock) because
    both anchors are measured for every configuration."""
    return (max(1, len(speeds)) * max(1, len(grips)) * max(1, len(pedals))
            * max(1, reps) * references)


def _hhmmss_to_secs(text):
    h, m, s = (int(p) for p in text.split(":"))
    return h * 3600 + m * 60 + s


# Every run appends to the same calibration.log, so a naive parse reports the
# previous run's crash as this run's status. The runner logs this line at
# startup, which is the only reliable "a new run begins here" marker.
RUN_START_RE = re.compile(r"\[refrun\]\s+compat:", re.M)


def tail_since_last_run(log_text):
    """Just the current run's portion of a shared, appended log."""
    starts = list(RUN_START_RE.finditer(log_text))
    return log_text[starts[-1].start():] if starts else log_text


def parse_progress(log_text, total_stops=None, whole_file=False):
    """What the log says so far. `total_stops` from planned_stops(); without it
    the fraction and ETA are unknown but counts still work.

    Only the current run is considered unless `whole_file`, calibration.log is
    appended across runs, so an earlier failure would otherwise be reported as
    this run's."""
    if not whole_file:
        log_text = tail_since_last_run(log_text)
    stops = STOP_RE.findall(log_text)
    rows = ROW_RE.findall(log_text)
    probes = len(PROBE_RE.findall(log_text))
    wrote = WROTE_RE.findall(log_text)
    failed = FAILED_RE.findall(log_text)

    times = [_hhmmss_to_secs(t) for t, _, _ in stops]
    durations = []
    for a, b in zip(times, times[1:]):
        gap = b - a
        if gap < 0:            # crossed midnight
            gap += 24 * 3600
        # A gap of many minutes means the run was interrupted or a car was
        # respawned, not that one stop took that long.
        if 0 < gap < 600:
            durations.append(gap)

    done = len(stops)
    result = {
        "stops_done": done,
        "rows_done": len(rows),
        "probes": probes,
        "seeking": probes > 0 and not SEEK_DONE_RE.search(log_text),
        "finished": bool(wrote),
        "out_path": wrote[-1] if wrote else None,
        "failed": failed[-1] if failed else None,
        "total_stops": total_stops,
        "fraction": None,
        "eta_seconds": None,
        "seconds_per_stop": None,
        "last_stop": None if not stops else f"{stops[-1][1]} @ {stops[-1][2]} mph",
    }
    if durations:
        # Median, not mean: one interrupted stop should not dominate, and stop
        # durations differ systematically by speed anyway.
        ordered = sorted(durations)
        mid = len(ordered) // 2
        per = (ordered[mid] if len(ordered) % 2
               else (ordered[mid - 1] + ordered[mid]) / 2.0)
        result["seconds_per_stop"] = per
        if total_stops:
            result["eta_seconds"] = max(0, (total_stops - done)) * per
    if total_stops:
        result["fraction"] = min(1.0, done / float(total_stops)) if total_stops else None
    return result


def format_eta(seconds):
    """Human ETA. None -> "estimating..." rather than a fake number."""
    if seconds is None:
        return "estimating..."
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def describe(progress):
    """One status line for the GUI."""
    if progress.get("failed"):
        kind, msg = progress["failed"]
        return f"FAILED ({kind}): {msg[:80]}"
    if progress.get("finished"):
        return f"done, {progress['stops_done']} stops measured"
    if progress.get("seeking"):
        return (f"seeking the steering angle for the corner "
                f"(probe {progress['probes']}), no stops measured yet")
    total = progress.get("total_stops")
    done = progress["stops_done"]
    if not done:
        return "starting BeamNG..."
    where = f" ({progress['last_stop']})" if progress.get("last_stop") else ""
    if total:
        return (f"stop {done} of {total}{where}, "
                f"{format_eta(progress['eta_seconds'])} left")
    return f"{done} stops measured{where}"


# Observed medians, 2026-08-30 A/B (60 mph, etk800). Stepping is dominated by
# the per-step round trip; fast mode is dominated by the acceleration run-up,
# which is why raising the speed factor past ~4 buys nothing.
SECONDS_PER_STOP = {"deterministic": 35.0, "fast": 4.0}


def estimate_seconds(stops, fast=False):
    return stops * SECONDS_PER_STOP["fast" if fast else "deterministic"]


def estimate_text(stops, fast=False):
    return format_eta(estimate_seconds(stops, fast))
