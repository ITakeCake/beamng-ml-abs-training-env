"""Live training statistics, derived from the episode log.

The trainer already writes one CSV row per finished episode, so nothing about
training needs to change to report progress -- this reads what is already
there. Keeping it a pure function of (rows, total_steps) means every number
below is testable without a game, which matters most for the two that are easy
to get quietly wrong: the trend and the ETA.

WHY THE TREND IS NOT "last minus first"
---------------------------------------
avg_g swings with the pedal draw far harder than it moves with learning. On
PPO-05 the pedal correlated 0.79 with avg_g, and comparing block medians showed
a rise of +0.03 that vanished entirely once full-pedal episodes were compared
against each other. So the trend here is a Theil-Sen slope over a window: the
median of every pairwise slope, which one lucky episode cannot drag the way it
drags a least-squares fit. See trend_slope().

WHY THE ETA IS NOT total_steps / mean_steps_per_second
------------------------------------------------------
Episode length is not constant: a stop that ends early has fewer steps, and the
wall-clock cost per episode is dominated by the acceleration run-up, which does
not scale with them. Estimating from recent episodes (rather than the run's
whole history) also lets the number react when the machine gets busier.
"""
import math

# Enough episodes to average out the pedal draw, few enough to still react when
# something changes. At ~80 s/episode this is roughly half an hour of history.
TREND_WINDOW = 40
RATE_WINDOW = 20

# g per 100 episodes below which the run reads as flat. See trend_word.
FLAT_BAND_PER_100 = 0.01


class TrainStats:
    """What the monitor window shows. Every field is safe to render."""

    def __init__(self, run_name="", episodes=0, steps_done=0, total_steps=0,
                 best=None, worst=None, last_g=None, last_outcome="--",
                 rolling_g=None, trend_per_100=None, eta_seconds=None,
                 steps_per_sec=None, series=None, crashes=0):
        self.run_name = run_name
        self.episodes = episodes
        self.steps_done = steps_done
        self.total_steps = total_steps
        self.best = best              # (avg_g, episode) or None
        self.worst = worst            # (avg_g, episode) or None
        self.last_g = last_g
        self.last_outcome = last_outcome
        self.rolling_g = rolling_g
        self.trend_per_100 = trend_per_100
        self.eta_seconds = eta_seconds
        self.steps_per_sec = steps_per_sec
        self.series = series or []    # [(episode, avg_g), ...]
        self.crashes = crashes

    @property
    def percent(self):
        if not self.total_steps:
            return 0.0
        return min(100.0, 100.0 * self.steps_done / float(self.total_steps))

    @property
    def trend_word(self):
        """Deliberately conservative: a slope smaller than the noise floor reads
        as 'flat' rather than as a direction the data does not support.

        The band is set from measured noise, not taste. PPO-05's avg_g median
        drifted about 0.03 g across 400 episodes -- roughly 0.0075 g per 100 --
        while the policy demonstrably did not improve (full-pedal episodes were
        flat at 0.983). So anything under 0.01 g per 100 episodes is inside
        what that run produced by chance alone."""
        if self.trend_per_100 is None:
            return "not enough episodes yet"
        if abs(self.trend_per_100) < FLAT_BAND_PER_100:
            return "flat"
        return "up" if self.trend_per_100 > 0 else "down"


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _i(row, key, default=0):
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def trend_slope(values):
    """Theil-Sen slope per element: the MEDIAN of all pairwise slopes.

    Least squares was the obvious choice and it is the wrong one here. A single
    outlier episode drags a least-squares fit: thirty episodes at 1.00 followed
    by one at 1.40 fits a slope of +0.24 g per 100 episodes, which would tell
    the user the run is improving on the strength of one lucky stop. The median
    of pairwise slopes ignores it, because most pairs still say zero.

    That matters more than usual for this signal. avg_g swings with the pedal
    draw (0.79 correlation on PPO-05) far harder than it moves with learning,
    so the outliers are not rare events -- they are most of the data.

    O(n^2) in the window, which at TREND_WINDOW=40 is 780 pairs: nothing, and
    paid once per two-second refresh.
    """
    n = len(values)
    if n < 3:
        return None
    slopes = [(values[j] - values[i]) / float(j - i)
              for i in range(n) for j in range(i + 1, n)]
    if not slopes:
        return None
    slopes.sort()
    mid = len(slopes) // 2
    if len(slopes) % 2:
        return slopes[mid]
    return (slopes[mid - 1] + slopes[mid]) / 2.0


def summarize(rows, total_steps=0, run_name=""):
    """Build the monitor's view from episode-log rows (dicts from csv.DictReader)."""
    rows = [r for r in rows if r.get("avg_g") not in (None, "")]
    if not rows:
        return TrainStats(run_name=run_name, total_steps=total_steps)

    series = [(_i(r, "episode", i + 1), _f(r, "avg_g")) for i, r in enumerate(rows)]
    gs = [g for _, g in series]

    best_i = max(range(len(gs)), key=lambda i: gs[i])
    worst_i = min(range(len(gs)), key=lambda i: gs[i])

    steps_done = sum(_i(r, "steps") for r in rows)
    rolling = sum(gs[-RATE_WINDOW:]) / len(gs[-RATE_WINDOW:])

    slope = trend_slope(gs[-TREND_WINDOW:])
    trend_per_100 = None if slope is None else slope * 100.0

    # Rate from recent episodes only: the run's lifetime average would be
    # dragged by the first episode, which includes the game boot.
    recent = rows[-RATE_WINDOW:]
    wall = sum(_f(r, "wall_clock_s") for r in recent)
    steps = sum(_i(r, "steps") for r in recent)
    steps_per_sec = (steps / wall) if wall > 0 and steps else None

    eta = None
    if steps_per_sec and total_steps and steps_done < total_steps:
        eta = (total_steps - steps_done) / steps_per_sec

    crashes = sum(1 for r in rows if str(r.get("outcome", "")).upper() != "STOP")

    return TrainStats(
        run_name=run_name, episodes=len(rows), steps_done=steps_done,
        total_steps=total_steps,
        best=(gs[best_i], series[best_i][0]),
        worst=(gs[worst_i], series[worst_i][0]),
        last_g=gs[-1], last_outcome=rows[-1].get("outcome", "--"),
        rolling_g=rolling, trend_per_100=trend_per_100,
        eta_seconds=eta, steps_per_sec=steps_per_sec,
        series=series, crashes=crashes)


def format_eta(seconds):
    """'4h 12m' / '38m' / '45s'. None becomes a dash rather than a fake zero."""
    if seconds is None or seconds <= 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def format_trend(stats):
    """One line a person can act on, in the units they think in."""
    if stats.trend_per_100 is None:
        return "trend: not enough episodes yet"
    if stats.trend_word == "flat":
        return (f"trend: flat ({stats.trend_per_100:+.4f} g per 100 episodes "
                f"over the last {min(stats.episodes, TREND_WINDOW)})")
    return (f"trend: {stats.trend_word} {abs(stats.trend_per_100):.4f} g per "
            f"100 episodes (last {min(stats.episodes, TREND_WINDOW)})")
