"""Shared logging for the ResidualABS GUI / trainer / env / probe.

Only ENTRY POINTS call setup_logging() (trainer main -> runs/<run>/train.log,
probe main -> logs/probe.log, GUI -> logs/gui.log). Library code (the env
subclass, callbacks) just does `logging.getLogger("residual.env")` and never
opens files, whoever configured the root handlers decides where lines land.

Format:  22:31:05.123 INFO  [env] message
"""
import collections
import logging
import os
import sys
import traceback

_FMT = "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s"
_DATEFMT = "%H:%M:%S"
_ROOT = "residual"


class _ShortName(logging.Filter):
    """Strip the 'residual.' prefix so lines read [env] / [trainer] / [gui]."""
    def filter(self, record):
        if record.name.startswith(_ROOT + "."):
            record.name = record.name[len(_ROOT) + 1:]
        elif record.name == _ROOT:
            record.name = "root"
        return True


# beamngpy logs one INFO line per bng.step() call. At FRAME_SKIP=1 that is one
# line per 5 ms of simulated time: PPO-05 wrote 475,355 "Advancing the simulation
# by 1 steps" lines into a 37 MB log for a single run, which buries the ~450 lines
# that actually say what happened and costs a formatted record + file write on
# every step of the hot loop.
BEAMNGPY_LOGGERS = ("beamngpy", "beamngpy.BeamNGpy", "beamngpy.Vehicle",
                    "beamngpy.Scenario", "beamngpy.Camera", "beamngpy.Sensor")


class _QuietBeamngpy(logging.Filter):
    """Drops beamngpy records below `level`, at the HANDLER.

    Setting the level on beamngpy's loggers is not enough and was observed not
    to work: BeamNGpy configures its own logging when it is constructed, which
    happens long after setup_logging() runs, and that resets whatever level was
    set beforehand. A filter on the handler cannot be undone that way, because
    the library never sees the handler.

    Matching on the record's name prefix rather than a fixed list also catches
    the loggers beamngpy creates that are not named here.
    """

    def __init__(self, level=logging.WARNING):
        super().__init__()
        self.level = level

    def filter(self, record):
        if record.name == "beamngpy" or record.name.startswith("beamngpy."):
            return record.levelno >= self.level
        return True


def quiet_beamngpy(level=logging.WARNING):
    """Stop beamngpy's per-step chatter from reaching the log.

    WARNING keeps connection failures and genuine problems visible; only the
    routine step/poll narration is dropped. Applied both as a logger level (so
    the cost is skipped early when it works) and as a handler filter (so it
    still works when beamngpy resets those levels underneath us)."""
    for name in BEAMNGPY_LOGGERS:
        logging.getLogger(name).setLevel(level)
    for h in logging.getLogger().handlers:
        if not any(isinstance(f, _QuietBeamngpy) for f in h.filters):
            h.addFilter(_QuietBeamngpy(level))
    return level


def setup_logging(file_path, component="main", level=logging.INFO, console=True):
    """Configure the root logger once: UTF-8 file + console, uncaught-exception
    hook. Idempotent, a second call with the same file adds no handlers."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)
    abs_path = os.path.abspath(file_path)

    have_file = any(isinstance(h, logging.FileHandler)
                    and getattr(h, "baseFilename", None) == abs_path
                    for h in root.handlers)
    if not have_file:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        # encoding matters: the default cp1252 on Windows throws inside the
        # logging module on any non-ASCII character and prints
        # "--- Logging error ---" instead of the message.
        fh = logging.FileHandler(abs_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        fh.addFilter(_ShortName())
        root.addHandler(fh)

    have_console = any(isinstance(h, logging.StreamHandler)
                       and not isinstance(h, logging.FileHandler)
                       and getattr(h, "_residual_console", False)
                       for h in root.handlers)
    if console and not have_console:
        ch = logging.StreamHandler(sys.stdout)
        ch._residual_console = True
        ch.setLevel(level)
        ch.setFormatter(fmt)
        ch.addFilter(_ShortName())
        root.addHandler(ch)

    log = logging.getLogger(f"{_ROOT}.{component}")

    def _hook(exc_type, exc, tb):
        log.critical("UNCAUGHT %s: %s\n%s", exc_type.__name__, exc,
                     "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _hook
    # After handlers exist, so a beamngpy logger configured by an earlier import
    # cannot re-lower itself past this point.
    quiet_beamngpy()
    return log


def get_logger(component):
    return logging.getLogger(f"{_ROOT}.{component}")


def tail_lines(path, n):
    """Last n lines of a text file ([] if missing/unreadable)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    return lines[-n:] if n > 0 else []


class StepRingBuffer:
    """Last N env steps, formatted one line each; dumped only on bad episodes
    (crash/timeout/exception) so 200 Hz per-step detail never floods the log."""

    def __init__(self, capacity=50):
        self._buf = collections.deque(maxlen=capacity)

    def push(self, step, action, brakes, gy, yaw, speed):
        a0, a1 = float(action[0]), float(action[1])
        fr, fl, rr, rl = (float(b) for b in brakes)
        self._buf.append(
            f"step={step} rel=({a0:.3f},{a1:.3f}) brk=({fr:.3f},{fl:.3f},{rr:.3f},{rl:.3f}) "
            f"gy={float(gy):+.2f} yaw={float(yaw):+.3f} spd={float(speed):.2f}")

    def dump(self):
        return list(self._buf)

    def clear(self):
        self._buf.clear()


class YawTrace:
    """Where an episode's yaw-error integral actually comes from.

    The terminal yaw bonus is a cliff at 0.1 rad of accumulated |yaw error|,
    and corner episodes land right on it (median 0.092, live 2026-08-29). Two
    very different causes produce that same number, and the total cannot tell
    them apart: error concentrated in the first fraction of a second (the car
    still turning in when the brakes hit, which is a PROCEDURE problem) versus
    error spread evenly across the stop (the threshold is genuinely too tight
    for a corner, which is a CONSTANT problem). Only one of those is fixed by
    changing 0.1.

    So the integral is accumulated into fixed-width time buckets as well as in
    total, and the head of the episode is measured separately. Nothing here
    decides anything, it reports, so the decision has evidence under it."""

    BUCKET_S = 0.5
    HEAD_S = 0.5          # "turn-in transient" window, from brake onset

    def __init__(self, bucket_s=BUCKET_S, head_s=HEAD_S):
        self.bucket_s = float(bucket_s)
        self.head_s = float(head_s)
        self.buckets = []          # integral of |yaw error| dt, per bucket
        self.total = 0.0
        self.head = 0.0            # integral over the first head_s seconds
        self.peak_error = 0.0
        self.peak_t = 0.0
        self.n = 0                 # steps pushed
        self.dt = 0.0              # step size, from the first push

    @property
    def t(self):
        """Elapsed episode time. Derived from an exact step COUNT rather than
        accumulated by `t += dt`: at 200 Hz over a 5 s stop the accumulated
        version drifts enough to move samples across bucket edges, which shows
        up as buckets of 100 and 102 samples in an episode of constant error."""
        return self.n * self.dt

    def push(self, yaw_error, dt):
        e = abs(float(yaw_error))
        self.dt = float(dt)
        t = self.t
        contrib = e * self.dt
        idx = int(t / self.bucket_s)
        while len(self.buckets) <= idx:
            self.buckets.append(0.0)
        self.buckets[idx] += contrib
        self.total += contrib
        if t < self.head_s:
            self.head += contrib
        if e > self.peak_error:
            self.peak_error, self.peak_t = e, t
        self.n += 1

    @property
    def head_fraction(self):
        """Share of the whole integral spent in the first head_s seconds. High
        (say >0.4 for a 5 s stop, where an even spread would give ~0.1) means
        turn-in transient dominates and the threshold is not the problem."""
        return 0.0 if self.total <= 0.0 else self.head / self.total

    def summary(self):
        """One line: total, where it came from, and the per-bucket shape."""
        shape = " ".join(f"{b:.3f}" for b in self.buckets)
        return (f"total={self.total:.4f} head({self.head_s:g}s)={self.head:.4f} "
                f"({self.head_fraction * 100:.0f}%) peak={self.peak_error:.3f}rad/s"
                f"@{self.peak_t:.2f}s per{self.bucket_s:g}s=[{shape}]")

    def clear(self):
        self.__init__(self.bucket_s, self.head_s)


class CornerDiag:
    """Per-episode corner diagnostics that the yaw integral cannot express.

    The integral answers "how much total rate error", which turned out to be
    the wrong question three times over: it cannot say whether the error was
    signed one way (the car simply not turning enough) or churning both ways
    (the brakes upsetting it), it keeps grading after the car has effectively
    stopped, and it hides the lateral grip state that decides whether a corner
    is even being driven. Each field below answers one of those directly."""

    STOPPED_MS = 1.0          # below this, the "stop" is over in any real sense

    def __init__(self, stopped_ms=STOPPED_MS):
        self.stopped_ms = float(stopped_ms)
        self.signed_sum = 0.0       # integral of SIGNED error: bias vs churn
        self.abs_sum = 0.0          # integral of |error|, for the ratio
        self.reversals = 0          # sign flips of the error: modulation churn
        self.after_stop = 0.0       # |error| integral accumulated below stopped_ms
        self.after_stop_s = 0.0
        self.peak_lat_g = 0.0
        self.lat_g_at_peak_err = 0.0
        self.peak_err = 0.0
        self._last_sign = 0

    def push(self, yaw_error, lateral_g, speed_ms, dt):
        e, dt = float(yaw_error), float(dt)
        self.signed_sum += e * dt
        self.abs_sum += abs(e) * dt
        sign = (e > 0) - (e < 0)
        if sign and self._last_sign and sign != self._last_sign:
            self.reversals += 1
        if sign:
            self._last_sign = sign
        if float(speed_ms) < self.stopped_ms:
            self.after_stop += abs(e) * dt
            self.after_stop_s += dt
        lat = abs(float(lateral_g))
        self.peak_lat_g = max(self.peak_lat_g, lat)
        if abs(e) > self.peak_err:
            self.peak_err, self.lat_g_at_peak_err = abs(e), lat

    @property
    def bias_ratio(self):
        """|signed| / absolute. Near 1.0 = the error is one-directional, the car
        is consistently not turning enough (a TRACKING failure). Near 0 = it is
        churning either side of the target (a DISTURBANCE), which is what brake
        modulation looks like. The integral alone cannot tell these apart."""
        return 0.0 if self.abs_sum <= 0 else abs(self.signed_sum) / self.abs_sum

    @property
    def after_stop_fraction(self):
        return 0.0 if self.abs_sum <= 0 else self.after_stop / self.abs_sum

    def summary(self):
        return (f"bias={self.bias_ratio:.2f} (signed={self.signed_sum:+.4f} "
                f"abs={self.abs_sum:.4f}) reversals={self.reversals} "
                f"after_stop={self.after_stop:.4f} ({self.after_stop_fraction * 100:.0f}%, "
                f"{self.after_stop_s:.2f}s) peak_lat_g={self.peak_lat_g:.2f} "
                f"lat_g@peak_err={self.lat_g_at_peak_err:.2f}")

    def clear(self):
        self.__init__(self.stopped_ms)
