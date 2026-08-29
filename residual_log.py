"""Shared logging for the ResidualABS GUI / trainer / env / probe.

Only ENTRY POINTS call setup_logging() (trainer main -> runs/<run>/train.log,
probe main -> logs/probe.log, GUI -> logs/gui.log). Library code (the env
subclass, callbacks) just does `logging.getLogger("residual.env")` and never
opens files -- whoever configured the root handlers decides where lines land.

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


def setup_logging(file_path, component="main", level=logging.INFO, console=True):
    """Configure the root logger once: UTF-8 file + console, uncaught-exception
    hook. Idempotent -- a second call with the same file adds no handlers."""
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
