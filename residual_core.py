"""Pure functions for residual (release-from-pedal) ABS training. NO game imports.

Action semantics (the whole point of this experiment): action = per-axle RELEASE.
zero action == full-pedal slam == the ~0.9g lockup baseline, so exploration starts
at the anti-lock boundary. brake = clamp(pedal*(1-release), 0.01, 1.0)."""

BRAKE_FLOOR = 0.01   # keeps the 2kHz metric state machine armed (env law)


def residual_to_brakes(action, pedal):
    """[front_release, rear_release] in [0,1] + pedal in (0,1] -> (fr, fl, rr, rl)."""
    fr_rel = min(1.0, max(0.0, float(action[0])))
    rr_rel = min(1.0, max(0.0, float(action[1])))
    front = min(1.0, max(BRAKE_FLOOR, float(pedal) * (1.0 - fr_rel)))
    rear = min(1.0, max(BRAKE_FLOOR, float(pedal) * (1.0 - rr_rel)))
    return (front, front, rear, rear)   # (fr, fl, rr, rl) — axle-locked


def parse_speeds(text):
    parts = [p.strip() for p in str(text).split(",")]
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"bad speed list: {text!r}")
    out = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"bad speed: {p!r}")
        v = int(p)
        if not 15 <= v <= 160:
            raise ValueError(f"speed out of range 15..160: {v}")
        out.append(v)
    return out


def parse_pedal_spec(text):
    t = str(text).strip().lower()
    if t in ("", "off", "none"):
        return None
    lo, _, hi = t.partition("-")
    lo = float(lo)
    hi = float(hi) if hi else lo
    if not (0.1 <= lo <= 1.0 and 0.1 <= hi <= 1.0 and lo <= hi):
        raise ValueError(f"bad pedal spec: {text!r}")
    return (lo, hi)


# ---------------------------------------------------------------- tire grip
GRIP_MIN, GRIP_MAX = 0.05, 2.0     # multiplier on the tire's jbeam friction
GRIP_DP = 3                        # must match calibration.config_key's rounding


class GripSpec:
    """How tire grip varies between episodes. Three modes:
      fixed  "0.6"            -- one concrete level
      list   "0.5,0.75,1.0"   -- randomized between runs, but only among these
      range  "0.4-1.0"        -- randomized continuously between runs

    `None` (not a GripSpec) means off/stock: never touch grip at all.

    Draws are rounded to the same precision calibration.config_key uses, so a
    drawn level can actually key a calibration row. A continuous range can
    still draw a value nothing was calibrated at -- hence
    `needs_continuous_calibration`, which the trainer checks before pairing it
    with a normalized reward."""

    def __init__(self, mode, values):
        self.mode = mode
        self.values = values

    @property
    def needs_continuous_calibration(self):
        return self.mode == "range" and self.values[0] != self.values[1]

    def levels(self):
        """The finite set of levels this can draw, or None if not enumerable."""
        if self.mode == "fixed":
            return [self.values[0]]
        if self.mode == "list":
            return list(self.values)
        lo, hi = self.values
        return [lo] if lo == hi else None

    def draw(self, rng=None):
        import random as _random
        rng = rng or _random
        if self.mode == "fixed":
            return self.values[0]
        if self.mode == "list":
            return rng.choice(list(self.values))
        lo, hi = self.values
        return lo if lo == hi else round(rng.uniform(lo, hi), GRIP_DP)

    def __repr__(self):
        return f"GripSpec({self.mode}, {self.values})"


def _grip_value(text):
    v = float(text)
    if not (GRIP_MIN <= v <= GRIP_MAX):
        raise ValueError(f"grip {v} out of range {GRIP_MIN}..{GRIP_MAX}")
    return round(v, GRIP_DP)


def parse_grip_spec(text):
    """None = off/stock (grip is never touched). Otherwise a GripSpec."""
    t = str(text).strip().lower()
    if t in ("", "off", "stock", "none"):
        return None
    if "," in t:
        vals = [_grip_value(p) for p in t.split(",") if p.strip() != ""]
        if not vals:
            raise ValueError(f"bad grip list: {text!r}")
        return GripSpec("list", vals)
    if "-" in t.lstrip("-"):
        lo_s, _, hi_s = t.partition("-")
        lo, hi = _grip_value(lo_s), _grip_value(hi_s)
        if lo > hi:
            raise ValueError(f"bad grip range (lo > hi): {text!r}")
        return GripSpec("range", (lo, hi))
    return GripSpec("fixed", (_grip_value(t),))
