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
    return (front, front, rear, rear)   # (fr, fl, rr, rl), axle-locked


def wheel_release_to_brakes(action, pedal):
    """[fr, fl, rr, rl] release in [0,1] + pedal -> (fr, fl, rr, rl), no axle lock."""
    out = []
    for i in range(4):
        rel = min(1.0, max(0.0, float(action[i])))
        out.append(min(1.0, max(BRAKE_FLOOR, float(pedal) * (1.0 - rel))))
    return tuple(out)


WHEEL_MODES = ("axle", "independent")
ACT_DIM_FOR = {"axle": 2, "independent": 4}


def axle_view(vec):
    """(front, rear) from either a 2-vector or a 4-vector (fr, fl, rr, rl)."""
    vals = [float(v) for v in vec]
    if len(vals) >= 4:
        return ((vals[0] + vals[1]) * 0.5, (vals[2] + vals[3]) * 0.5)
    if len(vals) == 2:
        return (vals[0], vals[1])
    return (float("nan"), float("nan"))


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


# Pedal is quantised to 2 decimals, and that is a HARD constraint rather than a
# tidiness choice: pedal position is part of the calibration key, so every
# distinct value needs its own measured slam/stock pair (~7 min each). At 3
PEDAL_DP = 2
PEDAL_MIN, PEDAL_MAX = 0.1, 1.0


def _pedal_value(text):
    v = round(float(text), PEDAL_DP)
    if not (PEDAL_MIN <= v <= PEDAL_MAX):
        raise ValueError(
            f"pedal {v} out of range {PEDAL_MIN}..{PEDAL_MAX}")
    return v


class PedalSpec:
    """How the driver's pedal position varies between episodes. One value is
    drawn per episode and HELD for the whole stop, it never moves mid-stop.

    Three modes, mirroring GripSpec:
      fixed  "0.6"            one concrete level
      list   "0.5,0.75,1.0"   randomized between runs, but only among these
      range  "0.5-1.0"        randomized continuously (51 levels at 2 dp)

    `None` (not a PedalSpec) means off: always full pedal.

    A tuple could not express this, because "0.5,1.0" (two levels) and
    "0.5-1.0" (everything between) would both be a 2-tuple."""

    def __init__(self, mode, values):
        self.mode = mode
        self.values = tuple(values)

    @property
    def needs_continuous_calibration(self):
        """A range spans every 2-dp step in it, which is 51 levels for
        0.5-1.0 and ~6 hours of calibration. Callers refuse this with a
        normalized reward and say to use a list instead."""
        return self.mode == "range" and self.values[0] != self.values[1]

    def levels(self):
        """The values this can draw, exactly what needs calibrating."""
        if self.mode in ("fixed", "list"):
            return list(self.values)
        lo, hi = self.values
        if lo == hi:
            return [lo]
        step = 10 ** -PEDAL_DP
        n = int(round((hi - lo) / step))
        return [round(lo + i * step, PEDAL_DP) for i in range(n + 1)]

    def draw(self, rng=None):
        import random as _random
        rng = rng or _random
        if self.mode == "fixed":
            return self.values[0]
        if self.mode == "list":
            return rng.choice(list(self.values))
        lo, hi = self.values
        return lo if lo == hi else round(rng.uniform(lo, hi), PEDAL_DP)

    def __repr__(self):
        return f"PedalSpec({self.mode}, {self.values})"

    def __eq__(self, other):
        return (isinstance(other, PedalSpec) and other.mode == self.mode
                and other.values == self.values)


def parse_pedal_spec(text):
    """None = off (always full pedal). Otherwise a PedalSpec."""
    t = str(text).strip().lower()
    if t in ("", "off", "none"):
        return None
    if "," in t:
        vals = sorted({_pedal_value(p) for p in t.split(",") if p.strip()})
        if not vals:
            raise ValueError(f"bad pedal list: {text!r}")
        return PedalSpec("list", vals)
    lo, _, hi = t.partition("-")
    if hi:
        lo, hi = _pedal_value(lo), _pedal_value(hi)
        if lo > hi:
            raise ValueError(f"bad pedal spec: {text!r} (low above high)")
        return PedalSpec("range", (lo, hi))
    v = _pedal_value(lo)
    return PedalSpec("fixed", (v,))


def pedal_levels(spec):
    """Every pedal value `spec` can draw. None for "off" (full pedal only)."""
    return None if spec is None else spec.levels()


# ---------------------------------------------------------------- tire grip
GRIP_MIN, GRIP_MAX = 0.05, 2.0     # multiplier on the tire's jbeam friction
GRIP_DP = 3                        # must match calibration.config_key's rounding


class GripSpec:
    """How tire grip varies between episodes. Three modes:
      fixed  "0.6"          , one concrete level
      list   "0.5,0.75,1.0" , randomized between runs, but only among these
      range  "0.4-1.0"      , randomized continuously between runs

    `None` (not a GripSpec) means off/stock: never touch grip at all.

    Draws are rounded to the same precision calibration.config_key uses, so a
    drawn level can actually key a calibration row. A continuous range can
    still draw a value nothing was calibrated at, hence
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


# --- network shape ---------------------------------------------------------
# Deployment ceiling, not a training one: the trained weights are exported into
# a Lua controller that runs the network by hand every 0.5 ms physics tick
NET_MAX_LAYERS = 24
NET_MAX_WIDTH = 2048
NET_MIN_WIDTH = 8
DEFAULT_NET_ARCH = [256, 256, 256]


def parse_net_arch(text):
    """"256,256,256" -> [256, 256, 256]. Blank/"default" -> DEFAULT_NET_ARCH.

    Also accepts "3x256" as shorthand for three layers of 256, since that is
    how these are usually spoken about."""
    t = str(text).strip().lower()
    if t in ("", "default", "none"):
        return list(DEFAULT_NET_ARCH)

    if "x" in t and "," not in t:
        count, _, width = t.partition("x")
        try:
            count, width = int(count), int(width)
        except ValueError:
            raise ValueError(f"bad network shape: {text!r} (expected e.g. \"3x256\")")
        if count < 1:
            raise ValueError(f"network needs at least 1 layer, got {count}")
        layers = [width] * count
    else:
        layers = []
        for part in t.split(","):
            part = part.strip()
            if not part:
                raise ValueError(f"bad network shape: {text!r} (empty layer)")
            try:
                layers.append(int(part))
            except ValueError:
                raise ValueError(f"bad layer size {part!r} in {text!r}")

    if not layers:
        raise ValueError(f"bad network shape: {text!r}")
    if len(layers) > NET_MAX_LAYERS:
        raise ValueError(
            f"{len(layers)} layers exceeds the {NET_MAX_LAYERS}-layer limit, the "
            f"exported network is evaluated by hand in Lua every 0.5 ms physics "
            f"tick, and a net too slow to keep up misses ticks silently.")
    for w in layers:
        if not (NET_MIN_WIDTH <= w <= NET_MAX_WIDTH):
            raise ValueError(
                f"layer width {w} out of range {NET_MIN_WIDTH}..{NET_MAX_WIDTH}")
    return layers


def net_arch_repr(layers):
    """Compact form for logs/labels: [256,256,256] -> "3x256"."""
    if layers and all(w == layers[0] for w in layers):
        return f"{len(layers)}x{layers[0]}"
    return ",".join(str(w) for w in layers)
