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
