"""Pure geometry/bookkeeping for braking-in-a-turn. NO game imports.

Scope (PLAN_V2 section 4): ONE constant-radius corner. The reward is not
rewritten, v5.0 already grades yaw as `|yaw_rate - target_yaw_rate|`, so a
corner is a *target* change. Two things have to move:

  1. `target_yaw_rate = +-v/R`, recomputed EVERY step from the current speed.
     A value fixed at brake onset would keep demanding entry-speed rotation
     from a car that is slowing to a stop.
  2. The crash rule (90 deg heading deviation) has to measure deviation from
     the ARC's tangent at the car's current position along the arc, not from
     the heading it entered with, a correctly driven corner legitimately
     rotates the car most of the way to 90 deg.

The tangent heading is just the integral of the target yaw rate:
`dtheta = v*dt/R = ds/R`, i.e. heading change per unit arc length. Integrating
`target_yaw_rate` with the car's ACTUAL speed therefore tracks the tangent at
the arc position the car has actually reached, no separate path tracking, and
a car that stops early simply stops rotating the target too.

Steering: open loop, one fixed angle held through the stop. A closed-loop arc
follower would react differently to each car's behaviour, which would turn
"stock ABS's advantage in a corner" into "the steering controller's reaction to
stock ABS", the calibration only means something if slam, stock and the model
all drive the identical procedure. The angle that yields a given radius is not
known a priori (wheelbase, understeer, grip), so it is SOUGHT once per
(car, speed, grip, radius) and cached: hold an angle, measure the steady-state
yaw rate, R = v / yaw_rate, correct, repeat.
"""
import math

RADIUS_MIN = 10.0        # tighter than this and a 60 mph entry is not a brake test
RADIUS_MAX = 500.0       # beyond this the arc is straight to within the yaw deadzone
RADIUS_DP = 1            # matches calibration.config_key's f"{radius:.1f}"

STEERING_MAX = 1.0       # normalized steering input, what vehicle.control takes
STEERING_SEEK_MAX_ITERS = 8
STEERING_SEEK_TOL = 0.03      # |R_measured/R_target - 1| this small = converged
STEERING_SEEK_GAIN = 0.8      # <1: understeer makes the map nonlinear, so undershoot

LEFT = 1
RIGHT = -1

# Guard for the parent's `heading_error = abs(current - target)`: both sides are
# raw obj:getDirection() radians, which wrap at +-pi. Straight-line episodes never
# noticed because the target never moved. A corner rotates the target up to ~90
# deg, so an episode that starts near +-pi would cross the branch cut mid-stop and
# the parent would read a ~2pi error, an instant, entirely fictional CRASH.
HEADING_BRANCH_MARGIN = 0.20  # rad of clearance demanded from +-pi


def wrap_pi(angle):
    """Fold an angle into (-pi, pi]."""
    a = math.fmod(float(angle) + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def target_yaw_rate(speed_ms, radius_m, direction=LEFT):
    """rad/s the car should be rotating at to hold the arc at this speed.
    radius None/0 (straight) => 0.0, which is exactly what v5.0 uses today."""
    if not radius_m:
        return 0.0
    return float(direction) * float(speed_ms) / float(radius_m)


def radius_from_yaw(speed_ms, yaw_rate):
    """R = v / omega. None when the yaw rate is too small to divide by (the car
    is going straight, so no finite radius is being held)."""
    if abs(float(yaw_rate)) < 1e-4:
        return None
    return abs(float(speed_ms) / float(yaw_rate))


def arc_radians(start_speed_ms, radius_m, avg_decel_ms2):
    """How far around the arc a stop from `start_speed_ms` will carry the car:
    arc length v^2/(2a) divided by R. Used to check the episode cannot cross the
    heading branch cut, and to sanity-check that a radius is even reachable."""
    if not radius_m or avg_decel_ms2 <= 0.0:
        return 0.0
    arc_len = (float(start_speed_ms) ** 2) / (2.0 * float(avg_decel_ms2))
    return arc_len / float(radius_m)


def heading_branch_is_safe(start_heading, arc_rad, direction=LEFT,
                           margin=HEADING_BRANCH_MARGIN):
    """True when a corner starting at `start_heading` and sweeping `arc_rad` in
    `direction` stays clear of the +-pi branch cut for the whole stop."""
    end = float(start_heading) + float(direction) * abs(float(arc_rad))
    return abs(start_heading) <= math.pi - margin and abs(end) <= math.pi - margin


class HeadingTracker:
    """Keeps UNWRAPPED current and target headings, and hands back the value to
    write into the parent's `self.target_heading` so that the parent's own
    `abs(current_raw - target_heading)` evaluates to the true error.

    The parent overwrites `current_heading` from telemetry inside its step() and
    then subtracts raw values, so the only lever available from a subclass is
    where the target sits. Placing the target on the same branch as the last raw
    reading makes the parent's subtraction continuous, and (because the tracker
    knows the true signed error) exactly right.

    A straight-line episode leaves the target at the start heading, i.e. the
    tracker returns `start_heading` forever, byte-identical to today."""

    def __init__(self, start_heading):
        self.start_heading = float(start_heading)
        self.raw = float(start_heading)          # last raw reading seen
        self.current = float(start_heading)      # unwrapped
        self.target = float(start_heading)       # unwrapped
        self.arc_rad = 0.0                       # |target rotation| so far

    def observe(self, raw_heading):
        """Feed the raw (wrapped) heading the game just reported."""
        raw = float(raw_heading)
        self.current += wrap_pi(raw - self.raw)
        self.raw = raw

    def advance_target(self, yaw_rate, dt):
        """Rotate the arc tangent by one step of the commanded yaw rate."""
        step = float(yaw_rate) * float(dt)
        self.target += step
        self.arc_rad += abs(step)

    @property
    def error(self):
        """Signed deviation of the car from the arc tangent, unwrapped."""
        return self.current - self.target

    def parent_target(self):
        """What to assign to the parent's `self.target_heading` for the step it
        is about to take: the last raw heading shifted by the true signed error,
        so `abs(next_raw - this)` == the true error at that step."""
        return self.raw - self.error


class CornerSpec:
    """One constant-radius corner. `steering` is the open-loop normalized wheel
    input that was measured to hold `radius_m`, None until a seek has run, and
    training refuses to start without it rather than guessing an angle."""

    def __init__(self, radius_m, direction=LEFT, steering=None):
        self.radius_m = round(float(radius_m), RADIUS_DP)
        self.direction = LEFT if int(direction) >= 0 else RIGHT
        self.steering = None if steering is None else float(steering)
        if not (RADIUS_MIN <= self.radius_m <= RADIUS_MAX):
            raise ValueError(
                f"radius {self.radius_m} out of range {RADIUS_MIN}..{RADIUS_MAX}")

    @property
    def signed_steering(self):
        if self.steering is None:
            raise ValueError(
                f"no steering angle known for radius {self.radius_m} m, run the "
                "steering seek (reference runner) before training this corner")
        return self.direction * abs(self.steering)

    def yaw_target(self, speed_ms):
        return target_yaw_rate(speed_ms, self.radius_m, self.direction)

    def __repr__(self):
        return (f"CornerSpec({self.radius_m}, "
                f"{'L' if self.direction == LEFT else 'R'}, steering={self.steering})")

    def __eq__(self, other):
        return (isinstance(other, CornerSpec)
                and other.radius_m == self.radius_m
                and other.direction == self.direction
                and other.steering == self.steering)


def parse_corner_spec(text):
    """None = straight (no corner at all, the v5.0 behaviour). Otherwise
    "50", "50L", "50R" (metres, L default)."""
    t = str(text).strip().lower()
    if t in ("", "off", "none", "straight"):
        return None
    direction = LEFT
    if t.endswith("l"):
        t = t[:-1]
    elif t.endswith("r"):
        t, direction = t[:-1], RIGHT
    try:
        radius = float(t)
    except ValueError:
        raise ValueError(f"bad corner spec: {text!r}")
    return CornerSpec(radius, direction)


def radius_for_lateral_g(speed_ms, lateral_g):
    """The radius that loads the tires to `lateral_g` at this speed.

    How a corner should be chosen. Braking in a turn is only a test of ABS if
    there is grip left to brake with: at the cornering limit the tires are
    already saturated laterally and no brake force is available at all. Real
    braking-in-a-turn procedures corner at a fraction of the limit (~0.4 g) for
    exactly this reason."""
    return (float(speed_ms) ** 2) / (float(lateral_g) * 9.81)


def lateral_g_for_radius(speed_ms, radius_m):
    """v^2/R in g. What a given corner actually asks of the tires."""
    return (float(speed_ms) ** 2) / (float(radius_m) * 9.81)


def seek_is_saturated(history, tol=0.02):
    """True when more steering has stopped buying radius, the car is
    understeering at its lateral limit, so the target is unreachable however
    far the wheel is turned. Detected rather than waited out: the remaining
    probes cost minutes and cannot succeed, and the smallest radius reached so
    far is the useful answer to report back."""
    if len(history) < 3:
        return False
    (_, r_prev), (_, r_last) = history[-2], history[-1]
    best = min(r for _, r in history)
    # radius no longer improving, and the last probe is not the best one
    return r_last >= r_prev * (1.0 - tol) and r_last > best * (1.0 - tol)


def steering_seek_update(steering, measured_radius, target_radius,
                         gain=STEERING_SEEK_GAIN):
    """Next angle to try. Radius falls as steering rises, so an angle that came
    out too wide needs to grow by the radius ratio. Damped by `gain` because
    understeer makes the relationship superlinear near the grip limit, an
    undamped step overshoots into a slide and measures nothing useful."""
    s = abs(float(steering))
    if s <= 0.0:
        raise ValueError("steering seek needs a non-zero starting angle")
    if not measured_radius or measured_radius <= 0.0:
        raise ValueError("no measured radius (car was not yawing), "
                         "increase the starting angle")
    ratio = float(measured_radius) / float(target_radius)
    nxt = s * (1.0 + gain * (ratio - 1.0))
    return min(STEERING_MAX, max(1e-3, nxt))


def steering_seek_converged(measured_radius, target_radius, tol=STEERING_SEEK_TOL):
    if not measured_radius or measured_radius <= 0.0:
        return False
    return abs(float(measured_radius) / float(target_radius) - 1.0) <= tol


def initial_steering_guess(radius_m, wheelbase_m=2.8, steering_lock_rad=0.52):
    """Ackermann first guess: road-wheel angle ~ wheelbase / R, expressed as a
    fraction of full lock. Only a seed for the seek, understeer means the real
    angle is always larger, which is why the seek exists."""
    angle = float(wheelbase_m) / float(radius_m)
    return min(STEERING_MAX, max(1e-3, angle / float(steering_lock_rad)))
