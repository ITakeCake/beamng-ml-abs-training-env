"""The measured per-configuration reference table the normalized reward anchors
on (PLAN_V2.md section 2).

Two references per configuration, both MEASURED, never assumed:
  slam_g  -- zero-action full lockup: the observed floor. Not an ABS, not
             claimed to be optimal; just what this car does on this surface
             with the wheels locked.
  stock_g -- the car's stock BeamNG ABS on the same config. A competitor used
             as a ruler, nothing more.

There is deliberately NO ceiling reference. BeamNG is emergent; no ABS mode
(arcade included) is trusted as a physical limit, so the score is unbounded
above -- nobody knows where the limit is.

Pure data/math: no game imports, no I/O beyond the table's own JSON file.
"""
import dataclasses
import hashlib
import json
import os
import statistics

STRAIGHT = None          # radius sentinel: a straight-line stop
REFERENCES = ("slam", "stock")

# Below this gap, (stock_g - slam_g) is too small to be a meaningful unit --
# stock did essentially nothing over lockup, so "advantage over stock,
# normalized by stock's own margin" is not a scale anyone should divide by.
MIN_REFERENCE_GAP_G = 0.01


# Pedal is quantised to 2 decimals (residual_core.PEDAL_DP) precisely because
# it is part of this key: every distinct value needs its own measured
# slam/stock pair, so 3 decimals would put a "0.5-1.0" range at 501
# uncalibratable levels.
PEDAL_DP = 2
FULL_PEDAL = 1.0


def config_key(grip, speed_mph, radius_m, pedal=FULL_PEDAL):
    """Canonical key for a configuration. Rounded so 1.0 and 1.000 (and 60 vs
    60.0) can never produce two rows for the same physical setup.

    `pedal` is part of the configuration because the references are measured at
    a specific pedal position: a half-pedal stop physically cannot reach the
    full-pedal lockup floor, so scoring it against those anchors marks a
    well-modulated stop as far worse than locking the wheels. Defaults to full
    pedal, which is what every row measured before this existed used, and the
    full-pedal key keeps its original text so those rows still resolve."""
    r = "straight" if radius_m in (None, 0) else f"{float(radius_m):.1f}"
    base = f"grip={float(grip):.3f}|speed={float(speed_mph):.1f}|radius={r}"
    if round(float(pedal), PEDAL_DP) == FULL_PEDAL:
        return base
    return f"{base}|pedal={float(pedal):.2f}"


DETERMINISTIC = "deterministic"


def regime_name(speed_factor=1.0, live=False):
    """How a row was measured. Recorded because two regimes in one table are
    not comparable to each other, and a table IS the ruler -- a silently mixed
    one would shift the zero point for some configurations and not others.

    Measured 2026-08-30, interleaved A/B at 60 mph full pedal: deterministic
    1.0135, live x4 1.0214, x10 1.0091, x25 1.0066 -- all within 0.8%, which is
    smaller than any single arm's own run-to-run spread (~0.02). So the regimes
    are equivalent at that configuration; the label exists because that was
    established for ONE configuration, not proven universally."""
    return DETERMINISTIC if not live else f"live_x{float(speed_factor):g}"


def summarize(values, regime=DETERMINISTIC):
    """Median (robust to the one bad episode a live run always produces) plus
    the raw values and spread, so a noisy config is visible rather than hidden
    behind a single number."""
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("summarize() needs at least one measurement")
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
        "values": vals,
        "regime": regime,
    }


def normalized_g(g, slam_g, stock_g):
    """Map a raw arc-length avg_g onto the config-relative scale:
        0.0 = the measured lockup floor
        1.0 = what stock ABS achieved here
        >1  = beating stock, in units of the margin stock itself had

    Dividing by (stock_g - slam_g) is what makes ice and dry asphalt
    comparable: on dry, stock might beat lockup by 0.40 g, on ice by 0.04 g,
    so the same *fraction* of that margin means the same achievement."""
    gap = float(stock_g) - float(slam_g)
    if gap < MIN_REFERENCE_GAP_G:
        raise ValueError(
            f"degenerate reference gap: stock_g={stock_g:.4f} is not meaningfully "
            f"above slam_g={slam_g:.4f} (gap {gap:.4f} < {MIN_REFERENCE_GAP_G}). "
            f"Nothing can be normalized against this config -- re-measure it, or "
            f"exclude it from the training matrix.")
    return (float(g) - float(slam_g)) / gap


@dataclasses.dataclass
class CalibrationTable:
    car: str
    rows: dict = dataclasses.field(default_factory=dict)
    # Corner configs only: the open-loop steering angle measured to hold the
    # row's radius, keyed the same way the rows are. Stored beside the
    # references because it is part of the procedure they were measured with --
    # a corner reference measured at one angle is not a ruler for a run driven
    # at another.
    steering: dict = dataclasses.field(default_factory=dict)

    def put(self, key, reference, summary):
        if reference not in REFERENCES:
            raise ValueError(f"unknown reference {reference!r}, expected one of {REFERENCES}")
        self.rows.setdefault(key, {})[reference] = summary

    def get(self, key):
        return self.rows.get(key)

    def references(self, key):
        """(slam_g, stock_g) medians for this config. Raises rather than
        falling back to any other row: training against another
        configuration's zero point silently produces meaningless scores."""
        row = self.rows.get(key)
        if row is None:
            raise KeyError(
                f"no calibration row for {key!r} on car {self.car!r} -- run "
                f"'Calibrate baselines' for this configuration before training "
                f"on it (refusing to reuse another config's anchors).")
        missing = [r for r in REFERENCES if r not in row]
        if missing:
            raise KeyError(
                f"calibration row {key!r} on car {self.car!r} is missing "
                f"{missing} -- both references are required to normalize.")
        return row["slam"]["median"], row["stock"]["median"]

    def put_steering(self, key, steering, measured_radius):
        self.steering[key] = {"steering": float(steering),
                              "measured_radius": float(measured_radius)}

    def steering_for(self, key):
        """The measured angle for this corner config. Raises rather than
        returning a guess: an unmeasured angle drives some other radius than
        the one the references were measured on."""
        entry = self.steering.get(key)
        if entry is None:
            raise KeyError(
                f"no steering angle for {key!r} on car {self.car!r} -- run the "
                f"steering seek for this corner before training it.")
        return entry["steering"]

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            payload = {"car": self.car, "rows": self.rows}
            if self.steering:
                payload["steering"] = self.steering
            json.dump(payload, fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(car=data.get("car", ""), rows=data.get("rows", {}),
                   steering=data.get("steering", {}))


def table_regimes(table):
    """Every distinct regime present. More than one means the table mixes
    measurement methods and its rows are not strictly comparable."""
    found = set()
    for row in table.rows.values():
        for entry in row.values():
            found.add(entry.get("regime", DETERMINISTIC))
    return sorted(found)


def table_hash(table):
    """Content hash, stamped into every run so a result can always be traced
    to the exact reference numbers it was scored against."""
    payload = {"car": table.car, "rows": table.rows}
    if table.steering:
        # Absent on a straight-line-only table, so tables stamped before corners
        # existed keep the hash they were stamped with.
        payload["steering"] = table.steering
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
