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


def config_key(grip, speed_mph, radius_m):
    """Canonical key for a configuration. Rounded so 1.0 and 1.000 (and 60 vs
    60.0) can never produce two rows for the same physical setup."""
    r = "straight" if radius_m in (None, 0) else f"{float(radius_m):.1f}"
    return f"grip={float(grip):.3f}|speed={float(speed_mph):.1f}|radius={r}"


def summarize(values):
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

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"car": self.car, "rows": self.rows}, fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(car=data.get("car", ""), rows=data.get("rows", {}))


def table_hash(table):
    """Content hash, stamped into every run so a result can always be traced
    to the exact reference numbers it was scored against."""
    blob = json.dumps({"car": table.car, "rows": table.rows},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
