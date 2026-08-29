"""calibration.py: the measured per-config reference table that the normalized
reward anchors on. Pure data/math -- no game imports."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import (
    config_key, summarize, normalized_g, CalibrationTable, table_hash,
    STRAIGHT,
)


def test_config_key_is_canonical_and_stable():
    a = config_key(grip=1.0, speed_mph=60, radius_m=None)
    b = config_key(grip=1.0, speed_mph=60, radius_m=STRAIGHT)
    assert a == b                       # None and STRAIGHT mean the same thing
    assert a == config_key(grip=1.000, speed_mph=60.0, radius_m=None)  # 1.0 == 1.000


def test_config_key_distinguishes_grip_speed_radius():
    base = config_key(grip=1.0, speed_mph=60, radius_m=None)
    assert config_key(grip=0.5, speed_mph=60, radius_m=None) != base
    assert config_key(grip=1.0, speed_mph=90, radius_m=None) != base
    assert config_key(grip=1.0, speed_mph=60, radius_m=50) != base


def test_summarize_uses_median_not_mean():
    # median is robust to the one bad episode a live run inevitably produces
    s = summarize([1.00, 1.02, 5.0])
    assert abs(s["median"] - 1.02) < 1e-9
    assert s["n"] == 3
    assert s["values"] == [1.00, 1.02, 5.0]


def test_summarize_spread_reports_min_max():
    s = summarize([0.9, 1.1, 1.0])
    assert abs(s["min"] - 0.9) < 1e-9
    assert abs(s["max"] - 1.1) < 1e-9


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_normalized_g_anchors_slam_at_zero_and_stock_at_one():
    assert abs(normalized_g(0.60, slam_g=0.60, stock_g=1.00) - 0.0) < 1e-9
    assert abs(normalized_g(1.00, slam_g=0.60, stock_g=1.00) - 1.0) < 1e-9


def test_normalized_g_scales_by_the_room_stock_had():
    # dry: stock beat slam by 0.40 g. Beating stock by 0.04 g -> +0.10 normalized.
    assert abs(normalized_g(1.04, slam_g=0.60, stock_g=1.00) - 1.10) < 1e-9
    # ice: stock beat slam by only 0.04 g, so a 0.004 g win is the same
    # FRACTION of stock's own margin -> also +0.10 normalized.
    assert abs(normalized_g(0.344, slam_g=0.30, stock_g=0.34) - 1.10) < 1e-9


def test_normalized_g_below_slam_goes_negative():
    assert normalized_g(0.50, slam_g=0.60, stock_g=1.00) < 0.0


def test_normalized_g_refuses_a_degenerate_reference_gap():
    # stock no better than lockup -> the scale is meaningless, must not divide by ~0
    with pytest.raises(ValueError):
        normalized_g(1.0, slam_g=1.00, stock_g=1.00)


def test_table_roundtrip(tmp_path):
    path = str(tmp_path / "etk800.json")
    t = CalibrationTable(car="etk800")
    key = config_key(grip=1.0, speed_mph=60, radius_m=None)
    t.put(key, "slam", summarize([1.02, 1.03, 1.01]))
    t.put(key, "stock", summarize([1.20, 1.22, 1.21]))
    t.save(path)

    loaded = CalibrationTable.load(path)
    assert loaded.car == "etk800"
    row = loaded.get(key)
    assert abs(row["slam"]["median"] - 1.02) < 1e-9
    assert abs(row["stock"]["median"] - 1.21) < 1e-9


def test_table_get_missing_returns_none(tmp_path):
    t = CalibrationTable(car="etk800")
    assert t.get(config_key(grip=1.0, speed_mph=60, radius_m=None)) is None


def test_table_references_returns_slam_and_stock_medians():
    t = CalibrationTable(car="etk800")
    key = config_key(grip=1.0, speed_mph=60, radius_m=None)
    t.put(key, "slam", summarize([1.0]))
    t.put(key, "stock", summarize([1.2]))
    slam, stock = t.references(key)
    assert (abs(slam - 1.0) < 1e-9) and (abs(stock - 1.2) < 1e-9)


def test_table_references_missing_row_raises_loudly():
    """A missing config must refuse, never silently fall back to another
    config's anchors -- that would train against the wrong zero point."""
    t = CalibrationTable(car="etk800")
    with pytest.raises(KeyError):
        t.references(config_key(grip=0.4, speed_mph=60, radius_m=None))


def test_table_references_incomplete_row_raises_loudly():
    t = CalibrationTable(car="etk800")
    key = config_key(grip=1.0, speed_mph=60, radius_m=None)
    t.put(key, "slam", summarize([1.0]))   # stock never measured
    with pytest.raises(KeyError):
        t.references(key)


def test_table_hash_is_stable_and_content_sensitive(tmp_path):
    t1 = CalibrationTable(car="etk800")
    key = config_key(grip=1.0, speed_mph=60, radius_m=None)
    t1.put(key, "slam", summarize([1.0]))
    t2 = CalibrationTable(car="etk800")
    t2.put(key, "slam", summarize([1.0]))
    assert table_hash(t1) == table_hash(t2)

    t2.put(key, "stock", summarize([1.2]))
    assert table_hash(t1) != table_hash(t2)
