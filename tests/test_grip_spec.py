"""Grip spec parsing: three modes Blake specified -- off/stock (never touch),
a concrete level, or randomized between runs. Pure, no game imports."""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residual_core import parse_grip_spec, GripSpec


def test_off_variants_mean_never_touch_grip():
    for text in ("", "off", "stock", "none", "OFF", "  Stock  "):
        assert parse_grip_spec(text) is None, text


def test_fixed_level():
    spec = parse_grip_spec("0.6")
    assert spec.mode == "fixed"
    assert spec.draw(random.Random(0)) == 0.6
    assert spec.draw(random.Random(99)) == 0.6      # never varies


def test_range_is_continuous_random_between_runs():
    spec = parse_grip_spec("0.4-1.0")
    assert spec.mode == "range"
    draws = {spec.draw(random.Random(s)) for s in range(20)}
    assert len(draws) > 1                            # actually varies
    assert all(0.4 <= d <= 1.0 for d in draws)


def test_list_picks_randomly_among_the_given_levels():
    spec = parse_grip_spec("0.5,0.75,1.0")
    assert spec.mode == "list"
    draws = {spec.draw(random.Random(s)) for s in range(30)}
    assert draws <= {0.5, 0.75, 1.0}
    assert len(draws) > 1


def test_draw_is_rounded_so_it_can_key_a_calibration_row():
    """config_key rounds to 3 dp; draws must round the same way or a
    normalized run would look up a row that can never match."""
    spec = parse_grip_spec("0.4-1.0")
    for s in range(20):
        d = spec.draw(random.Random(s))
        assert d == round(d, 3)


def test_rejects_out_of_range_and_malformed():
    for bad in ("0", "-0.5", "2.5", "abc", "0.4-", "0.9-0.4", "0.5,abc"):
        with pytest.raises(ValueError):
            parse_grip_spec(bad)


def test_above_stock_grip_is_allowed():
    """Multipliers > 1 are legitimate (stickier-than-stock tires), so the
    range is not capped at 1.0 -- only absurd values are rejected."""
    assert parse_grip_spec("1.5").draw(random.Random(0)) == 1.5


def test_range_and_list_of_one_are_effectively_fixed():
    assert parse_grip_spec("0.6-0.6").draw(random.Random(1)) == 0.6
    assert parse_grip_spec("0.6").draw(random.Random(1)) == 0.6


def test_needs_calibration_rows_flags_the_continuous_case():
    """A continuous range can draw a value no calibration row will ever match;
    fixed and list draws are enumerable, so they can be pre-calibrated."""
    assert parse_grip_spec("0.4-1.0").needs_continuous_calibration is True
    assert parse_grip_spec("0.5,0.75").needs_continuous_calibration is False
    assert parse_grip_spec("0.6").needs_continuous_calibration is False


def test_levels_enumerates_what_must_be_calibrated():
    assert parse_grip_spec("0.6").levels() == [0.6]
    assert parse_grip_spec("0.5,0.75,1.0").levels() == [0.5, 0.75, 1.0]
    assert parse_grip_spec("0.4-1.0").levels() is None   # not enumerable
