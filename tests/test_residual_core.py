import math, pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residual_core import residual_to_brakes, parse_speeds, parse_pedal_spec

def test_zero_release_is_full_pedal_slam():
    assert residual_to_brakes([0.0, 0.0], 1.0) == (1.0, 1.0, 1.0, 1.0)

def test_full_release_hits_floor_not_zero():
    fr, fl, rr, rl = residual_to_brakes([1.0, 1.0], 1.0)
    assert fr == fl == rr == rl == 0.01

def test_axle_locked_symmetry():
    fr, fl, rr, rl = residual_to_brakes([0.3, 0.7], 1.0)
    assert fr == fl and rr == rl and not math.isclose(fr, rr)

def test_multiplicative_scaling_with_partial_pedal():
    fr, _, rr, _ = residual_to_brakes([0.5, 0.0], 0.6)
    assert math.isclose(fr, 0.3, abs_tol=1e-9)
    assert math.isclose(rr, 0.6, abs_tol=1e-9)

def test_action_clipped_into_01():
    fr, _, rr, _ = residual_to_brakes([-0.5, 1.5], 1.0)
    assert fr == 1.0 and rr == 0.01

def test_parse_speeds_happy():
    assert parse_speeds("70, 120,150") == [70, 120, 150]

def test_parse_speeds_rejects_junk():
    for bad in ("", "70,,80", "abc", "10", "200"):
        with pytest.raises(ValueError):
            parse_speeds(bad)

def test_parse_pedal_off_and_range_and_scalar():
    assert parse_pedal_spec("off") is None
    assert parse_pedal_spec("") is None
    assert parse_pedal_spec("0.4-1.0") == (0.4, 1.0)
    assert parse_pedal_spec("0.7") == (0.7, 0.7)

def test_parse_pedal_rejects_bad():
    for bad in ("0.05", "1.2", "0.9-0.4", "x"):
        with pytest.raises(ValueError):
            parse_pedal_spec(bad)

def test_parent_action_inversion_roundtrip():
    # env mailboxes parent_action st. parent's 0.01+0.99*a reproduces our brakes
    import numpy as np
    fr, fl, rr, rl = residual_to_brakes([0.25, 0.6], 0.9)
    parent_action = (np.array([fr, fl, rr, rl]) - 0.01) / 0.99
    back = 0.01 + 0.99 * parent_action
    assert np.allclose(back, [fr, fl, rr, rl], atol=1e-12)
    assert (parent_action >= 0).all() and (parent_action <= 1).all()
