"""Pedal position is part of the calibration key.

The references are measured at a specific pedal position, and a half-pedal stop
physically cannot reach the full-pedal lockup floor, scored against those
anchors it lands near -2.9, far BELOW "locked wheels", however well it
modulates. Indistinguishable from failing.

Pedal is quantised to 2 decimals for the same reason: each distinct value needs
its own ~7-minute measurement, so 3 decimals would make a "0.5-1.0" range 501
uncalibratable levels."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residual_core import parse_pedal_spec, pedal_levels, PEDAL_DP
from calibration import config_key, normalized_g


# ------------------------------------------------------------ quantisation
def test_pedal_is_quantised_to_two_decimals():
    assert PEDAL_DP == 2
    assert parse_pedal_spec("0.456").values == (0.46,)
    assert parse_pedal_spec("0.454").values == (0.45,)


def test_a_range_spans_two_decimal_steps_not_three():
    """501 levels could never be calibrated; 51 at least can be reasoned about."""
    assert len(pedal_levels(parse_pedal_spec("0.5-1.0"))) == 51


def test_a_list_costs_only_what_was_asked_for():
    """The practical form: calibration cost is chosen, not dictated by width."""
    assert pedal_levels(parse_pedal_spec("0.5,0.6,0.7,0.8,0.9,1.0")) == [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_a_list_and_a_range_of_the_same_endpoints_are_different_things():
    """"0.5,1.0" is two levels; "0.5-1.0" is everything between. A tuple could
    not tell these apart, both are two numbers."""
    assert pedal_levels(parse_pedal_spec("0.5,1.0")) == [0.5, 1.0]
    assert len(pedal_levels(parse_pedal_spec("0.5-1.0"))) == 51


def test_only_a_range_is_flagged_as_uncalibratable():
    assert parse_pedal_spec("0.5-1.0").needs_continuous_calibration
    assert not parse_pedal_spec("0.5,0.75,1.0").needs_continuous_calibration
    assert not parse_pedal_spec("0.6").needs_continuous_calibration


def test_draws_land_on_calibratable_levels():
    import random
    spec = parse_pedal_spec("0.5-1.0")
    levels = set(spec.levels())
    rng = random.Random(0)
    for _ in range(500):
        assert spec.draw(rng) in levels


def test_a_list_only_ever_draws_its_own_levels():
    import random
    spec = parse_pedal_spec("0.5,0.75,1.0")
    rng = random.Random(0)
    assert {spec.draw(rng) for _ in range(200)} <= {0.5, 0.75, 1.0}


# --------------------------------------------------------------- the key
def test_full_pedal_keeps_the_original_key_text():
    """Rows measured before pedal existed were all full-pedal; they must still
    resolve rather than being orphaned."""
    assert config_key(1.0, 60, None) == "grip=1.000|speed=60.0|radius=straight"
    assert config_key(1.0, 60, None, pedal=1.0) == config_key(1.0, 60, None)


def test_a_different_pedal_is_a_different_configuration():
    assert config_key(1.0, 60, None, pedal=0.5) != config_key(1.0, 60, None)
    assert "pedal=0.50" in config_key(1.0, 60, None, pedal=0.5)


def test_the_key_rounds_to_two_decimals_so_one_row_serves_one_level():
    assert config_key(1.0, 60, 0.0, pedal=0.501) == config_key(1.0, 60, 0.0, pedal=0.5)


def test_why_the_key_needs_pedal_at_all():
    """The measured full-pedal anchors, applied to a stop a half-pedal episode
    could plausibly reach: far below the lockup floor, so a well-modulated
    half-pedal stop is scored worse than locking the wheels."""
    slam, stock = 1.0226, 1.1869          # measured, full pedal
    assert normalized_g(0.55, slam, stock) < -2.0


# ------------------------------------------------------ the calibrate button
from gui_cmd import build_calibration_cmd, validate_calibration_settings


def _settings(**over):
    s = dict(algo="sac", speeds="60", pedal_random=False, pedal_spec="0.4-1.0",
             grip="off", corner="straight", total_steps="1000", run_name="r",
             lr="1e-4", buffer_size="100000", tau="0.005", target_entropy="-2.0",
             learning_starts="5000", train_freq="2")
    s.update(over)
    return s


def test_calibrate_measures_the_pedal_levels_from_the_training_boxes():
    cmd = build_calibration_cmd(
        _settings(pedal_random=True, pedal_spec="0.5,0.6,0.7,0.8,0.9,1.0"),
        car="etk800")
    assert cmd[cmd.index("--pedals") + 1] == "0.5,0.6,0.7,0.8,0.9,1.0"


def test_no_pedal_flag_when_randomization_is_off():
    """Off means full pedal, which is the runner's default."""
    assert "--pedals" not in build_calibration_cmd(_settings(), car="etk800")


def test_a_range_is_judged_by_time_not_by_level_count():
    """Fast mode measures a stop in ~4s instead of ~35s, which turns the full
    0.5-1.0 range (51 levels, 306 stops) from about three hours into about
    twenty minutes. A blanket refusal on level count would now be wrong."""
    slow = validate_calibration_settings(
        _settings(pedal_random=True, pedal_spec="0.5-1.0", fast_calibration=False))
    fast = validate_calibration_settings(
        _settings(pedal_random=True, pedal_spec="0.5-1.0", fast_calibration=True))
    assert fast == []                      # ~20 min: fine
    assert slow == []                      # ~3 h: under the 4 h ceiling


def test_a_genuinely_impractical_matrix_is_still_refused():
    """Several speeds and grips multiply on top of the pedal levels; the guard
    is on total time, so it fires when the run stops being something a person
    starts and waits for."""
    problems = validate_calibration_settings(
        _settings(pedal_random=True, pedal_spec="0.4-1.0", speeds="60,90,120",
                  grip="0.5,0.75,1.0", fast_calibration=False))
    assert any("stops" in p and ("h " in p or "hour" in p) for p in problems)


def test_the_refusal_names_fast_mode_as_the_way_out():
    problems = validate_calibration_settings(
        _settings(pedal_random=True, pedal_spec="0.4-1.0", speeds="60,90,120",
                  grip="0.5,0.75,1.0", fast_calibration=False))
    assert any("Fast calibration" in p for p in problems)


def test_fast_mode_reaches_the_runner_with_its_speed_factor():
    cmd = build_calibration_cmd(
        _settings(fast_calibration=True, speed_factor="10"), car="etk800")
    assert "--live" in cmd
    assert cmd[cmd.index("--speed-factor") + 1] == "10"


def test_stepped_mode_passes_no_speed_flags():
    cmd = build_calibration_cmd(_settings(fast_calibration=False), car="etk800")
    assert "--live" not in cmd and "--speed-factor" not in cmd


# ------------------------------------------------------------- regime record
def test_a_row_records_how_it_was_measured():
    """Two regimes in one table are not strictly comparable, and the table IS
    the ruler, a silently mixed one would move the zero point for some
    configurations and not others."""
    from calibration import summarize, regime_name, DETERMINISTIC
    assert summarize([1.0, 1.1])["regime"] == DETERMINISTIC
    assert summarize([1.0], regime_name(10, True))["regime"] == "live_x10"
    assert regime_name(4, True) == "live_x4"
    assert regime_name(10, False) == DETERMINISTIC     # factor is moot when stepped


def test_table_regimes_reports_a_mix():
    from calibration import CalibrationTable, summarize, regime_name, table_regimes
    t = CalibrationTable(car="etk800")
    t.put("a", "slam", summarize([1.0]))
    t.put("b", "slam", summarize([1.0], regime_name(10, True)))
    assert table_regimes(t) == ["deterministic", "live_x10"]


def test_rows_written_before_regimes_existed_read_as_deterministic():
    """The three real rows already in etk800.json predate this field; they were
    measured stepped, so that is what absence must mean."""
    from calibration import CalibrationTable, table_regimes
    t = CalibrationTable(car="x", rows={"k": {"slam": {"median": 1.0}}})
    assert table_regimes(t) == ["deterministic"]


def test_a_list_is_accepted():
    assert validate_calibration_settings(
        _settings(pedal_random=True, pedal_spec="0.5,0.75,1.0")) == []


# ------------------------------------------------- the latch honours the pedal
def _lua():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return open(os.path.join(root, "abstelemetry.lua"), encoding="utf-8").read()


def test_the_slam_latch_holds_the_requested_pedal_not_a_hardcoded_1():
    """updateArmedSlam re-asserts input.brake every 0.5 ms tick. Hardcoded to 1
    it stamps full pedal over whatever Python sent, so a --pedals 0.5 run would
    measure a FULL-pedal stop and file it under a pedal=0.50 key: a silently
    wrong reference, which is worse than a missing one."""
    src = _lua()
    assert "input.brake = slamPedal" in src
    assert "input.brake = 1\n" not in src


def test_arm_brake_slam_takes_a_pedal_and_defaults_to_full():
    """Defaulting to 1 keeps every existing caller (all of training) unchanged."""
    src = _lua()
    assert "function armBrakeSlam(target_ms, pedal)" in src
    assert "slamPedal = pedal or 1" in src


def test_disarm_resets_the_pedal_so_it_cannot_leak_between_episodes():
    src = _lua()
    disarm = src[src.index("local function disarmBrakeSlam()"):]
    assert "slamPedal = 1" in disarm[:200]


def test_the_reference_runner_passes_the_pedal_to_the_latch():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "reference_runner.py"), encoding="utf-8").read()
    assert "armBrakeSlam({target_ms}, {float(pedal)})" in src


def test_the_shipped_mod_copy_has_the_same_fix():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundled = open(os.path.join(root, "assets", "mods", "mtb_ml_abs", "lua",
                                "vehicle", "extensions", "abstelemetry.lua"),
                   encoding="utf-8").read()
    assert "input.brake = slamPedal" in bundled
