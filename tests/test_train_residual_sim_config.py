"""build_sim_config is pure argparse-Namespace -> SimConfig logic -- test it
directly rather than exercising the full trainer (which needs a live game)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_residual import build_sim_config
from sim_config import SimConfig


def _ns(**over):
    base = dict(settings="/nonexistent/settings.json", game=None, game_folder=None,
                userpath=None, windowed=False, map=None, cpu_pinning=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_defaults_when_nothing_passed_and_no_settings_file():
    cfg = build_sim_config(_ns())
    assert cfg == SimConfig()


def test_cli_flags_override_defaults():
    cfg = build_sim_config(_ns(game="drive", game_folder="/g", userpath="/u",
                               windowed=True, map="italy", cpu_pinning=True))
    assert cfg.game == "drive"
    assert cfg.game_folder == "/g"
    assert cfg.userpath == "/u"
    assert cfg.headless is False
    assert cfg.map == "italy"
    assert cfg.cpu_pinning is True


def test_settings_file_used_as_base_then_cli_overrides_only_given_fields(tmp_path):
    from sim_config import save
    settings_path = str(tmp_path / "settings.json")
    save(SimConfig(game="drive", port=555, map="utah"), settings_path)
    cfg = build_sim_config(_ns(settings=settings_path, map="italy"))
    assert cfg.game == "drive"      # from file, not overridden
    assert cfg.port == 555          # from file
    assert cfg.map == "italy"       # CLI override wins


# --- calibration loading (reward normalization) ---
import json as _json
from train_residual import load_calibration
from calibration import CalibrationTable, config_key, summarize


def _cal_ns(**over):
    base = dict(calibration=None, calibration_car=None, vehicle_pc=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_load_calibration_missing_file_returns_none_and_the_path(tmp_path):
    table, path = load_calibration(_cal_ns(calibration=str(tmp_path / "nope.json")))
    assert table is None
    assert path.endswith("nope.json")


def test_load_calibration_reads_a_real_table(tmp_path):
    p = tmp_path / "etk800.json"
    t = CalibrationTable(car="etk800")
    key = config_key(grip=1.0, speed_mph=60, radius_m=None)
    t.put(key, "slam", summarize([1.03]))
    t.put(key, "stock", summarize([1.19]))
    t.save(str(p))
    table, path = load_calibration(_cal_ns(calibration=str(p)))
    assert table is not None
    assert table.references(key) == (1.03, 1.19)


def test_load_calibration_derives_the_car_from_vehicle_pc(tmp_path):
    _table, path = load_calibration(_cal_ns(vehicle_pc="vehicles/bx/MyCar.pc"))
    assert os.path.basename(path) == "bx.json"


def test_load_calibration_defaults_to_the_reference_car(tmp_path):
    _table, path = load_calibration(_cal_ns())
    assert os.path.basename(path) == "etk800.json"
