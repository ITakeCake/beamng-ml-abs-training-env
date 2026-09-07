"""A .pc without the ML ABS part cannot run in-car training: the env spawns the
car, slams the brakes, and waits for a controller that will never report in.
The failure surfaced as "active=None, warmup=None, tickseq=None" two minutes
after the game booted, naming the symptom, not the cause. Checked up front."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vehicle_scanner import check_ml_abs_car, pc_abs_part

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARS = os.path.join(REPO, "assets", "cars", "etk800")


def _pc(tmp_path, name, parts):
    p = tmp_path / name
    p.write_text(json.dumps({"format": 2, "parts": parts}), encoding="utf-8")
    return str(p)


def test_the_real_training_car_passes():
    assert check_ml_abs_car(os.path.join(CARS, "Machine-Trainer-Boy-V2-MLABS.pc")) is None


def test_the_car_that_actually_broke_a_run_is_refused():
    """Machine-Trainer-Boy.pc names no ABS slot at all, the run died on it."""
    problem = check_ml_abs_car(os.path.join(CARS, "Machine-Trainer-Boy.pc"))
    assert problem and "no ABS slot" in problem


def test_the_reference_cars_are_refused_with_their_own_reasons():
    """Both are legitimate cars for CALIBRATION, and useless for training."""
    empty = check_ml_abs_car(os.path.join(CARS, "Machine-Trainer-Boy-V2.pc"))
    stock = check_ml_abs_car(os.path.join(CARS, "Machine-Trainer-Boy-V2-STOCKABS.pc"))
    assert "empty" in empty and "lockup reference" in empty
    assert "etk_DSE_ABS" in stock and "not the ML ABS controller" in stock


def test_every_refusal_says_what_to_do_instead():
    """An error that only says what is wrong leaves the user where they were."""
    for name in ("Machine-Trainer-Boy.pc", "Machine-Trainer-Boy-V2.pc",
                 "Machine-Trainer-Boy-V2-STOCKABS.pc"):
        problem = check_ml_abs_car(os.path.join(CARS, name))
        assert "Pick a configuration whose ABS is set to ML ABS" in problem


def test_a_bom_prefixed_pc_is_read_not_rejected(tmp_path):
    """BeamNG writes these with a UTF-8 BOM often enough to matter, one broke
    the game's own vehicle cache earlier in this project."""
    p = tmp_path / "bom.pc"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(
        {"parts": {"etk_DSE_ABS": "etk_DSE_ABS_MTB_ML"}}).encode())
    assert check_ml_abs_car(str(p)) is None


def test_a_differently_named_abs_slot_still_resolves(tmp_path):
    """Slot names vary per car (etk800_ABS, bastion_DSE_ABS...); the check must
    not be hardcoded to the ETK one."""
    path = _pc(tmp_path, "other.pc", {"bastion_DSE_ABS": "etk_DSE_ABS_MTB_ML"})
    assert check_ml_abs_car(path) is None
    assert pc_abs_part(path) == "etk_DSE_ABS_MTB_ML"


def test_a_car_with_no_abs_slot_reports_none_rather_than_empty(tmp_path):
    """"no ABS slot" and "ABS slot set to empty" are different problems."""
    assert pc_abs_part(_pc(tmp_path, "n.pc", {"etk800_engine": "x"})) is None
    assert pc_abs_part(_pc(tmp_path, "e.pc", {"etk_DSE_ABS": ""})) == ""


def test_an_unreadable_file_is_reported_not_raised(tmp_path):
    p = tmp_path / "bad.pc"
    p.write_text("{not json", encoding="utf-8")
    assert "could not read" in check_ml_abs_car(str(p))
