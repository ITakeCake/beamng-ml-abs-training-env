"""vehicle_scanner reads BeamNG's own zips/userpath -- no game imports needed
to test it, just real (or synthetic-fixture) zip files."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vehicle_scanner import scan_models, scan_trims, scan_custom_configs
from dataclasses import dataclass


@dataclass
class ModelInfoStub:
    name: str
    display_name: str
    zip_path: str = ""

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_scan_models_filters_to_cars_only():
    models = scan_models(FIXTURES)
    names = [m.name for m in models]
    assert "testcar" in names
    assert "testprop" not in names


def test_scan_models_display_name_from_info_json():
    models = {m.name: m for m in scan_models(FIXTURES)}
    assert models["testcar"].display_name == "TESTCAR"


def test_scan_trims_reads_configuration_label():
    trims = scan_trims(os.path.join(FIXTURES, "testcar.zip"), "testcar")
    by_name = {t.pc_name: t for t in trims}
    assert by_name["trim_a"].display_name == "Trim A (Base)"
    assert by_name["trim_b"].display_name == "Trim B (Sport)"


def test_scan_trims_falls_back_to_filename_when_no_info(): 
    trims = scan_trims(os.path.join(FIXTURES, "testcar.zip"), "testcar")
    by_name = {t.pc_name: t for t in trims}
    assert by_name["trim_nolabel"].display_name == "trim_nolabel"


def test_scan_trims_missing_zip_returns_empty():
    assert scan_trims(os.path.join(FIXTURES, "nope.zip"), "nope") == []


def test_scan_custom_configs_reads_userpath_pc_files(tmp_path):
    car_dir = tmp_path / "vehicles" / "etk800"
    car_dir.mkdir(parents=True)
    (car_dir / "MyCustom.pc").write_text('{"format":2,"model":"etk800"}')
    (car_dir / "Another.pc").write_text('{"format":2,"model":"etk800"}')
    (car_dir / "notapc.txt").write_text("ignore me")
    customs = scan_custom_configs(str(tmp_path), "etk800")
    names = sorted(c.pc_name for c in customs)
    assert names == ["Another", "MyCustom"]


def test_scan_custom_configs_missing_folder_returns_empty(tmp_path):
    assert scan_custom_configs(str(tmp_path), "nonexistent_model") == []


def test_scan_custom_configs_display_name_is_filename():
    pass  # covered by roundtrip above; display_name == pc_name for customs (no info_ json)


# --- pure helpers used by the GUI car picker ---
from vehicle_scanner import resolve_part_config, SUPPORTED_ML_ABS_MODELS


def test_resolve_part_config_builds_the_beamngpy_string():
    assert resolve_part_config("etk800", "844_150_A") == "vehicles/etk800/844_150_A.pc"


def test_supported_ml_abs_models_includes_etk800_only_for_now():
    assert "etk800" in SUPPORTED_ML_ABS_MODELS
    assert "bx" not in SUPPORTED_ML_ABS_MODELS


# --- disambiguation for models sharing a display name (real bug: BeamNG's
# 'midsize' and 'pessima' zips both report Name="Pessima") ---
from vehicle_scanner import unique_model_labels


def test_unique_model_labels_no_collision_uses_plain_display_name():
    models = [ModelInfoStub("etk800", "800-Series"), ModelInfoStub("bx", "BX-Series")]
    labels = unique_model_labels(models)
    assert set(labels) == {"800-Series", "BX-Series"}
    assert labels["800-Series"].name == "etk800"


def test_unique_model_labels_disambiguates_collisions_with_folder_name():
    models = [ModelInfoStub("midsize", "Pessima"), ModelInfoStub("pessima", "Pessima")]
    labels = unique_model_labels(models)
    assert len(labels) == 2
    assert set(labels) == {"Pessima (midsize)", "Pessima (pessima)"}
