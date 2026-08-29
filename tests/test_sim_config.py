import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_config import (
    SimConfig, default_userpath, find_exe, guess_version_from_folder,
    validate, load, save, DEFAULTS,
)


def test_defaults_are_tech_headless_smallgrid():
    cfg = SimConfig()
    assert cfg.game == "tech"
    assert cfg.headless is True
    assert cfg.map == "smallgrid"
    assert cfg.port == 64291
    assert cfg.cpu_pinning is False


def test_default_userpath_drive_vs_tech():
    p_tech = default_userpath("tech")
    p_drive = default_userpath("drive")
    assert "BeamNG.tech" in p_tech
    assert "BeamNG.drive" in p_drive
    assert p_tech != p_drive


def test_find_exe_returns_none_for_missing_folder(tmp_path):
    assert find_exe(str(tmp_path), "tech") is None


def test_find_exe_finds_the_right_exe(tmp_path):
    (tmp_path / "BeamNG.tech.exe").write_text("")
    (tmp_path / "BeamNG.drive.exe").write_text("")
    assert find_exe(str(tmp_path), "tech").endswith("BeamNG.tech.exe")
    assert find_exe(str(tmp_path), "drive").endswith("BeamNG.drive.exe")


def test_guess_version_from_folder_name():
    assert guess_version_from_folder(r"C:\x\BeamNG.tech.v0.37.6.0") == "0.37.6.0"
    assert guess_version_from_folder(r"C:\x\SomeRandomFolder") is None


def test_validate_flags_missing_exe_and_headless_on_drive(tmp_path):
    cfg = SimConfig(game="drive", game_folder=str(tmp_path), headless=True)
    problems = validate(cfg)
    assert any("exe" in p.lower() for p in problems)
    assert any("headless" in p.lower() for p in problems)


def test_validate_passes_for_a_good_tech_config(tmp_path):
    (tmp_path / "BeamNG.tech.exe").write_text("")
    cfg = SimConfig(game="tech", game_folder=str(tmp_path), headless=True)
    assert validate(cfg) == []


def test_save_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "settings.json")
    cfg = SimConfig(game="drive", port=1234, headless=False, map="italy")
    save(cfg, path)
    loaded = load(path)
    assert loaded.game == "drive"
    assert loaded.port == 1234
    assert loaded.headless is False
    assert loaded.map == "italy"


def test_load_missing_file_returns_defaults(tmp_path):
    loaded = load(str(tmp_path / "nope.json"))
    assert loaded == SimConfig()


def test_load_corrupt_file_returns_defaults_not_crash(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    loaded = load(str(p))
    assert loaded == SimConfig()


# --- env-seam pure helpers (abs_env_residual.py) ---
from abs_env_residual import _resolve_launch_kwargs, _resolve_cpu_cores


def test_resolve_launch_kwargs_none_untouched_when_no_overrides_needed():
    cfg = SimConfig(port=999, game_folder="", userpath="/custom/path")
    headless, map_name, kwargs = _resolve_launch_kwargs(cfg, {})
    assert headless is True
    assert map_name == "smallgrid"
    assert kwargs["port"] == 999
    assert kwargs["user_path"] == "/custom/path"
    assert "bng_home" not in kwargs   # game_folder empty -> don't override the reference default


def test_resolve_launch_kwargs_sets_bng_home_when_game_folder_given():
    cfg = SimConfig(game_folder=r"C:\Games\BeamNG.tech")
    _, _, kwargs = _resolve_launch_kwargs(cfg, {})
    assert kwargs["bng_home"] == r"C:\Games\BeamNG.tech"


def test_resolve_launch_kwargs_explicit_caller_kwarg_wins():
    cfg = SimConfig(port=999)
    _, _, kwargs = _resolve_launch_kwargs(cfg, {"port": 111})
    assert kwargs["port"] == 111


def test_resolve_launch_kwargs_userpath_auto_guessed_when_blank():
    cfg = SimConfig(game="tech", userpath="")
    _, _, kwargs = _resolve_launch_kwargs(cfg, {})
    assert "BeamNG.tech" in kwargs["user_path"]


def test_resolve_cpu_cores_none_when_pinning_off():
    cfg = SimConfig(cpu_pinning=False)
    assert _resolve_cpu_cores(cfg, total_cores=16) is None


def test_resolve_cpu_cores_explicit_lists():
    cfg = SimConfig(cpu_pinning=True, python_cores=(0, 1), beamng_cores=(2, 3))
    assert _resolve_cpu_cores(cfg, total_cores=16) == ([0, 1], [2, 3])


def test_resolve_cpu_cores_auto_fills_beamng_cores():
    cfg = SimConfig(cpu_pinning=True, python_cores=(0, 1), beamng_cores=())
    py, bng = _resolve_cpu_cores(cfg, total_cores=4)
    assert py == [0, 1]
    assert bng == [2, 3]
