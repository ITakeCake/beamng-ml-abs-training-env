import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_config import (
    SimConfig, default_userpath, find_exe, guess_version_from_folder,
    validate, load, save, DEFAULTS, resolved_userpath,
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


def test_default_userpath_does_not_include_current():
    # BeamNG.exe manages a "current" version-subfolder ITSELF under whatever
    # -userpath it's given, passing a path that already ends in "current"
    # makes the game create/use <userpath>/current/current (confirmed live,
    # 2026-08-29: game log showed "userpath = ...\current\current\" after
    # launching with user=default_userpath("tech")).
    assert not default_userpath("tech").rstrip("\\/").endswith("current")
    assert not default_userpath("drive").rstrip("\\/").endswith("current")


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


def test_load_missing_file_returns_defaults(tmp_path, monkeypatch):
    """game_folder is excepted: load() fills a blank one by looking for the
    install, so this asserts the defaults with detection turned off."""
    import sim_config as sc
    monkeypatch.setattr(sc, "autodetect_game_folder", lambda g: None)
    loaded = load(str(tmp_path / "nope.json"))
    assert loaded == SimConfig()


def test_load_corrupt_file_returns_defaults_not_crash(tmp_path, monkeypatch):
    import sim_config as sc
    monkeypatch.setattr(sc, "autodetect_game_folder", lambda g: None)
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    loaded = load(str(p))
    assert loaded == SimConfig()


def test_a_corrupt_file_still_gets_a_detected_game_folder(tmp_path, monkeypatch):
    """Detection is not skipped on the error path, a broken settings.json
    should not also cost the user their install path."""
    import sim_config as sc
    monkeypatch.setattr(sc, "autodetect_game_folder", lambda g: r"C:\detected")
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert load(str(p)).game_folder == r"C:\detected"


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


# --- vehicle_pc override seam (abs_env_residual.py) ---
from abs_env_residual import _maybe_override_vehicle_pc
import abs_env_incar


def test_maybe_override_vehicle_pc_sets_module_global():
    original = abs_env_incar.VEHICLE_PC_INCAR
    try:
        _maybe_override_vehicle_pc("vehicles/bx/custom.pc")
        assert abs_env_incar.VEHICLE_PC_INCAR == "vehicles/bx/custom.pc"
    finally:
        abs_env_incar.VEHICLE_PC_INCAR = original


def test_maybe_override_vehicle_pc_none_leaves_default_untouched():
    original = abs_env_incar.VEHICLE_PC_INCAR
    _maybe_override_vehicle_pc(None)
    assert abs_env_incar.VEHICLE_PC_INCAR == original


# --- content_userpath: the folder where vehicles/mods actually live, one
# level under whatever base userpath the game is launched with ---
from sim_config import content_userpath


def test_content_userpath_appends_current_to_the_base():
    cfg = SimConfig(userpath=r"C:\Games\BeamNG.tech")
    assert content_userpath(cfg) == r"C:\Games\BeamNG.tech\current"


def test_resolved_userpath_normalizes_a_path_the_user_already_pointed_at_current():
    # Real-world mistake: users naturally browse to the *visible* folder
    # (…\current), since that's the one with content in Explorer. Must not
    # double-nest it at launch.
    cfg = SimConfig(userpath=r"C:\Games\BeamNG.tech\current")
    assert resolved_userpath(cfg) == r"C:\Games\BeamNG.tech"
    assert content_userpath(cfg) == r"C:\Games\BeamNG.tech\current"


def test_resolved_userpath_normalizes_trailing_slash_variants():
    cfg = SimConfig(userpath=r"C:\Games\BeamNG.tech\current\\")
    assert resolved_userpath(cfg) == r"C:\Games\BeamNG.tech"


# --- detect_game_version: the ini file BeamNG itself maintains is a far more
# reliable version source than guessing from the folder name (Steam installs
# like BeamNG.drive commonly have no version in their folder name at all --
# confirmed live 2026-08-29 against D:\SteamLibrary\...\BeamNG.drive) ---
from sim_config import detect_game_version


def test_detect_game_version_reads_the_ini_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    ini_dir = tmp_path / "BeamNG"
    ini_dir.mkdir()
    (ini_dir / "BeamNG.drive.ini").write_text("version = 0.39.4.0\ninstallPath = D:\\x\\\n")
    assert detect_game_version("drive", game_folder=r"D:\SteamLibrary\BeamNG.drive") == "0.39.4.0"


def test_detect_game_version_falls_back_to_folder_name_when_no_ini(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert detect_game_version("tech", game_folder=r"C:\x\BeamNG.tech.v0.37.6.0") == "0.37.6.0"


def test_detect_game_version_none_when_neither_source_available(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert detect_game_version("drive", game_folder=r"D:\SteamLibrary\BeamNG.drive") is None


# ----------------------------------------------------- game-folder autodetect
def test_autodetect_only_fills_a_blank_game_folder(monkeypatch):
    """An explicit setting is the user's, right or wrong, and is never
    second-guessed, otherwise pointing at a second install silently fails."""
    import sim_config as sc
    monkeypatch.setattr(sc, "autodetect_game_folder", lambda g: r"C:\detected")
    kept = sc._with_detected_game_folder(sc.SimConfig(game_folder=r"C:\mine"))
    assert kept.game_folder == r"C:\mine"
    filled = sc._with_detected_game_folder(sc.SimConfig(game_folder=""))
    assert filled.game_folder == r"C:\detected"


def test_a_machine_with_no_install_stays_blank_rather_than_guessing(monkeypatch):
    """None means 'ask the user', never an error and never a made-up path."""
    import sim_config as sc
    monkeypatch.setattr(sc, "autodetect_game_folder", lambda g: None)
    cfg = sc._with_detected_game_folder(sc.SimConfig(game_folder=""))
    assert cfg.game_folder == ""
    assert any("not found" in p for p in sc.validate(cfg))


def test_load_fills_the_game_folder_when_settings_json_is_missing(monkeypatch, tmp_path):
    """The first-run case: no settings.json anywhere."""
    import sim_config as sc
    monkeypatch.setattr(sc, "autodetect_game_folder", lambda g: r"C:\detected")
    cfg = sc.load(str(tmp_path / "nope.json"))
    assert cfg.game_folder == r"C:\detected"


def test_autodetect_returns_none_for_an_unknown_game():
    import sim_config as sc
    assert sc.autodetect_game_folder("playstation") is None
