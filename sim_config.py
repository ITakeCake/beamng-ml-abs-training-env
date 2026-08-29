"""Simulator configuration: which game, where it lives, how to launch it.

Pure/no game-import module (like residual_core.py) -- the GUI's Simulator tab,
the trainer, and the probe all read a SimConfig built here. Persisted to
settings.json next to the entry point (gitignored -- it's machine-local).
"""
import dataclasses
import json
import os
import re

EXE_NAME = {"tech": "BeamNG.tech.exe", "drive": "BeamNG.drive.exe"}


@dataclasses.dataclass
class SimConfig:
    game: str = "tech"                 # "tech" or "drive"
    game_folder: str = ""              # install root (contains BeamNG.<game>.exe)
    userpath: str = ""                 # "" = auto (default_userpath(game))
    headless: bool = True              # .tech only; .drive has no -gfx null
    port: int = 64291
    map: str = "smallgrid"
    attach_first: bool = True          # try to connect to a running instance first
    cpu_pinning: bool = False
    python_cores: tuple = (0, 1)
    beamng_cores: tuple = ()           # () = "all cores not in python_cores"


DEFAULTS = SimConfig()


def default_userpath(game):
    """The BASE userpath BeamNG.exe is launched with (its -userpath argument /
    beamngpy's `user=`). BeamNG manages a "current" version-subfolder ITSELF
    underneath this -- passing a path that already ends in "current" makes it
    create/use <path>/current/current instead (confirmed live 2026-08-29: the
    game's own log showed "userpath = ...\\current\\current\\" after a launch
    with the old, over-appended default). The folder where vehicles/mods
    actually live for reading/installing is this path's own "current"
    subfolder -- see vehicle_scanner.py / asset_installer.py, which take an
    already-resolved content folder as their argument, not this function's
    return value directly."""
    root = os.path.join(os.environ.get("LOCALAPPDATA", ""), "BeamNG")
    name = "BeamNG.tech" if game == "tech" else "BeamNG.drive"
    return os.path.join(root, name)


def _read_version_ini(game):
    """BeamNG writes %LOCALAPPDATA%\\BeamNG\\BeamNG.<game>.ini itself on every
    launch ("version = X.Y.Z.W") -- authoritative regardless of where or how
    the game is installed. None if the game has never been launched yet."""
    path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "BeamNG", f"BeamNG.{game}.ini")
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                if line.strip().startswith("version"):
                    return line.split("=", 1)[1].strip()
    except OSError:
        return None
    return None


def detect_game_version(game, game_folder):
    """Best available game version: the ini file BeamNG itself writes (works
    regardless of install location/naming -- e.g. Steam installs of
    BeamNG.drive commonly encode no version in the folder name at all),
    falling back to guessing from the folder name. None if neither source is
    available (e.g. a fresh install that has never been launched)."""
    return _read_version_ini(game) or guess_version_from_folder(game_folder)


def find_exe(game_folder, game):
    if not game_folder or not os.path.isdir(game_folder):
        return None
    candidate = os.path.join(game_folder, EXE_NAME[game])
    return candidate if os.path.isfile(candidate) else None


def guess_version_from_folder(game_folder):
    """Best-effort only: this install's folder happens to be named
    'BeamNG.tech.v0.37.6.0'. Steam installs (typical for .drive) usually
    don't encode a version in the folder name -- callers must treat None
    as 'unknown', not as an error."""
    m = re.search(r"\.v(\d+(?:\.\d+){2,3})(?:[\/]|$)", game_folder)
    return m.group(1) if m else None


def validate(cfg):
    """List of human-readable problems; [] means the config is launchable."""
    problems = []
    if cfg.game not in ("tech", "drive"):
        problems.append(f"unknown game {cfg.game!r} -- must be 'tech' or 'drive'")
    exe = find_exe(cfg.game_folder, cfg.game) if cfg.game in EXE_NAME else None
    if not exe:
        problems.append(
            f"{EXE_NAME.get(cfg.game, '?')} not found under game folder "
            f"{cfg.game_folder!r} -- point 'Game folder' at your BeamNG install")
    if cfg.headless and cfg.game == "drive":
        problems.append("headless is not supported on BeamNG.drive (no -gfx null) "
                        "-- turn it off for .drive")
    if not (1 <= cfg.port <= 65535):
        problems.append(f"port {cfg.port} out of range 1-65535")
    return problems


def resolved_userpath(cfg):
    """The BASE path to hand BeamNG at launch (beamngpy's user=/-userpath).
    Normalizes away a trailing "current" -- the natural real-world mistake is
    browsing to the *visible* folder (…\\current, the one Explorer shows with
    actual content in it) and pointing the Userpath field at that instead of
    its parent, which would double-nest it (see default_userpath)."""
    up = (cfg.userpath or default_userpath(cfg.game)).rstrip("\\/")
    if os.path.basename(up) == "current":
        up = os.path.dirname(up)
    return up


def content_userpath(cfg):
    """Where vehicles/mods actually live for reading/installing -- BeamNG's
    own "current" version-subfolder under the launch base. Use this, never
    resolved_userpath(), for anything that scans or writes vehicle/mod files
    (vehicle_scanner.py, asset_installer.py)."""
    return os.path.join(resolved_userpath(cfg), "current")


def save(cfg, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(cfg), fh, indent=2)


def load(path):
    """Missing or corrupt file -> defaults, never raises (this is a convenience
    load on GUI/trainer startup, not a place to surface a stack trace)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return SimConfig()
    known = {f.name for f in dataclasses.fields(SimConfig)}
    data = {k: (tuple(v) if isinstance(v, list) else v)
            for k, v in data.items() if k in known}
    return SimConfig(**data)
