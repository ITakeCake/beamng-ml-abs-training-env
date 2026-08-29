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
    root = os.path.join(os.environ.get("LOCALAPPDATA", ""), "BeamNG")
    name = "BeamNG.tech" if game == "tech" else "BeamNG.drive"
    return os.path.join(root, name, "current")


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
    return cfg.userpath or default_userpath(cfg.game)


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
