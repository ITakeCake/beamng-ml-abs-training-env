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
