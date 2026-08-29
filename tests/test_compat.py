"""compat.py checks the installed beamngpy against BeamNGpy's own published
compatibility table (BeamNGpy/COMPATIBILITY.md, fetched 2026-08-29) -- no game
imports needed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compat import required_beamngpy, check_compat, TABLE


def test_table_has_the_pinned_version_for_037():
    assert required_beamngpy("0.37") == "1.34.1"


def test_required_beamngpy_unknown_game_version_returns_none():
    assert required_beamngpy("9.99") is None


def test_check_compat_matching_versions_is_ok():
    result = check_compat(game_version="0.37.6.0", beamngpy_version="1.34.1")
    assert result.ok is True
    assert result.required == "1.34.1"


def test_check_compat_mismatch_is_not_ok_but_has_the_fix_command():
    result = check_compat(game_version="0.37.6.0", beamngpy_version="1.30.0")
    assert result.ok is False
    assert result.required == "1.34.1"
    assert "pip install" in result.fix_command
    assert "1.34.1" in result.fix_command


def test_check_compat_unknown_game_version_warns_but_does_not_hard_fail():
    result = check_compat(game_version=None, beamngpy_version="1.34.1")
    assert result.ok is None   # unknown, not a pass/fail
    assert "unknown" in result.message.lower()


def test_check_compat_missing_beamngpy():
    result = check_compat(game_version="0.37.6.0", beamngpy_version=None)
    assert result.ok is False
    assert "not installed" in result.message.lower() or "missing" in result.message.lower()
