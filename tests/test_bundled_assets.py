"""The mod under assets/ is what a GitHub user installs. Its copy of
abstelemetry.lua drifted 142 lines behind the root copy (missing the whole tire
grip feature) without anything noticing, because training deploys the ROOT copy
over the top of it -- so the staleness was invisible here and would only ever
have bitten someone else."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_LUA = os.path.join(REPO, "abstelemetry.lua")
BUNDLED_LUA = os.path.join(REPO, "assets", "mods", "mtb_ml_abs", "lua",
                           "vehicle", "extensions", "abstelemetry.lua")


def _normalized(path):
    """Line endings only -- the repo has CRLF at the root and LF under assets,
    and that difference is not drift."""
    with open(path, encoding="utf-8") as fh:
        return fh.read().replace("\r\n", "\n")


def test_the_shipped_mod_carries_the_same_telemetry_as_training_runs():
    assert _normalized(BUNDLED_LUA) == _normalized(ROOT_LUA), (
        "assets/mods/mtb_ml_abs/.../abstelemetry.lua has drifted from the root "
        "copy. abs_env.py copies the ROOT file into the userpath at startup, so "
        "training never notices -- but the bundled one is what ships. Re-sync it.")


def test_the_bundled_copy_has_the_features_the_env_calls():
    """Named explicitly so a partial sync fails loudly rather than silently
    shipping a telemetry file missing the calls the env makes."""
    src = _normalized(BUNDLED_LUA)
    for fn in ("armBrakeSlam", "armGripChange", "restoreGrip", "setGripMultiplier",
               "armRunupHandoff", "disarmRunupHandoff",
               "setTargetSpeed", "resetAccum", "releaseBrakes",
               "lastBrakeAvgGArc", "tel_yaw_rate_inst"):
        assert fn in src, f"bundled abstelemetry.lua is missing {fn}"
