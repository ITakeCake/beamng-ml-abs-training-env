"""The mod under assets/ is what a GitHub user installs. Its copy of
abstelemetry.lua drifted 142 lines behind the root copy (missing the whole tire
grip feature) without anything noticing, because training deploys the ROOT copy
over the top of it, so the staleness was invisible here and would only ever
have bitten someone else."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_LUA = os.path.join(REPO, "abstelemetry.lua")
BUNDLED_LUA = os.path.join(REPO, "assets", "mods", "mtb_ml_abs", "lua",
                           "vehicle", "extensions", "abstelemetry.lua")


def _normalized(path):
    """Line endings only, the repo has CRLF at the root and LF under assets,
    and that difference is not drift."""
    with open(path, encoding="utf-8") as fh:
        return fh.read().replace("\r\n", "\n")


def test_the_shipped_mod_carries_the_same_telemetry_as_training_runs():
    assert _normalized(BUNDLED_LUA) == _normalized(ROOT_LUA), (
        "assets/mods/mtb_ml_abs/.../abstelemetry.lua has drifted from the root "
        "copy. abs_env.py copies the ROOT file into the userpath at startup, so "
        "training never notices, but the bundled one is what ships. Re-sync it.")


def test_the_bundled_copy_has_the_features_the_env_calls():
    """Named explicitly so a partial sync fails loudly rather than silently
    shipping a telemetry file missing the calls the env makes."""
    src = _normalized(BUNDLED_LUA)
    for fn in ("armBrakeSlam", "armGripChange", "restoreGrip", "setGripMultiplier",
               "armRunupHandoff", "disarmRunupHandoff",
               "setTargetSpeed", "resetAccum", "releaseBrakes",
               "lastBrakeAvgGArc", "tel_yaw_rate_inst"):
        assert fn in src, f"bundled abstelemetry.lua is missing {fn}"


TRAINER_MOD = os.path.join(REPO, "assets", "mods", "dynamicabs_trainer")
TRAINER_CTRL = os.path.join(TRAINER_MOD, "lua", "vehicle", "controller",
                            "DynamicABS_Trainer.lua")
TRAINER_REC = os.path.join(TRAINER_MOD, "lua", "vehicle", "extensions",
                           "trainerRecorder.lua")
TRAINER_PC = os.path.join(REPO, "assets", "cars", "etk800", "DynamicABS_Trainer.pc")
STUDENT_PC = os.path.join(REPO, "assets", "cars", "etk800",
                          "Machine-Trainer-Boy-V2-MLABS.pc")


def test_the_trainer_publishes_the_electrics_the_recorder_reads():
    """record_dynamicabs_teacher.py labels each tick with the teacher's
    per-wheel brake command. The controller must publish it and the recorder
    must read it, both under the same names, or every label is zero."""
    ctrl = _normalized(TRAINER_CTRL)
    rec = _normalized(TRAINER_REC)
    for wheel in ("fr", "fl", "rr", "rl"):
        key = f"abs_cmd_{wheel}"
        assert f"electrics.values.{key}" in ctrl, f"controller never sets {key}"
        assert f"ev.{key}" in rec, f"recorder never reads {key}"


def test_the_recorder_header_matches_the_python_column_map():
    import re
    import record_dynamicabs_teacher as r
    src = _normalized(TRAINER_REC)
    start = src.index("local HEADER =")
    end = src.index("local FMT =", start)
    header = "".join(re.findall(r'"([^"]*)"', src[start:end]))
    cols = header.split(",")
    assert cols[r.C_GS] == "gs"
    assert cols[r.C_WS_FR] == "ws_fr"
    assert cols[r.C_BRK] == "brake_in"
    assert cols[r.C_ABSSPD] == "abs_speed"
    assert cols[r.C_BRK_NM_FR] == "brk_nm_fr"
    assert cols[r.C_CMD_FR] == "cmd_fr"
    assert cols[r.C_CMD_RL] == "cmd_rl"
    assert len(cols) == r.C_CMD_RL + 1


def test_the_teacher_car_is_the_student_car_with_only_the_abs_slot_swapped():
    """Teacher demonstrations are only valid if the chassis, tyres, brakes and
    mass match what the student trains on. Anything else is a different car."""
    import json
    with open(TRAINER_PC, encoding="utf-8") as fh:
        teacher = json.load(fh)["parts"]
    with open(STUDENT_PC, encoding="utf-8") as fh:
        student = json.load(fh)["parts"]
    diff = {k for k in set(teacher) | set(student) if teacher.get(k) != student.get(k)}
    assert diff == {"etk_DSE_ABS"}, f"teacher .pc differs from student in {sorted(diff)}"
    assert teacher["etk_DSE_ABS"] == "etk_DSE_DynABS_Trainer"
