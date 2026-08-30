import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui_cmd import build_cmd, validate_settings

BASE = dict(algo="sac", speeds="60,120", pedal_random=True, pedal_spec="0.4-1.0",
            total_steps=200000, run_name="r1", resume="",
            lr=1e-4, buffer_size=300000, tau=0.005, target_entropy=-2.0,
            learning_starts=5000, train_freq=2, n_steps=2048, batch_size=256,
            n_epochs=10, clip_range=0.2, gae_lambda=0.95, ent_coef=0.005)

def test_sac_cmd_has_sac_flags_not_ppo():
    cmd = build_cmd(BASE)
    s = " ".join(cmd)
    assert "--algo sac" in s and "--buffer-size 300000" in s
    assert "--n-steps" not in s and "--clip-range" not in s

def test_ppo_cmd_has_ppo_flags_not_sac():
    cmd = build_cmd({**BASE, "algo": "ppo"})
    s = " ".join(cmd)
    assert "--algo ppo" in s and "--n-steps 2048" in s
    assert "--buffer-size" not in s and "--target-entropy" not in s

def test_pedal_off_when_switch_disabled():
    s = " ".join(build_cmd({**BASE, "pedal_random": False}))
    assert "--pedal off" in s

def test_resume_included_only_when_set():
    assert "--resume" not in " ".join(build_cmd(BASE))
    s = " ".join(build_cmd({**BASE, "resume": r"runs\r0\final.zip"}))
    assert "--resume" in s

def test_validate_catches_bad_speeds_and_pedal():
    probs = validate_settings({**BASE, "speeds": "abc", "pedal_spec": "2.0"})
    assert len(probs) == 2
    assert validate_settings(BASE) == []


def test_vehicle_pc_included_only_when_set():
    assert "--vehicle-pc" not in " ".join(build_cmd(BASE))
    s = " ".join(build_cmd({**BASE, "vehicle_pc": "vehicles/bx/MyCar.pc"}))
    assert "--vehicle-pc vehicles/bx/MyCar.pc" in s


# ------------------------------------------------- grip / corner / reward
def _base():
    return dict(algo="sac", speeds="60", pedal_random=False, pedal_spec="0.4-1.0",
                total_steps="1000", run_name="r1", lr="3e-4", buffer_size="100000",
                tau="0.02", target_entropy="auto", learning_starts="1000",
                train_freq="1")


def test_omitted_keys_leave_the_trainers_own_defaults_alone():
    """A settings.json written before these fields existed must still build."""
    cmd = build_cmd(_base())
    assert "--grip" not in cmd and "--corner" not in cmd and "--reward" not in cmd


def test_grip_corner_and_reward_reach_the_command_line():
    s = _base()
    s.update(grip="0.5,1.0", corner="50L", reward="normalized")
    cmd = build_cmd(s)
    assert cmd[cmd.index("--grip") + 1] == "0.5,1.0"
    assert cmd[cmd.index("--corner") + 1] == "50L"
    assert cmd[cmd.index("--reward") + 1] == "normalized"


def test_validation_catches_a_bad_corner_before_the_game_launches():
    s = _base()
    s["corner"] = "wide"
    assert any("corner" in p for p in validate_settings(s))


def test_validation_catches_a_bad_grip():
    s = _base()
    s["grip"] = "9.0"
    assert any("grip" in p for p in validate_settings(s))


def test_a_straight_corner_and_stock_grip_are_valid_and_passed_through():
    s = _base()
    s.update(grip="off", corner="straight")
    assert validate_settings(s) == []
    cmd = build_cmd(s)
    assert cmd[cmd.index("--corner") + 1] == "straight"


# ----------------------------------------------- calibration command
from gui_cmd import build_calibration_cmd, validate_calibration_settings


def test_calibration_uses_the_same_configuration_the_training_tab_is_set_to():
    """A ruler measured on a different configuration than it scores is worse
    than no ruler -- so the fields are shared, not asked for twice."""
    s = _base()
    s.update(speeds="60,90", grip="0.5,1.0", corner="150L")
    cmd = build_calibration_cmd(s, car="etk800")
    assert cmd[1] == "reference_runner.py"
    assert cmd[cmd.index("--car") + 1] == "etk800"
    assert cmd[cmd.index("--speeds") + 1] == "60,90"
    assert cmd[cmd.index("--grips") + 1] == "0.5,1.0"
    assert cmd[cmd.index("--corner") + 1] == "150L"


def test_stock_grip_and_straight_add_no_flags():
    s = _base()
    s.update(grip="off", corner="straight")
    cmd = build_calibration_cmd(s, car="etk800")
    assert "--grips" not in cmd
    assert cmd[cmd.index("--corner") + 1] == "straight"


def test_a_continuous_grip_range_has_no_levels_to_calibrate():
    s = _base()
    s["grip"] = "0.4-1.0"
    cmd = build_calibration_cmd(s, car="etk800")
    assert "--grips" not in cmd
    assert any("continuously" in p for p in validate_calibration_settings(s))


def test_a_fixed_grip_is_a_single_calibratable_level():
    s = _base()
    s["grip"] = "0.6"
    assert build_calibration_cmd(s, car="etk800")[-1] == "0.6"
    assert validate_calibration_settings(s) == []


def test_reps_are_passed_through():
    assert build_calibration_cmd(_base(), car="etk800", reps=5)[-1] == "5"


# ------------------------------------------------- console interpreter
def test_the_trainer_never_inherits_pythonw(monkeypatch, tmp_path):
    """A run died with "Expected file or str, got None" after BeamNG had booted
    and driven a reset: the GUI runs under pythonw (no console window), the
    trainer inherited it, and pythonw sets sys.stdout to None -- which SB3's
    verbose=1 logger writes to unconditionally."""
    import gui_cmd
    (tmp_path / "python.exe").write_text("")
    monkeypatch.setattr(gui_cmd.sys, "executable", str(tmp_path / "pythonw.exe"))
    assert os.path.basename(gui_cmd._console_python()) == "python.exe"


def test_a_normal_python_is_left_alone(monkeypatch, tmp_path):
    import gui_cmd
    exe = str(tmp_path / "python.exe")
    monkeypatch.setattr(gui_cmd.sys, "executable", exe)
    assert gui_cmd._console_python() == exe


def test_pythonw_with_no_console_sibling_falls_back_rather_than_inventing_one(
        monkeypatch, tmp_path):
    """Returning a path that does not exist would fail worse than the original."""
    import gui_cmd
    exe = str(tmp_path / "pythonw.exe")
    monkeypatch.setattr(gui_cmd.sys, "executable", exe)
    assert gui_cmd._console_python() == exe
