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
