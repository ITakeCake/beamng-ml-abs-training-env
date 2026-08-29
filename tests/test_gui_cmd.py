import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui_cmd import build_cmd, validate_settings

BASE = dict(algo="sac", speeds="60,120", pedal_random=True, pedal_spec="0.4-1.0",
            total_steps=200000, run_name="r1", resume="",
            lr=1e-4, buffer_size=300000, tau=0.005, target_entropy=-2.0,
            learning_starts=5000, n_steps=2048, batch_size=256, n_epochs=10,
            clip_range=0.2, gae_lambda=0.95)

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
