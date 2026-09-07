import json

import pytest

from train_cosim import inherit_resume_contract, parse_seed


def test_gui_style_resume_inherits_parent_training_contract(tmp_path):
    run = tmp_path / "PPO-parent"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "PPO-parent_50000_steps.zip"
    checkpoint.write_bytes(b"placeholder")
    parent = {
        "reward": "v6.0", "vehicle_pc": "vehicles/etk800/original.pc",
        "speeds": [80.0], "runup_speed_factor": 16.0,
        "action_interface": "ppo_unit_tanh_release_v1",
        "deployment_interface": "cosim_axle_release_v1", "control_hz": 100,
        "ppo": {"lr": 0.0001, "gamma": 0.99, "n_steps": 2048},
    }
    (run / "config.json").write_text(json.dumps(parent), encoding="utf-8")
    requested = {
        "resume_checkpoint": str(checkpoint), "resume_inherit_config": True,
        "reward": "v5.0", "vehicle_pc": "wrong.pc", "speeds": [30],
        "runup_speed_factor": 4, "ppo": {"lr": 0.01},
        "run_name": "PPO-child", "total_steps": 100000,
        "device": "cpu", "port": 65000,
    }
    inherited = inherit_resume_contract(requested)
    assert inherited["reward"] == "v6.0"
    assert inherited["vehicle_pc"] == parent["vehicle_pc"]
    assert inherited["speeds"] == [80.0]
    assert inherited["runup_speed_factor"] == 16.0
    assert inherited["ppo"] == parent["ppo"]
    assert inherited["run_name"] == "PPO-child"
    assert inherited["total_steps"] == 100000
    assert inherited["device"] == "cpu"
    assert inherited["port"] == 65000


def test_manual_resume_without_inherit_keeps_explicit_values():
    cfg = {"resume_checkpoint": "x.zip", "reward": "v5.0"}
    assert inherit_resume_contract(cfg)["reward"] == "v5.0"


def test_seed_parser_records_auto_and_rejects_invalid_values():
    assert 0 <= parse_seed("auto") < 2 ** 32
    assert parse_seed("42") == 42
    with pytest.raises(SystemExit, match="seed"):
        parse_seed("nope")
