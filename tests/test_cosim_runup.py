"""Accelerated co-sim run-up stays out of the measured braking phase."""
from pathlib import Path

import numpy as np
import pytest

import abs_env_cosim
from abs_env_cosim import ABSCoSimEnv


def test_live_asset_install_preserves_a_content_addressed_rollback(tmp_path):
    source = tmp_path / "source.lua"
    destination = tmp_path / "live" / "extension.lua"
    source.write_text("new", encoding="utf-8")
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")
    assert ABSCoSimEnv._copy_with_rollback(source, destination) == "installed"
    assert destination.read_text(encoding="utf-8") == "new"
    backups = list(destination.parent.glob("extension.lua.backup-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old"
    assert ABSCoSimEnv._copy_with_rollback(source, destination) == "unchanged"
    assert len(list(destination.parent.glob("extension.lua.backup-*"))) == 1
from train_cosim import (RunLog, parse_ppo_config, parse_run_name,
                         parse_runup_speed_factor)


class _Vehicle:
    def __init__(self):
        self.controls = []
        self.lua = []

    def control(self, **values):
        self.controls.append(values)

    def queue_lua_command(self, command, response=False):
        self.lua.append((command, response))
        return True


def test_runup_factor_validation():
    assert parse_runup_speed_factor("4") == 4.0
    assert parse_runup_speed_factor(7.5) == 7.5
    for invalid in (0, 0.5, 1001, "fast", float("nan"), float("inf")):
        with pytest.raises(SystemExit, match="1 to 1000"):
            parse_runup_speed_factor(invalid)


def test_run_and_ppo_validation_happen_before_simulator_construction():
    assert parse_run_name("PPO-38") == "PPO-38"
    with pytest.raises(SystemExit, match="run_name"):
        parse_run_name("../escape")
    cfg = parse_ppo_config({"initial_release": 0.1, "log_std_init": -0.7,
                            "ent_coef": 0})
    assert cfg["ent_coef"] == 0.0
    with pytest.raises(SystemExit, match="batch_size"):
        parse_ppo_config({"n_steps": 32, "batch_size": 64})
    with pytest.raises(SystemExit, match="initial_release"):
        parse_ppo_config({"initial_release": 1.0})


def test_runup_events_reach_the_gui_training_log(tmp_path):
    path = tmp_path / "train.log"
    logger = RunLog(path)
    logger.info("RUNUP_HANDOFF ep=1 speed_ms=37.2")
    logger.info("BRAKE_ACTIVATION ep=1 speed_ms=38.7")
    assert path.read_text(encoding="utf-8").splitlines() == [
        "RUNUP_HANDOFF ep=1 speed_ms=37.2",
        "BRAKE_ACTIVATION ep=1 speed_ms=38.7",
    ]


def test_accelerated_runup_has_an_acknowledged_realtime_handoff(monkeypatch):
    env = object.__new__(ABSCoSimEnv)
    env.episode = 3
    env.runup_speed_factor = 4.0
    env.veh = _Vehicle()
    commands, factors, logs = [], [], []
    packets = iter([
        {"tel_runup_handoff_fired": 1,
         "tel_runup_handoff_fire_speed": 38.26,
         "tel_inst_speed": 38.26},
        {"tel_inst_speed": 38.20},
        {"tel_inst_speed": 39.80},
    ])
    env._q = commands.append
    env._set_physics_speed_factor = lambda value: factors.append(value) or True
    env._poll_electrics = lambda: next(packets)
    env._diag = logs.append
    monkeypatch.setattr(abs_env_cosim.time, "sleep", lambda _seconds: None)

    env._accelerated_runup(39.76)

    assert factors == [4.0, 0]
    assert any("armRunupHandoff" in command for command, _response in env.veh.lua)
    assert env.veh.lua[0][1] is True
    assert any("disarmRunupHandoff" in command for command in commands)
    assert any(line.startswith("RUNUP_HANDOFF") and "speed_ms=38.2600" in line
               for line in logs)
    assert any(line.startswith("RUNUP_REALTIME") and "ack=True" in line
               for line in logs)
    assert env.veh.controls[-1]["throttle"] == 1.0


def test_brake_activation_logs_the_lua_physics_tick_speed():
    env = object.__new__(ABSCoSimEnv)
    env.episode = 8
    env.target_ms = 35.7632
    env.start_speed_ms = 0.0
    env._brake_activation_logged = False
    logs = []
    env._diag = logs.append
    packet = np.zeros(len(abs_env_cosim.SIG_TO), dtype=float)
    packet[abs_env_cosim.I_FIRED] = 1
    packet[abs_env_cosim.I_FIRE_SPEED] = 38.7629

    env._log_brake_activation(packet)
    env._log_brake_activation(packet)

    assert len(logs) == 1
    assert logs[0].startswith("BRAKE_ACTIVATION")
    assert "speed_ms=38.7629" in logs[0]
    assert env.start_speed_ms == 38.7629


def test_first_moving_packet_ignores_the_previous_stop_snapshot():
    env = object.__new__(ABSCoSimEnv)
    env.episode = 2
    env._coupled_moving_packet_logged = False
    logs = []
    env._diag = logs.append
    stale = np.zeros(len(abs_env_cosim.SIG_TO), dtype=float)
    stale[abs_env_cosim.I_INST_SPEED] = 0.05
    moving = stale.copy()
    moving[abs_env_cosim.I_INST_SPEED] = 38.7
    moving[abs_env_cosim.I_GS] = 38.6
    moving[abs_env_cosim.I_FUSED] = 38.5

    env._log_first_moving_packet(stale)
    env._log_first_moving_packet(moving)
    env._log_first_moving_packet(moving)

    assert len(logs) == 1
    assert logs[0].startswith("COSIM_FIRST_MOVING_PACKET")
    assert "speed_ms=38.7000" in logs[0]


def test_lua_handoff_is_a_2khz_throttle_latch():
    source = Path(abs_env_cosim.__file__).with_name("abstelemetry.lua").read_text(
        encoding="utf-8")
    assert "local function updateRunupHandoff()" in source
    assert "instSpeed >= runupHandoff.target" in source
    assert "input.throttle = 0" in source
    assert source.index("updateRunupHandoff()", source.index("local function onPhysicsStep")) \
        < source.index("updateArmedSlam()", source.index("local function onPhysicsStep"))


def test_telemetry_installer_refreshes_global_and_mod_copies(tmp_path):
    ABSCoSimEnv._install_telemetry(tmp_path)
    expected = Path(abs_env_cosim.__file__).with_name("abstelemetry.lua").read_bytes()
    global_copy = tmp_path / "current" / "lua" / "vehicle" / "extensions" / "abstelemetry.lua"
    mod_copy = (tmp_path / "current" / "mods" / "unpacked" / "mtb_ml_abs" /
                "lua" / "vehicle" / "extensions" / "abstelemetry.lua")
    assert global_copy.read_bytes() == expected
    assert mod_copy.read_bytes() == expected


def _gear_env(mode):
    env = object.__new__(ABSCoSimEnv)
    env.episode = 7
    env.veh = _Vehicle()
    env.gear_mode = mode
    env._neutral_shifted = mode == "neutral"
    env._seen_moving = False
    env.logs = []
    env._diag = env.logs.append
    return env


def test_handoff_keeps_drive_gear_only_in_in_gear_mode():
    env = _gear_env("neutral")
    env._handoff_gear()
    assert env.veh.controls == [{"throttle": 0.0, "steering": 0, "gear": 0}]
    env = _gear_env("in_gear")
    env._handoff_gear()
    assert env.veh.controls == [{"throttle": 0.0, "steering": 0}]
    assert any("mode=in_gear" in line for line in env.logs)


def test_in_gear_shifts_to_neutral_once_at_or_below_five_mph():
    env = _gear_env("in_gear")
    env._maybe_shift_neutral(0.0)            # startup packets read ~0: ignore
    env._seen_moving = True
    for speed in (30.0, 2.3):                # 2.3 m/s is above 5 mph (2.235)
        env._maybe_shift_neutral(speed)
    assert env.veh.controls == []
    env._maybe_shift_neutral(2.2)
    env._maybe_shift_neutral(1.0)
    assert env.veh.controls == [{"gear": 0}]
    assert any(line.startswith("NEUTRAL_SHIFT ep=7") for line in env.logs)
    env = _gear_env("neutral")
    env._seen_moving = True
    env._maybe_shift_neutral(1.0)
    assert env.veh.controls == []


def test_target_kl_defaults_on_and_zero_disables():
    from train_cosim import parse_ppo_config
    assert parse_ppo_config({})["target_kl"] == 0.02
    assert parse_ppo_config({"target_kl": 0})["target_kl"] == 0.0
    with pytest.raises(SystemExit, match="target_kl"):
        parse_ppo_config({"target_kl": -1})
