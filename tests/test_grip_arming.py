"""The parent's reset() is monolithic: the teleport reloads the vehicle Lua
extension (losing module state) and there is no hook between that and the
armBrakeSlam call. So the grip arm is injected into the queued-command stream
right after armBrakeSlam -- which is safe because the slam only FIRES later,
when the coast-down crosses the target speed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abs_env_residual import GripArmInjector


class FakeVehicle:
    def __init__(self):
        self.sent = []

    def queue_lua_command(self, cmd):
        self.sent.append(cmd)


def test_injects_grip_arm_immediately_after_the_slam_arm():
    veh = FakeVehicle()
    inj = GripArmInjector(veh, grip=0.6, lead_seconds=0.0)
    with inj:
        veh.queue_lua_command("extensions.load('abstelemetry')")
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8224)")
        veh.queue_lua_command("something.else()")

    assert veh.sent[0] == "extensions.load('abstelemetry')"
    assert veh.sent[1] == "extensions.abstelemetry.armBrakeSlam(26.8224)"
    assert "armGripChange(0.6" in veh.sent[2]
    assert veh.sent[3] == "something.else()"
    assert inj.armed is True


def test_restores_the_original_method_afterwards():
    veh = FakeVehicle()
    original = veh.queue_lua_command
    with GripArmInjector(veh, grip=0.6, lead_seconds=0.0):
        pass
    assert veh.queue_lua_command == original


def test_restores_even_if_the_body_raises():
    veh = FakeVehicle()
    original = veh.queue_lua_command
    with pytest.raises(RuntimeError):
        with GripArmInjector(veh, grip=0.6, lead_seconds=0.0):
            raise RuntimeError("reset blew up")
    assert veh.queue_lua_command == original


def test_no_grip_means_no_injection_at_all():
    veh = FakeVehicle()
    inj = GripArmInjector(veh, grip=None, lead_seconds=0.0)
    with inj:
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
    assert veh.sent == ["extensions.abstelemetry.armBrakeSlam(26.8)"]
    assert inj.armed is False


def test_injects_only_once_even_if_the_slam_is_armed_twice():
    veh = FakeVehicle()
    with GripArmInjector(veh, grip=0.6, lead_seconds=0.0):
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
    assert sum("armGripChange" in c for c in veh.sent) == 1


def test_passes_the_lead_through():
    veh = FakeVehicle()
    with GripArmInjector(veh, grip=0.75, lead_seconds=0.01):
        veh.queue_lua_command("extensions.abstelemetry.armBrakeSlam(26.8)")
    injected = [c for c in veh.sent if "armGripChange" in c][0]
    assert "0.75" in injected and "0.01" in injected
