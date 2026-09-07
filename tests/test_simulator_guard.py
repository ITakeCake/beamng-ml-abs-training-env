import pytest

from simulator_guard import active_beamng_sessions, require_beamng_available


class _Process:
    def __init__(self, pid, name, command):
        self.pid = pid
        self.info = {"pid": pid, "name": name, "cmdline": command}


def test_guard_finds_main_drive_and_tech_but_not_chromium_children():
    processes = [
        _Process(10, "BeamNG.drive.x64.exe", ["BeamNG.drive.x64.exe", "-level", "smallgrid"]),
        _Process(11, "BeamNG.drive.x64.exe", ["BeamNG.drive.x64.exe", "--type=renderer"]),
        _Process(12, "BeamNG.tech.x64.exe", ["BeamNG.tech.x64.exe", "-headless"]),
        _Process(13, "python.exe", ["python", "train_cosim.py"]),
    ]
    assert [row["pid"] for row in active_beamng_sessions(processes)] == [10, 12]
    assert [row["pid"] for row in active_beamng_sessions(
        processes, game="tech")] == [12]
    assert [row["pid"] for row in active_beamng_sessions(
        processes, game="drive")] == [10]


def test_guard_refuses_without_terminating_or_mutating_processes():
    process = _Process(44, "BeamNG.drive.x64.exe", ["BeamNG.drive.x64.exe"])
    with pytest.raises(RuntimeError, match="already in use.*pid=44"):
        require_beamng_available([process])
    assert process.info["pid"] == 44


def test_guard_allows_launch_when_no_main_session_exists():
    processes = [_Process(11, "BeamNG.tech.x64.exe",
                          ["BeamNG.tech.x64.exe", "--type=gpu-process"])]
    assert require_beamng_available(processes)


def test_product_guard_allows_tech_while_drive_is_in_use():
    processes = [
        _Process(44, "BeamNG.drive.x64.exe", ["BeamNG.drive.x64.exe"]),
        _Process(45, "BeamNG.drive.x64.exe",
                 ["BeamNG.drive.x64.exe", "--type=gpu-process"]),
    ]
    assert require_beamng_available(processes, game="tech")


def test_product_guard_still_refuses_same_product():
    process = _Process(55, "BeamNG.tech.x64.exe", ["BeamNG.tech.x64.exe"])
    with pytest.raises(RuntimeError, match=r"BeamNG\.tech.*pid=55"):
        require_beamng_available([process], game="tech")


def test_product_guard_rejects_unknown_product():
    with pytest.raises(ValueError, match="game must be"):
        active_beamng_sessions([], game="racing")
