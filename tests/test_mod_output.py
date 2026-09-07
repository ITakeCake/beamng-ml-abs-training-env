"""mod_output writes the actual mod files: generate_all_cars() for the
per-car parent parts, export_model_to_game() merges a trained model's child
part in (never clobbering a previously-exported sibling model on the same
car). The exporter subprocess itself is mocked here -- test_export_policy_weights
style live parity is exercised separately against the real venv."""
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mod_output as mo
from model_registry import RunInfo

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_generate_all_cars_writes_parent_and_reports_skipped(tmp_path):
    written, skipped = mo.generate_all_cars(FIXTURES, str(tmp_path))
    assert "testcar" in written
    assert "testprop" not in written and "testprop" not in skipped  # not a Car at all
    assert "testcar_noabs" in skipped   # a real Car, just with no ABS slot
    parent_path = tmp_path / "vehicles" / "testcar" / "ml_abs_parent.jbeam"
    assert parent_path.is_file()
    json.loads(parent_path.read_text())   # must be valid jbeam JSON


def test_export_model_to_game_merges_without_clobbering_a_sibling(tmp_path, monkeypatch):
    calls = []

    def fake_run_exporter(cmd):
        # matches the REAL exporter's behavior: it does NOT create its own
        # output directory (bare `open(out_path, "w")`) -- the caller must.
        # A mock that auto-mkdir's here would hide exactly the bug this
        # caught live: export_model_to_game calling the exporter before the
        # directory existed.
        out_path = cmd[cmd.index("--out") + 1]
        with open(out_path, "w") as fh:
            fh.write("-- fake weights\n")
        calls.append(cmd)
        return True, "ok"

    monkeypatch.setattr(mo, "_run_exporter", fake_run_exporter)

    run_dir = tmp_path / "runs" / "runA"
    run_dir.mkdir(parents=True)
    (run_dir / "final.zip").write_bytes(b"x")
    (run_dir / "vecnormalize.pkl").write_bytes(b"x")
    run_a = RunInfo(run_name="runA", run_dir=str(run_dir), algo="sac",
                    vehicle_pc="vehicles/etk800/x.pc", model="etk800", best_avg_g=1.0)

    mod_dir = str(tmp_path / "mod")
    ok, msg = mo.export_model_to_game(run_a, mod_dir)
    assert ok is True

    run_dir_b = tmp_path / "runs" / "runB"
    run_dir_b.mkdir(parents=True)
    (run_dir_b / "final.zip").write_bytes(b"x")
    (run_dir_b / "vecnormalize.pkl").write_bytes(b"x")
    run_b = RunInfo(run_name="runB", run_dir=str(run_dir_b), algo="ppo",
                    vehicle_pc="vehicles/etk800/x.pc", model="etk800", best_avg_g=1.1)
    ok, msg = mo.export_model_to_game(run_b, mod_dir)
    assert ok is True

    models_path = os.path.join(mod_dir, "vehicles", "etk800", "ml_abs_models.jbeam")
    data = json.loads(open(models_path).read())
    assert len(data) == 2   # both runA and runB present -- runB didn't clobber runA
    assert any("runA" in k for k in data)
    assert any("runB" in k for k in data)


def test_export_model_to_game_reports_failure_from_exporter(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "_run_exporter", lambda cmd: (False, "boom"))
    run_dir = tmp_path / "runs" / "bad"
    run_dir.mkdir(parents=True)
    (run_dir / "final.zip").write_bytes(b"x")
    (run_dir / "vecnormalize.pkl").write_bytes(b"x")
    run = RunInfo(run_name="bad", run_dir=str(run_dir), algo="sac",
                 vehicle_pc="vehicles/etk800/x.pc", model="etk800", best_avg_g=None)
    ok, msg = mo.export_model_to_game(run, str(tmp_path / "mod"))
    assert ok is False
    assert "boom" in msg


def test_remove_model_from_game(tmp_path, monkeypatch):
    def fake_run_exporter(cmd):
        with open(cmd[cmd.index("--out") + 1], "w") as fh:
            fh.write("-- fake weights\n")
        return True, "ok"
    monkeypatch.setattr(mo, "_run_exporter", fake_run_exporter)
    run_dir = tmp_path / "runs" / "toRemove"
    run_dir.mkdir(parents=True)
    (run_dir / "final.zip").write_bytes(b"x")
    (run_dir / "vecnormalize.pkl").write_bytes(b"x")
    run = RunInfo(run_name="toRemove", run_dir=str(run_dir), algo="sac",
                 vehicle_pc="vehicles/etk800/x.pc", model="etk800", best_avg_g=None)
    mod_dir = str(tmp_path / "mod")
    mo.export_model_to_game(run, mod_dir)
    models_path = os.path.join(mod_dir, "vehicles", "etk800", "ml_abs_models.jbeam")
    assert len(json.loads(open(models_path).read())) == 1

    mo.remove_model_from_game(run, mod_dir)
    assert len(json.loads(open(models_path).read())) == 0


def test_cosim_export_selects_bounded_head_and_two_axle_controller(tmp_path,
                                                                  monkeypatch):
    calls = []
    def fake_run_exporter(cmd):
        with open(cmd[cmd.index("--out") + 1], "w") as fh:
            fh.write("-- fake weights\n")
        calls.append(cmd)
        return True, "PASS"
    monkeypatch.setattr(mo, "_run_exporter", fake_run_exporter)
    run_dir = tmp_path / "runs" / "cosim"
    run_dir.mkdir(parents=True)
    (run_dir / "final.zip").write_bytes(b"x")
    (run_dir / "vecnormalize.pkl").write_bytes(b"x")
    run = RunInfo("cosim", str(run_dir), "ppo", "vehicles/etk800/x.pc",
                  "etk800", 1.1, "cosim_axle_release_v1")
    mod_dir = str(tmp_path / "mod")
    ok, _ = mo.export_model_to_game(run, mod_dir)
    assert ok
    cmd = calls[0]
    assert cmd[cmd.index("--head") + 1] == "ppo_tanh_release01"
    assert cmd[cmd.index("--interface") + 1] == "cosim_axle_release_v1"
    data = json.loads(open(os.path.join(
        mod_dir, "vehicles", "etk800", "ml_abs_models.jbeam")).read())
    controller = next(iter(data.values()))["controller"][1][0]
    assert controller == "MTB-ML-ABS-CoSim"
    assert os.path.isfile(os.path.join(
        mod_dir, "lua", "vehicle", "controller", "MTB-ML-ABS-CoSim.lua"))


def test_incompatible_residual_export_fails_before_subprocess(tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(mo, "_run_exporter",
                        lambda cmd: (_ for _ in ()).throw(AssertionError(cmd)))
    run = RunInfo("res", str(tmp_path), "ppo", "vehicles/etk800/x.pc",
                  "etk800", 1.0, "residual_axle_release_v1")
    ok, message = mo.export_model_to_game(run, str(tmp_path / "mod"))
    assert not ok
    assert "two" in message and "invalid" in message
