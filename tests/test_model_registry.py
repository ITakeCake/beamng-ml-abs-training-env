"""model_registry lists finished training runs (runs/<name>/final.zip) with
enough metadata to export -- no game or SB3 imports, pure filesystem/text
parsing so it's fast and testable offline."""
import os
import json
import sys
import hashlib
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_registry import list_finished_runs, RunInfo


def _make_run(runs_dir, name, algo, vehicle_pc=None, best_avg_g=None, has_final=True):
    d = runs_dir / name
    d.mkdir(parents=True)
    if has_final:
        (d / "final.zip").write_bytes(b"fake")
        (d / "vecnormalize.pkl").write_bytes(b"fake")
    args_line = (f"args: {{'algo': '{algo}', 'vehicle_pc': {vehicle_pc!r}, "
                f"'run_name': '{name}'}}")
    (d / "train.log").write_text(
        f"12:00:00.000 INFO  [trainer] === train_residual start\n"
        f"12:00:00.001 INFO  [trainer] {args_line}\n")
    if best_avg_g is not None:
        (d / "episode_log_env0.csv").write_text(
            "episode,avg_g,outcome\n1,%.3f,STOP\n2,%.3f,STOP\n" % (best_avg_g - 0.05, best_avg_g))
    return d


def _commit_bounded_cosim_final(run):
    model = run / "final.zip"
    vec = run / "vecnormalize.pkl"
    with zipfile.ZipFile(model, "w") as archive:
        archive.writestr("policy.pth", b"valid")
    manifest = {
        "schema": 1,
        "model_path": model.name,
        "vecnormalize_path": vec.name,
        "model_bytes": model.stat().st_size,
        "vecnormalize_bytes": vec.stat().st_size,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "vecnormalize_sha256": hashlib.sha256(vec.read_bytes()).hexdigest(),
    }
    (run / "final.pair.json").write_text(json.dumps(manifest))


def test_list_finished_runs_finds_runs_with_final_zip(tmp_path):
    _make_run(tmp_path, "run1", "sac")
    runs = list_finished_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0].run_name == "run1"
    assert runs[0].algo == "sac"


def test_list_finished_runs_skips_runs_without_final_zip(tmp_path):
    _make_run(tmp_path, "unfinished", "ppo", has_final=False)
    assert list_finished_runs(str(tmp_path)) == []


def test_list_finished_runs_reads_vehicle_pc_from_train_log(tmp_path):
    _make_run(tmp_path, "run2", "ppo", vehicle_pc="vehicles/etk800/MyCar.pc")
    runs = list_finished_runs(str(tmp_path))
    assert runs[0].vehicle_pc == "vehicles/etk800/MyCar.pc"
    assert runs[0].model == "etk800"


def test_list_finished_runs_none_vehicle_pc_defaults_to_reference_car(tmp_path):
    _make_run(tmp_path, "run3", "sac", vehicle_pc=None)
    runs = list_finished_runs(str(tmp_path))
    assert runs[0].model == "etk800"   # the reference MLABS car is an etk800


def test_list_finished_runs_reads_best_avg_g(tmp_path):
    _make_run(tmp_path, "run4", "sac", best_avg_g=1.05)
    runs = list_finished_runs(str(tmp_path))
    assert abs(runs[0].best_avg_g - 1.05) < 1e-6


def test_list_finished_runs_missing_dir_returns_empty(tmp_path):
    assert list_finished_runs(str(tmp_path / "nope")) == []


def test_cosim_config_supplies_algo_episode_log_and_deployment_contract(tmp_path):
    run = tmp_path / "cosim"
    run.mkdir()
    (run / "final.zip").write_bytes(b"fake")
    (run / "vecnormalize.pkl").write_bytes(b"fake")
    (run / "config.json").write_text(json.dumps({
        "vehicle_pc": "vehicles/etk800/cosim.pc",
        "deployment_interface": "cosim_axle_release_v1",
    }))
    _commit_bounded_cosim_final(run)
    (run / "episode_log.csv").write_text(
        "episode,avg_g,outcome\n1,1.05,STOP\n2,1.12,STOP\n")
    info = list_finished_runs(str(tmp_path))[0]
    assert info.algo == "ppo"
    assert info.best_avg_g == 1.12
    assert info.deployment_interface == "cosim_axle_release_v1"


def test_bounded_cosim_final_without_commit_manifest_is_not_finished(tmp_path):
    run = _make_run(tmp_path, "partial_cosim", "ppo")
    (run / "config.json").write_text(json.dumps({
        "deployment_interface": "cosim_axle_release_v1",
    }))
    assert list_finished_runs(str(tmp_path)) == []


def test_bounded_cosim_final_with_mismatched_commit_is_not_finished(tmp_path):
    run = _make_run(tmp_path, "changed_after_commit", "ppo")
    (run / "config.json").write_text(json.dumps({
        "deployment_interface": "cosim_axle_release_v1",
    }))
    _commit_bounded_cosim_final(run)
    (run / "vecnormalize.pkl").write_bytes(b"changed")
    assert list_finished_runs(str(tmp_path)) == []


def test_final_without_vecnormalize_is_not_finished(tmp_path):
    run = _make_run(tmp_path, "missing_vec", "ppo")
    (run / "vecnormalize.pkl").unlink()
    assert list_finished_runs(str(tmp_path)) == []


def test_old_cosim_run_is_marked_unsafe_not_guessed_bounded(tmp_path):
    run = tmp_path / "old_cosim"
    run.mkdir()
    (run / "final.zip").write_bytes(b"fake")
    (run / "vecnormalize.pkl").write_bytes(b"fake")
    (run / "config.json").write_text("{}")
    info = list_finished_runs(str(tmp_path))[0]
    assert info.deployment_interface == "cosim_legacy_unbounded_v0"
