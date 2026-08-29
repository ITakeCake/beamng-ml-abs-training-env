"""model_registry lists finished training runs (runs/<name>/final.zip) with
enough metadata to export -- no game or SB3 imports, pure filesystem/text
parsing so it's fast and testable offline."""
import os
import sys

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
