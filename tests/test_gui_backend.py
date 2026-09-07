from gui_train import next_run_name


def test_next_run_name_counts_finished_and_unfinished_directories(tmp_path):
    (tmp_path / "PPO-01").mkdir()
    (tmp_path / "PPO-09").mkdir()
    (tmp_path / "PPO-not-a-number").mkdir()
    (tmp_path / "SAC-99").mkdir()
    assert next_run_name(str(tmp_path)) == "PPO-10"


def test_next_run_name_starts_at_one_for_missing_directory(tmp_path):
    assert next_run_name(str(tmp_path / "missing")) == "PPO-01"
