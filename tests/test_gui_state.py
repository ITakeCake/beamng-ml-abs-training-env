"""The Training tab forgot everything on close, and switching SAC->PPO->SAC
rebuilt the panel from defaults, wiping tuned hyperparameters. Losing them
silently is worse than a crash: a run launched with numbers the user did not
choose still trains, and its result looks legitimate."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gui_state


def test_a_missing_file_gives_empty_state_rather_than_raising(tmp_path):
    """Startup convenience -- a stack trace here is worse than defaults."""
    s = gui_state.load(str(tmp_path / "nope.json"))
    assert s == gui_state.default_state()


def test_a_corrupt_file_gives_empty_state(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert gui_state.load(str(p)) == gui_state.default_state()


def test_a_state_from_a_different_version_is_not_guessed_at(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"version": 999, "run": {"speeds": "90"}}), encoding="utf-8")
    assert gui_state.run_value(gui_state.load(str(p)), "speeds", "60") == "60"


def test_run_fields_survive_a_save_load_round_trip(tmp_path):
    p = str(tmp_path / "s.json")
    s = gui_state.remember_run(gui_state.default_state(),
                               {"speeds": "60,90", "reward": "normalized",
                                "corner": "150L", "grip": "0.5,1.0"})
    gui_state.save(s, p)
    back = gui_state.load(p)
    assert gui_state.run_value(back, "speeds", "") == "60,90"
    assert gui_state.run_value(back, "reward", "") == "normalized"
    assert gui_state.run_value(back, "corner", "") == "150L"


def test_unknown_keys_are_not_persisted():
    """Adding a widget should not silently start remembering it."""
    s = gui_state.remember_run(gui_state.default_state(),
                               {"speeds": "60", "something_else": "x"})
    assert "something_else" not in s["run"]


def test_sac_and_ppo_values_do_not_overwrite_each_other():
    """The reported bug: switch to SAC and the PPO values are gone."""
    s = gui_state.default_state()
    gui_state.remember_algo(s, "ppo", {"lr": "0.0002", "n_steps": "4096"})
    gui_state.remember_algo(s, "sac", {"lr": "0.0005", "tau": "0.01"})
    assert gui_state.algo_values(s, "ppo", {"lr": 1e-4, "n_steps": 2048})["lr"] == "0.0002"
    assert gui_state.algo_values(s, "sac", {"lr": 1e-4, "tau": 0.005})["lr"] == "0.0005"


def test_switching_away_and_back_restores_what_was_typed():
    s = gui_state.default_state()
    defaults = {"lr": 1e-4, "n_epochs": 10}
    gui_state.remember_algo(s, "ppo", {"lr": "0.0002", "n_epochs": "4"})
    gui_state.remember_algo(s, "sac", {"lr": "0.0009"})     # visit SAC
    assert gui_state.algo_values(s, "ppo", defaults) == {"lr": "0.0002", "n_epochs": "4"}


def test_a_field_added_after_the_state_was_written_still_gets_its_default():
    """Merged per key, not all-or-nothing, so a new hyperparameter appears with
    its default rather than vanishing from the panel."""
    s = gui_state.default_state()
    gui_state.remember_algo(s, "sac", {"lr": "0.0005"})
    vals = gui_state.algo_values(s, "sac", {"lr": 1e-4, "brand_new": 7})
    assert vals == {"lr": "0.0005", "brand_new": "7"}


def test_defaults_are_used_for_an_algorithm_never_visited():
    s = gui_state.default_state()
    assert gui_state.algo_values(s, "ppo", {"lr": 1e-4}) == {"lr": "0.0001"}


def test_values_are_stored_as_strings_because_the_boxes_are_text():
    s = gui_state.remember_algo(gui_state.default_state(), "sac", {"lr": 0.0001})
    assert s["algo"]["sac"]["lr"] == "0.0001"


def test_the_car_selection_is_remembered():
    """Re-picking model -> trim -> custom every launch is the most tedious part
    of the form."""
    s = gui_state.remember_run(gui_state.default_state(),
                               {"car_model": "ETK 800-Series",
                                "car_trim": "Custom...",
                                "car_custom": "Machine-Trainer-Boy-V2-MLABS"})
    assert gui_state.run_value(s, "car_custom", "") == "Machine-Trainer-Boy-V2-MLABS"


def test_a_boolean_survives_the_round_trip(tmp_path):
    p = str(tmp_path / "s.json")
    gui_state.save(gui_state.remember_run(gui_state.default_state(),
                                          {"pedal_random": True}), p)
    assert gui_state.run_value(gui_state.load(p), "pedal_random", False) is True
