"""Hover help is documentation that ships, so it is checked like code: every
field that has one gets one, and the text is actually explanatory rather than a
restatement of the label."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gui_help
from gui_cmd import SAC_KEYS, PPO_KEYS


def test_every_sac_and_ppo_field_has_help():
    """The algo panels are built straight from these key lists, so a field
    added there without help would silently ship a bare box."""
    for key in SAC_KEYS:
        assert gui_help.help_for("sac", key), f"no help for SAC field {key}"
    for key in PPO_KEYS:
        assert gui_help.help_for("ppo", key), f"no help for PPO field {key}"


def test_every_run_setting_has_help():
    for key in ("speeds", "grip", "corner", "reward", "run_name",
                "total_steps", "algo", "pedal_random", "pedal_spec"):
        assert gui_help.help_for("sac", key), f"no help for {key}"


def test_help_is_a_real_explanation_not_a_restatement():
    """A tooltip that just repeats the label teaches nothing."""
    for algo in ("sac", "ppo"):
        for key in (SAC_KEYS if algo == "sac" else PPO_KEYS):
            text = gui_help.help_for(algo, key)
            assert len(text) > 120, f"{key} help is too short to explain anything"
            assert "\n" in text, f"{key} help is a single blob, not readable"


def test_jargon_terms_are_defined_where_they_appear():
    """These labels are meaningless to a reader who has not done RL, which is
    the entire reason the tooltips exist."""
    assert "randomness" in gui_help.help_for("sac", "target_entropy").lower()
    assert "stable copy" in gui_help.help_for("sac", "tau").lower()
    assert "credit" in gui_help.help_for("ppo", "gae_lambda").lower()


def test_unknown_fields_return_none_rather_than_raising():
    """attach() treats a missing entry as 'no tooltip'; it must never be an
    error, or adding a widget before its help crashes the GUI."""
    assert gui_help.help_for("sac", "no_such_field") is None
    assert gui_help.help_for("ppo", "no_such_field") is None


def test_run_help_wins_over_algo_help_for_shared_names():
    """RUN_HELP is checked first, so a run-level key is never shadowed."""
    assert gui_help.help_for("sac", "speeds") == gui_help.RUN_HELP["speeds"]


def test_learning_rate_help_is_shared_by_both_algorithms():
    assert gui_help.help_for("sac", "lr") == gui_help.help_for("ppo", "lr")


def test_the_learning_starts_trap_is_documented():
    """Setting it above total steps means nothing is ever learned -- the exact
    mistake that produced a run of pure random actions in this project."""
    text = gui_help.help_for("sac", "learning_starts").lower()
    assert "never learns" in text or "never learn" in text


def test_pedal_help_states_it_is_constant_within_an_attempt():
    """The genuinely ambiguous one: "randomize pedal" reads equally well as
    "one value per stop" or "jittering during the stop". It is the former
    (drawn in reset(), held all episode), and the text has to say so."""
    text = gui_help.help_for("sac", "pedal_random").lower()
    assert "one value per" in text or "one value" in text
    assert "does not move during" in text
    assert gui_help.help_for("sac", "pedal_spec").lower().count("held") >= 1
