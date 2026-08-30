"""Network shape was hardcoded to 3x256 with no way to change it. It is now a
setting, which means it needs bounds: the trained net is evaluated by hand in
Lua every 0.5 ms physics tick, and one too slow to keep up misses ticks with
nothing logged anywhere."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residual_core import (parse_net_arch, net_arch_repr, DEFAULT_NET_ARCH,
                           NET_MAX_LAYERS, NET_MAX_WIDTH, NET_MIN_WIDTH)


@pytest.mark.parametrize("text", ["", "default", "none", "  DEFAULT "])
def test_blank_means_the_reference_shape(text):
    """Every result in this project so far used 3x256; blank must not quietly
    mean something else."""
    assert parse_net_arch(text) == DEFAULT_NET_ARCH


def test_explicit_list_and_shorthand_agree():
    assert parse_net_arch("256,256,256") == parse_net_arch("3x256") == [256] * 3


def test_a_deep_narrow_network_is_allowed():
    """The case that prompted this: "what if I wanted a 22 layer PPO"."""
    assert parse_net_arch("22x128") == [128] * 22


def test_uneven_widths_are_kept_in_order():
    assert parse_net_arch("512,256,128") == [512, 256, 128]


def test_whitespace_is_tolerated():
    assert parse_net_arch(" 512 , 256 ") == [512, 256]


def test_too_many_layers_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="physics tick"):
        parse_net_arch(f"{NET_MAX_LAYERS + 1}x64")


def test_the_layer_cap_itself_is_allowed():
    assert len(parse_net_arch(f"{NET_MAX_LAYERS}x64")) == NET_MAX_LAYERS


def test_widths_outside_the_range_are_refused():
    with pytest.raises(ValueError, match="out of range"):
        parse_net_arch(f"{NET_MAX_WIDTH + 1},64")
    with pytest.raises(ValueError, match="out of range"):
        parse_net_arch(f"{NET_MIN_WIDTH - 1}")


@pytest.mark.parametrize("bad", ["abc", "3x", "x256", "256,,256", "0x256", ","])
def test_garbage_is_refused_rather_than_guessed(bad):
    with pytest.raises(ValueError):
        parse_net_arch(bad)


def test_repr_round_trips_through_the_parser():
    for text in ("3x256", "22x128", "512,256,128"):
        assert net_arch_repr(parse_net_arch(text)) == text


def test_repr_compacts_uniform_layers_but_not_mixed():
    assert net_arch_repr([256, 256, 256]) == "3x256"
    assert net_arch_repr([512, 256]) == "512,256"


# --------------------------------------------------------- command plumbing
from gui_cmd import build_cmd, validate_settings


def _base():
    return dict(algo="ppo", speeds="60", pedal_random=False, pedal_spec="0.4-1.0",
                total_steps="1000", run_name="r1", lr="1e-4", n_steps="2048",
                batch_size="512", n_epochs="10", clip_range="0.2",
                gae_lambda="0.95", ent_coef="0.005")


def test_net_arch_reaches_the_trainer():
    s = _base()
    s["net_arch"] = "22x128"
    assert build_cmd(s)[build_cmd(s).index("--net-arch") + 1] == "22x128"


def test_an_impossible_network_is_caught_before_the_game_launches():
    """Otherwise the user waits for BeamNG to boot to find out."""
    s = _base()
    s["net_arch"] = "40x256"
    assert any("physics tick" in p for p in validate_settings(s))


def test_omitting_net_arch_keeps_the_trainers_own_default():
    assert "--net-arch" not in build_cmd(_base())
