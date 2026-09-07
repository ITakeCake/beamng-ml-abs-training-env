"""Wiring a RewardSpec into the protected env WITHOUT duplicating its step().

abs_env_incar binds every reward constant and _terminal_g_shape into its own
module namespace and reads them as globals inside step(), so rebinding those
names redirects the parent's own reward computation -- the same seam already
used for HEADLESS / MAP_NAME / VEHICLE_PC, and the reason the ~150-line
duplication (and its drift risk) is avoidable.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abs_env
import abs_env_incar
from reward_spec import RewardSpec
from calibration import CalibrationTable, config_key, summarize
from abs_env_residual import (
    ABSLearningEnvResidual,
    install_reward_spec,
    restore_reward_defaults,
    resolve_refs,
)

REFS = (1.0315, 1.1884)


@pytest.fixture(autouse=True)
def _restore_after_each_test():
    """These tests rebind module globals -- always put them back, or a later
    test (or a real run in the same process) inherits a patched reward."""
    yield
    restore_reward_defaults()


def test_defaults_start_equal_to_the_protected_file():
    assert abs_env_incar._terminal_g_shape is abs_env._terminal_g_shape
    assert abs_env_incar.PER_STEP_K == abs_env.PER_STEP_K


def test_installing_v5_leaves_every_value_identical():
    install_reward_spec(RewardSpec.v5(), lambda: None)
    for name in ("PER_STEP_K", "PER_STEP_G_GATE", "YAW_BONUS_K_STEP",
                 "YAW_BONUS_ALPHA", "YAW_BONUS_K_TERMINAL", "YAW_BONUS_THRESHOLD",
                 "YAW_PEN_K_TERMINAL", "YAW_RATE_DEADZONE_RAD_S", "CRASH_PENALTY"):
        assert getattr(abs_env_incar, name) == getattr(abs_env, name), name
    # and the shape still produces the protected file's numbers
    for g in (0.0, 0.3, 0.5, 1.0315, 1.06, 1.1884, 2.0):
        assert abs_env_incar._terminal_g_shape(g) == abs_env._terminal_g_shape(g)


def test_installing_normalized_redirects_the_shape():
    install_reward_spec(RewardSpec.normalized(), lambda: REFS)
    slam_g, stock_g = REFS
    assert abs_env_incar._terminal_g_shape(stock_g) == pytest.approx(0.0, abs=1e-9)
    assert abs_env_incar._terminal_g_shape(slam_g) < 0
    # the protected module itself is untouched -- only the importing namespace
    assert abs_env._terminal_g_shape(stock_g) > 900


def test_restore_puts_the_protected_values_back():
    install_reward_spec(RewardSpec.normalized(), lambda: REFS)
    restore_reward_defaults()
    assert abs_env_incar._terminal_g_shape is abs_env._terminal_g_shape
    assert abs_env_incar.PER_STEP_K == abs_env.PER_STEP_K


def test_normalized_shape_raises_when_refs_are_unavailable():
    """A missing calibration row must stop the run, not silently score against
    absolute anchors."""
    install_reward_spec(RewardSpec.normalized(), lambda: None)
    with pytest.raises(ValueError):
        abs_env_incar._terminal_g_shape(1.0)


def test_installing_v6_routes_dense_and_terminal_calls_correctly():
    state = {"episode": 1, "step": 1, "g": 0.9}
    spec = RewardSpec.v6()
    install_reward_spec(
        spec,
        lambda: None,
        episode_step_provider=lambda: (state["episode"], state["step"]),
        dt_provider=lambda: 0.01,
        step_g_provider=lambda: state["g"],
    )

    # First call under a new step key is the parent's dense reward call. The
    # passed value is intentionally bogus: the signed-G provider must win.
    assert abs_env_incar._terminal_g_shape(999.0) == pytest.approx(0.009)
    # A repeated key only occurs when the parent scores measured terminal avg G.
    assert abs_env_incar._terminal_g_shape(1.2) == pytest.approx(30.0)
    assert abs_env_incar.PER_STEP_K == 1.0
    assert abs_env_incar.YAW_BONUS_K_STEP == 0.0
    assert abs_env_incar.YAW_PEN_K_TERMINAL == 0.0

    state.update(episode=2, step=1, g=1.5)
    # A new episode clears the old rolling window.
    assert abs_env_incar._terminal_g_shape(999.0) == pytest.approx(0.015)


def test_installing_v6_requires_state_and_dt_providers():
    with pytest.raises(ValueError, match="episode/step and dt"):
        install_reward_spec(RewardSpec.v6(), lambda: None)


def test_residual_v6_applies_stability_only_to_a_successful_stop():
    env = object.__new__(ABSLearningEnvResidual)
    env._reward_spec = RewardSpec.v6()
    env.ep_yaw_abs_sum = 0.14
    assert env._terminal_reward_adjustment("STOP") == pytest.approx(-0.75)
    assert env._terminal_reward_adjustment("CRASH") == 0.0
    assert env._terminal_reward_adjustment("TIMEOUT") == -30.0


# --- resolve_refs ---------------------------------------------------------

def _table_with_row(mph=60):
    t = CalibrationTable(car="etk800")
    key = config_key(grip=1.0, speed_mph=mph, radius_m=None)
    t.put(key, "slam", summarize([REFS[0]]))
    t.put(key, "stock", summarize([REFS[1]]))
    return t


def test_resolve_refs_returns_none_for_a_non_normalized_spec():
    assert resolve_refs(RewardSpec.v5(), _table_with_row(), 60, 1.0, None) is None


def test_resolve_refs_looks_up_the_current_config():
    got = resolve_refs(RewardSpec.normalized(), _table_with_row(60), 60, 1.0, None)
    assert got == pytest.approx(REFS)


def test_resolve_refs_refuses_a_missing_row_loudly():
    table = _table_with_row(60)
    with pytest.raises(KeyError):
        resolve_refs(RewardSpec.normalized(), table, 120, 1.0, None)   # 120 never measured


def test_resolve_refs_refuses_when_no_table_at_all():
    with pytest.raises(KeyError):
        resolve_refs(RewardSpec.normalized(), None, 60, 1.0, None)
