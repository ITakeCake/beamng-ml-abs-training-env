"""The live measurement in reference_runner needs a running game, but its
guard rails and reference-car wiring are pure and must not regress -- a wrong
calibration row is worse than a missing one, since training consumes it
silently."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reference_runner import check_supported, REFERENCE_CARS, ABS_BEHAVIOR
from calibration import STRAIGHT

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_check_supported_allows_the_implemented_case():
    check_supported(grip=1.0, radius_m=STRAIGHT)   # must not raise


def test_check_supported_allows_grip_now_that_it_is_implemented():
    check_supported(grip=0.5, radius_m=STRAIGHT)    # must not raise


def test_check_supported_refuses_a_corner_with_no_measured_steering_angle():
    """A corner asked for with no angle would brake in a straight line and get
    written under a key claiming a radius -- the exact class of silently-wrong
    row the runner exists to refuse."""
    with pytest.raises(ValueError, match="steering seek"):
        check_supported(grip=1.0, radius_m=50)


def test_check_supported_allows_a_corner_once_the_angle_is_known():
    check_supported(grip=1.0, radius_m=50, steering=0.25)   # must not raise


def test_reference_cars_differ_only_in_the_abs_slot():
    """The whole calibration rests on ABS being the ONLY difference between the
    slam and stock cars -- if ESC/TC/tires/brakes ever diverge, the measured
    'stock ABS advantage' silently becomes an advantage of something else."""
    def parts(pc_rel):
        path = os.path.join(REPO, "assets", "cars", "etk800", os.path.basename(pc_rel))
        with open(path, encoding="utf-8-sig") as fh:
            return json.load(fh)["parts"]

    slam = parts(REFERENCE_CARS["slam"])
    stock = parts(REFERENCE_CARS["stock"])
    differing = {k for k in set(slam) | set(stock) if slam.get(k) != stock.get(k)}
    assert differing == {"etk_DSE_ABS"}
    assert slam["etk_DSE_ABS"] == ""                 # no ABS part at all -> lockup
    assert stock["etk_DSE_ABS"] == "etk_DSE_ABS"     # stock BeamNG ABS (enableABS:true)


def test_stock_reference_never_uses_arcade_abs():
    """BeamNG's own built-in brake test defaults to 'arcade', the idealized
    cheating mode. Anchoring the reward on that would be anchoring it on a
    fiction -- the stock reference must be the vehicle's real configured ABS."""
    assert ABS_BEHAVIOR["stock"] == "realistic"
    assert ABS_BEHAVIOR["slam"] == "off"
    assert "arcade" not in ABS_BEHAVIOR.values()
