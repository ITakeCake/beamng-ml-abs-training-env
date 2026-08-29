"""pedal_gate_verdict is the pure decision logic for GATE 3 (pedal scaling
reaches the wheels) -- no game imports, testable offline."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_probe import pedal_gate_verdict


def test_verdict_pass_when_low_pedal_clearly_brakes_less():
    ok, msg = pedal_gate_verdict([1.03, 1.02, 1.04], [0.55, 0.52, 0.58], margin=0.1)
    assert ok is True


def test_verdict_fail_when_pedal_has_no_effect():
    # if the slam-latch or driver-pedal override beats the mailboxed command,
    # peak_g stays ~identical regardless of the requested pedal
    ok, msg = pedal_gate_verdict([1.03, 1.02, 1.04], [1.01, 1.05, 1.00], margin=0.1)
    assert ok is False


def test_verdict_fail_on_empty_data():
    ok, msg = pedal_gate_verdict([], [0.5], margin=0.1)
    assert ok is False
    ok, msg = pedal_gate_verdict([1.0], [], margin=0.1)
    assert ok is False
