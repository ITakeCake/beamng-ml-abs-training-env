import logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residual_log import setup_logging, StepRingBuffer, tail_lines


def test_setup_logging_writes_utf8_file_and_is_idempotent(tmp_path):
    path = tmp_path / "t.log"
    log = setup_logging(str(path), component="test")
    log2 = setup_logging(str(path), component="test")   # second call must NOT double handlers
    log.info("arrow → dash, ok")
    for h in logging.getLogger().handlers:
        h.flush()
    text = path.read_text(encoding="utf-8")
    assert text.count("arrow → dash, ok") == 1
    assert "INFO" in text and "[test]" in text
    file_handlers = [h for h in logging.getLogger().handlers
                     if isinstance(h, logging.FileHandler) and h.baseFilename == str(path.resolve())]
    assert len(file_handlers) == 1


def test_ring_buffer_keeps_last_n_and_formats():
    rb = StepRingBuffer(capacity=3)
    for i in range(5):
        rb.push(step=i, action=(0.1 * i, 0.2), brakes=(0.9, 0.9, 0.8, 0.8), gy=-5.0, yaw=0.01, speed=20.0 - i)
    rows = rb.dump()
    assert len(rows) == 3
    assert rows[0].startswith("step=2") and rows[-1].startswith("step=4")
    assert "rel=" in rows[0] and "spd=" in rows[0]
    rb.clear()
    assert rb.dump() == []


def test_tail_lines_missing_file_and_last_n(tmp_path):
    assert tail_lines(str(tmp_path / "nope.log"), 5) == []
    p = tmp_path / "a.log"
    p.write_text("\n".join(f"L{i}" for i in range(30)), encoding="utf-8")
    assert tail_lines(str(p), 3) == ["L27", "L28", "L29"]
