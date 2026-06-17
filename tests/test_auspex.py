"""
Auspex test suite.

Runs with pytest (`pip install pytest && pytest`) or as a plain script with no
dependencies (`python tests/test_auspex.py`).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auspex.data import make_synthetic_channel, channel_type          # noqa: E402
from auspex.detector import detect                                     # noqa: E402
from auspex.evaluate import score, _overlaps                          # noqa: E402
from auspex.stream import OnlineDetector                              # noqa: E402
from auspex.explainer import explain, summarize                       # noqa: E402


def test_detector_catches_synthetic_dropout():
    vals, truth = make_synthetic_channel()
    res = detect(vals)
    assert _overlaps(truth[1], res.windows)        # the injected dropout window


def test_score_math():
    # predicted 2..5, truth 3..6 -> tp{3,4}=2, fp{2}=1, fn{5}=1
    s = score(10, [(2, 5)], [(3, 6)])
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    assert abs(s.precision - 2 / 3) < 1e-9
    assert s.detected_true == 1 and s.true_windows == 1


def test_stream_matches_batch_on_synthetic():
    vals, _ = make_synthetic_channel()
    batch = detect(vals).windows
    det, stream = OnlineDetector(), []
    for v in vals:
        _, _, ev = det.update(v)
        if ev:
            stream.append(ev)
    tail = det.finalize()
    if tail:
        stream.append(tail)
    for w in batch:                                 # streaming catches every batch window
        assert _overlaps(w, stream)


def test_explainer_dropout_direction():
    vals, truth = make_synthetic_channel()
    res = detect(vals)
    drop = [e for e in explain("P-SYN", vals, res) if _overlaps((e.start, e.end), [truth[1]])]
    assert drop and any(e.shape == "dropout" for e in drop)


def test_no_unknown_subsystem_leak():
    vals, _ = make_synthetic_channel()
    assert channel_type("D-99") == "unknown"
    for e in explain("D-99", vals, detect(vals)):   # never render a fake "(unknown)" subsystem
        assert "(unknown)" not in e.text


def test_summarize_runs():
    vals, truth = make_synthetic_channel()
    text = summarize("P-SYN", vals, detect(vals), truth)
    assert "In plain English" in text and len(text) > 100


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
