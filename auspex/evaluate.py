"""
Scoring against the labeled anomaly windows.

Two views, because "did we catch it" and "how noisy were we" are different
questions, and which one matters is a judgement call (YOUR DECISION):

  - point-wise precision / recall / F1: per-sample agreement.
  - window-level detection rate: fraction of TRUE anomalies that any predicted
    window overlapped (the metric closest to what an operator cares about),
    plus a false-alarm count for predicted windows that hit nothing.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    true_windows: int
    detected_true: int           # true windows with >=1 overlapping prediction
    pred_windows: int
    false_alarms: int            # predicted windows overlapping no true window
    tp: int = 0                  # point-wise counts, exposed so callers can pool
    fp: int = 0                  # (micro-average) results across many channels
    fn: int = 0

    def pretty(self) -> str:
        return (
            f"point-wise  P={self.precision:.2f}  R={self.recall:.2f}  F1={self.f1:.2f}\n"
            f"windows     caught {self.detected_true}/{self.true_windows} true "
            f"| {self.false_alarms} false alarm(s) of {self.pred_windows} predicted"
        )


def _mask(n: int, windows) -> list[bool]:
    m = [False] * n
    for a, b in windows:
        for i in range(max(0, a), min(n, b)):
            m[i] = True
    return m


def _overlaps(w, windows) -> bool:
    a, b = w
    return any(not (b <= c or a >= d) for c, d in windows)


def score(n: int, predicted, truth) -> Scores:
    pm, tm = _mask(n, predicted), _mask(n, truth)
    tp = sum(p and t for p, t in zip(pm, tm))
    fp = sum(p and not t for p, t in zip(pm, tm))
    fn = sum((not p) and t for p, t in zip(pm, tm))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    detected_true = sum(_overlaps(t, predicted) for t in truth)
    false_alarms = sum(not _overlaps(p, truth) for p in predicted)

    return Scores(
        precision=precision, recall=recall, f1=f1,
        true_windows=len(truth), detected_true=detected_true,
        pred_windows=len(predicted), false_alarms=false_alarms,
        tp=tp, fp=fp, fn=fn,
    )
