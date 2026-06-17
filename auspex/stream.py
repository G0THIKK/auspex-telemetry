"""
Online (streaming) version of the detector.

`detector.detect()` works on a whole array at once — fine for scoring a recorded
file, useless for live monitoring. This processes ONE reading at a time, keeping
the same logic in incremental form:

  forecast = trailing mean of prior values
  residual = |actual - forecast|, EWMA-smoothed
  threshold = (trailing mean + z * trailing std) of the smoothed residual

Because the batch detector already only ever looks backward, the streaming math
is faithful to it on the bulk of a channel. Two honest differences at the very
start: we can't back-fill threshold stats from the future, so nothing is flagged
until `threshold_window // 3` readings have arrived (a natural warm-up that
actually avoids some of the batch detector's start-up false alarms).

Usage:
    det = OnlineDetector()
    for value in stream:
        flagged, score, event = det.update(value)
        if event:                 # (start, end) of a just-closed anomaly window
            ...
    tail = det.finalize()         # flush an anomaly still open at end-of-stream
"""

from __future__ import annotations
from collections import deque

from .detector import DetectorConfig


class OnlineDetector:
    def __init__(self, config: DetectorConfig | None = None):
        cfg = config or DetectorConfig()
        self.cfg = cfg
        self.fw = cfg.forecast_window
        self.alpha = 2.0 / (cfg.smoothing + 1)          # EWMA, adjust=False
        self.tw = cfg.threshold_window
        self.min_periods = max(2, cfg.threshold_window // 3)
        self.z = cfg.z
        self.min_len = cfg.min_len

        self._fbuf: deque[float] = deque(maxlen=self.fw)   # trailing raw values
        self._fsum = 0.0
        self._smoothed_prev: float | None = None
        self._tbuf: deque[float] = deque(maxlen=self.tw)   # trailing smoothed values
        self._tsum = 0.0
        self._tsumsq = 0.0
        self._last_sd = 1e-9

        self.i = -1                 # index of the most recent reading
        self._run_start: int | None = None
        self._run_len = 0

    def update(self, value: float) -> tuple[bool, float, tuple[int, int] | None]:
        """Feed one reading. Returns (flagged, z_like_score, just_closed_event)."""
        self.i += 1
        i = self.i
        v = float(value)

        # 1. forecast = mean of trailing PRIOR values (never peeks at v)
        forecast = self._fsum / len(self._fbuf) if self._fbuf else v
        residual = abs(v - forecast)
        if len(self._fbuf) == self.fw:
            self._fsum -= self._fbuf[0]
        self._fbuf.append(v)
        self._fsum += v

        # 2. EWMA-smoothed residual
        if self._smoothed_prev is None:
            smoothed = residual
        else:
            smoothed = self.alpha * residual + (1.0 - self.alpha) * self._smoothed_prev
        self._smoothed_prev = smoothed

        # 3. dynamic threshold from trailing stats of the smoothed residual
        if len(self._tbuf) == self.tw:
            old = self._tbuf[0]
            self._tsum -= old
            self._tsumsq -= old * old
        self._tbuf.append(smoothed)
        self._tsum += smoothed
        self._tsumsq += smoothed * smoothed

        n = len(self._tbuf)
        flagged, score = False, 0.0
        if n >= self.min_periods:
            mu = self._tsum / n
            var = (self._tsumsq - self._tsum * self._tsum / n) / (n - 1) if n > 1 else 0.0
            sd = var ** 0.5 if var > 0 else 0.0
            if sd <= 0:
                sd = self._last_sd
            else:
                self._last_sd = sd
            score = (smoothed - mu) / sd if sd else 0.0
            flagged = smoothed > mu + self.z * sd

        # 4. track the current run of flags; emit a window when it ENDS
        event = None
        if flagged:
            if self._run_start is None:
                self._run_start, self._run_len = i, 1
            else:
                self._run_len += 1
        else:
            if self._run_start is not None and self._run_len >= self.min_len:
                event = (self._run_start, i)
            self._run_start, self._run_len = None, 0

        return flagged, score, event

    def finalize(self) -> tuple[int, int] | None:
        """Flush an anomaly still in progress when the stream ends."""
        if self._run_start is not None and self._run_len >= self.min_len:
            ev = (self._run_start, self.i + 1)
            self._run_start, self._run_len = None, 0
            return ev
        return None
