"""
Interpretable baseline anomaly detector.

Deliberately NOT an LSTM. The approach:
  1. One-step "forecast" of each value = trailing rolling mean.
  2. Residual = |actual - forecast|, then EWMA-smoothed.
  3. Dynamic threshold = rolling mean + z * rolling std of the smoothed residual.
  4. Points over threshold are flagged; consecutive flags merge into windows.

Every knob below (window sizes, z, min_len) is a decision you own. The Hundman
et al. (2018) LSTM baseline reaches ~0.85 F1 on this data — keep it as your point
of comparison and "future work", and see how close transparent stats get first.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DetectorConfig:
    forecast_window: int = 150   # trailing window for the rolling-mean forecast
    smoothing: int = 5           # EWMA span applied to the residual
    threshold_window: int = 500  # trailing window for the dynamic threshold stats
    z: float = 2.5               # how many std above local mean counts as anomalous
    min_len: int = 2             # discard flagged runs shorter than this


@dataclass
class DetectionResult:
    windows: list[tuple[int, int]]      # detected [start, end) index pairs
    score: np.ndarray                   # per-point anomaly score (z-like)
    smoothed_residual: np.ndarray
    threshold: np.ndarray
    config: DetectorConfig = field(default_factory=DetectorConfig)


def _merge_runs(flags: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Turn a boolean mask into [start, end) windows, dropping short runs."""
    windows: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                windows.append((start, i))
            start = None
    if start is not None and len(flags) - start >= min_len:
        windows.append((start, len(flags)))
    return windows


def detect(values: np.ndarray, config: DetectorConfig | None = None) -> DetectionResult:
    cfg = config or DetectorConfig()
    s = pd.Series(np.asarray(values, dtype=float))

    # 1. trailing-mean forecast (shifted so we never peek at the current value)
    forecast = s.shift(1).rolling(cfg.forecast_window, min_periods=1).mean()

    # 2. smoothed residual
    residual = (s - forecast).abs()
    smoothed = residual.ewm(span=cfg.smoothing, adjust=False).mean()

    # 3. dynamic threshold from trailing stats of the smoothed residual
    mu = smoothed.rolling(cfg.threshold_window, min_periods=cfg.threshold_window // 3).mean()
    sd = smoothed.rolling(cfg.threshold_window, min_periods=cfg.threshold_window // 3).std()
    mu = mu.bfill()
    sd = sd.bfill().replace(0, np.nan).ffill().fillna(1e-9)
    threshold = mu + cfg.z * sd

    # z-like score, useful for severity and plotting
    score = ((smoothed - mu) / sd).fillna(0.0).to_numpy()

    flags = (smoothed > threshold).to_numpy()
    windows = _merge_runs(flags, cfg.min_len)

    return DetectionResult(
        windows=windows,
        score=score,
        smoothed_residual=smoothed.to_numpy(),
        threshold=threshold.to_numpy(),
        config=cfg,
    )
