"""
Data access for Auspex.

Two sources:
  1. Real NASA SMAP/MSL telemetry (download separately into ./data — see README).
  2. A synthetic generator, so the full pipeline runs before you have the real data.

Real-data format (from the telemanom dataset):
  - data/test/<CHANNEL>.npy   shape (n_timesteps, n_inputs)
  - data/train/<CHANNEL>.npy  shape (n_timesteps, n_inputs)
  - The telemetry value being evaluated is the FIRST column: arr[:, 0].
    (Remaining columns are one-hot command context — ignored by this baseline.)
  - labeled_anomalies.csv with columns including:
        chan_id, spacecraft, anomaly_sequences, class, num_values
    where anomaly_sequences is a string like "[[1899, 2099], [4286, 4594]]"
    giving [start, end] index pairs in the TEST series.
"""

from __future__ import annotations
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Channel-ID prefix -> human-readable type.
# NOTE: the dataset is anonymized; only P (power) and R (radiation) are documented.
# The rest are guesses/placeholders — refine as you learn the data. (YOUR DECISION)
CHANNEL_TYPES = {
    "P": "power",
    "R": "radiation",
    "T": "temperature",
    "E": "energy",
    "A": "attitude",
    "S": "sensor",
}


def channel_type(channel_id: str) -> str:
    """Map a channel ID (e.g. 'P-1') to a human-readable subsystem type."""
    prefix = channel_id.strip()[:1].upper()
    return CHANNEL_TYPES.get(prefix, "unknown")


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def load_channel(channel_id: str, data_dir: str | Path = "data", split: str = "test") -> np.ndarray:
    """Load a single channel's telemetry series (1-D) from the NASA dataset."""
    path = Path(data_dir) / split / f"{channel_id}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the SMAP/MSL data into '{data_dir}/' "
            f"(see README) or use the synthetic generator."
        )
    arr = np.load(path)
    # telemetry value = first column; rest is command context
    return arr[:, 0].astype(float)


def load_labels(data_dir: str | Path = "data") -> dict[str, list[tuple[int, int]]]:
    """Parse labeled_anomalies.csv into {channel_id: [(start, end), ...]}."""
    path = Path(data_dir) / "labeled_anomalies.csv"
    if not path.exists():
        # also accept it at repo root, where some mirrors drop it
        alt = Path("labeled_anomalies.csv")
        path = alt if alt.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"labeled_anomalies.csv not found near '{data_dir}/'.")
    df = pd.read_csv(path)
    labels: dict[str, list[tuple[int, int]]] = {}
    for _, row in df.iterrows():
        seqs = ast.literal_eval(str(row["anomaly_sequences"]))
        labels[str(row["chan_id"])] = [(int(a), int(b)) for a, b in seqs]
    return labels


def list_channels(data_dir: str | Path = "data", split: str = "test") -> list[str]:
    d = Path(data_dir) / split
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.npy"))


# --------------------------------------------------------------------------- #
# Synthetic data (smoke test — runs with no download)
# --------------------------------------------------------------------------- #
def make_synthetic_channel(
    n: int = 4000,
    seed: int = 7,
    channel_id: str = "P-SYN",
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Build a believable telemetry series in [-1, 1] with four injected anomaly
    types (spike, dropout, level-shift, drift). Returns (values, true_windows).

    Useful for proving the pipeline end-to-end before the real data is in place.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # slow seasonal baseline + small noise, scaled into ~[-1, 1] like the real set
    base = 0.35 * np.sin(2 * np.pi * t / 600) + 0.12 * np.sin(2 * np.pi * t / 90)
    values = base + rng.normal(0, 0.03, n)

    truth: list[tuple[int, int]] = []

    # 1) point spike
    s = 800
    values[s:s + 5] += 0.9
    truth.append((s, s + 5))

    # 2) dropout toward the floor
    s = 1600
    values[s:s + 120] = -0.95 + rng.normal(0, 0.01, 120)
    truth.append((s, s + 120))

    # 3) sustained level shift
    s = 2500
    values[s:] += 0.4
    truth.append((s, s + 200))  # label the onset region

    # 4) gradual drift
    s, e = 3200, 3500
    values[s:e] += np.linspace(0, 0.6, e - s)
    truth.append((s, e))

    return np.clip(values, -1, 1), truth
