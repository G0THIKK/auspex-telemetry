"""
Auspex CLI.

    python run.py --synthetic              # runs end-to-end with no download
    python run.py --channel T-1            # real NASA channel (data in ./data)
    python run.py --channel P-1 --z 3.5    # override a detector knob
"""

from __future__ import annotations
import argparse

import numpy as np

from auspex.data import (
    load_channel, load_labels, make_synthetic_channel,
)
from auspex.detector import detect, DetectorConfig
from auspex.explainer import explain
from auspex.evaluate import score


def main() -> None:
    ap = argparse.ArgumentParser(description="Auspex — explain spacecraft telemetry anomalies")
    ap.add_argument("--channel", help="channel ID, e.g. T-1 (real data in ./data)")
    ap.add_argument("--synthetic", action="store_true", help="use generated data instead")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--forecast-window", type=int)
    ap.add_argument("--z", type=float)
    args = ap.parse_args()

    if args.synthetic or not args.channel:
        channel = "P-SYN"
        values, truth = make_synthetic_channel()
        print("Using synthetic telemetry (4 injected anomalies).")
    else:
        channel = args.channel
        values = load_channel(channel, args.data_dir)
        truth = load_labels(args.data_dir).get(channel, [])
        print(f"Loaded {channel}: {len(values)} samples, {len(truth)} labeled anomalies.")

    cfg = DetectorConfig()
    if args.forecast_window:
        cfg.forecast_window = args.forecast_window
    if args.z:
        cfg.z = args.z

    result = detect(values, cfg)
    explanations = explain(channel, values, result)

    print(f"\nDetected {len(result.windows)} anomaly window(s):\n")
    for e in explanations:
        print("  \u2022 " + e.text)

    if truth:
        print("\n" + score(len(values), result.windows, truth).pretty())


if __name__ == "__main__":
    main()
