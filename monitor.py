"""
Auspex live monitor.

Replays a telemetry channel as if it were arriving in real time and prints a
plain-English alert the moment each anomaly closes — the "constantly watching"
view of the same detect-and-explain pipeline.

    python monitor.py --channel A-1          # stream a real channel
    python monitor.py --synthetic            # no data needed
    python monitor.py --channel A-1 --delay 0.01   # slow it down to watch it scroll
    python monitor.py --list                 # show available channels

(The data is a recording, so this is a replay, not a live satellite link — but
the detector runs in true streaming mode, one reading at a time.)
"""

from __future__ import annotations
import argparse
import sys
import time
from types import SimpleNamespace

import numpy as np

from auspex.data import (
    list_channels, load_channel, make_synthetic_channel,
)
from auspex.detector import DetectorConfig
from auspex.explainer import explain
from auspex.stream import OnlineDetector

# the explanations contain σ / – ; make sure they print on any console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Auspex — live telemetry monitor (replay)")
    ap.add_argument("--channel", help="channel ID, e.g. A-1")
    ap.add_argument("--synthetic", action="store_true", help="use generated data")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to pause per reading (e.g. 0.01 to watch it scroll)")
    ap.add_argument("--heartbeat", type=int, default=1000,
                    help="print a status line every N readings (0 to silence)")
    ap.add_argument("--z", type=float, help="override detector sensitivity")
    ap.add_argument("--forecast-window", type=int)
    ap.add_argument("--list", action="store_true", help="list channels and exit")
    args = ap.parse_args()

    if args.list:
        chans = list_channels(args.data_dir)
        print(f"{len(chans)} channel(s): " +
              (", ".join(chans) if chans else "none — download data or use --synthetic"))
        return

    if args.synthetic or not args.channel:
        channel, values, src = "P-SYN", make_synthetic_channel()[0], "synthetic telemetry"
    else:
        channel = args.channel
        try:
            values = load_channel(channel, args.data_dir)
        except FileNotFoundError as e:
            print(e)
            print("Tip: run with --list to see channels, or --synthetic for the demo.")
            return
        src = f"channel {channel}"

    cfg = DetectorConfig()
    if args.z:
        cfg.z = args.z
    if args.forecast_window:
        cfg.forecast_window = args.forecast_window

    det = OnlineDetector(cfg)
    seen: list[float] = []
    scores: list[float] = []
    n_alerts = 0

    def describe(window: tuple[int, int]) -> str:
        det_obj = SimpleNamespace(windows=[window], score=np.asarray(scores))
        return explain(channel, np.asarray(seen), det_obj)[0].text

    print(f"> monitoring {src} - {len(values)} readings streaming in "
          f"(z={cfg.z}). Ctrl+C to stop.\n")
    try:
        for v in values:
            seen.append(float(v))
            flagged, score, event = det.update(v)
            scores.append(score)

            if event is not None:
                n_alerts += 1
                print(f"  [ALERT] {describe(event)}")

            if args.heartbeat and det.i and det.i % args.heartbeat == 0:
                status = "ANOMALY" if flagged else "ok"
                print(f"  ... {det.i:>5} readings scanned | {n_alerts} anomalies "
                      f"| now {float(v):+.2f} [{status}]")

            if args.delay:
                time.sleep(args.delay)

        tail = det.finalize()
        if tail is not None:
            n_alerts += 1
            print(f"  [ALERT] {describe(tail)}")

    except KeyboardInterrupt:
        print(f"\n> stopped after {det.i + 1} readings ({n_alerts} anomalies).")
        return

    print(f"\n> done - {n_alerts} anomalies flagged across {len(values)} readings.")


if __name__ == "__main__":
    main()
