"""
Auspex — live ISS telemetry monitor.

Connects to NASA's public, real-time International Space Station telemetry
(Lightstreamer's ISSLIVE feed — no API key, no hardware) and runs the streaming
anomaly detector on real, live channels, narrating anything unusual in plain
English.

    python iss.py                  # watch the default channel set, live
    python iss.py --list           # show which channels it watches
    python iss.py --duration 120   # run for 2 minutes, then stop

This is genuinely live — values come straight off the station via TDRS relay.
When the ISS is between relay satellites (Loss of Signal) the updates pause;
that's normal, and the monitor will say so. A healthy station is mostly quiet,
so alerts are rare by design — the point is a real spacecraft, watched and
explained as it streams.
"""

from __future__ import annotations
import argparse
import sys
import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from auspex.detector import DetectorConfig
from auspex.explainer import explain
from auspex.stream import OnlineDetector
from auspex.sources import ISSLiveSource, ISS_CHANNELS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def live_config() -> DetectorConfig:
    # ISS channels update slowly, so adapt faster than the file-replay defaults.
    return DetectorConfig(forecast_window=20, smoothing=5, threshold_window=60,
                          z=3.0, min_len=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auspex — live ISS telemetry monitor")
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds to run (0 = until Ctrl+C)")
    ap.add_argument("--heartbeat", type=float, default=15,
                    help="status line every N seconds")
    ap.add_argument("--min-severity", type=float, default=5.0,
                    help="only report alerts at/above this peak-sigma (filters quiet-channel "
                         "noise; use 0 to see everything)")
    ap.add_argument("--list", action="store_true", help="list watched channels and exit")
    args = ap.parse_args()

    if args.list:
        print(f"{len(ISS_CHANNELS)} channels:")
        for pui, name in ISS_CHANNELS.items():
            print(f"  {pui:14} {name}")
        return

    src = ISSLiveSource()
    cfg = live_config()
    dets: dict[str, OnlineDetector] = {}
    seen: dict[str, list] = defaultdict(list)
    scores: dict[str, list] = defaultdict(list)
    latest: dict[str, float] = {}
    n_alerts = 0

    print("> connecting to NASA ISS live telemetry (Lightstreamer ISSLIVE)…")
    try:
        src.start()
    except Exception as e:
        print("  could not connect:", e)
        return
    print(f"> watching {len(src.items)} channels. Ctrl+C to stop.\n")

    start = last_hb = last_data = time.time()
    stop = threading.Event()
    try:
        for upd in src.stream(stop):
            now = time.time()

            if upd is not None:
                pui, val = upd
                last_data = now
                latest[pui] = val
                det = dets.setdefault(pui, OnlineDetector(cfg))
                seen[pui].append(val)
                _, sc, ev = det.update(val)
                scores[pui].append(sc)
                if ev is not None:
                    a, b = ev
                    arr = np.asarray(scores[pui])
                    sev = float(np.max(np.abs(arr[a:b]))) if b > a else 0.0
                    if sev >= args.min_severity:        # gate out quiet-channel noise
                        n_alerts += 1
                        det_obj = SimpleNamespace(windows=[ev], score=arr)
                        e = explain(pui, np.asarray(seen[pui]), det_obj)[0]
                        text = e.text.replace(pui, src.label(pui)).replace("samples", "readings")
                        print(f"  [ALERT] {text}")

            if now - last_hb >= args.heartbeat:
                last_hb = now
                quiet = now - last_data
                if quiet > 25:
                    print(f"  … {int(now - start):>4}s · signal quiet for {int(quiet)}s "
                          f"(ISS likely between relays) · {n_alerts} anomalies")
                else:
                    shown = list(latest.items())[:3]
                    sample = " · ".join(f"{src.label(p)}={v:+.2f}" for p, v in shown)
                    print(f"  … {int(now - start):>4}s · {len(latest)} channels live "
                          f"· {n_alerts} anomalies · {sample}")

            if args.duration and now - start >= args.duration:
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        src.stop()

    print(f"\n> stopped after {int(time.time() - start)}s · {n_alerts} anomalies "
          f"across {len(dets)} channels.")


if __name__ == "__main__":
    main()
