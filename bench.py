"""
Auspex benchmark — run the detector across every channel and aggregate the
results the README table wants, split by spacecraft (SMAP vs MSL).

    python bench.py                       # all channels in ./data, default knobs
    python bench.py --z 3.0               # sweep a detector knob
    python bench.py --limit 5             # quick smoke run on the first 5 channels
    python bench.py --data-dir /path/data

Two aggregates per spacecraft, matching evaluate.py's two views:

  - "Anomalies caught": window-level detection rate, summed across channels
    (true windows with >=1 overlapping prediction / all true windows). This is
    the headline operator metric.
  - "Point F1": point-wise precision/recall/F1, *micro-averaged* — TP/FP/FN are
    pooled across all of a spacecraft's channels, then P/R/F1 computed once. This
    weights each sample equally and avoids the distortion a macro-average of
    per-channel F1s would introduce on channels with few anomalous points.

The script prints a human-readable summary and, at the end, Markdown rows you
can paste straight into the README results table.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from auspex.data import load_channel
from auspex.detector import detect, DetectorConfig
from auspex.evaluate import score


@dataclass
class Agg:
    """Running totals for one spacecraft."""
    channels: int = 0
    skipped: list[str] = field(default_factory=list)   # channels with no .npy on disk
    true_windows: int = 0
    detected_true: int = 0
    pred_windows: int = 0
    false_alarms: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _labels_path(data_dir: Path) -> Path:
    """labeled_anomalies.csv lives under data/ on most mirrors, repo root on some."""
    for cand in (data_dir / "labeled_anomalies.csv", Path("labeled_anomalies.csv")):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"labeled_anomalies.csv not found in '{data_dir}/' or repo root. "
        "Download the SMAP/MSL data first (see README)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Auspex results by spacecraft")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test", help="which split the labels index (default: test)")
    ap.add_argument("--forecast-window", type=int)
    ap.add_argument("--z", type=float)
    ap.add_argument("--limit", type=int, help="only run the first N channels (quick check)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    df = pd.read_csv(_labels_path(data_dir))
    if args.limit:
        df = df.head(args.limit)

    cfg = DetectorConfig()
    if args.forecast_window:
        cfg.forecast_window = args.forecast_window
    if args.z:
        cfg.z = args.z

    aggs: dict[str, Agg] = {}
    import ast

    for _, row in df.iterrows():
        chan = str(row["chan_id"])
        craft = str(row.get("spacecraft", "?")).strip() or "?"
        agg = aggs.setdefault(craft, Agg())

        try:
            values = load_channel(chan, data_dir, split=args.split)
        except FileNotFoundError:
            agg.skipped.append(chan)
            continue

        truth = [(int(a), int(b)) for a, b in ast.literal_eval(str(row["anomaly_sequences"]))]
        result = detect(values, cfg)
        s = score(len(values), result.windows, truth)

        agg.channels += 1
        agg.true_windows += s.true_windows
        agg.detected_true += s.detected_true
        agg.pred_windows += s.pred_windows
        agg.false_alarms += s.false_alarms
        agg.tp += s.tp
        agg.fp += s.fp
        agg.fn += s.fn

    # ----- report -----
    print(f"\nDetector knobs: forecast_window={cfg.forecast_window} z={cfg.z} "
          f"smoothing={cfg.smoothing} threshold_window={cfg.threshold_window} "
          f"min_len={cfg.min_len}\n")

    rows = []
    for craft in sorted(aggs):
        a = aggs[craft]
        print(f"== {craft} ==")
        print(f"  channels scored : {a.channels}" +
              (f"  ({len(a.skipped)} skipped, no .npy)" if a.skipped else ""))
        print(f"  anomalies caught: {a.detected_true}/{a.true_windows} windows")
        print(f"  false alarms    : {a.false_alarms} of {a.pred_windows} predicted windows")
        print(f"  point-wise      : P={a.precision:.2f}  R={a.recall:.2f}  F1={a.f1:.2f}\n")
        notes = f"{a.channels} channels, {a.false_alarms} false alarms"
        rows.append(
            f"| {craft + ' (real)':<18} | {f'{a.detected_true} / {a.true_windows}':<16} "
            f"| {a.f1:<8.2f} | {notes:<30} |"
        )

    if rows:
        print("README table rows (paste into the Results section):\n")
        print("| Dataset            | Anomalies caught | Point F1 | Notes                          |")
        print("|--------------------|------------------|----------|--------------------------------|")
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
