# Auspex

**Interpretable anomaly detection and explanation for spacecraft telemetry.**

A detector tells you *where* something went wrong. Auspex also tells you *what*
happened, in language an operator could read — it flags anomalies in a telemetry
channel and emits a plain-English summary of each one (shape, severity, and how
the signal deviated from its baseline).

It started as a question: after building an AI log-analysis tool for software
incidents, would the same *detect-and-explain* approach work on real spacecraft
telemetry? This is the answer.

> **It now watches a real spacecraft, live.** `python iss.py` — or the dashboard's
> *Live ISS feed* — streams NASA's public International Space Station telemetry
> (no API key, no hardware), runs the same streaming detector on it, and explains
> each channel and any anomaly in plain English.

---

## Data

Auspex runs against NASA's public **SMAP / MSL** telemetry set — the de-facto
benchmark for spacecraft anomaly detection — consisting of real telemetry and
labeled anomalies from the Soil Moisture Active Passive satellite and the
Curiosity rover (Mars Science Laboratory). The data is anonymized: timestamps
are removed, values are pre-scaled to [-1, 1], and channel IDs are masked (the
first letter indicates subsystem type, e.g. `P` = power, `R` = radiation). It is
therefore a **methods demonstration**, not real mission insight.

> Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T.
> (2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
> Thresholding.* KDD 2018. The original LSTM baseline reaches ~0.85 F1 on this set.

**Get the data** (Kaggle mirror referenced by the original repo):

```bash
pip install kaggle   # requires a Kaggle API token
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
unzip nasa-anomaly-detection-dataset-smap-msl.zip -d data
# expected layout: data/train/<CHANNEL>.npy, data/test/<CHANNEL>.npy,
#                  data/labeled_anomalies.csv
```

You do **not** need the data to try Auspex — see the synthetic mode below.

---

## Approach

Deliberately **not** an LSTM. The point was to see how far a transparent,
laptop-friendly method gets before reaching for a black box.

1. **Forecast** each value with a trailing rolling mean.
2. **Residual** = |actual − forecast|, EWMA-smoothed.
3. **Dynamic threshold** = rolling mean + *z*·std of the smoothed residual.
4. **Merge** over-threshold points into windows.
5. **Explain** each window: classify its shape (spike / dropout / level-shift /
   drift), measure severity and deviation, and render a sentence.

The LSTM benchmark stays as the comparison point and future work.

---

## Run it

```bash
pip install -r requirements.txt

# 1. End-to-end on synthetic data (no download needed):
python run.py --synthetic

# 2. On a real channel once data is in ./data:
python run.py --channel T-1

# 3. Live monitor — replays a recorded channel as a real-time stream, alerting
#    as each anomaly closes (true streaming detector, one reading at a time):
python monitor.py --channel A-1
python monitor.py --channel A-1 --delay 0.01   # slow it down to watch it scroll

# 4. LIVE telemetry from the actual ISS — no API key, no hardware:
python iss.py                  # watch real ISS channels stream in, narrated
python iss.py --list           # the channels it watches

# 5. Interactive dashboard:
streamlit run app.py
```

> **`iss.py` is the real thing.** It connects to NASA's public International
> Space Station telemetry (Lightstreamer's `ISSLIVE` feed) and runs the same
> streaming detector on live channels — attitude, solar beta angle, gyro
> temperatures, array voltages — narrating anything unusual in plain English.
> Values arrive straight off the station via TDRS relay; when the ISS is between
> relays (Loss of Signal) updates pause, and the monitor says so. A healthy
> station is mostly quiet, so alerts are rare *by design* — the point is a real
> spacecraft, watched and explained as it streams. The same `OnlineDetector`
> powers both the file replay and the live feed; only the data source differs
> (`auspex/sources.py`).

---

## Results

Window-level detection (did we catch the anomaly an operator cares about) is the
headline metric; point-wise P/R/F1 is reported alongside.

| Dataset            | Anomalies caught | Point F1 | Notes                              |
|--------------------|------------------|----------|------------------------------------|
| Synthetic (4 inj.) | 3 / 4            | 0.20     | demo; 2 warm-up false alarms       |
| SMAP (real)        | 59 / 69          | 0.09     | 55 channels; 1007 false alarms     |
| MSL (real)         | 31 / 36          | 0.18     | 27 channels; 129 false alarms      |

Both real splits use default knobs (`forecast_window=150`, `z=2.5`); reproduce
with `python bench.py`. Window-catch holds at ~86% on each spacecraft, but the
high false-alarm count and low point-wise F1 are the trailing-mean baseline's
known failure mode — see Limitations.

_(Add a screenshot of the dashboard here.)_

---

## Limitations & next steps

- The trailing-mean residual flags **transitions**, so it catches the *onset* of
  sustained dropouts/level-shifts but under-covers their bodies — which is why
  point-wise recall is low. Comparing against a longer seasonal/global baseline
  is the obvious next iteration.
- Warm-up false alarms before the threshold stats stabilize; a burn-in period
  would clean these up.
- The explainer's shape rules are heuristics. A natural next step is wiring the
  optional LLM hook in `explainer.to_natural_language` for richer phrasing.

---

## License

MIT © 2026 Curtis Lord. Telemetry data © NASA/JPL-Caltech, used per its public
release terms.
