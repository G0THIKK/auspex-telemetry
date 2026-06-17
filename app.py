"""
Auspex dashboard.

    streamlit run app.py

Three views (sidebar):
  - Static analysis: plot a channel, shade true vs. detected anomalies, list the
    plain-English explanations. Optional 12-year-old-friendly synopsis.
  - Simulated livestream: replay a recorded channel reading-by-reading through
    the streaming detector, with Play / Pause / Reset.
  - Live ISS feed: connect to NASA's public ISS telemetry (Lightstreamer
    ISSLIVE — no key, no hardware) and watch the streaming detector run on real,
    live channels.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from auspex.data import (
    list_channels, load_channel, load_labels, make_synthetic_channel,
)
from auspex.detector import detect, DetectorConfig
from auspex.explainer import explain, summarize
from auspex.evaluate import score
from auspex.stream import OnlineDetector
from auspex.sources import ISSLiveSource, ISS_CHANNELS, ISS_CHANNEL_INFO

TRUE_COLOR, DET_COLOR = "#2e9e5b", "#d1495b"

st.set_page_config(page_title="Auspex", layout="wide")
st.title("Auspex — spacecraft telemetry anomaly explainer")

# ----- controls ------------------------------------------------------------ #
channels = list_channels()
with st.sidebar:
    st.header("View")
    view = st.radio("view", ["Static analysis", "Simulated livestream", "Live ISS feed"],
                    label_visibility="collapsed")

    synopsis = False
    if view != "Live ISS feed":
        st.header("Data")
        if channels:
            source = st.radio("Source", ["Synthetic", "Real channel"])
            channel = (st.selectbox("Channel", channels)
                       if source == "Real channel" else "P-SYN")
        else:
            st.caption("No real data found in ./data — using synthetic.")
            source, channel = "Synthetic", "P-SYN"

        st.header("Detector")
        fw = st.slider("Forecast window", 20, 600, DetectorConfig.forecast_window, 10)
        sm = st.slider("Residual smoothing", 1, 50, DetectorConfig.smoothing, 1)
        z = st.slider("Threshold z", 1.5, 6.0, DetectorConfig.z, 0.1)

        if view == "Static analysis":
            synopsis = st.toggle("Plain-English synopsis", value=False,
                                 help="A short, 12-year-old-friendly summary.")
        else:
            speed = st.slider("Readings per frame", 5, 200, 40, 5)
            frame_pause = st.slider("Frame pause (s)", 0.0, 0.30, 0.05, 0.01)
            window = st.slider("Visible window (readings)", 200, 4000, 1500, 100)


# --------------------------------------------------------------------------- #
# Simulated livestream (file replay) — pausable, session-state driven
# --------------------------------------------------------------------------- #
def _stream_sig():
    return (source, channel, fw, sm, z, len(values))


def _reset_stream():
    S = st.session_state
    S.stream_sig = _stream_sig()
    S.stream_det = OnlineDetector(cfg)
    S.stream_pos = 0
    S.stream_seen, S.stream_scores = [], []
    S.stream_events, S.stream_alerts = [], []
    S.stream_playing, S.stream_done = False, False
    S.stream_last_flag, S.stream_last_val = False, 0.0


def run_live() -> None:
    S = st.session_state
    if S.get("stream_sig") != _stream_sig():
        _reset_stream()
    n = len(values)

    c1, c2, c3 = st.columns([1, 1, 2])
    if S.stream_done:
        c1.button("▶ Play", disabled=True, use_container_width=True)
    elif S.stream_playing:
        if c1.button("⏸ Pause", use_container_width=True):
            S.stream_playing = False
            st.rerun()
    else:
        if c1.button("▶ Play", type="primary", use_container_width=True):
            S.stream_playing = True
            st.rerun()
    if c2.button("⏹ Reset", use_container_width=True):
        _reset_stream()
        st.rerun()
    state = ("playing" if S.stream_playing else "done" if S.stream_done
             else "paused" if S.stream_pos else "ready")
    c3.markdown(f"&nbsp;&nbsp;**{state}**", unsafe_allow_html=True)

    def explain_window(win):
        det_obj = SimpleNamespace(windows=[win], score=np.asarray(S.stream_scores))
        return explain(channel, np.asarray(S.stream_seen), det_obj)[0].text

    def render():
        count = min(S.stream_pos, n)
        if count < 1:
            st.info("Press ▶ Play to replay this channel as a live stream.")
            return
        i = count - 1
        lo, hi = max(0, i - window + 1), count
        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.plot(range(lo, hi), S.stream_seen[lo:hi], lw=0.7, color="#222")
        for a, b in truth:
            if b > lo and a < hi:
                ax.axvspan(max(a, lo), min(b, hi), color=TRUE_COLOR, alpha=0.18)
        for a, b in S.stream_events:
            if b > lo and a < hi:
                ax.axvspan(max(a, lo), min(b, hi), color=DET_COLOR, alpha=0.30)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("sample"); ax.set_ylabel("value")
        ax.set_title(f"{channel} — live")
        st.pyplot(fig)
        plt.close(fig)

        dot = "🔴 anomaly" if S.stream_last_flag else "🟢 ok"
        st.markdown(
            f"**{count:,}/{n:,}** readings · **{len(S.stream_events)}** anomalies "
            f"· value `{S.stream_last_val:+.2f}` {dot}"
        )
        st.progress(count / n)
        st.subheader("Live alerts")
        st.markdown("\n".join(S.stream_alerts[-15:]) if S.stream_alerts else "_none yet_")

    tick = frame_pause if frame_pause >= 0.02 else 0.02
    run_every = tick if S.stream_playing else None

    @st.fragment(run_every=run_every)
    def stream_frame():
        if S.stream_playing and S.stream_pos < n:
            end = min(S.stream_pos + speed, n)
            for j in range(S.stream_pos, end):
                v = float(values[j])
                S.stream_seen.append(v)
                flagged, sc, ev = S.stream_det.update(v)
                S.stream_scores.append(sc)
                if ev is not None:
                    S.stream_events.append(ev)
                    S.stream_alerts.append(f"- **[{ev[0]}–{ev[1]}]** {explain_window(ev)}")
                S.stream_last_flag, S.stream_last_val = flagged, v
            S.stream_pos = end
            if S.stream_pos >= n:
                tail = S.stream_det.finalize()
                if tail is not None:
                    S.stream_events.append(tail)
                    S.stream_alerts.append(f"- **[{tail[0]}–{tail[1]}]** {explain_window(tail)}")
                S.stream_playing, S.stream_done = False, True
                st.rerun()
        render()

    stream_frame()


# --------------------------------------------------------------------------- #
# Live ISS feed — real telemetry via Lightstreamer
# --------------------------------------------------------------------------- #
def _fmt(v):
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.3f}"


def run_iss() -> None:
    S = st.session_state
    S.setdefault("iss_source", None)
    S.setdefault("iss_connected", False)
    for k in ("iss_dets", "iss_seen", "iss_scores", "iss_events"):
        S.setdefault(k, {})
    S.setdefault("iss_latest", {})
    S.setdefault("iss_alerts", [])
    S.setdefault("iss_started", 0.0)
    S.setdefault("iss_last_data", 0.0)

    iss_cfg = DetectorConfig(forecast_window=20, smoothing=5, threshold_window=60,
                             z=3.0, min_len=2)

    S.setdefault("iss_selected", list(ISS_CHANNELS)[0])

    c1, c2 = st.columns([1, 2])
    if not S.iss_connected:
        if c1.button("🔌 Connect", type="primary", use_container_width=True):
            try:
                src = ISSLiveSource()
                src.start()
                S.iss_source = src
                S.iss_connected = True
                S.iss_dets, S.iss_seen, S.iss_scores, S.iss_events = {}, {}, {}, {}
                S.iss_latest, S.iss_alerts = {}, []
                S.iss_started = S.iss_last_data = time.time()
                st.rerun()
            except Exception as e:
                st.error(f"Connect failed: {e}")
    else:
        if c1.button("⏹ Disconnect", use_container_width=True):
            try:
                S.iss_source.stop()
            except Exception:
                pass
            S.iss_connected, S.iss_source = False, None
            st.rerun()
    min_sev = c2.slider("Alert threshold (σ)", 0.0, 12.0, 5.0, 0.5)

    if not S.iss_connected:
        st.info("Press **Connect** to stream live telemetry from the International "
                "Space Station. No key, no hardware — straight off the station via TDRS.")
        return

    win = 300

    @st.fragment(run_every=1.0)
    def iss_frame():
        src = S.iss_source
        if src is None:
            return
        for pui, val in src.drain():
            S.iss_last_data = time.time()
            S.iss_latest[pui] = val
            det = S.iss_dets.get(pui)
            if det is None:
                det = OnlineDetector(iss_cfg)
                S.iss_dets[pui] = det
                S.iss_seen[pui], S.iss_scores[pui], S.iss_events[pui] = [], [], []
            S.iss_seen[pui].append(val)
            _, sc, ev = det.update(val)
            S.iss_scores[pui].append(sc)
            if ev is not None:
                a, b = ev
                arr = np.asarray(S.iss_scores[pui])
                sev = float(np.max(np.abs(arr[a:b]))) if b > a else 0.0
                if sev >= min_sev:
                    S.iss_events[pui].append((a, b))
                    det_obj = SimpleNamespace(windows=[ev], score=arr)
                    e = explain(pui, np.asarray(S.iss_seen[pui]), det_obj)[0]
                    S.iss_alerts.append(
                        e.text.replace(pui, ISS_CHANNELS[pui]).replace("samples", "readings"))

        up = int(time.time() - S.iss_started)
        quiet = time.time() - S.iss_last_data
        sig = "🟢 receiving" if quiet < 20 else f"🟠 signal quiet {int(quiet)}s (between relays)"
        st.markdown(f"**Live ISS telemetry** · {len(S.iss_latest)}/{len(ISS_CHANNELS)} channels "
                    f"· up {up}s · {sig} · **{len(S.iss_alerts)}** alerts")
        st.caption("Click any channel below to see what the number means.")

        puis = list(ISS_CHANNELS)
        for r in range(0, len(puis), 4):
            cols = st.columns(4)
            for col, p in zip(cols, puis[r:r + 4]):
                v = S.iss_latest.get(p)
                label = f"{ISS_CHANNELS[p]}\n\n{_fmt(v) if v is not None else '—'}"
                if col.button(label, key=f"sel_{p}", use_container_width=True,
                              type="primary" if p == S.iss_selected else "secondary"):
                    S.iss_selected = p

        sel = S.iss_selected
        info = ISS_CHANNEL_INFO.get(sel, {})
        v = S.iss_latest.get(sel)
        with st.container(border=True):
            now = f" — currently `{_fmt(v)}` {info.get('unit', '')}".rstrip() if v is not None else ""
            st.markdown(f"#### {ISS_CHANNELS[sel]}{now}")
            if info:
                st.markdown(f"**What it is** — {info['what']}")
                st.markdown(f"**In plain words** — {info['plain']}")
                st.markdown(f"**Why it matters** — {info['matters']}")
            else:
                st.caption("No description for this channel yet.")

            seen = S.iss_seen.get(sel, [])
            if len(seen) >= 2:
                lo, hi = max(0, len(seen) - win), len(seen)
                fig, ax = plt.subplots(figsize=(12, 2.6))
                ax.plot(range(lo, hi), seen[lo:hi], lw=0.8, color="#222")
                for a, b in S.iss_events.get(sel, []):
                    if b > lo and a < hi:
                        ax.axvspan(max(a, lo), min(b, hi), color=DET_COLOR, alpha=0.30)
                ax.set_xlabel("reading"); ax.set_ylabel("value")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.caption("waiting for readings…")

        st.subheader("Live alerts")
        if S.iss_alerts:
            st.markdown("\n".join(f"- {a}" for a in S.iss_alerts[-12:]))
        else:
            st.caption(f"none yet — a healthy ISS is mostly quiet (showing ≥ {min_sev:.0f}σ).")

    iss_frame()


# --------------------------------------------------------------------------- #
# Static mode (full analysis at once)
# --------------------------------------------------------------------------- #
def run_static() -> None:
    result = detect(values, cfg)
    explanations = explain(channel, values, result)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(values, lw=0.7, color="#222")
    for a, b in truth:
        ax.axvspan(a, b, color=TRUE_COLOR, alpha=0.18,
                   label="true" if (a, b) == truth[0] else None)
    for a, b in result.windows:
        ax.axvspan(a, b, color=DET_COLOR, alpha=0.30,
                   label="detected" if (a, b) == result.windows[0] else None)
    ax.set_xlabel("sample"); ax.set_ylabel("value"); ax.set_title(channel)
    if truth or result.windows:
        ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Score")
        if truth:
            st.code(score(len(values), result.windows, truth).pretty())
        else:
            st.caption("No labels for this channel — detection only.")
    with c2:
        st.subheader(f"Explanations ({len(explanations)})")
        for e in explanations:
            st.markdown(f"- {e.text}")


# ----- render -------------------------------------------------------------- #
# leaving ISS mode? close the live connection.
if view != "Live ISS feed" and st.session_state.get("iss_source"):
    try:
        st.session_state.iss_source.stop()
    except Exception:
        pass
    st.session_state.iss_source, st.session_state.iss_connected = None, False

if view == "Live ISS feed":
    run_iss()
else:
    if source == "Real channel":
        values = load_channel(channel)
        truth = load_labels().get(channel, [])
    else:
        values, truth = make_synthetic_channel()
    cfg = DetectorConfig(forecast_window=fw, smoothing=sm, z=z)

    if synopsis:
        st.markdown(summarize(channel, values, detect(values, cfg), truth))
        st.divider()

    if view == "Simulated livestream":
        run_live()
    else:
        run_static()
