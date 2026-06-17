"""
Telemetry sources for Auspex.

A source produces a stream of (channel, value) readings that the streaming
detector can consume. This keeps Auspex decoder-agnostic: a recorded file, a
live API, or anything that emits numbers all look the same downstream.

  - ISSLiveSource: NASA's public, real-time ISS telemetry via Lightstreamer's
    ISSLIVE feed (no API key, no hardware). Values come straight off the station
    through TDRS relay; when the ISS is between relays (Loss of Signal) updates
    simply pause.

Channel names below are from the public SatNOGS/ISS telemetry dictionaries.
"""

from __future__ import annotations
import queue

from .lightstreamer import LSClient, Subscription

# Curated, well-documented ISS telemetry channels (PUI -> friendly name).
# Chosen to span subsystems and to actually vary over an orbit.
ISS_CHANNELS = {
    "USLAB000018": "Attitude Quaternion Q0",
    "USLAB000019": "Attitude Quaternion Q1",
    "USLAB000020": "Attitude Quaternion Q2",
    "USLAB000021": "Attitude Quaternion Q3",
    "USLAB000040": "Solar Beta Angle (deg)",
    "USLAB000045": "CMG-1 bearing temperature (C)",
    "USLAB000046": "CMG-2 bearing temperature (C)",
    "USLAB000047": "CMG-3 bearing temperature (C)",
    "S0000004":    "SARJ port joint angle (deg)",
    "S4000001":    "Solar array 1A drive voltage",
    "P4000001":    "Solar array 2A drive voltage",
    "NODE3000005": "Urine tank quantity (%)",
}


# Plain-English, accessible explanations for each channel: what it is, a
# friendly analogy, and why an operator cares (so an anomaly has meaning).
ISS_CHANNEL_INFO = {
    "USLAB000018": {
        "unit": "",
        "what": "One of four numbers (a 'quaternion') that together say which way "
                "the station is pointing in space. Q0 is the 'scalar' part.",
        "plain": "You can't read a direction from one of these alone — the four work "
                 "as a team, like the math version of a compass plus a tilt sensor. "
                 "When the station holds its normal orientation, Q0 sits near 1 and "
                 "the other three near 0 (which is just what you see).",
        "matters": "Attitude control keeps the solar wings facing the Sun, antennas "
                   "facing Earth, and dockings lined up. A sudden jump usually means a "
                   "planned maneuver, a thruster firing, or a control glitch.",
    },
    "USLAB000019": {"unit": "", "what": "The Q1 'vector' part of the station's orientation quaternion — roughly its rotation about one axis.",
        "plain": "Part of the four-number team describing which way the station points. It hovers near 0 while the station holds a steady attitude.",
        "matters": "Drifts in the vector parts (Q1–Q3) show the station rotating. Big moves mean a maneuver or a disturbance worth checking."},
    "USLAB000020": {"unit": "", "what": "The Q2 'vector' part of the orientation quaternion — rotation about a second axis.",
        "plain": "One of the four numbers describing the station's pointing direction; near 0 when holding steady.",
        "matters": "Watched together with Q0/Q1/Q3 to see how the station is turning."},
    "USLAB000021": {"unit": "", "what": "The Q3 'vector' part of the orientation quaternion — rotation about a third axis.",
        "plain": "The last of the four pointing numbers; near 0 in the normal attitude.",
        "matters": "Together the four quaternion numbers fully describe the station's orientation."},
    "USLAB000040": {
        "unit": "degrees",
        "what": "The angle between the Sun and the plane of the station's orbit.",
        "plain": "It tells you how 'sideways' sunlight hits the orbit. Near 0° the station "
                 "dips through Earth's shadow every lap (lots of day/night); near ±75° the "
                 "orbit is tilted so the station sits in almost constant sunshine. It drifts "
                 "slowly over weeks.",
        "matters": "Drives how much solar power is available and how hot or cold things get — "
                   "operators plan power and thermal around it.",
    },
    "USLAB000045": {"unit": "°C", "what": "Temperature of the spin bearing inside Control Moment Gyroscope #1.",
        "plain": "CMGs are big spinning wheels (~100 kg, ~6,600 rpm) that steer the station without burning fuel — like a skater pulling in their arms. The bearings warm up as they spin; this is how warm #1 is.",
        "matters": "A bearing trending hot is an early sign of wear. The ISS has lost CMGs before, so this temperature is watched closely."},
    "USLAB000046": {"unit": "°C", "what": "Spin-bearing temperature of Control Moment Gyroscope #2.",
        "plain": "Same idea as CMG-1 — one of the fuel-free spinning wheels that aim the station. This is its bearing temperature.",
        "matters": "Rising temperature can foreshadow a bearing problem; catching the trend early matters."},
    "USLAB000047": {"unit": "°C", "what": "Spin-bearing temperature of Control Moment Gyroscope #3.",
        "plain": "The third steering gyro's bearing temperature. The station normally runs several CMGs together to point itself.",
        "matters": "Together the CMG temperatures show the health of the station's fuel-free steering."},
    "S0000004": {"unit": "degrees", "what": "The rotation angle of the port (left) Solar Alpha Rotary Joint.",
        "plain": "A giant joint that slowly spins the left solar wings to track the Sun as the station orbits — like a sunflower turning to follow the light. This number is where it's currently pointed.",
        "matters": "If it stops tracking or grinds, solar power drops. The port joint famously had a contamination problem back in 2007."},
    "S4000001": {"unit": "volts", "what": "The drive voltage coming off solar array 1A.",
        "plain": "How much electrical 'push' that solar wing is producing right now. It rises in sunlight and falls toward zero when the station passes into Earth's shadow.",
        "matters": "The station runs on this power. A wing under-producing while in full sun would be a red flag."},
    "P4000001": {"unit": "volts", "what": "The drive voltage coming off solar array 2A.",
        "plain": "Like array 1A — the electrical output of another solar wing, swinging up in sunlight and down in eclipse.",
        "matters": "Part of the power picture; a sudden drop in daylight would stand out."},
    "NODE3000005": {"unit": "%", "what": "How full the urine collection tank in Node 3 is.",
        "plain": "Yes, really: the station recycles urine into drinking water, and this is the tank level before processing. It fills up, then gets pumped down when the recycler runs.",
        "matters": "It has to be processed before it overflows. A level that's stuck could mean the water-recycling system isn't keeping up."},
}


class ISSLiveSource:
    SERVER = "http://push.lightstreamer.com"
    ADAPTER_SET = "ISSLIVE"

    def __init__(self, items=None):
        self.items = list(items or ISS_CHANNELS.keys())
        self._q: queue.Queue = queue.Queue()
        self._client: LSClient | None = None

    def label(self, pui: str) -> str:
        return ISS_CHANNELS.get(pui, pui)

    def start(self) -> None:
        self._client = LSClient(self.SERVER, self.ADAPTER_SET)
        self._client.connect()
        sub = Subscription("MERGE", self.items, ["Value", "TimeStamp"])
        sub.addlistener(self._on_update)
        self._client.subscribe(sub)

    def _on_update(self, info) -> None:
        raw = info["values"].get("Value")
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return                       # non-numeric status value — skip
        self._q.put((info["name"], val))

    def stream(self, stop=None):
        """Yield (pui, value) as updates arrive, or None on an idle tick so the
        consumer can run a heartbeat. Stop via the optional threading.Event."""
        while stop is None or not stop.is_set():
            try:
                yield self._q.get(timeout=0.5)
            except queue.Empty:
                yield None

    def drain(self):
        """Return all currently-queued (pui, value) updates without blocking.
        Used by polling consumers like the Streamlit dashboard."""
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self) -> None:
        if self._client:
            self._client.disconnect()
