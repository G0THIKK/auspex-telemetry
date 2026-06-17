"""
The explainer — Auspex's reason for existing.

A detector tells you *where*. This tells you *what happened*, in language an
operator could read. For each detected window it:
  - identifies the channel and its subsystem type,
  - classifies the SHAPE (spike / dropout / level-shift / drift),
  - measures severity and deviation from the local baseline,
  - emits a structured record AND a plain-English sentence.

The shape rules below are deliberate heuristics — tuning them is your call.
`to_natural_language` is template-based by default; an optional LLM hook is
marked so you can wire in your own model for richer phrasing.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np

from .data import channel_type
from .evaluate import score as _score


@dataclass
class Explanation:
    channel: str
    channel_type: str
    start: int
    end: int
    length: int
    shape: str
    severity: float          # peak score within the window (z-like)
    direction: str           # "rise", "drop", or "mixed"
    local_baseline: float
    window_mean: float
    text: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _classify_shape(values: np.ndarray, start: int, end: int, context: int = 150) -> tuple[str, str, float, float]:
    """Return (shape, direction, local_baseline, window_mean)."""
    n = len(values)
    pre = values[max(0, start - context):start]
    post = values[end:min(n, end + context)]
    seg = values[start:end]

    baseline = float(np.median(pre)) if len(pre) else float(np.median(values))
    win_mean = float(np.mean(seg)) if len(seg) else baseline
    floor, ceil = float(np.min(values)), float(np.max(values))
    span = (ceil - floor) or 1.0
    length = end - start

    direction = "rise" if win_mean > baseline else "drop"

    # dropout: the signal FELL to near the global floor from a clearly higher
    # baseline. Near-floor alone isn't enough — a channel already resting at its
    # floor that lifts off is a rise/drift, not a dropout.
    near_floor = (win_mean - floor) / span < 0.1
    fell_from_above = (baseline - floor) / span > 0.15
    if near_floor and fell_from_above:
        return "dropout", "drop", baseline, win_mean

    # level shift: sustained change that persists after the window
    if len(post):
        post_mean = float(np.mean(post))
        if abs(post_mean - baseline) > 0.25 * span and abs(post_mean - win_mean) < 0.15 * span:
            return "level_shift", direction, baseline, win_mean

    # drift: strong monotonic trend across the segment
    if length >= 10:
        x = np.arange(length)
        slope = np.polyfit(x, seg, 1)[0]
        if abs(slope) * length > 0.2 * span:
            return "drift", ("rise" if slope > 0 else "drop"), baseline, win_mean

    # default: a (relatively) short, sharp excursion
    return "spike", direction, baseline, win_mean


# direction -> human adjective; a "mixed" spike reads best with no word at all
_DIR_WORD = {"rise": "upward", "drop": "downward", "mixed": ""}

_TEMPLATES = {
    "spike": "a {brevity} {dir} {kind}",
    "dropout": "a dropout — the signal fell from its baseline to near its floor",
    "level_shift": "a sustained {dir} shift in level that persisted afterward",
    "drift": "a gradual {dir} drift across the interval",
}


def to_natural_language(e: Explanation) -> str:
    """Template phrasing. (LLM HOOK: replace/augment this with a model call.)"""
    # the spike bucket is the catch-all, so let length pick honest wording:
    # a short sharp "spike" vs. a longer "excursion" (never a 116-sample "brief spike")
    brevity, kind = ("brief", "spike") if e.length <= 15 else ("prolonged", "excursion")
    shape_phrase = _TEMPLATES[e.shape].format(
        dir=_DIR_WORD.get(e.direction, ""), brevity=brevity, kind=kind
    )
    shape_phrase = " ".join(shape_phrase.split())   # tidy double spaces when dir is empty
    sev = "high" if e.severity >= 8 else "moderate" if e.severity >= 5 else "low"
    # the dataset is anonymized; only name a subsystem when we actually know it
    chan = f"Channel {e.channel}"
    if e.channel_type != "unknown":
        chan += f" ({e.channel_type})"
    return (
        f"{chan} shows {shape_phrase} "
        f"between samples {e.start}\u2013{e.end} ({e.length} pts). "
        f"Severity {sev} (peak {e.severity:.1f}\u03c3); window mean "
        f"{e.window_mean:+.2f} vs local baseline {e.local_baseline:+.2f}."
    )

    # --- Optional richer phrasing -------------------------------------------
    # from your_llm import complete
    # prompt = f"Summarize this spacecraft telemetry anomaly for an operator: {e.as_dict()}"
    # return complete(prompt)


def explain(channel: str, values: np.ndarray, detection) -> list[Explanation]:
    out: list[Explanation] = []
    for (start, end) in detection.windows:
        shape, direction, baseline, win_mean = _classify_shape(values, start, end)
        severity = float(np.max(np.abs(detection.score[start:end]))) if end > start else 0.0
        e = Explanation(
            channel=channel,
            channel_type=channel_type(channel),
            start=int(start),
            end=int(end),
            length=int(end - start),
            shape=shape,
            severity=severity,
            direction=direction,
            local_baseline=round(baseline, 3),
            window_mean=round(win_mean, 3),
        )
        e.text = to_natural_language(e)
        out.append(e)
    return out


# --------------------------------------------------------------------------- #
# Plain-English synopsis — the "explain it to a 12-year-old" view
# --------------------------------------------------------------------------- #
_KID_SHAPE = {
    "spike": "had a sudden jump and snapped right back — like a hiccup",
    "dropout": "dropped to its lowest possible value and went quiet, like something switched off",
    "level_shift": "jumped to a new level and just stayed there — a new normal",
    "drift": "slowly slid away from where it should be, little by little",
}

_KID_MEANING = {
    "spike": "Quick hiccups like this are usually a momentary glitch, not something breaking.",
    "dropout": "A signal going quiet like this can mean a sensor or system briefly switched off.",
    "level_shift": "When a reading moves to a new level and stays there, something may have actually changed.",
    "drift": "A slow slide can be an early warning that something is gradually wearing out.",
}


def _kid_shape(e: Explanation) -> str:
    # a long "spike" is really a wander-and-return, so phrase it honestly
    if e.shape == "spike" and e.length > 15:
        return "wandered away from normal for a while, then came back"
    return _KID_SHAPE.get(e.shape, "did something unusual")


def _kid_meaning(e: Explanation) -> str:
    if e.shape == "spike" and e.length > 15:
        return ("A reading that drifts off and then returns is often a passing disturbance "
                "— worth watching if it keeps happening.")
    return _KID_MEANING.get(e.shape, "It's worth a closer look by an engineer.")


def summarize(channel: str, values: np.ndarray, detection, truth=None) -> str:
    """A short, plain-English writeup of a channel's results — aimed at a curious
    12-year-old, not an operator. Returns Markdown."""
    exps = explain(channel, values, detection)
    n = len(values)
    ctype = channel_type(channel)
    sub = (f"the **{ctype}** system" if ctype != "unknown"
           else "a system we can't name (the data is anonymized)")

    parts = [
        "#### In plain English",
        (f"This is one stream of measurements from a spacecraft — think of it as a single "
         f"gauge on a giant dashboard. From its name (`{channel}`) we can tell it belongs to "
         f"{sub}. It sent **{n:,}** readings in a row."),
    ]

    if not exps:
        parts.append("**Auspex didn't find anything odd.** The whole stream looked normal "
                     "from start to finish. 👍")
        return "\n\n".join(parts)

    worst = max(exps, key=lambda e: e.severity)
    by_sev = sorted(exps, key=lambda e: -e.severity)

    parts.append(f"**What happened:** Auspex spotted **{len(exps)}** moment(s) that didn't fit "
                 "the usual pattern. The standouts:")
    parts.append("\n".join(
        f"- Around reading **{e.start:,}**, the signal {_kid_shape(e)}." for e in by_sev[:4]
    ))

    times = max(1, round(worst.severity))
    parts.append(f"**The biggest one** was near reading **{worst.start:,}** — about **{times}×** "
                 "further off than this channel's normal little wiggles, so it really stood out.")

    parts.append("**What it might mean:** " + _kid_meaning(worst)
                 + " On a real mission, an operator would decide whether to investigate.")

    if truth:
        s = _score(n, detection.windows, truth)
        line = (f"**Did it get them right?** People marked **{s.true_windows}** real problem(s) "
                f"in this channel by hand, and Auspex caught **{s.detected_true}** of them")
        line += (f" — though it also raised **{s.false_alarms}** false alarm(s) along the way."
                 if s.false_alarms else ".")
        parts.append(line)
    else:
        parts.append("**Note:** there's no answer key for this channel, so this is Auspex "
                     "finding things entirely on its own.")

    return "\n\n".join(parts)
