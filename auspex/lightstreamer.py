"""
Minimal Lightstreamer client (legacy TLCP text protocol), Python 3.

Adapted and trimmed from the Apache-2.0 licensed Lightstreamer examples
(StockList Python client by Lightstreamer Srl, and the ISS variant by
liamkennedy). Kept to exactly what Auspex needs: open a session against an
adapter set, subscribe in MERGE mode, and receive item updates via a callback.
Original copyright (c) Lightstreamer Srl, Apache License 2.0.
"""

from __future__ import annotations
import threading
from urllib.request import urlopen
from urllib.parse import urlparse, urljoin, urlencode

# Standard Lightstreamer demo client id (required by the public server).
_CID = "mgQkwtwdysogQz2BJ4Ji kOj2Bg"


class Subscription:
    def __init__(self, mode, items, fields, adapter=""):
        self.mode = mode
        self.items = list(items)
        self.fields = list(fields)
        self.adapter = adapter
        self._items_map: dict[int, dict] = {}
        self._listeners = []

    def addlistener(self, fn):
        self._listeners.append(fn)

    @staticmethod
    def _decode(value, last):
        # Lightstreamer text protocol: "$" empty, "#" null, "" unchanged.
        if value == "$":
            return ""
        if value == "#":
            return None
        if not value:
            return last
        if value[0] in "#$":
            return value[1:]
        return value

    def notifyupdate(self, item_line):
        toks = item_line.rstrip("\r\n").split("|")
        pos = int(toks[0])
        cur = self._items_map.get(pos, {})
        vals = {f: self._decode(raw, cur.get(f)) for f, raw in zip(self.fields, toks[1:])}
        self._items_map[pos] = vals
        info = {"name": self.items[pos - 1], "values": vals}
        for fn in self._listeners:
            fn(info)


class LSClient:
    def __init__(self, base_url, adapter_set=""):
        self._base = urlparse(base_url)
        self._adapter = adapter_set
        self._session: dict[str, str] = {}
        self._subs: dict[int, Subscription] = {}
        self._key = 0
        self._conn = None
        self._thread = None
        self._control_base = base_url
        self._stop = threading.Event()

    def _post(self, base, path, params):
        url = urljoin(base, path)
        body = urlencode({k: v for k, v in params.items() if v != ""}).encode("utf-8")
        return urlopen(url, data=body, timeout=30)

    def _readline(self):
        return self._conn.readline().decode("utf-8").rstrip()

    def _read_session_header(self):
        while True:
            line = self._readline()
            if not line:
                break
            if ":" in line:
                k, v = line.split(":", 1)
                self._session[k] = v

    def connect(self):
        self._conn = self._post(
            self._base.geturl(), "lightstreamer/create_session.txt",
            {"LS_op2": "create", "LS_cid": _CID, "LS_adapter_set": self._adapter},
        )
        first = self._readline()
        if first != "OK":
            rest = self._conn.read(400).decode(errors="replace")
            raise IOError(f"Lightstreamer create_session failed: {first} {rest}")
        self._read_session_header()
        if self._session.get("ControlAddress"):
            self._control_base = self._base.scheme + "://" + self._session["ControlAddress"]
        self._thread = threading.Thread(target=self._receive, name="LS-rx", daemon=True)
        self._thread.start()

    def _bind(self):
        self._conn = self._post(
            self._control_base, "lightstreamer/bind_session.txt",
            {"LS_session": self._session["SessionId"]},
        )
        first = self._readline()
        if first != "OK":
            raise IOError(f"Lightstreamer bind failed: {first}")
        self._read_session_header()

    def subscribe(self, sub: Subscription):
        self._key += 1
        self._subs[self._key] = sub
        r = self._post(
            self._control_base, "lightstreamer/control.txt",
            {"LS_session": self._session["SessionId"], "LS_Table": str(self._key),
             "LS_op": "add", "LS_data_adapter": sub.adapter, "LS_mode": sub.mode,
             "LS_snapshot": "true", "LS_requested_max_frequency": "1",
             "LS_schema": " ".join(sub.fields), "LS_id": " ".join(sub.items)},
        )
        resp = r.readline().decode("utf-8").rstrip()
        if resp != "OK":
            raise IOError(f"Lightstreamer subscribe failed: {resp}")
        return self._key

    def _receive(self):
        while not self._stop.is_set():
            try:
                msg = self._readline()
            except Exception:
                return
            if not msg or msg == "PROBE" or msg.startswith("Preamble"):
                continue
            if msg == "LOOP":                      # session cycled — rebind and continue
                try:
                    self._bind()
                    continue
                except Exception:
                    return
            if msg.startswith(("ERROR", "END", "SYNC")):
                return
            try:
                tbl, item = msg.split(",", 1)
                sub = self._subs.get(int(tbl))
                if sub:
                    sub.notifyupdate(item)
            except Exception:
                continue

    def disconnect(self):
        self._stop.set()
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
