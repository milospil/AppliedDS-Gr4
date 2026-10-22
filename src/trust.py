"""Shared helpers for the sensor-trust notebooks (notebooks/sensor_trust/).

Every notebook loads data, labels operating modes and records findings the same way.
Needs data/processed/raw_long/ from src/load_raw.py.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "..", "data", "processed")
RAW = os.path.join(HERE, "..", "data", "raw")
CACHE = os.path.join(PROCESSED, "trust_cache")

# Physical limits (see notebooks/sensor_trust/physical_limits.md): (check above, hard limit above)
LIMITS = {
    "shaft_rpm": (105, 110),
    "me_rpm": (105, 110),
    "shaft_kw": (8075, 10000),
    "shaft_torque_nm": (864, 950),
    "fo_kgh": (1750, 2000),
    "me_load_pct": (100, 110),
    "stw_kn": (16, 18),
    "sog_kn": (18, 21),
}
LIMITS_LOW = {"shaft_rpm": (-90, -110), "me_rpm": (-90, -110),   # (check below, hard limit below): astern
              "shaft_kw": (-8075, -10000), "shaft_torque_nm": (-864, -950)}
SMCR_KW, SMCR_RPM = 9500, 105
PITCH_M = 6.8 * 0.6742
KN = 1.94384


# ---------------------------------------------------------------- loading
def load_wide(signals, name=None):
    """15 s table: one row per moment, one column per signal.

    Timestamps are rounded to the nearest 15 s (median offset 0.01 s). If two readings of the same
    signal land on the same moment (0.01 % of readings), their mean is used. Nothing else is changed.
    With `name`, the result is cached in data/processed/trust_cache/<name>.parquet.
    """
    path = os.path.join(CACHE, f"{name}.parquet") if name else None
    if path and os.path.exists(path):
        w = pd.read_parquet(path)
        if set(signals) <= set(w.columns):
            return w[signals]
    raw = pd.read_parquet(os.path.join(PROCESSED, "raw_long"), columns=["time", "signal", "value"],
                          filters=[("signal", "in", list(signals))])
    raw["slot"] = raw.time.dt.round("15s").dt.tz_convert(None)
    w = raw.groupby(["slot", "signal"], observed=True).value.mean().unstack()
    w = w.reindex(columns=list(signals))
    w.index.name = "time"
    w.columns.name = None
    if path:
        os.makedirs(CACHE, exist_ok=True)
        w.to_parquet(path)
    return w


def load_noon():
    """Daily noon reports, indexed by date."""
    import glob
    files = sorted(glob.glob(os.path.join(RAW, "Noon Reports", "*.csv")))
    n = pd.concat([pd.read_csv(f) for f in files])
    n["date"] = pd.to_datetime(n["Date"], format="%d/%m/%Y")
    return n.set_index("date").sort_index()


# ---------------------------------------------------------------- operating mode
def operating_mode(w, stop_sog=0.5, stop_min_h=2, port_heading_deg=5):
    """Label every moment: 'port', 'port, propeller turning', 'anchor', 'sailing', or 'unknown' (no SOG).

    Stop = SOG below `stop_sog` kn for at least `stop_min_h` hours. Port if the heading spread
    (circular std) during the stop stays below `port_heading_deg`, otherwise anchor (it swings).
    Needs columns sog_kn, heading_deg and shaft_rpm (me_rpm is used where shaft_rpm is missing).
    """
    sog = w.sog_kn.dropna()
    slow = sog < stop_sog
    run = (~slow | (sog.index.to_series().diff() > pd.Timedelta("1min"))).cumsum()
    t = sog.index.to_series()[slow]
    stops = t.groupby(run[slow]).agg(["min", "max"]).rename(columns={"min": "start", "max": "end"})
    stops["hours"] = (stops.end - stops.start).dt.total_seconds() / 3600 + 15 / 3600
    stops = stops[stops.hours >= stop_min_h]

    def spread(a, b):
        h = np.deg2rad(w.heading_deg.loc[a:b].dropna())
        if h.empty:
            return np.nan
        r = np.hypot(np.cos(h).mean(), np.sin(h).mean())
        return np.degrees(np.sqrt(-2 * np.log(max(r, 1e-12))))

    stops["type"] = ["port" if spread(a, b) < port_heading_deg else "anchor" for a, b in zip(stops.start, stops.end)]
    rpm = w.shaft_rpm.fillna(w.me_rpm) if "me_rpm" in w else w.shaft_rpm
    stop_type = run[slow].map(stops.type).reindex(w.index)
    mode = pd.Series("sailing", index=w.index)
    mode[w.sog_kn.isna()] = "unknown"
    mode[stop_type == "anchor"] = "anchor"
    mode[(stop_type == "port") & ~(rpm > 0)] = "port"
    mode[(stop_type == "port") & (rpm > 0)] = "port, propeller turning"
    return mode


# ---------------------------------------------------------------- helpers
def runs(mask):
    """Unbroken runs where `mask` is True (15 s steps; a gap over 1 min ends a run): start, end, hours."""
    m = mask.fillna(False)
    idx = m.index.to_series()
    new = ~m | (idx.diff() > pd.Timedelta("1min"))
    rid = new.cumsum()[m]
    if rid.empty:
        return pd.DataFrame(columns=["start", "end", "hours", "readings"])
    g = idx[m].groupby(rid)
    out = pd.DataFrame({"start": g.min(), "end": g.max(), "readings": g.size()})
    out["hours"] = (out.end - out.start).dt.total_seconds() / 3600 + 15 / 3600
    return out.reset_index(drop=True)


def stuck_runs(s, mask=None, min_hours=0.5):
    """Runs where the value does not change for at least `min_hours`, optionally only where `mask` is True."""
    s = s.dropna()
    if mask is not None:
        s = s[mask.reindex(s.index, fill_value=False)]
    t = s.index.to_series()
    same = s.eq(s.shift()) & (t.diff() <= pd.Timedelta("1min"))
    rid = (~same).cumsum()
    # vectorised: only keep runs with at least 2 readings before grouping
    size = rid.map(rid.value_counts())
    keep = size >= max(2, int(min_hours * 240))
    if not keep.any():
        return pd.DataFrame(columns=["start", "end", "value", "readings", "hours"])
    df = pd.DataFrame({"rid": rid[keep], "t": t[keep], "v": s[keep]})
    out = df.groupby("rid").agg(start=("t", "min"), end=("t", "max"), value=("v", "first"), readings=("v", "size"))
    out["hours"] = (out.end - out.start).dt.total_seconds() / 3600
    return out[out.hours >= min_hours].reset_index(drop=True)


def daily(s, how="median", min_count=20):
    """Daily statistic of a 15 s series, only days with at least `min_count` readings."""
    d = s.dropna()
    g = d.groupby(d.index.normalize())
    out = g.agg(how)
    return out[g.size() >= min_count].asfreq("D")


def find_steps(s, penalty=None, min_size=7):
    """Change points in a daily series (ruptures, Pelt, l2 cost). Returns a table of segments."""
    import ruptures as rpt
    d = s.dropna()
    if len(d) < 2 * min_size:
        return pd.DataFrame()
    x = d.values.reshape(-1, 1)
    if penalty is None:
        penalty = 3 * np.log(len(d)) * np.var(x)        # BIC-like default
    bk = rpt.Pelt(model="l2", min_size=min_size).fit(x).predict(pen=penalty)
    seg, start = [], 0
    for end in bk:
        part = d.iloc[start:end]
        seg.append({"from": part.index[0].date(), "to": part.index[-1].date(), "days": len(part),
                    "median": part.median()})
        start = end
    out = pd.DataFrame(seg)
    out["change_vs_previous_%"] = 100 * out["median"].pct_change()
    return out


# ---------------------------------------------------------------- findings
class Findings:
    """Collects findings in the same columns as docs/data_quality_log.md."""

    COLS = ["signal", "period", "type", "what", "evidence", "affected", "impact", "options"]

    def __init__(self, notebook):
        self.notebook = notebook
        self.rows = []

    def add(self, signal, period, type, what, evidence, affected, impact="?", options="?"):
        self.rows.append(dict(zip(self.COLS, [signal, period, type, what, f"{self.notebook} {evidence}",
                                              affected, impact, options])))

    def table(self):
        return pd.DataFrame(self.rows, columns=self.COLS)

    def markdown_rows(self, first_id, who="mp", date=None):
        """Rows ready to paste into docs/data_quality_log.md."""
        date = date or pd.Timestamp.today().strftime("%d %b")
        lines = []
        for i, r in enumerate(self.rows):
            lines.append(f"| DQ-{first_id + i:02d} | {r['signal']} | {r['period']} | {r['type']} | {r['what']} | "
                         f"{r['evidence']} | {r['affected']} | {r['impact']} | {r['options']} | | open | {who}, {date} |")
        return "\n".join(lines)
