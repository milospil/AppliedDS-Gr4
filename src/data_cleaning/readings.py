"""Cleaning of the raw 15 s readings, used by make_10min.py before the 10-minute averages are taken.

Follows the outlier blocks of Dalheim & Steen (2020): obvious outliers (limits) and repeated values. Their spike
filter was tested and left out: no better fit, and virtual power got worse (see the notes in the data-quality log).
Bad readings are removed, not replaced: the 10-minute mean is taken over what is left.
Also holds their steady-state test (t-test on the slope in a moving window).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trust import LIMITS, LIMITS_LOW

STUCK_COLS = ["sog_kn", "stw_kn"]   # DQ-22. Other signals: their long flat runs are the torquemeter dropouts (DQ-14)
STUCK_MIN_READINGS = 120            # 30 minutes
STUCK_ABOVE_KN = 3                  # a flat 0 in port is real
RPM_WINDOW, HEADING_WINDOW = 20, 80  # readings: 5 and 20 minutes (Dalheim & Steen)
ALPHA = 0.01


def bad_readings(w):
    """w: 15 s grid, one column per signal. Returns a boolean table (limit or stuck) and the counts."""
    bad = pd.DataFrame(False, index=w.index, columns=w.columns)
    count = {"limit": 0, "stuck": 0}
    for c in w.columns:
        s = w[c]
        hi, lo = LIMITS.get(c, (None, np.inf))[1], LIMITS_LOW.get(c, (None, -np.inf))[1]
        limit = (s > hi) | (s < lo)
        stuck = pd.Series(False, index=w.index)
        if c in STUCK_COLS:
            same = (s != s.shift()).cumsum()
            stuck = (same.map(same.value_counts()) >= STUCK_MIN_READINGS) & (s > STUCK_ABOVE_KN)
        bad[c] = limit | stuck
        count["limit"] += int(limit.sum())
        count["stuck"] += int(stuck.sum())
    return bad, count


def slope_steady(s, n, alpha=ALPHA):
    """Dalheim & Steen: fit a line in every full window of n readings and t-test the slope.
    A window with no significant slope is accepted, and all its readings count as steady."""
    x = np.arange(n) - (n - 1) / 2
    sxx = (x ** 2).sum()
    v = s.to_numpy(dtype=float)
    ok = ~np.isnan(v)
    v0 = np.where(ok, v, 0.0)
    ones = np.ones(n)
    full = np.convolve(ok.astype(float), ones, "valid") == n
    sy, syy = np.convolve(v0, ones, "valid"), np.convolve(v0 ** 2, ones, "valid")
    b = np.convolve(v0, x[::-1], "valid") / sxx
    sse = np.maximum(syy - sy ** 2 / n - b ** 2 * sxx, 0)
    se = np.sqrt(sse / (n - 2) / sxx)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, np.abs(b) / se, np.where(b != 0, np.inf, 0.0))
    accepted = full & (t < stats.t.ppf(1 - alpha / 2, n - 2))
    return pd.Series(np.convolve(accepted.astype(int), np.ones(n, int), "full") > 0, index=s.index)


def steady_share(w):
    """Share of each 10-minute block that is steady in both engine rpm and heading."""
    if not {"me_rpm", "heading_deg"} <= set(w.columns):
        return None
    h = w.heading_deg.dropna()
    unwrapped = pd.Series(np.degrees(np.unwrap(np.radians(h.to_numpy()))), index=h.index).reindex(w.index)
    steady = slope_steady(w.me_rpm, RPM_WINDOW) & slope_steady(unwrapped, HEADING_WINDOW)
    return steady.astype(float).groupby(w.index.floor("10min")).mean()
