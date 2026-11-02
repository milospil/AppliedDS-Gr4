"""Figure of the two sensor faults that matter most for the fouling result (DQ-02 and DQ-05).

Run from the repo root:   python src/sensor_faults_figure.py
Reads data/processed/model_table.csv, writes plots4report/sensor_faults.png.

Daily medians of steady sailing rows:
  A. torquemeter power / power estimated from scavenge pressure and engine rpm (about 1 while healthy)
  B. speed log minus GPS speed over ground (about +0.3 kn normally; currents add day-to-day scatter)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "..", "data", "processed")
PLOTS = os.path.join(HERE, "..", "plots4report")
MIN_ROWS = 6   # at least one hour of steady sailing per day
INK, MUTED, FAULT = "#1b1b1b", "#7a7a7a", "#c0392b"

t = pd.read_csv(os.path.join(PROCESSED, "model_table.csv"), index_col="time", parse_dates=True)
s = t[t.steady & (t.index < "2025-01-01")].copy()
s["log_minus_gps"] = s.stw_log_kn - s.sog_kn
day = s.groupby(s.index.normalize()).agg(ratio=("torque_ratio", "median"), log=("log_minus_gps", "median"),
                                         n=("steady", "size"))
day = day[day.n >= MIN_ROWS]
tz = day.index.tz

fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
a.plot(day.index, day.ratio, ".", ms=4, color=INK)
a.axhline(1, color=MUTED, lw=0.8)
a.axvline(pd.Timestamp("2024-07-10", tz=tz), color=FAULT, ls="--", lw=1)
a.text(pd.Timestamp("2024-07-14", tz=tz), 1.12, "torquemeter fault from 10 Jul", color=FAULT, fontsize=9)
a.set_ylabel("torquemeter ÷\nestimated power")
a.set_ylim(0.6, 1.2)
a.set_title("A. Torquemeter power against power estimated from scavenge pressure and engine rpm", loc="left", fontsize=10)

b.plot(day.index, day.log, ".", ms=4, color=INK)
b.axhline(0.3, color=MUTED, lw=0.8)
b.axvspan(pd.Timestamp("2024-07-29", tz=tz), pd.Timestamp("2024-10-16", tz=tz), color=FAULT, alpha=0.12, lw=0)
b.text(pd.Timestamp("2024-08-02", tz=tz), 2.1, "speed-log offset 29 Jul – 16 Oct", color=FAULT, fontsize=9)
b.set_ylabel("speed log −\nGPS speed [kn]")
b.set_ylim(-1.5, 2.5)
b.set_title("B. Speed log against GPS speed over ground (grey line: +0.3 kn, the normal difference)", loc="left", fontsize=10)

for ax in (a, b):
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "sensor_faults.png"), dpi=150)

before, after = day.ratio[day.index < pd.Timestamp("2024-07-10", tz=tz)], day.ratio[day.index >= pd.Timestamp("2024-07-10", tz=tz)]
inside = (day.index >= pd.Timestamp("2024-07-29", tz=tz)) & (day.index < pd.Timestamp("2024-10-16", tz=tz))
print(f"days: {len(day)}; torque ratio median before 10 Jul {before.median():.2f}, after {after.median():.2f}")
print(f"speed log - GPS median inside offset window {day.log[inside].median():+.2f} kn, outside {day.log[~inside].median():+.2f} kn")
