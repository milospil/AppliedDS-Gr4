'''
identify_trips.py
-----------------   
Script to segement time series data into different trips based on the trip start and end times. 
The trip start and end times are defined in the trips.csv file. The script will create a new table in the DuckDB database called trips, which will contain the segmented data for each trip.
'''

import duckdb as db
import polars as pl
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.join("..", "..", "src"))
from cfg import DROP, STD_COLS, COUNT_COLS, SHORT, HERE, DATASET, PROCESSED, DB_PATH, load_extensions

"""
Speed over time with docking / fouling-growth zones and numbered trips.

- Shaded band below 4 kn  -> docking
- Shaded band below 3 kn  -> fouling growth zone (overlaid inside the docking band)
- Vertical lines at the start (green) and end (grey dotted) of each trip
  between dockings, labelled Trip 1, Trip 2, ...

Edit the CONFIG block for your file, column names and thresholds.
"""

# ----------------------------------------------------------------- CONFIG
FILE = "vessel_data.parquet"
TIME = "time"
SPEED = "speed_over_ground_kn"   # SOG is more reliable than STW in port
RESAMPLE = "10m"
DOCK_KN = 4.0                    # below this = docking
FOULING_KN = 3.0                 # below this = fouling growth zone
MIN_DOCK_H = 2                   # slow periods shorter than this (hours) are NOT dockings
MIN_TRIP_H = 3                   # sailing periods shorter than this (hours) are NOT trips
FFILL_LIMIT = 6                  # bridge up to 6 missing windows (1 h) for state detection

# ----------------------------------------------------------------- LOAD
conn = db.connect(database=DB_PATH)
metrics_raw_long = conn.execute("SELECT timestamp, short_name, value FROM metrics_raw").pl()
noon_reports = conn.execute("SELECT * FROM noon_reports").pl()
jan25 = conn.execute("SELECT * FROM jan25").pl()
metrics_reg = conn.execute("SELECT * FROM metrics_registration").pl()
metrics_wide = conn.execute("SELECT * FROM metrics_wide").pl()
conn.close()






df = (pl.read_parquet(FILE)          # or pl.read_csv(FILE, try_parse_dates=True)
        .select(pl.col(TIME).cast(pl.Datetime), pl.col(SPEED))
        .sort(TIME)
        .group_by_dynamic(TIME, every=RESAMPLE).agg(pl.col(SPEED).mean())
        .upsample(TIME, every=RESAMPLE))   # gaps become nulls -> line breaks in plot

# ---------------------------------------------------- DOCKED / SAILING STATE
def add_runs(frame: pl.DataFrame, flag: str) -> pl.DataFrame:
    """Label consecutive runs of a boolean column and attach each run's duration."""
    return (frame
            .with_columns(run=(pl.col(flag) != pl.col(flag).shift()).cum_sum().fill_null(0))
            .with_columns(dur=(pl.col(TIME).max() - pl.col(TIME).min()).over("run")))

state = (df.with_columns(docked=pl.col(SPEED).forward_fill(limit=FFILL_LIMIT) < DOCK_KN)
           .drop_nulls("docked"))

# 1) Brief slow-downs (shorter than MIN_DOCK_H) are treated as sailing
state = add_runs(state, "docked").with_columns(
    docked=pl.col("docked") & (pl.col("dur") >= pl.duration(hours=MIN_DOCK_H)))

# 2) Re-label runs; trips = sailing runs that are long enough
state = add_runs(state, "docked")
trips = (state.filter(~pl.col("docked") & (pl.col("dur") >= pl.duration(hours=MIN_TRIP_H)))
              .group_by("run", maintain_order=True)
              .agg(start=pl.col(TIME).first(), end=pl.col(TIME).last())
              .with_row_index("trip", offset=1)
              .with_columns(duration=pl.col("end") - pl.col("start")))
print(trips.select("trip", "start", "end", "duration"))

# ----------------------------------------------------------------- PLOT
fig, ax = plt.subplots(figsize=(14, 5))

ax.axhspan(0, DOCK_KN, color="tab:orange", alpha=0.15, label=f"Docking (< {DOCK_KN:g} kn)")
ax.axhspan(0, FOULING_KN, color="tab:red", alpha=0.18,
           label=f"Fouling growth zone (< {FOULING_KN:g} kn)")
ax.axhline(DOCK_KN, color="tab:orange", ls="--", lw=1)
ax.axhline(FOULING_KN, color="tab:red", ls="--", lw=1)

ax.plot(df[TIME], df[SPEED], lw=0.7, color="tab:blue", label="Speed")

lab_tf = ax.get_xaxis_transform()        # x in data coords, y as fraction of axes height
rotate = 90 if trips.height > 15 else 0  # rotate labels when there are many trips
for trip, start, end in trips.select("trip", "start", "end").iter_rows():
    ax.axvline(start, color="tab:green", lw=1)
    ax.axvline(end, color="0.4", lw=1, ls=":")
    ax.text(start + (end - start) / 2, 0.98, f"Trip {trip}", transform=lab_tf,
            ha="center", va="top", fontsize=8, rotation=rotate)

ax.set_ylim(bottom=0)
ax.set_ylabel("Speed (kn)")
loc = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig("speed_trips.png", dpi=150)
print("Saved speed_trips.png")