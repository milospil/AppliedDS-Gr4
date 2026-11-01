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
from src.cfg import DROP, STD_COLS, COUNT_COLS, SHORT, HERE, DATASET, PROCESSED, DB_PATH, load_extensions, PLOTS_DIR

"""
Speed over time with docking / fouling-growth zones and numbered trips.

- Shaded band below 4 kn  -> docking
- Shaded band below 3 kn  -> fouling growth zone (overlaid inside the docking band)
- Vertical lines at the start (green) and end (grey dotted) of each trip
  between dockings, labelled Trip 1, Trip 2, ...

Edit the CONFIG block for your file, column names and thresholds.
"""
# -------------------------------------------------------------------------------------------------------------------------------------------------#
# Configuration
# -------------------------------------------------------------------------------------------------------------------------------------------------# 
TIME = "timestamp"          # timestamp column
SOG = "sog_kn"              # speed over ground (GPS)
STW = "stw_kn"              # speed through water (speed log)
RESAMPLE = "10m"
DOCK_KN = 4.0               # below this = docking
FOULING_KN = 3.0            # below this = fouling growth zone
MIN_DOCK_H = 24             # slow periods shorter than this (hours) are NOT dockings
MIN_TRIP_H = 48              # sailing periods shorter than this (hours) are NOT trips
FFILL_LIMIT = 6             # bridge up to 6 missing windows (1 h) for state detection
MEDIAN_WIN = "30m"          # rolling median for docking detection only
MAX_KN =16              # above vessel's max service speed = invalid
ZERO_KN = 0.1               # |speed| below this counts as "0"
PLOT_STW = True             # also draw STW as a thin grey line

 
LABELS = ["Both 0", "Both moving", "SOG 0, STW not 0", "STW 0, SOG not 0","SOG only", "STW only", "No speed"]

# -------------------------------------------------------------------------------------------------------------------------------------------------#
# LOAD DATA
# -------------------------------------------------------------------------------------------------------------------------------------------------# 
conn = db.connect(database=DB_PATH)
metrics_wide = conn.execute("SELECT * FROM metrics_wide").pl()
conn.close()

trips_speeds = metrics_wide.select(pl.col(TIME),pl.col("stw_kn"),pl.col("sog_kn")).drop_nulls().sort(TIME).group_by_dynamic(TIME, every=RESAMPLE).agg(pl.col("stw_kn").mean(),pl.col("sog_kn").mean())
trips_sog = metrics_wide.select(pl.col(TIME),pl.col("sog_kn")).drop_nulls().sort(TIME).group_by_dynamic(TIME, every=RESAMPLE).agg(pl.col("sog_kn").mean())
trips_stw = metrics_wide.select(pl.col(TIME),pl.col("stw_kn")).drop_nulls().sort(TIME).group_by_dynamic(TIME, every=RESAMPLE).agg(pl.col("stw_kn").mean())
# -------------------------------------------------------------------------------------------------------------------------------------------------#
#Docked / Sailing State Detection
# -------------------------------------------------------------------------------------------------------------------------------------------------#

def add_runs(df: pl.DataFrame, flag: str) -> pl.DataFrame:
    """Label consecutive runs of a boolean column and attach each run's duration."""
    return (df
            .with_columns(run=(pl.col(flag) != pl.col(flag).shift()).cum_sum().fill_null(0))
            .with_columns(dur=(pl.col(TIME).max() - pl.col(TIME).min()).over("run")))

def categorize_speeds(df: pl.DataFrame) -> pl.DataFrame:
    """Categorize speed data into 7 categories based on SOG and STW values."""
    sog0 = pl.col(SOG).abs() < ZERO_KN
    stw0 = pl.col(STW).abs() < ZERO_KN
    has_sog, has_stw = pl.col(SOG).is_not_null(), pl.col(STW).is_not_null()

    df = df.with_columns(cat=(
        pl.when(has_sog & has_stw & sog0 & stw0).then(pl.lit("Both 0"))
        .when(has_sog & has_stw & ~sog0 & ~stw0).then(pl.lit("Both moving"))
        .when(has_sog & has_stw & sog0).then(pl.lit("SOG 0, STW not 0"))
        .when(has_sog & has_stw & stw0).then(pl.lit("STW 0, SOG not 0"))
        .when(has_sog).then(pl.lit("SOG only"))
        .when(has_stw).then(pl.lit("STW only"))
        .otherwise(pl.lit("No speed"))
    ).cast(pl.Enum(LABELS)))
    df = df.with_columns(pl.col("stw_kn").rolling_median_by(TIME, MEDIAN_WIN).alias("stw_kn_romed"),
                         pl.col("sog_kn").rolling_median_by(TIME, MEDIAN_WIN).alias("sog_kn_romed"))
    return df


def states(df: pl.DataFrame, SPEED: str = "stw_kn"):
    """Detect docked vs sailing states based on speed thresholds and durations."""
    state = (df.with_columns(docked=pl.col("sog_kn_romed").forward_fill(limit=FFILL_LIMIT) < DOCK_KN).drop_nulls("docked"))
    #Brief slow-downs (shorter than MIN_DOCK_H) are treated as sailing
    state = add_runs(state, "docked").with_columns(docked=pl.col("docked") & (pl.col("dur") >= pl.duration(hours=MIN_DOCK_H)))

    #Re-label runs; trips = sailing runs that are long enough
    state = add_runs(state, "docked")
    trips = (state.filter(~pl.col("docked") & (pl.col("dur") >= pl.duration(hours=MIN_TRIP_H)))
                .group_by("run", maintain_order=True)
                .agg(start=pl.col(TIME).first(), end=pl.col(TIME).last())
                .with_row_index("trip", offset=1)
                .with_columns(duration=pl.col("end") - pl.col("start")))
    trips = trips.select("trip", "start", "end", "duration")
    trips_db = trips.with_columns(duration_h=(pl.col("duration").dt.total_seconds() / 3600)).drop("duration")
    conn = db.connect(database=DB_PATH)
    conn.execute(f"CREATE OR REPLACE TABLE trips AS SELECT * FROM trips_db")
    conn.execute(f"CREATE OR REPLACE TABLE trip_states AS SELECT * FROM state")
    conn.close()
    return state, trips

# -------------------------------------------------------------------------------------------------------------------------------------------------#
#plotting
# -------------------------------------------------------------------------------------------------------------------------------------------------#
def plot_speed_trips(df: pl.DataFrame, trips: pl.DataFrame, DOCK_KN: float, FOULING_KN: float, PLOTS_DIR: str, ):
    """Plot speed over time with docking / fouling-growth zones and numbered trips."""
    
    plt.rcParams['axes.grid'] = True          # Enable the grid by default
    plt.rcParams['grid.color'] = '#d3d3d3'  # soft light gray
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhspan(0, DOCK_KN, color="tab:orange", alpha=0.15, label=f"Non Trips (< {DOCK_KN:g} kn)")
    ax.axhspan(0, FOULING_KN, color="tab:red", alpha=0.18,
            label=f"Fouling growth zone (< {FOULING_KN:g} kn)")
    ax.axhline(DOCK_KN, color="tab:orange", ls="--", lw=1)
    ax.axhline(FOULING_KN, color="tab:red", ls="--", lw=1)

    ax.plot(df[TIME], df["stw_kn_romed"], lw=0.7, color="tab:blue", label="STW")
    ax.plot(df[TIME], df["sog_kn_romed"], lw=0.7, color="tab:blue", label="SOG", alpha=0.5)
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
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=6, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/trips_made.png", dpi=150)

if __name__ == "__main__":
    df = categorize_speeds(trips_speeds)
    state, trips = states(df, SPEED="stw_kn_romed")
    plot_speed_trips(state, trips, DOCK_KN, FOULING_KN, PLOTS_DIR)