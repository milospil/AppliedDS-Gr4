'''
time_sync - script to detect jumps in time vector, and sync the timestamps across observations recorded by different sensors

Based on processes described in:
1. Dalheim and Steen (2020), "Preparation of in-service measurement data for ship operation and performance analysis", Ocean Engineering
2. Gupta et al. (2022), "Data Processing Framework for Ship Performance Analysis", arvix
'''

import duckdb as db
import polars as pl
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from ..cfg import DB_PATH, get_metrics_reg, get_metrics_wide, PLOTS_DIR, PROCESSED

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#JUMP DETECTION IN TIME STAMPS FOR DIFFERENT SENSORS
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def detect_jumps(DB_PATH, create_table = False, save_csv = False) -> dict:
    '''
    detect jumps in the time stamps for different sensors
    There are 13 sensors in metrics_registration with recorded time stamps in metrics_wide
    Jumps are detected using standard deviation of First ordered differenced series   
    '''
    metrics_wide = get_metrics_wide(DB_PATH)
    metrics_reg = get_metrics_reg(DB_PATH)
    sensors = metrics_reg.group_by(by="source_name").agg(pl.col("short_name")) # sensors and recorded signals, technically all signals recorded by one sensor should be observed at same time stamp
    timestamp_dict = {}
    os.makedirs(os.path.join(PROCESSED, "sensor_timestamp"), exist_ok=True)
    for row in sensors.iter_rows(named=True):
        df = metrics_wide.select(pl.col(["timestamp"] + row["short_name"]))
        if create_table:
            conn = db.connect(database=DB_PATH)
            conn.execute(f"CREATE TABLE metrics_{row['by'].replace(" ","_").lower()} AS SELECT * FROM df")
            conn.close()
        df = df.drop_nulls().sort("timestamp").with_columns(pl.col("timestamp").diff().dt.total_seconds().alias("time_diff")) 
        df = df.select(pl.col("timestamp", "time_diff"))
        df = df.with_columns((pl.col("time_diff") - pl.col("time_diff").mode()).alias("y_s")) \
                .with_columns((pl.col("y_s").std().alias("sigma_s"))) \
                .with_columns((pl.col("y_s").abs() > pl.col("sigma_s")).alias("jump"))
        print(f"{row['by']} :: No. of rows : {len(df)}, NP STD : {np.std(df["y_s"].drop_nulls().to_numpy())}, Polar STD: {df["sigma_s"].mode().max()}")
        ## mode is assumed to be the real 1/frequency of observation, some of the sensors have high y_s
        if save_csv:
            df.write_csv(os.path.join(PROCESSED, "sensor_timestamp", f"{row['by'].replace(" ","_").lower()}.csv"))
        timestamp_dict[row['by'].replace(" ","_").lower()] = df
    return timestamp_dict

def detect_jumps_excluding_oct(td : dict[str, pl.DataFrame]) -> dict:
    td_wo ={}
    for n, df in td.items():
        df_ = df.filter(pl.col("timestamp") < pl.date(2024,10,1))
        df_ = df_.select(pl.col("timestamp")).with_columns(pl.col("timestamp").diff().dt.total_seconds().alias("time_diff")) 
        df_ = df_.with_columns((pl.col("time_diff") - pl.col("time_diff").mode()).alias("y_s")) \
                        .with_columns((pl.col("y_s").std().alias("sigma_s"))) \
                        .with_columns((pl.col("y_s").abs() > pl.col("sigma_s")).alias("jump"))
        td_wo[n] = df_
    return td_wo

def _unique_ts(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.col("timestamp").unique().sort())


def offsets_ms(td: dict, reference:str = "instrument_torquemeter") -> pl.DataFrame :
    '''
    td: timestamp_dict
    reference : reference sensor
    '''
    ref =_unique_ts(td[reference]).with_columns(pl.col("timestamp").alias("t_ref"))
    half = ref["timestamp"].diff().dt.total_microseconds().median() / 2 / 1e3   # half interval, ms
    parts = []
    for name, df in td.items():
        if name == reference:
            continue
        parts.append(
            _unique_ts(df)
            .join_asof(ref, on="timestamp", strategy="nearest")
            .with_columns(((pl.col("timestamp") - pl.col("t_ref")).dt.total_microseconds() / 1e3)
                          .alias("offset_ms"))
            .filter(pl.col("offset_ms").abs() <= half)
            .select(pl.lit(name).alias("instrument"), "timestamp", "offset_ms")
        )
    return pl.concat(parts)

def plot_offsets_ms(td: dict, reference: str, event : datetime, zoom_ms: float | None = None, plotdir : str = PLOTS_DIR, fname: str = "timestamp_offsets_ms.pdf"):

    off = offsets_ms(td, reference)
    stats = (off.group_by("instrument", maintain_order=True)
                .agg(pl.col("offset_ms").median().alias("median_ms"),
                     pl.col("offset_ms").quantile(0.05).alias("p05_ms"),
                     pl.col("offset_ms").quantile(0.95).alias("p95_ms"),
                     pl.len().alias("n"))
                .sort("median_ms"))
    others = stats["instrument"].to_list()
    names = [reference] + others                        # rows ordered by median offset
    rng = np.random.default_rng(0)
 
    # x-range: wide enough for all instruments
    lim = zoom_ms or 1.15 * max(abs(stats["p05_ms"].min()), abs(stats["p95_ms"].max()), 1.0)
 
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 0.45 * len(names) + 1.8), sharey=True,
                                     facecolor="white")
 
    # (a) one logging event
    ref_ts = _unique_ts(td[reference])
    t_ev = (pl.DataFrame({"timestamp": [event]}).cast(ref_ts.schema)
              .join_asof(ref_ts.with_columns(pl.col("timestamp").alias("t_ref")),
                         on="timestamp", strategy="nearest"))["t_ref"][0]
    for i, n in enumerate(names):
        ts = _unique_ts(td[n])
        d = ((ts["timestamp"] - t_ev).dt.total_microseconds() / 1e3).to_numpy()
        d = d[np.abs(d) <= lim]
        ax_a.vlines(d, i - 0.35, i + 0.35, color=COLORS[i % 8], lw=2.5)
    ax_a.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax_a.set_xlim(-lim, lim)
    ax_a.set_title(f"(a) One logging event: {t_ev:%Y-%m-%d %H:%M:%S} UTC", loc="left", fontsize=11)
    ax_a.set_xlabel(rf"time relative to {reference} timestamp (ms)")
 
    # (b) all logging events
    for i, n in enumerate(others, start=1):
        o = off.filter(pl.col("instrument") == n)["offset_ms"].to_numpy()
        y = i + rng.uniform(-0.25, 0.25, o.size)
        ax_b.plot(o, y, ".", ms=2, alpha=0.25, color=COLORS[i % 8], rasterized=True)
        med = np.median(o)
        ax_b.vlines(med, i - 0.38, i + 0.38, color=INK, lw=2)
        ax_b.annotate(f"{med:+.0f} ms", (med, i - 0.38), xytext=(0, 2), textcoords="offset points",
                      ha="center", va="bottom", fontsize=8, color=INK2)
    ax_b.axvline(0, color=COLORS[0], lw=1.2)
    ax_b.set_xlim(-lim, lim)
    ax_b.set_title(rf"(b) All logging events: offset $\Delta t$ (bar = median)", loc="left", fontsize=11)
    ax_b.set_xlabel(rf"$\Delta t$ to nearest {reference} timestamp (ms)")
 
    ax_a.set_yticks(range(len(names)), [f"{reference} (ref.)"] + others)
    ax_a.invert_yaxis()
    for ax in (ax_a, ax_b):
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir,fname), dpi=300)
    #plt.show()
    print(stats)
    return stats


def plot_timestamp_offsets(DB_PATH, PLOTS_DIR): #didn't work due to scale, offset is not noticeable in plot
    names= get_metrics_reg(DB_PATH)["source_name"].unique()
    w0 = pl.datetime(2024, 1, 13, 12, 0, 0, 0, time_zone="Europe/Oslo")  # time is in UTC, however, by default it was set at Europe/Oslo. 
    w1 = pl.datetime(2024, 1, 13, 12, 1, 0, 0, time_zone="Europe/Oslo") 

    conn = db.connect(database=DB_PATH)
    fig = plt.figure(figsize = (12,6), layout = "tight")
    gs = fig.add_gridspec(1,4)
    ax_a = fig.add_subplot(gs[0,0:2])
    ax_b = fig.add_subplot(gs[0,3])
    # (a) raster
    for i, sn in enumerate(names):
        t = conn.execute(f"SELECT * FROM metrics_{sn.replace(" ","_").lower()}").pl().drop_nulls().filter(pl.col("timestamp").is_between(w0, w1))["timestamp"].unique()
        ax_a.vlines(t, i - 0.35, i + 0.35, color=COLORS[i % 8], lw=2)
    ax_a.set_yticks(range(len(names)), [f"{n} (ref.)" for n in names])
    #ax_a.set_xlim(w0.item(), w1.item())
    #plt.show() #didn't work due to scale, offset is not noticeable in plot
    plt.savefig(os.path.join(PLOTS_DIR, "sensor_offsets.png"), dpi = 300)
    conn.close()

def plot_jumps(timestamp_dict, td_wo): #td_wo : time stamp dictionary upto sept
    os.makedirs(os.path.join(PLOTS_DIR, "time_vector"), exist_ok=True)
    plt.rcParams['axes.grid'] = True          # Enable the grid by default
    plt.rcParams['grid.color'] = '#d3d3d3'  # soft light gray
    for k,v in timestamp_dict.items():
        fig,ax = plt.subplots(1,2, figsize = (15,6))
        
        ax[0].plot(v["timestamp"].to_numpy(),v["time_diff"].to_numpy(), color="cyan", lw=0.5, ls= "--", label=r"$\Delta T$")
        ax[0].plot(v["timestamp"].to_numpy(),v["y_s"].to_numpy(), color="blue", lw=1, label=r"$\mathbf{Z} - 1/f_s$")
        ax[0].axhline(v["sigma_s"].mode().max(), color="red", ls="-.", lw=0.8, label=r"$\pm\sigma_{\mathbf{z}}$")
        ax[0].axhline(v["sigma_s"].mode().max(), color="red", ls="-.", lw=0.8)
        ax[0].legend(loc="upper right", fontsize=12, edgecolor="black", fancybox=False, framealpha=1)
        ax[0].set_ylabel(r"$1^{st}$ order difference [seconds]", family="serif", fontsize=14)
        ax[0].set_title(k)

        v1 = td_wo[k]
        ax[1].plot(v1["timestamp"].to_numpy(),v1["time_diff"].to_numpy(), color="cyan", lw=0.5, ls= "--", label=r"$\Delta T$")
        ax[1].plot(v1["timestamp"].to_numpy(),v1["y_s"].to_numpy(), color="blue", lw=1, label=r"$\mathbf{Z} - 1/f_s$")
        ax[1].axhline(v1["sigma_s"].mode().max(), color="red", ls="-.", lw=0.8, label=r"$\pm\sigma_{\mathbf{z}}$")
        ax[1].axhline(v1["sigma_s"].mode().max(), color="red", ls="-.", lw=0.8)
        ax[1].legend(loc="upper right", fontsize=12, edgecolor="black", fancybox=False, framealpha=1)
        ax[1].set_ylabel(r"$1^{st}$ order difference [seconds]", family="serif", fontsize=14)
        ax[1].set_title(f"{k} - till sept")

        #plt.show()
        fig.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,  "time_vector", f"{k.replace(" ","_").lower()}.png"))
        plt.close()
        

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TIME SYNCHRONIZATION ACROSS SENSORS
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 

def same_physical_quantities(DB_PATH):
    metrics_wide = get_metrics_wide(DB_PATH)
    me_rpm_df = metrics_wide.select(pl.col(["timestamp","me_rpm"]))
    shaft_rpm_df = metrics_wide.select(pl.col(["timestamp","shaft_rpm"]))
    me_rpm_df = me_rpm_df.filter(pl.col("timestamp").is_between(pl.date(2024, 1, 13), pl.date(2024, 1, 14))).drop_nulls()
    shaft_rpm_df = shaft_rpm_df.filter(pl.col("timestamp").is_between(pl.date(2024, 1, 13), pl.date(2024, 1, 14))).drop_nulls()
    fig = plt.figure(figsize=(12,6))
    plt.grid(True)
    plt.plot(me_rpm_df["timestamp"].to_numpy(),me_rpm_df["me_rpm"].drop_nulls().to_numpy(), color = "red", lw = 0.5 )
    plt.plot(shaft_rpm_df["timestamp"].to_numpy(),shaft_rpm_df["shaft_rpm"].to_numpy(), color = "blue", lw = 0.5 )
    plt.show()



if __name__ == "__main__":
    timestamp_d = detect_jumps(DB_PATH, create_table=False)
    td1 = detect_jumps_excluding_oct(td=timestamp_d)
    plot_jumps(timestamp_d, td1)
    plot_timestamp_offsets(DB_PATH, PLOTS_DIR)
    plot_offsets_ms(timestamp_d, reference="instrument_torquemeter", event=datetime(2024, 1, 13, 11, 0, 0, tzinfo=timezone.utc))
    #same_physical_quantities(DB_PATH)

