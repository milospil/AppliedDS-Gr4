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

from ..cfg import DB_PATH, get_metrics_reg, get_metrics_wide, PLOTS_DIR, PROCESSED

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

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
        timestamp_dict[row["by"]] = df
    return timestamp_dict

def plot_jumps(timestamp_dict):
    os.makedirs(os.path.join(PLOTS_DIR, "time_vector"), exist_ok=True)
    for k,v in timestamp_dict.items():
        fig,ax = plt.subplots(figsize = (12,6))
        ax.grid(True)
        #ax.plot(v["timestamp"].to_numpy(), v["y_s"].to_numpy(), color = "blue", linewidth = 0.25)
        ax.plot(v["timestamp"].to_numpy(),v["time_diff"].to_numpy(), color="cyan", lw=0.5, ls= "--", label=r"$\Delta T$")
        ax.plot(v["timestamp"].to_numpy(),v["y_s"].to_numpy(), color="blue", lw=1, label=r"$\mathbf{Z} - 1/f_s$")
        ax.axhline(v["sigma_s"].mode().max(), color="red", ls="-.", lw=0.8, label=r"$\pm\sigma_{\mathbf{z}}$")
        ax.axhline(v["sigma_s"].mode().max(), color="red", ls="-.", lw=0.8)
        ax.legend(loc="upper right", fontsize=12, edgecolor="black", fancybox=False, framealpha=1)
        ax.set_ylabel(r"$1^{st}$ order difference [seconds]", family="serif", fontsize=14)
        ax.set_title(k)
        plt.savefig(os.path.join(PLOTS_DIR,  "time_vector", f"{k.replace(" ","_").lower()}.png"))
        #plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TIME SYNCHRONIZATION ACROSS SENSORS
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 
















if __name__ == "__main__":
    timestamp_d = detect_jumps(DB_PATH, create_table=False)
    plot_jumps(timestamp_d)



