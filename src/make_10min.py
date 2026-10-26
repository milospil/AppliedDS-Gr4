"""Rebuild kensas_10min.csv from the raw Kensas files.

Run from the repo root:   python src/make_10min.py
Needs: pandas, openpyxl (both come with Anaconda). Takes a few minutes and about 1 GB of memory.

Input (in data/raw):
  1.csv ... 12.csv          Jan to Dec 2024. No header. Columns: ISO timestamp, signal id, value
  Jan25.csv                  Jan 2025. Header row, day-first dates, an extra 4th column
  Metrics registration.xlsx  Maps each signal id to a name, source and unit

Output:
  data/processed/kensas_10min.csv  one row per 10 minutes (UTC), one column per signal (mean of the raw readings),
                    plus _sd (standard deviation inside the 10 minutes) and _n (number of readings)
                    for a few signals used to find steady sailing. Draft sensors are dropped (always 0).
"""
import glob
import os
import pandas as pd
import duckdb as db
import polars as pl
from .cfg import DROP, STD_COLS, COUNT_COLS, SHORT, HERE, DATASET, PROCESSED, DB_PATH, load_extensions


load_extensions()  # load DuckDB extensions for reading Excel, Parquet, etc.

def csv_files(path2dir):
    return [os.path.join(dirname, f) for dirname, _,files in os.walk(path2dir) for f in files if f.endswith('.csv')]

def signal_names():
    """signal id -> short name, using Metrics registration.xlsx."""
    m = pd.read_excel(os.path.join(DATASET, "Metrics registration.xlsx"))
    full = m.quantity_name + " | " + m.source_name
    return {qid: SHORT[name] for qid, name in zip(m.qid_mapping, full)}


def read_raw(path):
    """Read one raw file into columns ts (ISO text, UTC), q (signal id), v (value)."""
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    if first.startswith("F1,"):  # Jan25.csv: header row, 'DD/MM/YYYY HH:MM:SS', 4th column
        d = pd.read_csv(path, encoding="utf-8-sig")
        d.columns = ["ts", "q", "v", "n"]
        d["ts"] = pd.to_datetime(d.ts, format="%d/%m/%Y %H:%M:%S").dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return d[["ts", "q", "v"]]
    return pd.read_csv(path, header=None, names=["ts", "q", "v"], dtype={"ts": str, "q": str, "v": "float64"})


def create_database(path2metric_registration, path2data_dir, db_path):
    # Create a DuckDB database connection
    conn = db.connect(database=db_path)  
    col_names = {k.split(" | ")[0]: v for k, v in SHORT.items()}  
    conn.execute("SET TimeZone = 'UTC'")
    conn.execute(f"""CREATE TABLE metrics_registration AS SELECT * FROM read_xlsx('{path2metric_registration}')""")
    conn.execute("""ALTER TABLE metrics_registration
                ADD COLUMN short_name VARCHAR""")

    # Populate it using the dictionary
    conn.executemany(
        """
        UPDATE metrics_registration
        SET short_name = ?
        WHERE quantity_name = ?
        """,
        [(label, key) for key, label in col_names.items()],)
    
    # 
    data_csvs = csv_files(f"{path2data_dir}/timeseriesdata/")
    conn.execute("""
    CREATE TABLE metrics_raw AS 
    SELECT 
        CAST(csv.column0 AS TIMESTAMPTZ) AS timestamp,
        csv.column1 AS qid_mapping,
        dict.quantity_name,
        dict.short_name,
        csv.column2 AS value
    FROM read_csv($1::VARCHAR[], header=False) csv
    LEFT JOIN metrics_registration dict ON csv.column1 = dict.qid_mapping;
    """, [data_csvs])

    #Noon reports
    noon_csvs = csv_files(f"{path2data_dir}/Noon Reports/")
    conn.execute("""
    CREATE TABLE noon_reports AS 
    SELECT *
    FROM read_csv($1::VARCHAR[]) csv
    """, [noon_csvs])

    #Jan 25 
    jan25 = csv_files(f"{path2data_dir}/Jan25/")
    conn.execute("""
    CREATE TABLE jan25 AS 
    SELECT *
    FROM read_csv($1::VARCHAR[]) csv
    """, [jan25])
    print("Database Tables:", [row[0] for row in conn.execute("SHOW TABLES;").fetchall()])
    conn.close()

def long_to_wide(db_path):
    conn = db.connect(database=db_path)
    metrics_long = conn.execute("SELECT timestamp, short_name, value FROM metrics_raw").pl()
    conn.close()
    print("Long DataFrame shape:", metrics_long.shape)
    
    #there are some duplicate timestamps, so we need to aggregate them first
    wide_df = metrics_long.pivot(index="timestamp", on="short_name", values="value", aggregate_function="mean",).sort("timestamp") 
    
    print("Wide DataFrame shape:", wide_df.shape)
    return wide_df

def ten_minutes(d, names):
    """Long format -> wide 10-minute table for one file."""
    d["sig"] = d.q.map(names)
    unknown = d.sig.isna().sum()
    if unknown:
        print(f"  warning: {unknown} rows with an unknown signal id were skipped")
    d = d.dropna(subset=["sig"])
    d["slot"] = d.ts.str[:15] + "0"          # '2024-01-25T14:52' -> '2024-01-25T14:50'
    g = d.groupby(["slot", "sig"]).v
    wide = g.mean().unstack()
    sd = g.std().unstack()
    n = g.count().unstack()
    for c in STD_COLS:
        if c in sd:
            wide[c + "_sd"] = sd[c]
    for c in COUNT_COLS:
        if c in n:
            wide[c + "_n"] = n[c]
    return wide


def main():
    names = signal_names()
    files = sorted(glob.glob(os.path.join(DATASET, "*.csv")))
    parts = []
    for f in files:
        print("reading", os.path.basename(f), flush=True)
        parts.append(ten_minutes(read_raw(f), names))
    t = pd.concat(parts)
    t.index = pd.to_datetime(t.index, format="%Y-%m-%dT%H:%M", utc=True)
    t.index.name = "time"
    t = t.groupby(level=0).mean().sort_index()   # a slot that appears in two files (month border) is averaged
    t = t.drop(columns=[c for c in DROP if c in t.columns])
    t.round(4).to_csv(os.path.join(PROCESSED, "kensas_10min.csv"))
    print(f"kensas_10min.csv: {len(t):,} rows, {t.shape[1]} columns, {t.index.min()} to {t.index.max()}")


if __name__ == "__main__":
    #main()
    #create_database(os.path.join(DATASET, "Metrics registration.xlsx"), DATASET, os.path.join(DATASET, "sintef_ocean.db"))
    wide_df = long_to_wide(os.path.join(DATASET, "sintef_ocean.db"))
