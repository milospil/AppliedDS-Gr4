"""Convert the raw monthly files to Parquet at native resolution (no averaging, nothing dropped).

Run from the repo root:   python src/load_raw.py
Needs: pandas, openpyxl, pyarrow. Takes a few minutes, about 2 GB of memory.

Input (in data/raw):
  1.csv ... 12.csv, Jan25.csv   raw long-format files (timestamp, signal id, value)
  Metrics registration.xlsx     signal id -> quantity, source, unit

Output (in data/processed/raw_long/, one file per raw file):
  <name>.parquet  columns: time (UTC), signal (short name), value (float64, exactly as logged),
                  n_records (Jan25 only, otherwise empty)
  data/processed/signals.csv   signal metadata: short name, quantity, source, unit, id

Every reading is kept, including zeros, negatives, duplicates and readings with an unknown id
(signal = "unknown:<id>"). Read it back with:  pd.read_parquet("data/processed/raw_long")
"""
import glob
import os

import pandas as pd

from cfg import DATASET, SHORT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "processed", "raw_long")


def signals():
    m = pd.read_excel(os.path.join(DATASET, "Metrics registration.xlsx"))
    m["signal"] = (m.quantity_name + " | " + m.source_name).map(SHORT)
    return m


def read_file(path):
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    if first.startswith("F1,"):  # Jan25.csv: header row, 'DD/MM/YYYY HH:MM:SS', 4th column
        d = pd.read_csv(path, encoding="utf-8-sig")
        d.columns = ["ts", "q", "value", "n_records"]
        d["time"] = pd.to_datetime(d.ts, format="%d/%m/%Y %H:%M:%S", utc=True)
    else:
        d = pd.read_csv(path, header=None, names=["ts", "q", "value"], dtype={"ts": str, "q": str, "value": "float64"})
        d["time"] = pd.to_datetime(d.ts, format="ISO8601", utc=True)
        d["n_records"] = pd.NA
    return d


def main():
    meta = signals()
    os.makedirs(OUT, exist_ok=True)
    meta.to_csv(os.path.join(OUT, "..", "signals.csv"), index=False)
    names = dict(zip(meta.qid_mapping, meta.signal))
    total = 0
    for path in sorted(glob.glob(os.path.join(DATASET, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        d = read_file(path)
        d["signal"] = d.q.map(names).fillna("unknown:" + d.q)
        d = d[["time", "signal", "value", "n_records"]].astype({"signal": "category", "n_records": "Int32"})
        d["source_file"] = name
        d.to_parquet(os.path.join(OUT, f"{name}.parquet"), index=False)
        total += len(d)
        print(f"{name}.csv: {len(d):,} readings, {d.signal.nunique()} signals", flush=True)
    print(f"total: {total:,} readings -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
