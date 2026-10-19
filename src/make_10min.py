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

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "raw")
PROCESSED = os.path.join(HERE, "..", "data", "processed")

# Short column names for every signal (quantity name | source name, as in Metrics registration.xlsx)
SHORT = {
    "Main Engine Scavenging Air Pressure | Transducer Pressure": "me_scav_bar",
    "Main Engine Turbocharger Rotational Speed | Transducer RPM": "me_tc_rpm_tr",
    "Main Engine Fuel Load % | Transducer Fuel Load": "me_load_pct",
    "Vessel External Conditions Wind Relative Speed | Instrument Anemometer": "wind_rel_kn",
    "Vessel External Conditions Wind Relative Angle | Instrument Anemometer": "wind_rel_deg",
    "Vessel Propeller Shaft Rotational Speed | Instrument Torquemeter": "shaft_rpm",
    "Vessel Propeller Shaft Revolutions | Instrument Torquemeter": "shaft_revs",
    "Vessel Propeller Shaft Torque | Instrument Torquemeter": "shaft_torque_nm",  # values are really kNm
    "Vessel Propeller Shaft Mechanical Power | Instrument Torquemeter": "shaft_kw",
    "Vessel Propeller Shaft Mechanical Energy | Instrument Torquemeter": "shaft_kwh",
    "Vessel Propeller Shaft Thrust Force | Instrument Torquemeter": "thrust_kn",
    "Main Engine Fuel Oil Inlet Mass Flow | Instrument Torquemeter": "fo_kgh",
    "Main Engine Turbocharger Rotational Speed | Instrument RPM Indicator": "me_tc_rpm",  # no data in the files
    "Vessel Hull Relative To Transducer Water Depth | Instrument Echosounder": "depth_m",
    "Vessel Hull Heading True Angle | Instrument Gyrocompass": "heading_deg",
    "Vessel Hull Heading Turn Rate | Instrument Gyrocompass": "turn_rate",
    "Vessel Hull Over Ground Speed | Instrument GPS 1": "sog_kn",
    "Vessel Hull Through Water Longitudinal Speed | Instrument Speedlog": "stw_kn",
    "Vessel Hull Fore Draft | Control Alarm Monitoring System": "draft_fwd",
    "Vessel Hull Aft Draft | Control Alarm Monitoring System": "draft_aft",
    "Vessel Hull MidP Draft | Control Alarm Monitoring System": "draft_midp",
    "Vessel Hull MidS Draft | Control Alarm Monitoring System": "draft_mids",
    "Main Engine Rotational Speed | Control Alarm Monitoring System": "me_rpm",
    "Vessel External Conditions Wind True Speed | Provider MB": "mb_wind_kn",
    "Vessel External Conditions Wind True Angle | Provider MB": "mb_wind_deg",
    "Vessel External Conditions Wave Significant Height | Provider MB": "mb_hs",
    "Vessel External Conditions Swell Significant Height | Provider MB": "mb_swell",
    "Vessel External Conditions Eastward Sea Water Velocity | Provider MB": "mb_cur_e",
    "Vessel External Conditions Northward Sea Water Velocity | Provider MB": "mb_cur_n",
    "Vessel External Conditions Wave True Angle | Provider S": "s_wave_deg",
    "Vessel External Conditions Sea Water Temperature | Provider S": "s_sst",
    "Vessel External Conditions Wave Period | Provider S": "s_tp",
    "Vessel External Conditions Wave Significant Height | Provider S": "s_hs",
    "Vessel External Conditions Eastward Sea Water Velocity | Provider S": "s_cur_e",
    "Vessel External Conditions Northward Sea Water Velocity | Provider S": "s_cur_n",
    "Vessel External Conditions Eastward Wind Velocity | Provider S": "s_wind_e",
    "Vessel External Conditions Northward Wind Velocity | Provider S": "s_wind_n",
}
STD_COLS = ["stw_kn", "sog_kn", "shaft_rpm", "shaft_kw", "heading_deg", "turn_rate"]  # -> _sd columns
COUNT_COLS = ["stw_kn", "shaft_kw"]                                                    # -> _n columns
DROP = ["draft_fwd", "draft_aft", "draft_midp", "draft_mids"]                          # always 0


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
    main()
