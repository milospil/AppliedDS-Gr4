import os
import duckdb as db

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


HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "raw")
PROCESSED = os.path.join(HERE, "..", "data", "processed")
DB_PATH = os.path.join(HERE, "..", "data", "raw", "sintef_ocean.db")

def load_extensions():
    for ext in ["parquet", "json", "spatial", "excel"]:
        db.sql(f"INSTALL {ext}; LOAD {ext};")