"""Turn kensas_10min.csv into a model-ready table: model_table.csv

Run from the repo root:   python src/prepare_features.py
Needs: pandas, numpy, scikit-learn. Reads data/processed/kensas_10min.csv and data/raw/Noon Reports

What it does
  1. Loads the 10-minute sensor table and the daily noon reports
  2. Fixes the known sensor faults
       - drafts: sensors read 0 all year -> displacement, drafts and trim from the noon reports
       - power: torquemeter wrong from 10 Jul 2024 and when it drops out to 0 -> virtual power from
         scavenge air pressure and engine rpm (fuel flow is not used: the fuel meter drifts from Oct, DQ-18)
       - weather: Provider S placeholder rows (DQ-06) and Provider MB all-zero currents (DQ-26) -> NaN
       - speed: speed log (measured speed through water) with its offset removed on 29 Jul - 16 Oct (DQ-05);
         the modelled currents explain too little of STW - SOG to rebuild speed from GPS (DQ-19).
         GPS speed minus current is kept as stw_gps_kn and used where the speed log is missing
       - waves: Provider S first (it is the only source of wave direction); MB first was tested and fits worse (DQ-28)
  3. Builds physics-based features (log power, log speed, wind and wave components)
  4. Adds filter flags (steady sailing, calm weather) and split helpers (reference period, week)
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

HERE = os.path.dirname(os.path.abspath(__file__))
NOON = os.path.join(HERE, "..", "data", "raw", "Noon Reports")
PROCESSED = os.path.join(HERE, "..", "data", "processed")
KN = 1.94384  # m/s -> knots
TORQUE_OK_UNTIL = "2024-07-10"   # torquemeter trusted before this date
REFERENCE_END = "2024-06-24"     # reference period for the baseline: 1 Jan to 23 Jun 2024
STW_OFFSET = ("2024-07-29", "2024-10-17", 0.73)  # DQ-05: speed log − SOG is +1.16 kn here vs +0.43 kn laden elsewhere
                                                  # (the period is all laden; ballast elsewhere +0.12 kn)

# ---------------------------------------------------------------- 1. load
m = pd.read_csv(os.path.join(PROCESSED, "kensas_10min.csv"), index_col="time", parse_dates=True)
# placeholder weather -> NaN, before the hourly values are interpolated into their neighbours
s_fill = (m.s_cur_e == -0.4) & (m.s_cur_n == -0.4) & (m.s_sst.round(1) == 6.0)          # DQ-06
m.loc[s_fill, ["s_sst", "s_cur_e", "s_cur_n", "s_tp", "s_hs", "s_wave_deg"]] = np.nan
mb_zero = (m.mb_cur_e == 0) & (m.mb_cur_n == 0)                                            # DQ-26
m.loc[mb_zero, ["mb_cur_e", "mb_cur_n"]] = np.nan
weather = [c for c in m.columns if c.startswith(("s_", "mb_"))]
m[weather] = m[weather].interpolate(limit=6, limit_area="inside")  # hourly -> 10 min, gaps up to 1 h

noon = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(NOON, "*.csv")))])
noon["date"] = pd.to_datetime(noon["Date"], format="%d/%m/%Y")
noon = noon.set_index("date")
day = m.index.tz_localize(None).normalize()

out = pd.DataFrame(index=m.index)

# ---------------------------------------------------------------- 2a. loading condition (noon reports)
out["disp_t"] = noon["Displacement"].reindex(day).values
out["draft_fwd"] = noon["Fwd Draft"].reindex(day).values
out["draft_aft"] = noon["Aft Draft"].reindex(day).values
out["trim_m"] = out.draft_aft - out.draft_fwd             # + = by the stern
out["laden"] = out.disp_t > 55000

# ---------------------------------------------------------------- 2b. speed through water
h = np.deg2rad(m.heading_deg)
cur_s = (m.s_cur_e * np.sin(h) + m.s_cur_n * np.cos(h)) * KN
cur_mb = (m.mb_cur_e * np.sin(h) + m.mb_cur_n * np.cos(h)) * KN
out["current_along_kn"] = pd.concat([cur_s, cur_mb], axis=1).mean(axis=1)  # + = current pushing the ship
out["sog_kn"] = m.sog_kn
out["stw_log_kn"] = m.stw_kn
out["stw_gps_kn"] = m.sog_kn - out.current_along_kn                      # GPS speed minus modelled current
out["log_bias_kn"] = out.stw_log_kn - out.stw_gps_kn                      # sensor check
# speed used by the models: the speed log, its offset removed where it reads high (DQ-05), GPS-based where it is missing
off_from, off_to, off_kn = STW_OFFSET
stw = out.stw_log_kn.copy()
stw[(out.index >= pd.Timestamp(off_from, tz=out.index.tz)) & (out.index < pd.Timestamp(off_to, tz=out.index.tz))] -= off_kn
out["speed_kn"] = stw.fillna(out.stw_gps_kn)
out["speed_source"] = np.where(stw.notna(), "speed log", np.where(out.stw_gps_kn.notna(), "gps - current", "none"))

# ---------------------------------------------------------------- 2c. power: measured, virtual, and the one to use
out["shaft_rpm"] = m.shaft_rpm.abs()
out["me_rpm"] = m.me_rpm.abs()
out["scav_bar"] = m.me_scav_bar
out["fo_kgh"] = m.fo_kgh
out["power_meas_kw"] = m.shaft_kw.abs()
# DQ-14: torquemeter writes 0 while the engine turns -> a dropout, not real power. Engine rpm stands in for shaft rpm
dropout = (out.power_meas_kw < 50) & (out.me_rpm > 50)
out.loc[dropout, "power_meas_kw"] = np.nan
out.loc[dropout, "shaft_rpm"] = out.me_rpm[dropout]
fit_rows = (out.me_rpm > 50) & (m.shaft_rpm_sd < 1) & (out.power_meas_kw > 800) & (m.index < TORQUE_OK_UNTIL)


def virtual_power(cols, ok):
    """power from engine-side signals, fitted where the torquemeter is trusted. Prints the May-Jul holdout MAPE."""
    X = out[cols]
    mdl = make_pipeline(PolynomialFeatures(2), LinearRegression())
    train, test = fit_rows & (m.index < "2024-05-01"), fit_rows & (m.index >= "2024-05-01")
    mdl.fit(X[train], out.power_meas_kw[train])
    mape = 100 * (mdl.predict(X[test]) / out.power_meas_kw[test] - 1).abs().mean()
    print(f"virtual power from {' + '.join(cols)}: holdout MAPE (May - 9 Jul) {mape:.1f}%")
    mdl.fit(X[fit_rows], out.power_meas_kw[fit_rows])
    p = pd.Series(np.nan, index=out.index)
    ok = ok & X.notna().all(axis=1)
    p[ok] = mdl.predict(X[ok])
    return p


# DQ-02 / DQ-18: torquemeter reads low from 10 Jul, and the fuel meter reads high from mid Oct, so the virtual
# power uses scavenge air pressure + engine rpm (neither sensor involved). The fuel-based version is kept for comparison
out["power_virtual_kw"] = virtual_power(["scav_bar", "me_rpm"], (out.me_rpm > 50) & (out.scav_bar > 0.1))
out["power_virtual_fuel_kw"] = virtual_power(["fo_kgh", "me_rpm"], (out.me_rpm > 50) & (out.fo_kgh > 200))
out["torque_ratio"] = out.power_meas_kw / out.power_virtual_kw            # sensor check, ~1.0 when healthy
use_meas = (m.index < TORQUE_OK_UNTIL) & ~dropout
out["power_kw"] = np.where(use_meas, out.power_meas_kw, out.power_virtual_kw)
out["power_source"] = np.where(use_meas, "torquemeter", "scavenge")

# ---------------------------------------------------------------- 3. physics features
out["log_power"] = np.log(out.power_kw.where(out.power_kw > 0))
out["log_speed"] = np.log(out.speed_kn.where(out.speed_kn > 0))
out["log_disp"] = np.log(out.disp_t)
rel = np.deg2rad(m.wind_rel_deg)
out["wind_rel_kn"] = m.wind_rel_kn
out["wind_head_kn"] = m.wind_rel_kn * np.cos(rel)                          # + = head wind
out["wind_side_kn"] = (m.wind_rel_kn * np.sin(rel)).abs()
out["wind_head2"] = out.wind_head_kn * out.wind_rel_kn                     # ~ wind resistance, keeps the sign
out["wind_true_kn"] = np.hypot(m.s_wind_e, m.s_wind_n) * KN
out["hs_m"] = m.s_hs.fillna(m.mb_hs)   # wave height: S first. MB first was tested (DQ-28) and fits worse: only S has wave direction
wave_rel = np.deg2rad((m.s_wave_deg - m.heading_deg) % 360)
out["wave_head_m"] = out.hs_m * np.cos(wave_rel)                            # + = head seas
out["hs2"] = out.hs_m ** 2
out["wave_period_s"] = m.s_tp
out["sea_temp_c"] = m.s_sst            # NOT for the baseline model (acts as a clock), OK for the growth model
out["depth_m"] = m.depth_m             # echosounder, sparse

# ---------------------------------------------------------------- 4. flags and split helpers
out["steady"] = ((out.speed_kn > 8) & (out.shaft_rpm > 55) & (m.shaft_rpm_sd < 1) & (m.stw_kn_sd < 0.3)
                 & (m.sog_kn_sd < 0.3) & (m.heading_deg_sd < 2) & (m.turn_rate.abs() < 0.5)
                 & out.disp_t.notna() & (out.power_kw > 800))
out["calm"] = (out.wind_true_kn < 16) & (out.hs_m < 2.0)
out["reference_period"] = out.index < REFERENCE_END
out["torque_trusted"] = out.index < TORQUE_OK_UNTIL
out["week"] = out.index.strftime("%G-W%V")                                 # group for week-by-week validation
out["month"] = out.index.strftime("%Y-%m")
out.round(4).to_csv(os.path.join(PROCESSED, "model_table.csv"))
s = out[out.steady]
print(f"model_table.csv: {len(out):,} rows, {out.shape[1]} columns")
print(f"steady sailing rows: {len(s):,} (reference period: {s.reference_period.sum():,}, calm: {s.calm.sum():,})")
