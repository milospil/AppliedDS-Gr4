"""Benchmark B2: LightGBM power model (the failure case).

Run from the repo root:   python src/models/lightgbm_benchmark.py
Input:  data/processed/model_table.csv, reference/towing_with_psv.csv
Output: printed only

Same rows (steady, reference period) and the same week-based GroupKFold as the speed baseline.
LightGBM fits well inside the data (low MAPE), but trees cannot extrapolate: outside the trained
speed range the predicted power goes flat, while the towing-tank curve keeps rising roughly with V^3.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
PROCESSED = os.path.join(ROOT, "data", "processed")
TOWING = os.path.join(ROOT, "reference", "towing_with_psv.csv")
FEATURES = ["stw_gps_kn", "disp_t", "trim_m", "wind_rel_kn", "wind_head_kn", "hs_m", "wave_head_m", "wave_period_s"]
TARGET = "power_kw"
N_FOLDS = 6
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
COND_DISP = {"BALLAST": 36134, "DESIGN": 73346, "SCANTLING": 86941}  # m3, from the towing tank report


def towing_curve():
    """Clean-hull power (kW) from speed (kn) and displacement: log-log fit per condition,
    interpolated (linear beyond the ends) on log displacement."""
    t = pd.read_csv(TOWING)
    keys = ["BALLAST", "DESIGN", "SCANTLING"]
    fits = {c: np.polyfit(np.log(g.vs), np.log(g.pd_kw), 1) for c, g in t.groupby("cond")}  # slope, intercept
    xs = np.log([COND_DISP[k] for k in keys])

    def power(v, d):
        lv, ld = np.log(np.asarray(v, float)), np.log(np.asarray(d, float))
        ys = np.stack([fits[k][1] + fits[k][0] * lv for k in keys])  # 3 x n
        out = np.empty_like(lv)
        for i in range(len(lv)):
            if ld[i] < xs[0]:
                out[i] = ys[0, i] + (ld[i] - xs[0]) * (ys[1, i] - ys[0, i]) / (xs[1] - xs[0])
            elif ld[i] > xs[2]:
                out[i] = ys[2, i] + (ld[i] - xs[2]) * (ys[2, i] - ys[1, i]) / (xs[2] - xs[1])
            else:
                out[i] = np.interp(ld[i], xs, ys[:, i])
        return np.exp(out)
    return power


def main():
    t = pd.read_csv(os.path.join(PROCESSED, "model_table.csv"), index_col="time", parse_dates=True)
    ref = t[t.steady & t.reference_period & t[TARGET].notna()]
    print(f"reference rows: {len(ref):,} in {ref.week.nunique()} weeks")

    # ---- week-based cross-validation
    pred = pd.Series(np.nan, index=ref.index)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(ref, groups=ref.week):
        a, b = ref.iloc[tr], ref.iloc[te]
        pred.iloc[te] = lgb.LGBMRegressor(**PARAMS).fit(a[FEATURES], a[TARGET]).predict(b[FEATURES])
    e = 100 * np.abs(pred - ref[TARGET]) / ref[TARGET]
    print(f"power MAPE, GroupKFold by week: {e.mean():.2f}%  | laden {e[ref.laden].mean():.2f}%"
          f"  | ballast {e[~ref.laden].mean():.2f}%")

    # ---- extrapolation test: fixed laden condition, mild weather, speed 9 to 13.5 kn
    model = lgb.LGBMRegressor(**PARAMS).fit(ref[FEATURES], ref[TARGET])
    grid = pd.DataFrame({"stw_gps_kn": np.arange(9.0, 13.51, 0.5), "disp_t": 70228.0, "trim_m": 0.0,
                         "wind_rel_kn": 10.0, "wind_head_kn": 10.0, "hs_m": 1.0, "wave_head_m": 1.0,
                         "wave_period_s": 6.0})[FEATURES]
    table = pd.DataFrame({"speed_kn": grid.stw_gps_kn,
                          "lightgbm_kw": model.predict(grid).round(0),
                          "towing_tank_kw": towing_curve()(grid.stw_gps_kn, grid.disp_t).round(0)})
    print("\nextrapolation test (laden 70,228 t, trim 0, wind 10 kn head, Hs 1 m, Tp 6 s)")
    print(table.to_string(index=False))
    lo, hi = ref.stw_gps_kn.quantile([0.05, 0.95])
    print(f"reference speed range (5-95%): {lo:.1f} to {hi:.1f} kn")


if __name__ == "__main__":
    main()
