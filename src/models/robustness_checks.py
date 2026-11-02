"""Robustness checks of the fouling result: per trip, and refitted with different filters and inputs.

Run from the repo root:   python src/models/robustness_checks.py
Input:  data/processed/model_table.csv, data/processed/kensas_10min.csv, reference/towing_with_psv.csv
Output: printed only

The fouling indicator is refitted for each variant exactly as in submission.py (towing-tank curve x Huber
correction fitted on steady reference rows, fouling = 100 (P / P_expected - 1), median of clean rows), so the
variants differ only in the rows or inputs being tested.
  - Trips: stays are runs below 4 kn for at least 24 h (as in notebooks/energy_comparison/identify_trips.py).
    Trips are also split where the ship data stop for 24 h or more and where the loading condition changes,
    so each trip is one observed voyage in one loading condition. Trips shorter than 48 h are dropped.
  - Settling: a row also needs the three rows before it (30 minutes) to be steady (Prytz et al., 2026).
  - MB waves: wave height from provider MB first. Direction still comes from provider S, the only one that has it,
    so rows without an S direction are left out. Tested with the calm-weather rule on S and on MB wave height.
  - Light wind: clean rows also need true wind below 10 kn (the calm rule allows 16 kn).
  - GPS speed: speed through water from GPS speed minus the modelled current, instead of the speed log.
Intervals are 90 % from resampling whole weeks.
"""
import contextlib
import io
import os
import runpy

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

pd.set_option("display.width", 200)
HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "..", "..", "data", "processed")
STAY_KN, MIN_STAY_H, MIN_GAP_H, MIN_TRIP_H = 4.0, 24, 24, 48
SETTLE_ROWS, LIGHT_WIND_KN, MIN_WEEK_ROWS, N_BOOT = 3, 10, 12, 1000

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(HERE, "submission.py"))   # reuse the clean curves and constants
interp_disp, fit_p, F = g["interp_disp"], g["fit_p"], g["F"]
OUTLIER_SD, VIRTUAL_POWER_ERR = g["OUTLIER_SD"], g["VIRTUAL_POWER_ERR"]


def fouling(m):
    """Same steps as submission.py. Returns the fouling indicator, the clean-row flag and the residual spread."""
    v, p, disp = m.speed_kn, m.power_kw, m.disp_t
    usable = (v > 5) & (p > 500) & disp.notna()
    p_clean = pd.Series(np.nan, index=m.index)
    p_clean[usable] = np.exp(interp_disp(fit_p, v[usable], disp[usable]))
    X = m[F].astype(float)
    ok = usable & X.notna().all(axis=1)
    train = m.steady & m.reference_period & ok
    corr = HuberRegressor(max_iter=500).fit(X[train], np.log(p / p_clean)[train])
    p_exp = pd.Series(np.nan, index=m.index)
    p_exp[ok] = p_clean[ok] * np.exp(corr.predict(X[ok]))
    res = np.log(p / p_exp)
    sd_ref = 1.4826 * (res[train] - res[train].median()).abs().median()
    sd_row = pd.Series(np.where(m.power_source == "torquemeter", sd_ref, np.hypot(sd_ref, VIRTUAL_POWER_ERR)), index=m.index)
    roll = res.where(m.steady).rolling("3D", min_periods=12).median()
    outlier = ok & (((res - roll).abs() > OUTLIER_SD * sd_row) | (m.torque_trusted & ((m.torque_ratio - 1).abs() > 0.15))
                    | (m.log_bias_kn.abs() > 2))
    clean = ok & m.steady & m.calm.fillna(False) & ~outlier & (m.index < "2025-01-01")
    return 100 * (p / p_exp - 1), clean, sd_ref


def week_interval(f):
    """Median with a 90 % interval from resampling whole weeks."""
    weeks = [w.to_numpy() for _, w in f.groupby(pd.Grouper(freq="W")) if len(w)]
    rng = np.random.default_rng(0)
    boot = [np.median(np.concatenate([weeks[i] for i in rng.integers(0, len(weeks), len(weeks))])) for _ in range(N_BOOT)]
    return np.median(f), np.quantile(boot, 0.05), np.quantile(boot, 0.95), len(weeks)


def runs(flag):
    """Length in hours of the run each row belongs to."""
    run = (flag != flag.shift()).cumsum()
    return flag.groupby(run).transform(lambda s: (s.index[-1] - s.index[0]).total_seconds() / 3600 + 1 / 6)


def trips(m):
    """Trip number per row (0 = not on a trip)."""
    slow = m.sog_kn.rolling("30min").median() < STAY_KN
    no_data = m.sog_kn.isna() & m.power_kw.isna()
    stop = (slow & (runs(slow) >= MIN_STAY_H)) | (no_data & (runs(no_data) >= MIN_GAP_H))
    laden = m.laden.where(~stop).ffill()
    seg = ((stop != stop.shift()) | (laden != laden.shift())).cumsum().where(~stop)
    span = m.index.to_series().groupby(seg).agg(lambda s: (s.iloc[-1] - s.iloc[0]).total_seconds() / 3600)
    number = {s: i + 1 for i, s in enumerate(span[span >= MIN_TRIP_H].index)}
    return seg.map(number).fillna(0).astype(int)


def weekly_range(f, clean, mask):
    w = f[clean & mask].groupby(pd.Grouper(freq="W")).agg(["median", "size"])
    w = w[w["size"] >= MIN_WEEK_ROWS]["median"]
    return f"{w.min():.0f} to {w.max():.0f}"


def main():
    m = pd.read_csv(os.path.join(PROCESSED, "model_table.csv"), index_col="time", parse_dates=True)
    base, clean, sd = fouling(m)
    print(f"baseline: robust residual sd {100 * sd:.1f}%, {clean.sum():,} clean rows")

    # ---- 1. per trip
    m["trip"] = trips(m)
    rows = []
    for k, x in base[clean & (m.trip > 0)].groupby(m.trip):
        med, lo, hi, n = week_interval(x)
        rows.append(dict(trip=k, start=x.index.min().strftime("%d %b"), end=x.index.max().strftime("%d %b"),
                         loading="laden" if m.laden[x.index].mean() > 0.5 else "ballast", rows=len(x), weeks=n,
                         median=med, low=lo, high=hi, weekly_range=weekly_range(base, clean, m.trip == k)))
    print(f"\n1. Fouling [%] per trip, clean rows, 90 % interval from resampling weeks")
    print(pd.DataFrame(rows).set_index("trip").round(1).to_string())

    # ---- 2. variants
    variants = {"baseline": m}
    prev = m.steady.astype(float)
    consecutive = m.index.to_series().diff().dt.total_seconds().rolling(SETTLE_ROWS).max() <= 600
    settled = m.steady & consecutive & np.logical_and.reduce([prev.shift(i).eq(1) for i in range(1, SETTLE_ROWS + 1)])
    variants["settled 30 min"] = m.assign(steady=settled)

    mb_hs = pd.read_csv(os.path.join(PROCESSED, "kensas_10min.csv"), index_col="time", parse_dates=True,
                        usecols=["time", "mb_hs"]).mb_hs.reindex(m.index).interpolate(limit=6, limit_area="inside")
    direction = (m.wave_head_m / m.hs_m).where(m.hs_m > 0)          # cosine of the S wave direction; NaN = unknown
    hs = mb_hs.fillna(m.hs_m)
    mb = m.assign(hs2=(hs ** 2).where(direction.notna()), wave_head_m=hs * direction)
    variants["MB waves, calm on S"] = mb
    variants["MB waves, calm on MB"] = mb.assign(calm=(m.wind_true_kn < 16) & (hs < 2.0))
    variants["light wind"] = m.assign(calm=m.calm.fillna(False) & (m.wind_true_kn < LIGHT_WIND_KN))
    variants["GPS speed"] = m.assign(speed_kn=m.stw_gps_kn)

    monthly, summary = {}, []
    for name, v in variants.items():
        f, c, s = fouling(v)
        x = pd.DataFrame({"f": f, "laden": m.laden})[c]
        t = x.groupby([x.index.strftime("%Y-%m"), "laden"]).f.agg(["median", "size"])
        monthly[name] = t[t["size"] >= 30]["median"]
        summary.append(dict(variant=name, sd=100 * s, clean_rows=int(c.sum()),
                            laden_aug_sep=weekly_range(f, c, m.laden & (m.index >= "2024-08-01") & (m.index < "2024-10-01")),
                            ballast_oct_nov=weekly_range(f, c, ~m.laden & (m.index >= "2024-10-01") & (m.index < "2024-12-01"))))
    print(f"\n2. Variants. sd = robust residual spread in the reference period [%]; weekly ranges of the indicator [%]")
    print(f"   settling keeps {settled.sum():,} of {m.steady.sum():,} steady rows")
    print(pd.DataFrame(summary).set_index("variant").round(1).to_string())
    print("\nMonthly fouling [%], clean rows (months with at least 30 rows)")
    print(pd.DataFrame(monthly).round(1).to_string())


if __name__ == "__main__":
    main()
