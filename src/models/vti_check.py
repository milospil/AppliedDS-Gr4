"""Checks of the VTI-style index from submission.py: how much it depends on speed, the speed source and wind.

Run from the repo root:   python src/models/vti_check.py
Runs submission.py first, then prints the tables only. On clean steady rows, all of 2024.
"""
import contextlib
import io
import os
import runpy

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
HERE = os.path.dirname(os.path.abspath(__file__))
MIN_ROWS = 30

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(HERE, "submission.py"))
c = g["c"][g["c"].index < "2025-01-01"].copy()
interp_disp, fit_p = g["interp_disp"], g["fit_p"]

month = c.index.month
c["period"] = np.select([c.reference_period, month == 7, np.isin(month, [8, 9]), np.isin(month, [10, 11]), month == 12],
                        ["ref", "Jul", "Aug-Sep", "Oct-Nov", "Dec"], "other")
c["speed_bin"] = pd.cut(c.speed_kn, [0, 9, 10, 11, 12, 13, 16], labels=["<9", "9-10", "10-11", "11-12", "12-13", "13+"])
c["power_bin"] = pd.cut(c.power_kw, [0, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 9000])
gps = c.stw_gps_kn > 5
c["vti_gps"] = np.nan   # same index with GPS speed minus current in place of the speed log
c.loc[gps, "vti_gps"] = c.vti[gps] * c.p_clean[gps] / np.exp(interp_disp(fit_p, c.stw_gps_kn[gps], c.disp_t[gps]))


def table(x, index, value):
    t = x.pivot_table(index=index, columns="period", values=value, aggfunc="median", observed=True)
    n = x.pivot_table(index=index, columns="period", values=value, aggfunc="size", observed=True)
    return t.where(n >= MIN_ROWS).round(2)


for laden, name in [(True, "LADEN"), (False, "BALLAST")]:
    x = c[c.laden == laden]
    ref = x[x.reference_period]
    print(f"\n===== {name}")
    print(x.groupby("period").agg(rows=("vti", "size"), disp_t=("disp_t", "median"), speed_kn=("speed_kn", "median"),
                                  power_kw=("power_kw", "median"), vti=("vti", "median")).round(2).to_string())
    print("\nVTI at equal speed\n" + table(x, "speed_bin", "vti").to_string())
    print("\nspeed at equal power\n" + table(x, "power_bin", "speed_kn").to_string())
    slope = np.polyfit(np.log(ref.speed_kn), np.log(ref.vti), 1)[0]
    print(f"\nreference period: VTI ~ speed^{slope:.2f}, {(ref.speed_kn < 12).sum()} rows below 12 kn")
    w = x.groupby(pd.Grouper(freq="W")).agg(vti=("vti", "median"), vti_gps=("vti_gps", "median"), speed_kn=("speed_kn", "median"),
                                            stw_gps_kn=("stw_gps_kn", "median"), me_rpm=("me_rpm", "median"),
                                            power_kw=("power_kw", "median"), wind_head_kn=("wind_head_kn", "median"),
                                            rows=("vti", "size"))
    w = w[w.rows >= 12]
    print("\nweekly, from 20 Jul\n" + w[w.index >= "2024-07-20"].round(2).to_string())

by_month = c.groupby([c.index.strftime("%Y-%m"), "laden"]).agg(speed_log=("vti", "median"), gps=("vti_gps", "median"), rows=("vti", "size"))
print("\nVTI by month, speed log vs GPS-based speed\n" + by_month[by_month.rows >= MIN_ROWS].round(2).to_string())
