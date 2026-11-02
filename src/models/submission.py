"""Fill the two challenge templates (Templates/BillTsars_Model1_*.csv) from the first-pass model table.

Run from the repo root:   python src/models/submission.py
Reads data/processed/model_table.csv and reference/towing_with_psv.csv, writes to data/processed/submission/:
  - Kensas_Model1_CleanBaselineScantling.csv: clean-hull speed / power / rpm at scantling draft, 5 to 16 kn
  - Kensas_Model1_Timeseries.csv: one row per 10 minutes with expected power and rpm, fouling, speed loss,
    uncertainty and flags
and data/processed/vti_daily.csv, plots4report/fouling_trend.png, plots4report/vti_trend.png

Method
  1. Clean hull = towing-tank curves with PSV (P = a V^b per draft, interpolated on log displacement)
  2. Expected power = clean curve x a correction fitted on steady sailing in the reference period
     (1 Jan - 23 Jun 2024): log(P / P_clean) ~ laden + trim + wind + waves. This absorbs model-to-full-scale
     offsets and the weather, so fouling is measured against the ship as it was in the reference period
  3. Fouling [%Power] = 100 (P / P_expected - 1); speed loss = speed the reference hull would make at the
     measured power minus measured speed, using the local exponent b of the clean curve
  4. Uncertainty = robust spread of the residuals in the reference period, plus the error of the
     virtual power (scavenge air pressure + engine rpm) where the torquemeter is not used
  5. VTI-style index, a second baseline (idea from DNV-RP-0675): power with the fitted weather terms removed, divided
     by the clean curve at the same speed and displacement. 1.00 = clean hull. It is not in the template.
     It uses the same fit as fouling, so it is not an independent check, and sea temperature is not corrected
  Speed is `speed_kn` from prepare_features.py (speed log, offset removed on 29 Jul - 16 Oct, DQ-05).
  Jan 2025 (blind month) has no power, torque or engine rpm at sea (DQ-07), so its rows carry speed only:
  no expected power, fouling or speed loss can be computed there. The 'power source' column says 'none'.
  The wind terms use the anemometer as measured; its mast-height bias (DQ-27) is constant all year, so the
  reference-period fit absorbs it.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
PROCESSED = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(PROCESSED, "submission")
PLOTS = os.path.join(ROOT, "plots4report")
DISP = {"BALLAST": 36134, "DESIGN": 73346, "SCANTLING": 86941}  # towing tank report
KEYS = ["BALLAST", "DESIGN", "SCANTLING"]
WEATHER = ["wind_head2", "wind_side_kn", "hs2", "wave_head_m"]
VIRTUAL_POWER_ERR = 0.01  # holdout MAPE of the scavenge pressure + rpm power model (prepare_features.py)
OUTLIER_SD = 3.5

# ---- 1. clean-hull curves: P = a V^b and rpm = c V^d per draft, interpolated on log displacement
tow = pd.read_csv(os.path.join(ROOT, "reference", "towing_with_psv.csv"))
fit_p = {k: np.polyfit(np.log(g.vs), np.log(g.pd_kw), 1) for k, g in tow.groupby("cond")}   # (b, log a)
fit_n = {k: np.polyfit(np.log(g.vs), np.log(g.rpm), 1) for k, g in tow.groupby("cond")}
xs = np.log([DISP[k] for k in KEYS])


def interp_disp(fits, v, disp):
    """log of the curve value at speed v and displacement disp, linear in log disp (extrapolates)."""
    lv, ld = np.log(np.asarray(v, float)), np.log(np.asarray(disp, float))
    ys = np.stack([fits[k][1] + fits[k][0] * lv for k in KEYS])
    lo = ld < xs[1]  # use ballast-design segment below design, design-scantling above
    i0, i1 = np.where(lo, 0, 1), np.where(lo, 1, 2)
    cols = np.arange(len(lv))
    w = (ld - xs[i0]) / (xs[i1] - xs[i0])
    return ys[i0, cols] + w * (ys[i1, cols] - ys[i0, cols])


def exponent(disp):
    """local speed exponent b of the clean power curve at this displacement."""
    bs = np.array([fit_p[k][0] for k in KEYS])
    return np.interp(np.log(disp), xs, bs)


os.makedirs(OUT, exist_ok=True)
grid = np.arange(5.0, 16.0 + 1e-9, 0.25)
d_s = np.full_like(grid, DISP["SCANTLING"])
pd.DataFrame({
    "Speed [kn]": grid.round(2),
    "Power [KW]": np.exp(interp_disp(fit_p, grid, d_s)).round(0).astype(int),
    "RPM": np.exp(interp_disp(fit_n, grid, d_s)).round(1),
}).to_csv(os.path.join(OUT, "Kensas_Model1_CleanBaselineScantling.csv"), index=False)

# ---- 2. reference-period correction on top of the clean curve
m = pd.read_csv(os.path.join(PROCESSED, "model_table.csv"), index_col="time", parse_dates=True)
v, p, disp = m.speed_kn, m.power_kw, m.disp_t
usable = (v > 5) & (p > 500) & disp.notna()
m["p_clean"] = np.nan
m.loc[usable, "p_clean"] = np.exp(interp_disp(fit_p, v[usable], disp[usable]))
m["rpm_clean"] = np.nan
m.loc[usable, "rpm_clean"] = np.exp(interp_disp(fit_n, v[usable], disp[usable]))

F = ["laden", "trim_m", "wind_head2", "wind_side_kn", "hs2", "wave_head_m"]
X = m[F].astype(float)
has_x = X.notna().all(axis=1)
y = np.log(p / m.p_clean)
train = m.steady & m.reference_period & usable & has_x
corr = HuberRegressor(max_iter=500).fit(X[train], y[train])
rpm_corr = HuberRegressor(max_iter=500).fit(X[train], np.log(m.shaft_rpm / m.rpm_clean)[train])

ok = usable & has_x
m["p_exp"] = np.nan
m["rpm_exp"] = np.nan
m.loc[ok, "p_exp"] = m.p_clean[ok] * np.exp(corr.predict(X[ok]))
m.loc[ok, "rpm_exp"] = m.rpm_clean[ok] * np.exp(rpm_corr.predict(X[ok]))

# ---- 3. fouling and speed loss
ratio = p / m.p_exp
m["fouling_pct"] = 100 * (ratio - 1)
b = pd.Series(np.nan, index=m.index)
b[ok] = exponent(disp[ok])
m["speed_loss_kn"] = v * ratio ** (1 / b) - v   # + = slower than the reference hull at this power

# ---- VTI-style index: weather removed, relative to the clean curve
weather = sum(corr.coef_[F.index(f)] * X[f] for f in WEATHER)
m["vti"] = p * np.exp(-weather) / m.p_clean
m["below_tank"] = v < np.interp(np.log(disp), xs, [tow[tow.cond == k].vs.min() for k in KEYS])   # curve extrapolated

# ---- 4. uncertainty and flags
res = np.log(ratio)
ref_res = res[train]
sd_ref = 1.4826 * (ref_res - ref_res.median()).abs().median()      # robust sd, log units ~ fraction
sd_row = np.where(m.power_source == "torquemeter", sd_ref, np.hypot(sd_ref, VIRTUAL_POWER_ERR))
sd_row = np.where(ok, sd_row, np.nan)
m["unc_pct"] = 100 * sd_row
m["unc_kn"] = v * (np.exp(sd_row) ** (1 / b) - 1)

# outlier: far from a 3-day rolling median of steady rows, or sensors disagree while the torquemeter is trusted
roll = res.where(m.steady).rolling("3D", min_periods=12).median()
m["outlier"] = ok & (((res - roll).abs() > OUTLIER_SD * sd_row)
                     | (m.torque_trusted & ((m.torque_ratio - 1).abs() > 0.15))
                     | (m.log_bias_kn.abs() > 2))
m["bad_weather"] = ~m.calm.fillna(False)
m["clean"] = ok & m.steady & ~m.bad_weather & ~m.outlier

out = pd.DataFrame({
    "Timestamp": m.index.tz_convert(None).strftime("%Y-%m-%dT%H:%M") if m.index.tz else m.index.strftime("%Y-%m-%dT%H:%M"),
    "Speed [kn]": v.round(2),
    "Power [KW]": p.round(0),
    "RPM": m.shaft_rpm.round(1),
    "Power Expected": m.p_exp.round(0),
    "RPM Expected": m.rpm_exp.round(1),
    "Fouling [%Power]": m.fouling_pct.round(2),
    "fouling Speed loss": m.speed_loss_kn.round(3),
    "Uncertainty %": m.unc_pct.round(2),
    "Uncertainty[kn]": m.unc_kn.round(3),
    "clean ": m.clean,
    "bad weather": m.bad_weather,
    "outlier": m.outlier,
    # the template's free columns AAA / BBB / XXX
    "steady": m.steady,
    "power source": np.where(p.notna(), m.power_source, "none"),   # torquemeter / scavenge / none (e.g. Jan 2025)
    "laden": m.laden,
})
out.to_csv(os.path.join(OUT, "Kensas_Model1_Timeseries.csv"), index=False)

# ---- summary
c = m[m.clean]
print("reference fit on %d rows, robust residual sd %.1f%%" % (train.sum(), 100 * sd_ref))
print("correction coefficients:", dict(zip(F, corr.coef_.round(4))), "intercept %.3f" % corr.intercept_)
print("rows: %d total, %d with expected power, %d clean" % (len(m), ok.sum(), m.clean.sum()))
blind = m.index >= pd.Timestamp("2025-01-01", tz=m.index.tz)
print("Jan 2025: %d rows, %d moving above 5 kn, %d with expected power (no power or engine rpm at sea, DQ-07)"
      % (blind.sum(), (blind & (v > 5)).sum(), (blind & ok).sum()))
mon = c.groupby([c.index.strftime("%Y-%m"), "laden"]).agg(
    fouling_pct=("fouling_pct", "median"), speed_loss_kn=("speed_loss_kn", "median"), vti=("vti", "median"),
    n=("fouling_pct", "size"))
print(mon[mon.n >= 30].round(2).to_string())
print("VTI on clean rows: %.0f%% below the lowest towing-tank speed (curve extrapolated)" % (100 * c.below_tank.mean()))

# ---- daily trend plot of clean rows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
day = c.groupby(c.index.normalize()).agg(f=("fouling_pct", "median"), u=("unc_pct", "median"),
                                         n=("fouling_pct", "size"), laden=("laden", "mean"))
day = day[day.n >= 6]
day["se"] = day.u / np.sqrt(day.n)
fig, ax = plt.subplots(figsize=(11, 4.5))
for lad, col, lab in [(True, "tab:blue", "laden"), (False, "tab:orange", "ballast")]:
    d = day[(day.laden > 0.5) == lad]
    ax.errorbar(d.index, d.f, yerr=2 * d.se, fmt="o", ms=3, color=col, alpha=0.7, label=lab)
roll = day.f.rolling(15, center=True, min_periods=5).median()
ax.plot(day.index, roll, "k-", lw=2, label="15-day rolling median")
ax.axhline(0, color="grey", lw=0.8)
ax.axvspan(day.index.min(), pd.Timestamp("2024-06-24", tz=day.index.tz), color="green", alpha=0.07, label="reference period")
ax.axvline(pd.Timestamp("2024-07-10", tz=day.index.tz), color="red", ls="--", lw=1, label="power from scavenge pressure after")
ax.set_ylabel("excess power vs reference hull [%]")
ax.set_title("Kensas: daily fouling indicator, clean steady rows (error bars = 2 standard errors)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fouling_trend.png"), dpi=130)

# ---- daily VTI
c_vti = c.groupby([c.index.normalize(), "laden"]).agg(vti=("vti", "median"), below_tank=("below_tank", "mean"),
                                                      n=("vti", "size"))
c_vti = c_vti[c_vti.n >= 6].round(3)
c_vti.to_csv(os.path.join(PROCESSED, "vti_daily.csv"))
fig, ax = plt.subplots(figsize=(11, 4.5))
for lad, col, lab in [(True, "tab:blue", "laden"), (False, "tab:orange", "ballast")]:
    d = c_vti.xs(lad, level="laden")
    ax.plot(d.index, d.vti, "o", ms=3, color=col, alpha=0.7, label=lab)
ax.axhline(1, color="grey", lw=0.8)
ax.axvspan(day.index.min(), pd.Timestamp("2024-06-24", tz=day.index.tz), color="green", alpha=0.07, label="reference period")
ax.axvline(pd.Timestamp("2024-07-10", tz=day.index.tz), color="red", ls="--", lw=1, label="power from scavenge pressure after")
ax.set_ylabel("VTI-style index (1.00 = clean hull)")
ax.set_title("Kensas: power relative to the clean hull, weather removed, clean steady rows (daily median)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "vti_trend.png"), dpi=130)
