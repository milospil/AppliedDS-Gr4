"""Speed-power curves per draft for the clean and the current condition (the challenge's main task).

Run from the repo root:   python src/models/draft_curves.py
Input:  data/processed/model_table.csv, reference/towing_with_psv.csv (through submission.py)
Output: data/processed/curves/Kensas_<Condition>_<Draft>.csv (Speed [kn], Power [KW], RPM, as in the template;
        Current also has Power low / Power high), plots4report/speed_power_curves.png

Three curves per draft, all for calm water (no wind or waves) and the typical trim of the reference period:
  - Towing tank: the clean-hull model-test curve with PSV, the newbuild reference.
  - Reference: the ship as it was in the reference period (1 Jan - 23 Jun 2024) = towing tank x the correction
    fitted in submission.py. It includes the model-to-full-scale difference, so it is the "clean" curve of our
    fouling indicator.
  - Current: the latest voyage in each loading condition (laden 30 Nov - 28 Dec, ballast 16 Oct - 15 Nov) =
    Reference x (1 + median fouling of that voyage). This assumes fouling adds the same share of power at every
    speed, and the laden factor (measured at about 59,000 t) is also used for design and scantling.
    The band is the 90 % interval of that median from resampling whole weeks.
Speeds outside what the ship sailed in steady conditions (5-95 % of clean rows of that loading) are dashed.
"""
import contextlib
import io
import os
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(ROOT, "data", "processed", "curves")
PLOTS = os.path.join(ROOT, "plots4report")
LATEST = {True: ("2024-11-30", "2024-12-29"), False: ("2024-10-16", "2024-11-16")}   # latest voyage per loading
GRID = np.arange(8.0, 15.0 + 1e-9, 0.25)
N_BOOT = 1000
COLORS = {"Towing tank": "#7f7f7f", "Reference": "tab:blue", "Current": "tab:orange"}

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(HERE, "submission.py"))   # clean curves, fitted correction and fouling per row
m, corr, rpm_corr, F = g["m"], g["corr"], g["rpm_corr"], g["F"]
interp_disp, fit_p, fit_n, DISP, tow = g["interp_disp"], g["fit_p"], g["fit_n"], g["DISP"], g["tow"]
TANK_MIN = {k: tow[tow.cond == k].vs.min() for k in DISP}
TANK_MAX = {k: tow[tow.cond == k].vs.max() for k in DISP}


def calm_X(laden, trim):
    """Correction inputs for calm water at a given loading and trim, one row per grid speed."""
    row = {"laden": float(laden), "trim_m": trim, "wind_head2": 0.0, "wind_side_kn": 0.0, "hs2": 0.0, "wave_head_m": 0.0}
    return pd.DataFrame([row] * len(GRID))[F]


def week_interval(x):
    """Median with a 90 % interval from resampling whole weeks."""
    weeks = [w.to_numpy() for _, w in x.groupby(pd.Grouper(freq="W")) if len(w)]
    rng = np.random.default_rng(0)
    boot = [np.median(np.concatenate([weeks[i] for i in rng.integers(0, len(weeks), len(weeks))])) for _ in range(N_BOOT)]
    return np.median(x), np.quantile(boot, 0.05), np.quantile(boot, 0.95), len(weeks)


def main():
    os.makedirs(OUT, exist_ok=True)
    clean = m[m.clean & (m.index < "2025-01-01")]
    ref = m[m.steady & m.reference_period]
    trim = ref.groupby("laden").trim_m.median()
    sailed = clean.groupby("laden").speed_kn.quantile([0.05, 0.95]).unstack()

    shift, rpm_shift = {}, {}
    for laden, (a, b) in LATEST.items():
        x = clean[(clean.laden == laden) & (clean.index >= a) & (clean.index < b)]
        shift[laden] = week_interval(x.fouling_pct / 100)
        rpm_shift[laden] = np.median(np.log(x.shaft_rpm / x.rpm_exp).dropna())
        med, lo, hi, n = shift[laden]
        print(f"current {'laden' if laden else 'ballast'} ({a} to {b}): {len(x):,} clean rows in {n} weeks, "
              f"power +{100 * med:.1f} % (90 % interval {100 * lo:.1f} to {100 * hi:.1f}), rpm {100 * rpm_shift[laden]:+.1f} %")

    panels = []
    for draft, disp in DISP.items():
        laden = disp > 55000
        d = np.full_like(GRID, disp)
        X = calm_X(laden, trim[laden])
        tank = np.exp(interp_disp(fit_p, GRID, d))
        tank_rpm = np.exp(interp_disp(fit_n, GRID, d))
        reference = tank * np.exp(corr.predict(X))
        reference_rpm = tank_rpm * np.exp(rpm_corr.predict(X))
        med, lo, hi, _ = shift[laden]
        current_rpm = reference_rpm * np.exp(rpm_shift[laden])
        frames = {
            "TowingTank": pd.DataFrame({"Speed [kn]": GRID, "Power [KW]": tank, "RPM": tank_rpm}),
            "Reference": pd.DataFrame({"Speed [kn]": GRID, "Power [KW]": reference, "RPM": reference_rpm}),
            "Current": pd.DataFrame({"Speed [kn]": GRID, "Power [KW]": reference * (1 + med), "RPM": current_rpm,
                                     "Power low": reference * (1 + lo), "Power high": reference * (1 + hi)}),
        }
        for cond, f in frames.items():
            f.round({"Speed [kn]": 2, "Power [KW]": 0, "RPM": 1, "Power low": 0, "Power high": 0}).to_csv(
                os.path.join(OUT, f"Kensas_{cond}_{draft.capitalize()}.csv"), index=False)
        panels.append((draft, disp, laden, frames))
        at = {s: np.interp(s, GRID, reference) for s in (11.0, 12.0, 13.0)}
        print(f"{draft.capitalize():9s} {disp:,} t: reference / towing tank = "
              + ", ".join(f"{s:g} kn {at[s] / np.interp(s, GRID, tank):.2f}" for s in at))

    # ---- figure: one panel per draft
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (draft, disp, laden, frames) in zip(axes, panels):
        lo_v, hi_v = sailed.loc[laden, 0.05], sailed.loc[laden, 0.95]
        inside = (GRID >= lo_v) & (GRID <= hi_v)
        tank_in = (GRID >= TANK_MIN[draft]) & (GRID <= TANK_MAX[draft])
        for cond, key, rng in [("Towing tank", "TowingTank", tank_in), ("Reference", "Reference", inside),
                               ("Current", "Current", inside)]:
            f = frames[key]
            ax.plot(f["Speed [kn]"], f["Power [KW]"], color=COLORS[cond], lw=1, ls="--")
            ax.plot(f["Speed [kn]"].where(rng), f["Power [KW]"].where(rng), color=COLORS[cond], lw=2, label=cond)
        cur = frames["Current"]
        ax.fill_between(cur["Speed [kn]"], cur["Power low"], cur["Power high"], color=COLORS["Current"], alpha=0.15,
                        lw=0, label="Current, 90 % interval")
        ax.set_title(f"{draft.capitalize()} ({disp:,} t)", loc="left", fontsize=11)
        ax.set_xlabel("Speed through water [kn]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Delivered power, calm water [kW]")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Speed-power curves per draft. Solid: inside the towing-tank range (tank) or the speeds the ship "
                 "sailed (reference, current); dashed: extrapolated", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "speed_power_curves.png"), dpi=150)

    # ---- check: reference curve against clean reference rows near each draft, with their weather and trim
    # taken out using the same correction, so the ratio should be close to 1
    print("\ncheck, measured (corrected to calm water and typical trim) / reference curve, clean reference rows "
          "within 5 % of each draft's displacement:")
    for draft, disp, laden, frames in panels:
        x = clean[clean.reference_period & ((clean.disp_t / disp - 1).abs() < 0.05)]
        if len(x) < 30:
            print(f"  {draft.capitalize()}: only {len(x)} rows, no check")
            continue
        X = x[F].astype(float)
        calm = calm_X(laden, trim[laden]).iloc[:1].to_numpy()
        to_calm = np.exp(-(X.to_numpy() - calm) @ corr.coef_)
        curve = np.interp(x.speed_kn, GRID, frames["Reference"]["Power [KW]"])
        scale = np.exp(interp_disp(fit_p, x.speed_kn, np.full(len(x), disp)) - interp_disp(fit_p, x.speed_kn, x.disp_t))
        print(f"  {draft.capitalize()}: {len(x):,} rows, median ratio {np.median(x.power_kw * to_calm * scale / curve):.3f}")


if __name__ == "__main__":
    main()
