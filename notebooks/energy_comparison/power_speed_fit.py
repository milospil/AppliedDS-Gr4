"""
Fit a power law  P = k * V^n  to speed-power data and plot it.

Two fits are made and compared:
  1. log-space linear regression (scipy.stats.linregress)  -> minimises relative error
  2. non-linear least squares     (scipy.optimize.curve_fit) -> minimises absolute error (kW)

Plots:
  left  : normal axes, scatter + both fitted curves
  right : log-log axes, where a power law is a straight line (diagnostic)

Only rows with V > 0 and P > 0 are used (the logarithm is undefined otherwise).
Set MIN_SPEED_KN to your docking threshold to fit on sailing data only.

Usage with a DataFrame already in memory:
    from speed_power_fit import fit_power_law, plot_fit
    res = fit_power_law(df)
    plot_fit(df, res)
"""
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

SPEED = "stw_kn"
POWER = "shaft_kw"
MIN_SPEED_KN = 2          # docking threshold; set to 0 to use all positive speeds
OUT_PNG = "speed_power_fit.png"
DEGREE = 2

def power_law(v, k, n):
    return k * v ** n

def poly(v, *coef):
    """Polynomial with coefficients highest power first: a·V^n + ... + const."""
    return np.polyval(coef, v)

def _equation(coef: np.ndarray) -> str:
    parts = []
    for i, c in enumerate(coef):
        pw = DEGREE - i
        var = f"·V^{pw}" if pw > 1 else ("·V" if pw == 1 else "")
        if i == 0:
            parts.append(f"{'-' if c < 0 else ''}{abs(c):.4g}{var}")
        else:
            parts.append(f" {'-' if c < 0 else '+'} {abs(c):.4g}{var}")
    return "P = " + "".join(parts)

def _xy(df: pl.DataFrame, SPEED = SPEED) -> tuple[np.ndarray, np.ndarray]:
    d = (df.select(pl.col(SPEED).fill_nan(None), pl.col(POWER).fill_nan(None))
           .drop_nulls()
           .filter(pl.col(SPEED) > MIN_SPEED_KN))
    return d[SPEED].to_numpy(), d[POWER].to_numpy()

def _xyln(df: pl.DataFrame, SPEED = SPEED) -> tuple[np.ndarray, np.ndarray]:
    x, y = _xy(df, SPEED)
    return np.log(x), np.log(y)

def fit_log_space(df: pl.DataFrame, SPEED = SPEED) -> dict:
    x, y = _xy(df, SPEED)
    if len(x) < 3:
        raise ValueError("Not enough rows with positive speed and power to fit.")

    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
    k = np.exp(intercept)
    n = slope
    pred = power_law(x, k, n)
    res = {
        "k": k, "n": n, "R2": r_value ** 2,
        "RMSE": np.sqrt(np.mean((y - pred) ** 2)),
        "n_points": len(x),
    }
    print(f"Points used: {len(x)}")
    print(f"Fit: P = {k:.4g}·V^{n:.4g}")
    print(f"R² = {res['R2']:.4f},  RMSE = {res['RMSE']:.1f} kW")
    return res

def fit_poly(df: pl.DataFrame, SPEED= SPEED) -> dict:
    x, y = _xy(df, SPEED)
    lnx, _ = _xyln(df, SPEED)
    if len(x) < 3:
        raise ValueError("Not enough rows with positive speed and power to fit.")

    p0 = np.polyfit(x, y, DEGREE)   
    coef, cov = curve_fit(poly, x, y, p0=p0)
    err = np.sqrt(np.diag(cov))   
    pred = poly(x, *coef)
    res = {
        "coef": coef, "cov": cov, "err": err, "n_points": len(x),
        "R2": 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2),
        "RMSE": np.sqrt(np.mean((y - pred) ** 2)),
    }
    names = [f"V^{DEGREE - i}" if DEGREE - i > 1 else ("V" if DEGREE - i == 1 else "const")
             for i in range(DEGREE + 1)]
    print(f"Points used: {len(x)}")
    print(f"Fit: {_equation(coef)}")
    for nm, c, e in zip(names, coef, err):
        print(f"  {nm:>6}: {c:12.4g} ± {e:.3g}")
    print(f"R² = {res['R2']:.4f},  RMSE = {res['RMSE']:.1f} kW")
    return res

def fit_polyln(df: pl.DataFrame, SPEED= SPEED) -> dict:
    x, y = _xyln(df, SPEED)
    if len(x) < 3:
        raise ValueError("Not enough rows with positive speed and power to fit.")

    p0 = np.polyfit(x, y, DEGREE)   
    coef, cov = curve_fit(poly, x, y, p0=p0)
    err = np.sqrt(np.diag(cov))   
    pred = poly(x, *coef)
    res = {
        "coef": coef, "cov": cov, "err": err, "n_points": len(x),
        "R2": 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2),
        "RMSE": np.sqrt(np.mean((y - pred) ** 2)),
    }
    names = [f"V^{DEGREE - i}" if DEGREE - i > 1 else ("V" if DEGREE - i == 1 else "const")
             for i in range(DEGREE + 1)]
    print(f"Points used: {len(x)}")
    print(f"Fit: {_equation(coef)}")
    for nm, c, e in zip(names, coef, err):
        print(f"  {nm:>6}: {c:12.4g} ± {e:.3g}")
    print(f"R² = {res['R2']:.4f},  RMSE = {res['RMSE']:.1f} kW")
    return res

def plot_fit(df: pl.DataFrame, res: dict, out: str = OUT_PNG):
    x, y = _xy(df)
    xs = np.linspace(x.min(), x.max(), 200)
    lg, nl = res["log"], res["nls"]
    curves = [
        (lg, "tab:red", "-",  f"Log-space fit: P = {lg['k']:.3g}·V^{lg['n']:.2f}  (R² = {lg['R2']:.3f})"),
        (nl, "tab:blue", "--", f"Direct fit: P = {nl['k']:.3g}·V^{nl['n']:.2f}  (R² = {nl['R2']:.3f})"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax in (ax1, ax2):
        ax.scatter(x, y, s=4, alpha=0.3, color="0.5", label=f"Data (n = {len(x)})")
        for r, col, ls, lab in curves:
            ax.plot(xs, power_law(xs, r["k"], r["n"]), color=col, ls=ls, lw=2, label=lab)
        ax.set_xlabel("Speed (kn)")
        ax.set_ylabel("Power (kW)")

    ax1.set_title("Speed–power, normal axes")
    ax1.legend(loc="upper left", fontsize=9)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    for axis in (ax2.xaxis, ax2.yaxis):                  # plain numbers, not 8×10^0
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    step = max(1, int(np.ceil((x.max() - x.min()) / 10)))  # whole-knot ticks, ≤ ~10 labels
    ax2.xaxis.set_minor_locator(plt.NullLocator())
    ax2.set_xticks(np.arange(np.floor(x.min()), np.ceil(x.max()) + 1, step))
    ax2.set_xlim(x.min() * 0.97, x.max() * 1.03)
    ax2.set_title("Log–log axes (power law = straight line)")

    fig.tight_layout()
    #fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    return fig


if __name__ == "__main__":
    #data = pl.read_parquet(FILE)
    #result = fit_power_law(data)
    #plot_fit(data, result)
    pass