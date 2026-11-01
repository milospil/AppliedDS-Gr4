"""Correlation of the operating variables with power, for report Section 2. What drives power?

Run from the repo root:   python src/correlation_drivers.py
Input:   data/processed/model_table.csv
Output:  plots4report/correlation_by_loading.png   laden (lower triangle) vs ballast (upper triangle)
         plots4report/correlation_drivers.png      raw and partial correlation with power, with week-bootstrap intervals

The sensor check before and after 10 Jul is in src/correlation_matrix.py.

Choices:
  rows        steady sailing, reference period (1 Jan - 23 Jun 2024), power from the torquemeter. The torquemeter is
              trusted there, so no estimated power enters, and the hull is assumed clean, so fouling is not mixed in.
              The calm-weather filter is NOT applied: it would remove the wind and wave variation we want to see.
  space       Pearson correlation after the same log transforms the models use (ln power, ln speed, ln rpm, ln displacement).
              Power is a power law of speed and rpm, so the relationship is linear in logs. Spearman is printed as a check.
  no p-values neighbouring 10-minute rows are near copies (lag-1 autocorrelation of power 0.99), so the real sample size
              is the number of weeks (22), not the number of rows. Intervals come from resampling whole weeks.
  partial     correlation with ln power after removing ln speed and ln displacement from both sides.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TABLE = os.path.join(ROOT, "data", "processed", "model_table.csv")
PLOTS = os.path.join(ROOT, "plots4report")
N_BOOT = 500
SEED = 1

# column, label, transform, group
VARIABLES = [
    ("power_kw", "ln power", "log", "Engine output"),
    ("fo_kgh", "ln fuel flow", "log", "Engine output"),
    ("me_rpm", "ln engine rpm", "log", "Engine output"),
    ("speed_kn", "ln speed through water", "log", "Speed"),
    ("stw_gps_kn", "ln speed (GPS - current)", "log", "Speed"),
    ("sog_kn", "ln speed over ground", "log", "Speed"),
    ("disp_t", "ln displacement", "log", "Loading"),
    ("trim_m", "trim (stern +)", None, "Loading"),
    ("wind_head_kn", "head wind (relative)", None, "Wind"),
    ("wind_side_kn", "side wind (relative)", None, "Wind"),
    ("hs_m", "wave height", None, "Waves"),
    ("wave_head_m", "head-sea wave height", None, "Waves"),
    ("wave_period_s", "wave period", None, "Waves"),
    ("current_along_kn", "current along heading", None, "Current"),
    ("sea_temp_c", "sea temperature", None, "Time proxies"),
    ("days", "days since 1 Jan", None, "Time proxies"),
]
COLS = [v[0] for v in VARIABLES]
LABEL = {v[0]: v[1] for v in VARIABLES}
GROUP = {v[0]: v[3] for v in VARIABLES}
DRIVERS = ["speed_kn", "disp_t", "trim_m", "wind_head_kn", "wind_side_kn", "hs_m", "wave_head_m", "wave_period_s",
           "current_along_kn", "sea_temp_c", "days"]
CONTROLS = ["speed_kn", "disp_t"]

# colours: the reference palette's blue <-> red diverging pair with a gray midpoint; blue and orange for laden / ballast
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
CMAP = LinearSegmentedColormap.from_list("div", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a89f", "#d03b3b", "#8e1f1f"])
C_ALL, C_LADEN, C_BALLAST = "#52514e", "#2a78d6", "#eb6834"


def load():
    t = pd.read_csv(TABLE, index_col="time", parse_dates=True)
    r = t[t.steady & t.reference_period & (t.power_source == "torquemeter")].copy()
    r["days"] = (r.index - pd.Timestamp("2024-01-01", tz=r.index.tz)).days
    for col, _, tr, _ in VARIABLES:
        if tr == "log":
            r[col] = np.log(r[col])
    r = r.dropna(subset=COLS)
    r["is_laden"] = r.laden.astype(bool)
    return r


def partial_r(d, target, controls, variables):
    X = np.c_[np.ones(len(d)), d[controls].values]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ty = res(d[target].values)
    return pd.Series({v: np.corrcoef(ty, res(d[v].values))[0, 1] for v in variables})


def week_bootstrap(d, fn, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    groups = {w: g for w, g in d.groupby("week")}
    weeks = list(groups)
    out = [fn(pd.concat([groups[w] for w in rng.choice(weeks, len(weeks))])) for _ in range(n)]
    out = pd.DataFrame(out)
    return out.quantile(0.05), out.quantile(0.95)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def group_bounds():
    """index ranges of each group, in the order of VARIABLES."""
    out, start = [], 0
    for i in range(1, len(COLS) + 1):
        if i == len(COLS) or GROUP[COLS[i]] != GROUP[COLS[start]]:
            out.append((GROUP[COLS[start]], start, i))
            start = i
    return out


def cell_text(ax, i, j, v):
    ax.text(j, i, f"{v:.2f}".replace("-", "\u2212"), ha="center", va="center", fontsize=6.6,
            color="white" if abs(v) > 0.62 else INK)


def draw_matrix(ax, lower, upper=None):
    """lower only: the empty first row and last column are cut away. With `upper`: the full square, grey diagonal."""
    n = len(COLS)
    show = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i > j:
                show[i, j] = lower.values[i, j]
            elif upper is not None and i < j:
                show[i, j] = upper.values[i, j]
    r0, c1 = (0, n) if upper is not None else (1, n - 1)
    sub = show[r0:, :c1]
    ax.imshow(sub, cmap=CMAP, vmin=-1, vmax=1)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if not np.isnan(sub[i, j]):
                cell_text(ax, i, j, sub[i, j])
            elif upper is not None and i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRID))
    for _, a, b in group_bounds():
        for k in (a, b):
            ax.axhline(k - r0 - 0.5, color="white", lw=2.4)
            ax.axvline(k - 0.5, color="white", lw=2.4)
    ax.set_xticks(range(c1))
    ax.set_yticks(range(sub.shape[0]))
    ax.set_xticklabels([LABEL[c] for c in COLS[:c1]], rotation=55, ha="right", rotation_mode="anchor", fontsize=8.5, color=INK)
    ax.set_yticklabels([LABEL[c] for c in COLS[r0:]], fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return r0, c1


def group_labels(ax, r0, c1):
    """group names to the right of the matrix, one bracket per group."""
    x = c1 - 0.35
    for name, a, b in group_bounds():
        top, bot = max(a - r0, 0) - 0.4, b - 1 - r0 + 0.4
        if bot < top:
            continue
        ax.plot([x, x], [top, bot], color=GRID, lw=1.4, clip_on=False)
        ax.text(x + 0.25, (top + bot) / 2, name, ha="left", va="center", fontsize=8, color=INK2, clip_on=False)


def colorbar(fig, rect, label="Pearson correlation r"):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-1, 1)), cax=cax, orientation="horizontal")
    cb.set_label(label, color=INK2, fontsize=8)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=7.5, colors=INK2, length=0)
    cb.outline.set_visible(False)


def fig_by_loading(r, Cl, Cb):
    fig = plt.figure(figsize=(10.4, 9.8))
    ax = fig.add_axes([0.2, 0.27, 0.6, 0.62])
    r0, c1 = draw_matrix(ax, Cl, upper=Cb)
    group_labels(ax, r0, c1)
    fig.text(0.02, 0.965, "Correlation of operating variables by loading condition, steady sailing in the reference period", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.02, 0.93, "lower triangle: laden", fontsize=9.5, color=C_LADEN, fontweight="bold")
    fig.text(0.25, 0.93, "upper triangle: ballast", fontsize=9.5, color=C_BALLAST, fontweight="bold")
    colorbar(fig, [0.34, 0.06, 0.3, 0.014])
    nl, nb = int(r.is_laden.sum()), int((~r.is_laden).sum())
    fig.text(0.02, 0.012, f"Laden: n = {nl:,} rows, {r[r.is_laden].week.nunique()} weeks.  Ballast: n = {nb:,} rows, "
             f"{r[~r.is_laden].week.nunique()} weeks.  Laden = displacement above 55,000 t.", fontsize=7.5, color=INK2)
    fig.savefig(os.path.join(PLOTS, "correlation_by_loading.png"), dpi=170)


def fig_drivers(r):
    pops = [("All", r, C_ALL), ("Laden", r[r.is_laden], C_LADEN), ("Ballast", r[~r.is_laden], C_BALLAST)]
    partial_vars = [v for v in DRIVERS if v not in CONTROLS]
    rows = DRIVERS
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.4), sharey=True)
    y = np.arange(len(rows))[::-1]
    res = {}
    for k, (name, d, col) in enumerate(pops):
        raw = d[DRIVERS + ["power_kw"]].corr().power_kw[DRIVERS]
        lo, hi = week_bootstrap(d, lambda x: x[DRIVERS + ["power_kw"]].corr().power_kw[DRIVERS])
        par = partial_r(d, "power_kw", CONTROLS, partial_vars)
        plo, phi = week_bootstrap(d, lambda x: partial_r(x, "power_kw", CONTROLS, partial_vars))
        res[name] = (raw, lo, hi, par, plo, phi)
        off = (1 - k) * 0.22
        for ax, val, a, b, vs in ((axes[0], raw, lo, hi, rows), (axes[1], par, plo, phi, partial_vars)):
            yy = np.array([y[rows.index(v)] for v in vs]) + off
            ax.hlines(yy, a[vs], b[vs], color=col, lw=1.6, alpha=0.55)
            ax.plot(val[vs], yy, "o", color=col, ms=5.5, mec="white", mew=0.8, label=f"{name} ({d.week.nunique()} weeks)")
    for ax, t in zip(axes, ["Correlation with ln power", "Partial correlation with ln power\n(ln speed and ln displacement removed)"]):
        style(ax)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel(t, fontsize=8.5, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[v] for v in rows], fontsize=8.5, color=INK)
    for v in CONTROLS:
        axes[1].text(0, y[rows.index(v)], "held fixed", ha="center", va="center", fontsize=7.5, color=MUTED, style="italic")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.55, 1.0))
    fig.text(0.01, 0.005, "Dots: correlation. Lines: 90 % interval from resampling whole weeks (500 draws). "
             "An interval that crosses 0 means we cannot tell the effect from zero with this many weeks.", fontsize=7, color=INK2)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.17, wspace=0.14)
    fig.savefig(os.path.join(PLOTS, "correlation_drivers.png"), dpi=170)
    return res


def main():
    r = load()
    print(f"rows {len(r):,}, weeks {r.week.nunique()} (laden {int(r.is_laden.sum()):,} / {r.week[r.is_laden].nunique()} wk, "
          f"ballast {int((~r.is_laden).sum()):,} / {r.week[~r.is_laden].nunique()} wk)")
    C = r[COLS].corr()
    Cl, Cb = r[r.is_laden][COLS].corr(), r[~r.is_laden][COLS].corr()
    fig_by_loading(r, Cl, Cb)
    res = fig_drivers(r)

    sp = r[COLS].corr(method="spearman")
    diff = (sp - C).abs().where(np.triu(np.ones(C.shape), 1).astype(bool)).stack()
    print("\nlargest Pearson / Spearman gaps (pooled):")
    for (a, b), v in diff.sort_values(ascending=False).head(6).items():
        print(f"  {LABEL[a]} ~ {LABEL[b]}: pearson {C.loc[a, b]:+.2f}, spearman {sp.loc[a, b]:+.2f}")
    print("\npairs with |r| > 0.7 (pooled), excluding the output group:")
    inputs = [c for c in COLS if GROUP[c] != "Engine output"]
    for i, a in enumerate(inputs):
        for b in inputs[i + 1:]:
            if abs(C.loc[a, b]) > 0.7:
                print(f"  {LABEL[a]} ~ {LABEL[b]}: {C.loc[a, b]:+.2f}")
    f = ["speed_kn", "disp_t", "trim_m", "wind_head_kn", "wind_side_kn", "hs_m", "wave_head_m", "wave_period_s", "current_along_kn"]
    Z = r[f]
    vif = pd.Series(np.diag(np.linalg.inv(np.corrcoef(((Z - Z.mean()) / Z.std()).values.T))), index=f)
    print("\nVIF of the model-style inputs:\n" + vif.round(1).to_string())
    print("\nraw / partial correlation with ln power (90 % week-bootstrap interval):")
    for name, (raw, lo, hi, par, plo, phi) in res.items():
        print(f"-- {name}")
        for v in DRIVERS:
            p = f"{par[v]:+.2f} [{plo[v]:+.2f}, {phi[v]:+.2f}]" if v in par.index else "held fixed"
            print(f"  {LABEL[v]:28s} raw {raw[v]:+.2f} [{lo[v]:+.2f}, {hi[v]:+.2f}]   partial {p}")


if __name__ == "__main__":
    main()
