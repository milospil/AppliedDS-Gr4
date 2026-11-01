"""Correlation matrix of the main signals, for the report.

Run from the repo root:   python src/correlation_matrix.py
Reads data/processed/model_table.csv, writes plots4report/correlation_matrix.png and
data/processed/correlation_reference.csv

Spearman (rank) correlation on steady sailing rows, because the relations are curved (P ~ V^3).
Two periods side by side: while the torquemeter is trusted, and after it. Virtual power is left out: it is
built from scavenge pressure and rpm, so its correlation with them says nothing new.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "..", "data", "processed")
PLOTS = os.path.join(HERE, "..", "plots4report")
COLS = {"power_meas_kw": "Torquemeter power", "scav_bar": "Scavenge pressure", "fo_kgh": "Fuel flow", "me_rpm": "Engine rpm",
        "speed_kn": "Speed through water", "sog_kn": "Speed over ground", "disp_t": "Displacement", "trim_m": "Trim",
        "wind_head_kn": "Head wind", "wind_side_kn": "Side wind", "hs_m": "Wave height", "wave_period_s": "Wave period",
        "sea_temp_c": "Sea temperature"}
CMAP = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])
INK, INK2 = "#0b0b0b", "#52514e"

t = pd.read_csv(os.path.join(PROCESSED, "model_table.csv"), index_col="time", parse_dates=True)
s = t[t.steady & (t.index < "2025-01-01")]
periods = [("Torquemeter trusted (Jan to 9 Jul)", s[s.torque_trusted]), ("After 10 Jul", s[~s.torque_trusted])]

fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")
for ax, (title, part) in zip(axes, periods):
    c = part[list(COLS)].corr(method="spearman")
    ax.imshow(c, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(COLS)), COLS.values(), rotation=45, ha="right", color=INK2, fontsize=9)
    ax.set_yticks(range(len(COLS)), COLS.values(), color=INK2, fontsize=9)
    for i in range(len(COLS)):
        for j in range(len(COLS)):
            if not np.isnan(c.iloc[i, j]):
                ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(c.iloc[i, j]) > 0.7 else INK)
    ax.set_title(f"{title}, {len(part):,} rows", loc="left", color=INK, fontsize=11)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    print(title, f"{len(part):,} rows")
    print(c.round(2).to_string())
    if part is periods[0][1]:
        c.round(3).to_csv(os.path.join(PROCESSED, "correlation_reference.csv"))
fig.suptitle("Spearman correlation on steady sailing: blue = negative, red = positive", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "correlation_matrix.png"), dpi=150)
