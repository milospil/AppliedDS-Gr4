"""Speed baseline (main model): clean-reference speed from power, loading and weather.

Run from the repo root:   python src/models/speed_baseline.py
Input:  data/processed/model_table.csv
Output: data/processed/speed_loss_daily.csv  (date, speed_loss_pct, laden, n_rows)

Model: log-linear regression, log(speed) ~ log(power) + log(displacement) + trim + wind + waves,
trained on steady sailing in the reference period (1 Jan to 23 Jun 2024).
Validated with GroupKFold by ISO week, never random splits (neighbouring 10-min rows are near copies).
sea_temp_c, month and dates are NOT inputs: they act as a clock and would absorb the fouling trend.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "..", "..", "data", "processed")
FEATURES = ["log_power", "log_disp", "trim_m", "wind_head2", "wind_side_kn", "hs2", "wave_head_m"]
TARGET = "log_speed"
N_FOLDS = 6


def mape(y, yhat):
    return 100 * np.mean(np.abs(yhat - y) / y)


def main():
    t = pd.read_csv(os.path.join(PROCESSED, "model_table.csv"), index_col="time", parse_dates=True)
    usable = t.steady & t[FEATURES + [TARGET]].notna().all(axis=1)
    ref = t[usable & t.reference_period]
    print(f"reference rows: {len(ref):,} in {ref.week.nunique()} weeks (laden {ref.laden.sum():,}, ballast {(~ref.laden).sum():,})")

    # ---- week-based cross-validation
    speed = np.exp(ref[TARGET])
    pred = pd.Series(np.nan, index=ref.index)
    rows = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(ref, groups=ref.week), 1):
        a, b = ref.iloc[tr], ref.iloc[te]
        p = np.exp(LinearRegression().fit(a[FEATURES], a[TARGET]).predict(b[FEATURES]))
        pred.iloc[te] = p
        lad = b.laden.values
        rows.append(dict(fold=k, weeks=b.week.nunique(), rows=len(b), MAPE=mape(speed.iloc[te], p),
                         MAPE_laden=mape(speed.iloc[te][lad], p[lad]) if lad.any() else np.nan,
                         MAPE_ballast=mape(speed.iloc[te][~lad], p[~lad]) if (~lad).any() else np.nan))
    print("\nspeed MAPE (%) per fold, GroupKFold by week")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"overall: {mape(speed, pred):.2f}%  | laden {mape(speed[ref.laden], pred[ref.laden]):.2f}%"
          f"  | ballast {mape(speed[~ref.laden], pred[~ref.laden]):.2f}%")

    # ---- final fit on the whole reference period
    model = LinearRegression().fit(ref[FEATURES], ref[TARGET])
    coef = dict(zip(FEATURES, model.coef_.tolist()))
    print("\ncoefficients:", {k: round(v, 4) for k, v in coef.items()})
    print(f"coef on log_power: {coef['log_power']:.3f}  -> implied P ~ V^{1 / coef['log_power']:.2f}")

    # ---- daily speed loss for all steady rows in 2024
    x = t[usable & (t.index < "2025-01-01")].copy()
    x["speed_loss_pct"] = 100 * (np.exp(x[TARGET]) / np.exp(model.predict(x[FEATURES])) - 1)  # negative = slower
    daily = x.groupby(x.index.tz_localize(None).normalize()).agg(
        speed_loss_pct=("speed_loss_pct", "median"), laden=("laden", "mean"), n_rows=("speed_loss_pct", "size"))
    daily["laden"] = daily.laden >= 0.5
    daily.index.name = "date"
    daily.round(3).to_csv(os.path.join(PROCESSED, "speed_loss_daily.csv"))
    print(f"\nspeed_loss_daily.csv: {len(daily)} days, {daily.n_rows.sum():,} rows")
    print(daily.groupby([daily.index.to_period("M"), "laden"]).speed_loss_pct.median().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
