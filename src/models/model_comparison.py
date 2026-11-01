"""Compare all power models on the same rows, in the same unit.

Run from the repo root:   python src/models/model_comparison.py
Input:  data/processed/model_table.csv, reference/towing_with_psv.csv
Output: printed only

Rows: steady sailing in the reference period where every model has its inputs. Metric: power MAPE, with a 90 %
interval from resampling whole weeks. Two validations: GroupKFold by ISO week, and forward in time
(train Jan - Apr, test May - 23 Jun). The speed model predicts speed, so its prediction is turned into power by
solving its equation for log power.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import GroupKFold

from lightgbm_benchmark import PARAMS, towing_curve

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "..", "..", "data", "processed")
WEATHER = ["trim_m", "wind_head2", "wind_side_kn", "hs2", "wave_head_m"]
SPEED_X = ["log_power", "log_disp"] + WEATHER     # same inputs as speed_baseline.py
POWER_X = ["log_speed", "log_disp"] + WEATHER
GREY_X = ["laden"] + WEATHER                      # same inputs as submission.py
TREE_X = ["speed_kn", "disp_t", "trim_m", "wind_rel_kn", "wind_head_kn", "hs_m", "wave_head_m", "wave_period_s"]
N_FOLDS, N_BOOT, SPLIT = 6, 2000, "2024-05-01"


def predict_all(a, b):
    """Train on a, return predicted power [kW] on b for every model."""
    out = {"Mean power per loading": b.laden.map(a.groupby("laden").power_kw.mean()).to_numpy(float),
           "Towing-tank curve alone": b.p_clean.to_numpy()}
    grey = HuberRegressor(max_iter=500).fit(a[GREY_X].astype(float), np.log(a.power_kw / a.p_clean))
    out["Towing tank + correction"] = b.p_clean.to_numpy() * np.exp(grey.predict(b[GREY_X].astype(float)))
    direct = LinearRegression().fit(a[POWER_X], np.log(a.power_kw))
    out["Log-linear, power from speed"] = np.exp(direct.predict(b[POWER_X]))
    m = LinearRegression().fit(a[SPEED_X], a.log_speed)
    rest = m.intercept_ + b[SPEED_X[1:]].to_numpy() @ m.coef_[1:]
    out["Log-linear, speed from power (main)"] = np.exp((b.log_speed.to_numpy() - rest) / m.coef_[0])
    out["LightGBM"] = lgb.LGBMRegressor(**PARAMS).fit(a[TREE_X], a.power_kw).predict(b[TREE_X])
    return out


def score(y, pred, ref):
    """MAPE overall, laden and ballast, and a 90 % interval from resampling weeks."""
    e = pd.Series(100 * np.abs(pred - y) / y, index=ref.index)
    by_week = e.groupby(ref.week).agg(["sum", "size"])
    rng = np.random.default_rng(0)
    draw = rng.integers(0, len(by_week), (N_BOOT, len(by_week)))
    boot = by_week["sum"].to_numpy()[draw].sum(1) / by_week["size"].to_numpy()[draw].sum(1)
    return dict(MAPE=e.mean(), low=np.quantile(boot, 0.05), high=np.quantile(boot, 0.95),
                laden=e[ref.laden].mean(), ballast=e[~ref.laden].mean())


def main():
    t = pd.read_csv(os.path.join(PROCESSED, "model_table.csv"), index_col="time", parse_dates=True)
    need = list(dict.fromkeys(SPEED_X + POWER_X + GREY_X + TREE_X + ["power_kw"]))
    ref = t[t.steady & t.reference_period & (t.speed_kn > 5) & t[need].notna().all(axis=1)].copy()
    ref["p_clean"] = towing_curve()(ref.speed_kn, ref.disp_t)
    print(f"{len(ref):,} rows in {ref.week.nunique()} weeks (laden {ref.laden.sum():,}, ballast {(~ref.laden).sum():,})")

    exponent = LinearRegression().fit(ref[POWER_X], np.log(ref.power_kw)).coef_[0]
    slope = LinearRegression().fit(ref[SPEED_X], ref.log_speed).coef_[0]
    print(f"implied exponent: power from speed P ~ V^{exponent:.2f}, speed from power P ~ V^{1 / slope:.2f}")

    pred = {}
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(ref, groups=ref.week):
        for k, v in predict_all(ref.iloc[tr], ref.iloc[te]).items():
            pred.setdefault(k, np.full(len(ref), np.nan))[te] = v
    names = list(pred)
    print("\nGroupKFold by week, power MAPE [%]")
    print(pd.DataFrame({k: score(ref.power_kw.to_numpy(), pred[k], ref) for k in names}).T.round(2).to_string())

    a, b = ref[ref.index < SPLIT], ref[ref.index >= SPLIT]
    print(f"\nForward in time: train {len(a):,} rows before {SPLIT}, test {len(b):,} rows (laden {b.laden.sum():,}, ballast {(~b.laden).sum():,})")
    out = predict_all(a, b)
    print(pd.DataFrame({k: score(b.power_kw.to_numpy(), out[k], b) for k in names}).T.round(2).to_string())


if __name__ == "__main__":
    main()
