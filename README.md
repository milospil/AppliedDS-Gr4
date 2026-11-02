# TDT4259 – Kensas hull and propeller cleaning

Group project, TDT4259 Applied Data Science (NTNU). One year of sensor data (Jan 2024 to Jan 2025) from the bulk carrier *Kensas* (SINTEF / HullPIC Real Vessel Data Challenge). Goal: decide when the hull and propeller should be cleaned.

## Setup

1. Use the Anaconda base environment, then run:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the contents of the Drive `Dataset` folder into `data/raw/`. That's the 13 monthly CSVs, `Metrics registration.xlsx` and the `Noon Reports` folder. `data/raw` and `data/processed` are gitignored, so never commit data.

## Run order

From the repo root:

```bash
python src/make_10min.py
python src/prepare_features.py
python src/models/speed_baseline.py
python src/models/lightgbm_benchmark.py
python src/models/submission.py
python src/correlation_matrix.py
python src/correlation_drivers.py
```

## Scripts

| Script | What it does |
|---|---|
| `src/make_10min.py` | Raw monthly files (long format, 15 s) → `data/processed/kensas_10min.csv`: removes bad readings (`src/data_cleaning/readings.py`), then 10-minute means, one column per signal (circular mean for angles) |
| `src/correlation_matrix.py` | Sensor check: Spearman correlation of the main signals before and after 10 Jul → `plots4report/sensor_correlation_periods.png` |
| `src/correlation_drivers.py` | What drives power, reference period: laden vs ballast matrix and partial correlations with week-resampled intervals → `plots4report/correlation_by_loading.png`, `correlation_drivers.png` |
| `src/sensor_faults_figure.py` | The two main sensor faults (torquemeter ratio, speed log minus GPS) as daily medians → `plots4report/sensor_faults.png` |
| `src/prepare_features.py` | 10-min table + noon reports → `data/processed/model_table.csv`: fixes sensor faults, adds physics features, `steady`, `calm`, `reference_period` and `week` columns |
| `src/models/speed_baseline.py` | **Main model.** Log-linear speed model fitted on the reference period, validated by week. Writes daily speed loss to `data/processed/speed_loss_daily.csv` |
| `src/models/lightgbm_benchmark.py` | Benchmark B2. A LightGBM power model with a good MAPE that goes flat outside the trained speed range (compared with the towing-tank curve) |
| `src/models/submission.py` | Fills the challenge templates: clean scantling baseline and the 10-min timeseries (expected power, fouling, speed loss, uncertainty, flags) in `data/processed/submission/`, plus `plots4report/fouling_trend.png`. Also a VTI-style clean-hull index, not in the template: `data/processed/vti_daily.csv`, `plots4report/vti_trend.png` |
| `src/models/vti_check.py` | Prints checks of the VTI-style index: by speed and power, speed log vs GPS speed, and weekly values |
| `src/models/model_comparison.py` | Prints power error for all models on the same rows: by week and forward in time |
| `src/models/robustness_checks.py` | Prints the fouling indicator per trip and refitted with a 30-min settling filter, MB waves, light wind only and GPS speed |
| `src/models/draft_curves.py` | Speed-power curves per draft (towing tank, reference, current) → `data/processed/curves/` and `plots4report/speed_power_curves.png` |
| `src/models/trend_changepoints.py` | Stub. Rolling median plus CUSUM change points (ruptures) on daily speed loss |
| `src/models/fouling_growth.py` | Stub. OLS growth rate of speed loss, July to December |
| `reference/towing_with_psv.csv` | Clean-hull speed-power tables from the towing tank report (ballast, design, scantling) |

Put personal exploration in `notebooks/`, named with your initials (e.g. `mp_eda.ipynb`).

## Sensor faults

- **Drafts:** the draft sensors read 0 all year, so drafts, trim and displacement come from the noon reports.
- **Power:** from 10 Jul 2024 the torquemeter under-reads by about 18–24% (depending on the reference signal), and by about 23% from mid Oct (DQ-02). After 10 Jul, and when the torquemeter drops out to 0, we use virtual power from scavenge air pressure and engine rpm (1.0% holdout error). Fuel flow is not used: the fuel meter reads high from mid Oct (DQ-18).
- **Speed:** the speed log reads about 0.73 kn too high from 29 Jul to 16 Oct (DQ-05). Speed (`speed_kn`) is the speed log with that offset removed; GPS speed minus the modelled current is only a fallback, because the weather providers' currents explain too little (DQ-19).
- **Wind angle:** averaged with a circular mean, because a plain mean turns head wind into tail wind (DQ-30).
- **Bad readings:** physical limits (DQ-15) and SOG stuck for 30 min or more while moving (DQ-22) are removed before averaging.
- **Weather:** Provider S placeholder rows and Provider MB all-zero currents are set to missing (DQ-06, DQ-26).

## Team rules

- The reference period is 1 Jan to 23 Jun 2024.
- Split by week only (GroupKFold on `week`), never randomly.
- Never use sea temperature, month or any date as a baseline input, because they absorb the fouling trend.
- Never fill missing power or fuel values.
