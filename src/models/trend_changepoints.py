"""Trend and change points in the daily speed loss (stub, not implemented yet).

Purpose: find when the hull and propeller performance degraded, and any sudden steps
         (e.g. a cleaning, a sensor fault or a long port stay), to support the cleaning decision.
Input:   data/processed/speed_loss_daily.csv  (from src/models/speed_baseline.py)
Output:  data/processed/speed_loss_trend.csv (rolling median per day) and a list of change-point
         dates with the mean speed loss before and after each, for laden and ballast separately.
Method:  laden and ballast handled separately (different baselines and noise).
         - Rolling median of speed_loss_pct (e.g. 14-day window, weighted by n_rows) as the trend.
         - CUSUM-style change-point detection with the ruptures package (e.g. Pelt or Binseg with
           an l2 cost), penalty tuned so the reference period shows no change points.
"""
