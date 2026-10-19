"""Fouling growth rate (stub, not implemented yet).

Purpose: estimate how fast speed loss builds up from fouling, to project when it crosses a cleaning threshold.
Input:   data/processed/speed_loss_daily.csv  (from src/models/speed_baseline.py)
Output:  slope of speed loss in % per day (and per 30 days) with a 95% confidence interval,
         for laden and ballast, printed and saved to data/processed/fouling_growth.csv.
Method:  statsmodels OLS of speed_loss_pct on days since 1 July 2024, using days from July to December 2024.
         Laden and ballast fitted separately, or together with a laden dummy. Weight days by n_rows and use
         HAC (Newey-West) standard errors because neighbouring days are correlated.
         Sea temperature may be added here as an explanatory variable (it is excluded only from the baseline).
"""
