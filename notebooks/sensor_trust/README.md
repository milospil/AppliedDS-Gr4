# Sensor trust (Phase 1)

**Goal:** decide which sensors we can trust, and when, before choosing features or building models.

> "Data trustworthiness often prevails over model sophistication." (Tsarsitalidis et al., Real Vessel Data Challenge results, HullPIC 2026)

## Notebooks

| Notebook | Signals | Owner | Status |
|---|---|---|---|
| `01_propulsion.ipynb` | shaft power, rpm, torque, thrust, fuel, engine, counters | mp | done |
| `01b_torquemeter_overview.ipynb` | torquemeter through the year against operating mode; dropout evidence | mp | done |
| `02_speed_navigation.ipynb` | STW, SOG, currents, heading, turn rate, depth; Jan 2025 at sea | mp | done |
| `03_weather.ipynb` | anemometer, Provider S, Provider MB | mp | done |
| `04_noon_reports.ipynb` | noon reports internally and against sensors; fuel meter vs air side | mp | done |
| `05_summary.ipynb` | trust table for all signals × months, all findings | | planned |

**Main results so far** (details and evidence in `docs/data_quality_log.md`):
- Torquemeter power reads ~23 % low from July (DQ-02), drops out to 0 at sea for 656 h (DQ-14), sign fault 10–16 Jul (DQ-03).
- Fuel meter reads ~16 % high from mid-October (DQ-18); noon fuel is not an independent check (DQ-29).
- Speed log reads +1.2 kn high 29 Jul – 16 Oct, proven to be the sensor, not a current (DQ-05).
- Reliable all year: engine rpm, GPS speed, scavenge pressure and turbocharger speed (air side), noon displacement.
- Jan 2025 (blind month): only speed and weather remain while sailing (DQ-07).

Shared code is in `src/trust.py`, so every notebook loads data and defines port, anchor and sailing the same way. Limits are in `physical_limits.md`.

**Before running:** `python src/load_raw.py` (once). The first run of each notebook builds a cache in `data/processed/trust_cache/`.

## Mindset

1. **Guilty until proven innocent.** A value is trusted when an independent source agrees: another sensor, physics or the noon report.
2. **Physics is the referee.** Linked signals that disagree mean at least one is wrong.
3. **Look over time, not only at distributions.** Faults hide in *when*.
4. **Unusual is not wrong.** Astern, port and storms are real. Judge every check per operating mode.
5. **Don't trust labels.** Units, names and signs can be wrong.
6. **Watch for fake values.** Missing data is often written as 0 or a default.
7. **Quantify.** Signal, period, size, number of readings.
8. **Flag, don't fix.** Fixes are a team decision in `docs/data_quality_log.md`.
9. **Check your checks.** Results that look too good or too bad are often a bug.

## The same 8 steps in every notebook

| Step | Question |
|---|---|
| 1. Missing | When and how long is it missing? Several signals at once? |
| 2. Fake values | Zeros or defaults where a real value should be? |
| 3. Stuck | Same value too long **while sailing**? |
| 4. Physical range | Outside what the ship can physically do? |
| 5. Spikes | Impossible jumps between consecutive readings? |
| 6. Cross-checks | Does it agree with physics and other sensors? |
| 7. Step changes | When does a relationship shift suddenly, and by how much? |
| 8. Findings | Trust per month (OK / suspect / bad) and new issue-log rows |
