# Data-quality log

One row per issue found in the raw data. Nothing gets cleaned, dropped or corrected in the pipeline unless it is in this log with status **agreed**.

**How to add an issue**
1. Give it the next ID (DQ-xx) and fill in every column. Write "?" if you don't know yet.
2. **Evidence** must point to something others can rerun: a notebook and section, or a short code snippet.
3. Leave **Decision** and **Status** empty until the team has discussed it.
4. When the fix is in the code, set the status to **implemented** and name the script.

**Columns**

| Column | What to write |
|---|---|
| ID | DQ-01, DQ-02, ... |
| Signal(s) | Short names from `data/processed/signals.csv` (e.g. `shaft_kw`) |
| Period | From – to (UTC), or "all year" |
| Type | One of: missing / gap · stuck or placeholder value · out of range · sign or unit · drift or bias · duplicate · timestamp · other |
| What we see | One or two sentences, with numbers (how many readings, how big) |
| Evidence | Notebook + section, or a code snippet |
| Readings affected | Count, and % of that signal |
| Impact | What breaks if we ignore it (e.g. "power 20% too low after Jul → looks like fouling") |
| Options | Keep · flag only · set to NaN · correct (how) · drop rows · use other signal |
| Decision | Filled in at the team meeting |
| Status | open · agreed · implemented (script) · rejected |
| Found by / date | Initials, date |

## Log

The first rows are **candidates from the first-pass scripts (19 Oct)**. They have not been checked at native resolution yet. Verify or reject each one in the Phase 1 inventory before relying on it.

| ID | Signal(s) | Period | Type | What we see | Evidence | Readings affected | Impact | Options | Decision | Status | Found by / date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DQ-01 | `draft_fwd`, `draft_aft`, `draft_midp`, `draft_mids` | all year | stuck or placeholder value | Draft sensors read 0 | first pass | ? | No draft/displacement from sensors | Use noon-report drafts | | open (to verify) | mp, 19 Oct |
| DQ-02 | `shaft_kw`, `shaft_torque_nm` | from ~10–19 Jul 2024, worse from 16 Oct 2024 | drift or bias | Power reads low, in two steps. Fuel per shaft kWh (SFOC) rises from 203 to 261 g/kWh (+29 %) and then 331 g/kWh (+64 %). The propeller-law coefficient k = P/rpm³ (power-only reference) falls −17 % and then −23 %. Fouling would raise k, but a hull cleaning lowers it, and the drop follows the 25 Jun – 10 Jul port stay (DQ-12). The SFOC jump decides it: a cleaning lowers fuel and power together, so SFOC would stay flat. The torquemeter also returned from that stay with the wrong sign (DQ-03), pointing to work on the sensor. Power reads roughly 17–22 % low from July and more from October; a cleaning in the same stay cannot be ruled out | `01_propulsion.ipynb` §6d, §6i | all power readings after 19 Jul | Power too low → fouling hidden | Correct power per period (e.g. virtual power fitted before the step); see DQ-18 before using fuel for it | | open (verified) | mp, 19 Oct; updated 22 Oct |
| DQ-03 | `shaft_rpm`, `shaft_kw`, `shaft_torque_nm` | 53 days, 8 Jan – 29 Dec 2024; 118 h of 123 h in July (10–16 Jul) | sign or unit | Torquemeter reports astern (rpm about −90, negative power and torque) while the engine runs ahead (`me_rpm` about +90) and the ship sails at 13.5 kn. Engine rpm is the reference: only 722 moments (3 h) are real astern, with engine and shaft both negative | `01_propulsion.ipynb` §2, §6c | 29,415 moments, 123 h | abs() on all negatives would also flip real astern; astern counts are inflated | Flip the sign only where the engine runs ahead; keep real astern | | open (verified) | mp, 19 Oct; updated 22 Oct |
| DQ-04 | `shaft_torque_nm` | all year | sign or unit | Logged as N·m, but power / (2π·rpm/60·torque) = 1.0000 with torque in kN·m (p1–p99: 0.973–1.028) | `01_propulsion.ipynb` §6a | all readings (1.25 M checked) | Wrong by ×1000 if used as N·m | Treat as kN·m | | open (verified) | mp, 19 Oct; updated 22 Oct |
| DQ-05 | `stw_kn` | Aug–Sep 2024 | drift or bias | Speed log about 1.2 kn above GPS speed minus current | first pass, `log_bias_kn` | ? | Speed too high → hides fouling | Use GPS speed minus current | | open (to verify) | mp, 19 Oct |
| DQ-06 | `s_sst`, `s_cur_e`, `s_cur_n`, `s_tp`, `s_hs`, `s_wave_deg` | 65 days, mostly in port | stuck or placeholder value | Identical values in every column: 6.0, −0.4, −0.4, 1.1, 0.01, 10 | `mp_raw_inventory.ipynb` §4 | 1,212 of 9,304 hourly rows, on 65 days | Fake weather/current → wrong corrections | Set to NaN; use Provider MB | | open (to verify) | mp, 20 Oct |
| DQ-07 | all | Jan 2025 | timestamp | `Jan25.csv` has a header, day-first dates, an extra column; no torque, power, fuel or engine load | first pass | ? | No power in Jan 2025 | Exclude from power models | | open (to verify) | mp, 19 Oct |
| DQ-08 | all ship sensors | 24 Sep – 16 Oct 2024 | missing / gap | No sensor data for 22 days (longest gap for shaft, engine and fuel signals: 538.6 h). Weather continues | `mp_raw_inventory.ipynb` §1–2 | 22 days | Hole in the fouling trend right when it matters most | Keep the gap; never interpolate across it | | open | mp, 20 Oct |
| DQ-09 | `shaft_rpm`, `shaft_kw`, `shaft_torque_nm` | — | — | **Merged into DQ-03.** The earlier reading (most negatives outside July are astern manoeuvres) was wrong: with engine rpm as the reference, most negatives are the sign fault and real astern is only 722 moments | `01_propulsion.ipynb` §2 | — | — | — | | rejected (merged into DQ-03) | mp, 20 Oct; updated 22 Oct |
| DQ-10 | `shaft_kwh` | 13 Jan, 21 Jan, 14 Feb, 24 Feb, 1 Mar, 20 Mar, 26 May, 27 Aug 2024 | other | Energy counter goes down 8 times (largest −277,624 kWh on 20 Mar). Otherwise it matches integrated shaft power (ratio 0.999 per hour) | `01_propulsion.ipynb` §6g | 8 steps | Wrong if energy is computed from counter differences | Treat a decrease as a counter reset | | open (verified) | mp, 20 Oct; updated 22 Oct |
| DQ-11 | `stw_kn` | all stops | other | When the ship is still (SOG = 0), the speed log is NOT 0 in 64% of moments (931 h vs 522 h with both 0). Mostly 0.1 kn noise; up to 2–2.8 kn on 25–28 May and 21–23 Nov (water flowing past an anchored or moored ship, or a speed-log offset) | `mp_speed_check.ipynb` §2, §4 | 223,527 moments | Using STW = 0 to find stops misses most of them | Define "still" from SOG = 0 (and rpm = 0), not from STW | | open | mp, 20 Oct |
| DQ-12 | all | 25 Jun – 10 Jul 2024 | other | Two long stops (127 h + 192 h, propeller stopped) right after the reference period ends and right before the torquemeter fault starts (10 Jul). Possible port stay with maintenance, repair or cleaning | `mp_speed_check.ipynb` §4 | about 13 days | If the hull or propeller was cleaned here, the fouling trend restarts on 10 Jul | Ask course staff for the cleaning-events file; compare speed loss before and after | | open | mp, 20 Oct |
| DQ-13 | `fo_kgh`, `me_load_pct` | port / anchor stays | other | With engine and shaft both stopped (engine rpm as reference), fuel flow still reads above 50 kg/h for 759 h (median 303 kg/h, p95 399 kg/h). The main engine drives the shaft directly, so it cannot burn fuel without turning: likely fuel circulating or heated, not consumption | `01_propulsion.ipynb` §2 | 182,186 moments, 759 h | Counted as consumption if summed; hides real warm-up | Use fuel as consumption only when the engine turns; ask what the meter measures | | open (verified) | mp, 20 Oct; updated 22 Oct |
| DQ-14 | `shaft_rpm`, `shaft_kw`, `shaft_torque_nm` (and `thrust_kn`) | 44 periods, mainly 10–23 Jun and 27 Aug – 9 Sep 2024 | missing / gap | Torquemeter reads exactly 0 while the ship sails (median 10.7 kn) with the engine at 88 rpm, 1,126 kg/h fuel and 49 % load: a dropout written as 0. Longest: 169 h unbroken (10–17 Jun). This is the 0 kW line in a power vs engine-rpm plot | `01_propulsion.ipynb` §2, §3 | 656 h (157,494 moments) | Zero power while sailing; overlaps the last weeks of the reference period (June) | Set to NaN; never treat as real power | | open | mp, 22 Oct |
| DQ-15 | `fo_kgh` | 29 Jan, 5 Feb, 20 May, 27 Jun, 29 Jun, 24 Dec 2024 | out of range | Fuel flow up to 4,153 kg/h, above the 2,000 kg/h hard limit (engine max ≈ 1,710 kg/h). 36 of 47 readings on 27 and 29 Jun, during the long port stay (see DQ-12) | `01_propulsion.ipynb` §4 | 47 readings | Spikes distort daily sums | Set to NaN | | open | mp, 22 Oct |
| DQ-16 | `thrust_kn` | all year | sign or unit | Logged thrust about −9,172 'kN' at 4–6 MW, where physics (towing-tank resistance / (1 − t)) gives +700 to +900 kN: negative and about 11× too large. Unit or calibration unknown. Also goes to 0 with the torquemeter dropouts (DQ-14) | `01_propulsion.ipynb` §6f | all readings | Absolute thrust unusable | Use only as a relative signal, or not at all | | open | mp, 22 Oct |
| DQ-17 | `shaft_revs` | all year | sign or unit | Revolution counter rises 59.9× faster than integrated shaft rpm: it counts 1/60 revolution (or rpm-seconds), not revolutions. No unit in the metadata | `01_propulsion.ipynb` §6g | all readings | Wrong if read as revolutions | Divide by 60, or don't use | | open | mp, 22 Oct |
| DQ-18 | `fo_kgh` | from 16 Oct 2024 (after the Sep–Oct data gap, DQ-08) | drift or bias | The second SFOC step (+64 % vs Jan–Jul) is only partly power: k says power reads −23 %, so fuel must read about +26 % higher. Scavenge air pressure per fuel falls −33 % at the same time, which supports a fuel-meter (or engine) change. After the July step fuel moves only +7 % | `01_propulsion.ipynb` §6i | all fuel readings after 16 Oct | A fuel-based power correction would over-correct after October | Check against noon-report fuel (04_noon_reports); treat fuel after 16 Oct as suspect | | open | mp, 22 Oct |
| DQ-19 | | | | | | | | | | | |
